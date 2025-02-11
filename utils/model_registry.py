import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import glob
from typing import Any, Callable, Iterable, Optional, Union
from google.cloud import storage
import requests
import taskcluster
import warnings
import shelve
from taskgraph.util.taskcluster import get_artifact, get_artifact_url

# This script is not integrated into a production environment, so suppress the auth warning.
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth._default")

project_name = "translations-data-prod"
bucket_name = "moz-fx-translations-data--303e-prod-translations-data"
root_dir = Path(__file__).parent.parent
model_registry_site = root_dir / "site/model-registry"
training_runs_folder = model_registry_site / "training-runs"

os.environ["TASKCLUSTER_ROOT_URL"] = "https://firefox-ci-tc.services.mozilla.com"


def get_subdirectories(
    bucket: storage.Bucket, prefix: str, cache: Optional[shelve.Shelf]
) -> set[str]:
    cache_key = f"get_subdirectories-{bucket_name}-{prefix}"
    if cache is not None:
        data = cache.get(cache_key, None)
        if data:
            return data

    print(f"Listing {bucket_name}/{prefix}")
    blobs = bucket.list_blobs(
        prefix=prefix,
        # Specify a delimiter to only return the objects in the directory
        delimiter="/",
    )

    prefixes: set[str] = set()

    for page in blobs.pages:
        prefixes.update(page.prefixes)

    if cache is not None:
        cache[cache_key] = prefixes

    return prefixes


@dataclass
class Evaluation:
    chrf: Optional[float]
    bleu: Optional[float]
    comet: Optional[float]

    @staticmethod
    def create():
        return Evaluation(
            chrf=None,
            bleu=None,
            comet=None,
        )


@dataclass
class Corpus:
    """
    Each file contains a newline separated "sentence" in the language. Each line
    in the source matches the translation in the target sentence. There is no tokenization
    that is applied to this corpus.
    """

    source_url: str
    source_bytes: int
    target_url: str
    target_bytes: int

    @staticmethod
    def from_task(
        training_run: "TrainingRun",
        task: Optional[dict],
    ) -> Optional["Corpus"]:
        if task is None:
            print("  [corpus] task missing")
            return None

        taskId = task["status"]["taskId"]
        print("  [corpus]", task_name(task))
        print("  [corpus]", task_url(taskId))
        source_url = get_artifact_url(
            taskId,
            f"public/build/corpus.{training_run.source_lang}.zst",
        )
        target_url = get_artifact_url(
            taskId,
            f"public/build/corpus.{training_run.target_lang}.zst",
        )

        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not source_head.ok or not target_head.ok:
            print("  [corpus] corpus missing")
            return None

        return Corpus(
            source_url=source_url,
            target_url=target_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
        )

    @staticmethod
    def from_mono_tasks(
        training_run: "TrainingRun",
        tasks: list[dict],
    ) -> Optional["Corpus"]:
        """
        The monolingual files are in separate tasks, so the lookups are a bit different.
        """
        source_task = find_latest_task(tasks, match_by_label(r"^collect-mono-trg-"))
        target_task = find_latest_task(tasks, match_by_label(r"^collect-mono-src-"))

        if source_task is None or target_task is None:
            print("  [corpus] mono tasks missing")
            return None

        print("  [corpus]", task_name(source_task))
        print("  [corpus]", task_url(source_task))
        source_url = get_artifact_url(
            source_task["status"]["taskId"],
            f"public/build/mono.{training_run.source_lang}.zst",
        )

        print("  [corpus]", task_name(target_task))
        print("  [corpus]", task_url(target_task))
        target_url = get_artifact_url(
            target_task["status"]["taskId"],
            f"public/build/mono.{training_run.target_lang}.zst",
        )

        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not source_head.ok or not target_head.ok:
            print("  [corpus] corpus missing")
            return None

        return Corpus(
            source_url=source_url,
            target_url=target_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
        )


@dataclass
class WordAlignedCorpus:
    """
    Each file contains a newline separated "sentence" in the language. Each line
    in the source matches the translation in the target sentence. The text is tokenized
    based on the words, where " ▁ " represents a logical word break that has whitespace
    in the original, while " " represents a logical word break that did not have
    whitespace in the original text.

    Example tokenizations:
    "machine translation" -> "machine ▁ translation"
    "机器翻译" -> "机器 翻译"

    The alignments represent how the source sentence's words are aligned to the target
    sentence words. They are tuples of word indexes. A word on the source sentence
    can map to multiple words in the target and vice versa.

    0-3 1-2 1-4 2-0 2-1 2-5
    0-0 1-1 1-2
    0-0 1-0 1-1 2-1 3-2
    """

    source_url: str
    target_url: str
    alignments_url: str

    source_bytes: int
    target_bytes: int
    alignments_bytes: int

    @staticmethod
    def from_task(
        training_run: "TrainingRun", alignments_task: Optional[dict]
    ) -> Optional["WordAlignedCorpus"]:
        if alignments_task is None:
            print("  [word-aligned-corpus] No alignments task")
            return None

        task_id = alignments_task["status"]["taskId"]
        alignments_url = get_artifact_url(alignments_task, "public/build/corpus.aln.zst")
        print("  [word-aligned-corpus]", task_name(alignments_task), task_url(task_id))
        source_url = get_artifact_url(
            task_id,
            f"public/build/corpus.tok-icu.{training_run.source_lang}.zst",
        )
        target_url = get_artifact_url(
            task_id,
            f"public/build/corpus.tok-icu.{training_run.target_lang}.zst",
        )

        alignments_head = requests.head(alignments_url, allow_redirects=True)
        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not alignments_head.ok or not source_head.ok or not target_head.ok:
            print("  [word-aligned-corpus] could not find the files from task")
            return None

        return WordAlignedCorpus(
            source_url=source_url,
            target_url=target_url,
            alignments_url=alignments_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
            alignments_bytes=int(alignments_head.headers.get("content-length", 0)),
        )


@dataclass
class Model:
    date: Optional[datetime]
    config: Optional[dict]
    task_group_id: Optional[dict]
    flores: Optional[Evaluation]
    artifact_folder: Optional[str]
    artifact_urls: list[str]

    @staticmethod
    def create():
        return Model(
            date=None,
            config=None,
            task_group_id=None,
            flores=None,
            artifact_folder=None,
            artifact_urls=[],
        )


@dataclass
class TrainingRun:
    """
    A training run has a unique name, and language pair. It can take multiple task groups
    to complete a training run. This struct represents the collection of all tasks sorted
    by date, with the most recent task being picked for the final artifacts.
    """

    name: str  # e.g. "spring-2024"
    langpair: str
    source_lang: str
    target_lang: str
    task_group_ids: list[str]
    date_started: Optional[datetime]

    # e.g. { "google": 0.8708, ... }
    comet_flores_comparison: dict[str, float]
    bleu_flores_comparison: dict[str, float]

    # Aligned Corpora
    parallel_corpus_aligned: Optional[WordAlignedCorpus]
    backtranslations_corpus_aligned: Optional[WordAlignedCorpus]
    distillation_corpus_aligned: Optional[WordAlignedCorpus]

    # Non-aligned Corpora
    parallel_corpus: Optional[Corpus]
    backtranslations_corpus: Optional[Corpus]
    distillation_corpus: Optional[Corpus]

    # Models
    teacher_1: Optional[Model]
    teacher_2: Optional[Model]
    teacher_ensemble_flores: Optional[Evaluation]

    student: Optional[Model]
    student_finetuned: Optional[Model]
    student_quantized: Optional[Model]
    student_exported: Optional[Model]

    @staticmethod
    def create(name: str, task_group_ids: list[str], langpair):
        source_lang, target_lang = langpair.split("-")
        return TrainingRun(
            name=name,
            langpair=langpair,
            source_lang=source_lang,
            target_lang=target_lang,
            task_group_ids=task_group_ids,
            date_started=None,
            comet_flores_comparison={},
            bleu_flores_comparison={},
            # Aligned Corpora
            parallel_corpus_aligned=None,
            backtranslations_corpus_aligned=None,
            distillation_corpus_aligned=None,
            # Non-aligned Corpora
            parallel_corpus=None,
            backtranslations_corpus=None,
            distillation_corpus=None,
            # Models
            teacher_1=None,
            teacher_2=None,
            teacher_ensemble_flores=None,
            student=None,
            student_finetuned=None,
            student_quantized=None,
            student_exported=None,
        )


class JsonEncoder(json.JSONEncoder):
    """Converts a dataclass into a JSON serializable struct"""

    def default(self, o: Any):
        if is_dataclass(o):
            return asdict(o)  # type: ignore
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def get_training_runs_by_langpair(
    cache: Optional[shelve.Shelf],
) -> dict[str, list[TrainingRun]]:
    """
    Training runs are stored in the following structure. Extract out the information into
    a structured format.

    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_DtSyAeaVRoGNZDnUKscGWw/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_NPlcq4JZRRCj0ksitTDSVw/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_Ov3G4D_DRJa-4qTlILkPhg/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_TSvbd6EuTmGayUQtIP3Lbg/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_bQQme71PS4eZRDl3NM-kgA/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_bbjDBFoDTNGSUo2if3ET_A/
    """
    client = storage.Client(project=project_name)
    bucket = client.get_bucket(bucket_name)

    # e.g. ['en-fi', 'en-da', 'en-hu', 'en-sr', ...]
    langpairs = [
        model_prefix.split("/")[1] for model_prefix in get_subdirectories(bucket, "models/", cache)
    ]

    training_runs_by_langpair: dict[str, list[TrainingRun]] = {}
    training_runs_by_name: dict[str, TrainingRun] = {}

    for langpair in langpairs:
        training_runs: list[TrainingRun] = []
        training_runs_by_langpair[langpair] = training_runs

        # e.g { "models/en-lv/spring-2024_J3av8ewURni5QQqP2u3QRg/", ... }
        for task_group_prefix in get_subdirectories(bucket, f"models/{langpair}/", cache):
            # e.g. "spring-2024_J3av8ewURni5QQqP2u3QRg"
            name_task_group_tuple = task_group_prefix.split("/")[2]

            # Task Group IDs are 22 letters long, and contain "_", so don't split on "_"
            # which is used as a delimiter. Only rely on the hard coded length, which
            # is simpler than using this regex:
            # https://github.com/taskcluster/taskcluster/blob/3249015448f795d30ebbc3c3304c3b6d86c39284/services/auth/schemas/constants.yml#L11-L12
            name = name_task_group_tuple[:-23]
            task_group_id = name_task_group_tuple[-22:]
            key = f"{langpair} {name}"

            training_task_group = training_runs_by_name.get(key, None)
            if training_task_group:
                training_task_group.task_group_ids.append(task_group_id)
            else:
                training_task_group = TrainingRun.create(
                    name=name, task_group_ids=[task_group_id], langpair=langpair
                )
                training_runs.append(training_task_group)
                training_runs_by_name[key] = training_task_group

    return training_runs_by_langpair


def print_training_runs_tree(training_runs_by_langpair: dict[str, list[TrainingRun]]):
    last_langpair_index = len(training_runs_by_langpair) - 1

    print("\nTraining Runs")
    for langpair_index, (langpair, training_runs) in enumerate(training_runs_by_langpair.items()):
        prefix_langpair = "└──" if langpair_index == last_langpair_index else "├──"
        print(f"{prefix_langpair} {langpair}")

        last_run_index = len(training_runs) - 1
        for run_index, training_run in enumerate(training_runs):
            prefix_run = "└──" if run_index == last_run_index else "├──"
            connector = "    " if langpair_index == last_langpair_index else "│   "
            print(f"{connector}{prefix_run} {training_run.name}")

            task_groups = training_run.task_group_ids
            last_task_index = len(task_groups) - 1
            for task_index, task_group_id in enumerate(task_groups):
                prefix_task = "└──" if task_index == last_task_index else "├──"
                sub_connector = "    " if run_index == last_run_index else "│   "
                print(
                    f"{connector}{sub_connector}{prefix_task} https://firefox-ci-tc.services.mozilla.com/tasks/groups/{task_group_id}"
                )


# The JSON from the evaluations repo.
# {
#   "en-af": {
#     "flores-dev": { "nllb": 0.8566, "google": 0.8708, ... }
#     "flores-test": {...}
#     ...
#   },
#   ...
# }
EvaluationJson = dict[str, dict[str, dict[str, float]]]

Task = dict[str, dict]


def iterate_training_runs(
    training_runs_by_langpair: dict[str, list[TrainingRun]], cache: Optional[shelve.Shelf]
):
    """
    Reduce the complexity required for iterating over the training runs and their tasks.
    """
    for training_runs in training_runs_by_langpair.values():
        for training_run in training_runs:
            yield training_run, get_tasks_in_all_runs(training_run, cache)


def collect_training_runs(
    training_runs_by_langpair: dict[str, list[TrainingRun]], cache: Optional[shelve.Shelf]
):
    client = storage.Client(project=project_name)
    bucket = client.get_bucket(bucket_name)

    comet_results_by_langpair: EvaluationJson = fetch_json(
        "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/comet-results.json"
    )
    bleu_results_by_langpair: EvaluationJson = fetch_json(
        "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/bleu-results.json"
    )
    # chrF is not computed

    for training_run, tasks in iterate_training_runs(training_runs_by_langpair, cache):
        training_run_json = (
            training_runs_folder / f"{training_run.name}-{training_run.langpair}.json"
        )
        if training_run_json.exists():
            print("Already processed", training_run.name, training_run.langpair)
            continue

        print("Processing", training_run.name, training_run.langpair)
        collect_models(tasks, training_run, bucket)
        collect_flores_comparisons(
            training_run, comet_results_by_langpair, bleu_results_by_langpair
        )
        collect_corpora(training_run, tasks)

        with open(training_run_json, "w") as file:
            json.dump(training_run, file, cls=JsonEncoder, indent=2)


def get_tasks_in_all_runs(training_run: TrainingRun, cache: Optional[shelve.Shelf]) -> list[Task]:
    queue = taskcluster.Queue(options={"rootUrl": "https://firefox-ci-tc.services.mozilla.com"})

    tasks_in_all_runs: list[Task] = []
    for task_group_id in training_run.task_group_ids:
        cache_key = f"list_task_group-{task_group_id}"
        tasks = None
        prefix = "Fetched"
        if cache is not None:
            tasks = cache.get(cache_key, None)
            if tasks is not None:
                prefix = "Cached"

        if tasks is None:
            tasks = []
            try:
                list_task_group: Any = queue.listTaskGroup(task_group_id)
                tasks.extend(list_task_group["tasks"])

                # Do a bounded lookup of more tasks. 10 should be a reasonable limit.
                for _ in range(10):
                    if not list_task_group.get("continuationToken", None):
                        break
                    list_task_group: Any = queue.listTaskGroup(
                        task_group_id,
                        continuationToken=list_task_group["continuationToken"],
                    )
                    tasks.extend(list_task_group["tasks"])
            except taskcluster.exceptions.TaskclusterRestFailure as error:
                # 404 errors indicate expired task groups.
                if error.status_code == 404:
                    print("Task group expired:", task_group_id)
                else:
                    raise error

        if cache is not None:
            cache[cache_key] = tasks

        tasks_in_all_runs.extend(tasks)

        print(f"{prefix} {len(tasks)} tasks from {task_group_id}")
        for task in tasks:
            date = str_to_datetime(task["task"]["created"])

            if training_run.date_started is None or date < training_run.date_started:
                training_run.date_started = date

    return tasks_in_all_runs


def collect_flores_comparisons(
    training_run: TrainingRun,
    comet_results_by_langpair: EvaluationJson,
    bleu_results_by_langpair: EvaluationJson,
):
    comet_results = comet_results_by_langpair.get(training_run.langpair, None)
    if comet_results:
        training_run.comet_flores_comparison = comet_results["flores-dev"]
    bleu_results = bleu_results_by_langpair.get(training_run.langpair, None)
    if bleu_results:
        training_run.bleu_flores_comparison = bleu_results["flores-dev"]


def collect_models(tasks: list[Task], training_run: TrainingRun, bucket: storage.Bucket):
    """
    Lookup models from Google Cloud Storage.
    """
    train_teacher_1 = find_latest_task(tasks, match_by_label(r"^train-teacher-.*-1"))
    if train_teacher_1:
        training_run.teacher_1 = get_model(
            train_teacher_1,
            training_run,
            bucket,
            tasks,
            tc_model_name="teacher",
            gcs_model_name="teacher0",
            gcs_eval_name="teacher0",
        )

    train_teacher_2 = find_latest_task(tasks, match_by_label(r"^train-teacher-.*-2"))
    if train_teacher_2:
        training_run.teacher_2 = get_model(
            train_teacher_2,
            training_run,
            bucket,
            tasks,
            tc_model_name="teacher",
            gcs_model_name="teacher1",
            gcs_eval_name="teacher1",
        )

    train_student_task = find_latest_task(tasks, match_by_label(r"^train-student-"))
    if train_student_task:
        training_run.student = get_model(
            train_student_task,
            training_run,
            bucket,
            tasks,
            tc_model_name="student",
            gcs_model_name="student",
            gcs_eval_name="student",
        )
    student_quantize_task = find_latest_task(tasks, match_by_label(r"^quantize-"))
    if student_quantize_task:
        training_run.student_quantized = get_model(
            student_quantize_task,
            training_run,
            bucket,
            tasks,
            tc_model_name="quantized",
            gcs_model_name="quantized",
            gcs_eval_name="speed",
        )
    student_export_task = find_latest_task(tasks, match_by_label(r"^export-"))
    if student_export_task:
        training_run.student_exported = get_model(
            student_export_task,
            training_run,
            bucket,
            tasks,
            tc_model_name="export",
            gcs_model_name="exported",
            gcs_eval_name="exported",
        )
        if training_run.student_quantized:
            # The export step doesn't have an explicit eval, so take
            # the one from the quantized step.
            training_run.student_exported.flores = training_run.student_quantized.flores


def collect_corpora(training_run: TrainingRun, tasks: list[Task]):
    # Find the word aligned corpora.
    training_run.parallel_corpus_aligned = WordAlignedCorpus.from_task(
        training_run,
        find_latest_task(tasks, match_by_label(r"^alignments-original-")),
    )
    training_run.backtranslations_corpus_aligned = WordAlignedCorpus.from_task(
        training_run,
        find_latest_task(tasks, match_by_label(r"^alignments-backtranslated-")),
    )
    training_run.distillation_corpus_aligned = WordAlignedCorpus.from_task(
        training_run,
        find_latest_task(tasks, match_by_label(r"^alignments-student-")),
    )

    # Find the raw corpora
    training_run.parallel_corpus = Corpus.from_task(
        training_run,
        find_latest_task(tasks, match_by_label(r"^merge-corpus-")),
    )
    training_run.backtranslations_corpus = Corpus.from_mono_tasks(
        training_run,
        tasks,
    )
    training_run.distillation_corpus = Corpus.from_task(
        training_run,
        find_latest_task(tasks, match_by_label(r"^cefilter-")),
    )


def get_model(
    task: dict,
    training_run: TrainingRun,
    bucket: storage.Bucket,
    tasks_in_all_runs: list[dict],
    # The model name in Taskcluster tasks.
    tc_model_name: str,
    # The model directory name in GCS.
    gcs_model_name: str,
    # The model directory name in GCS.
    gcs_eval_name: str,
):
    task_group_id = task["status"]["taskGroupId"]

    model = Model.create()
    model.config = get_config(task_group_id)
    model.task_group_id = task_group_id
    model.date = get_completed_time(task)

    flores_blob = get_flores_eval_blob(
        training_run,
        task_group_id,
        gcs_eval_name,
        tc_model_name,
        bucket,
    )
    if not flores_blob:
        # The eval wasn't in the same task group as the training.
        label_regex = f"^evaluate-{tc_model_name}-flores-"
        # These don't follow the same format, so adjust the regex. These need to match
        # the current naming convention, and the older one:
        #  - evaluate-teacher-flores-devtest-sk-en-1
        #  - evaluate-teacher-flores-devtest-sk-en-1/2
        if gcs_model_name == "teacher0":
            label_regex = r"^evaluate-teacher-flores-.*1"
        if gcs_model_name == "teacher1":
            label_regex = r"^evaluate-teacher-flores-.*2"

        eval_task = find_latest_task(tasks_in_all_runs, match_by_label(label_regex))
        if eval_task:
            flores_blob = get_flores_eval_blob(
                training_run,
                eval_task["status"]["taskGroupId"],
                gcs_eval_name,
                tc_model_name,
                bucket,
            )

    if flores_blob:
        print(f"  [model] loading {tc_model_name} evals")
        flores_results = json.loads(flores_blob.download_as_text())

        # Older evaluations may not have COMET integrated.
        comet = None
        if "comet" in flores_results:
            comet = flores_results["comet"]["score"] * 100.0

        model.flores = Evaluation(
            chrf=flores_results["chrf"]["score"],
            bleu=flores_results["bleu"]["score"],
            comet=comet,
        )
    else:
        print(f"  [model] {tc_model_name} evals not found")

    prefix = (
        f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/{gcs_model_name}/"
    )
    model.artifact_folder = f"https://storage.googleapis.com/{bucket_name}/{prefix}"

    # List all of the artifacts.
    print(f"  [model] listing {tc_model_name} files - {model.artifact_folder}")
    blobs: Optional[Iterable[storage.Blob]] = bucket.list_blobs(prefix=prefix)
    if blobs:
        model.artifact_urls = [
            f"https://storage.googleapis.com/{bucket_name}/{blob.name}" for blob in blobs
        ]
    else:
        print(f"  [model] no {tc_model_name} files found")

    return model


def fetch_json(url: str):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_flores_eval_blob(
    training_run: TrainingRun,
    task_group_id: str,
    gcs_eval_name: str,
    tc_model_name: str,
    bucket: storage.Bucket,
) -> Optional[storage.Blob]:
    # First try with just the source language.
    blob_url = (
        f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"
        f"evaluation/{gcs_eval_name}/"
        f"{tc_model_name}-flores-devtest-{training_run.source_lang}_devtest.metrics.json"
    )
    blob = bucket.get_blob(blob_url)
    if not blob:
        # Also check with the langpair.
        blob_url = (
            f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"
            f"evaluation/{gcs_eval_name}/"
            f"{tc_model_name}-flores-devtest-{training_run.langpair}_devtest.metrics.json"
        )
        blob = bucket.get_blob(blob_url)

    return blob


def get_completed_time(task: dict) -> Optional[datetime]:
    for run in reversed(task["status"]["runs"]):
        if run["state"] == "completed":
            return str_to_datetime(run["resolved"])


def get_config(action_task_id: dict) -> Optional[dict]:
    try:
        return get_artifact(action_task_id, "public/parameters.yml")["training_config"]
    except Exception:
        return None


def _match_by_label(task: dict, pattern: str) -> bool:
    runs = task["status"]["runs"]
    if not runs:
        return False

    last_run = runs[-1]
    if last_run["state"] != "completed":
        return False

    return re.match(pattern, task["task"]["metadata"]["name"]) is not None


def match_by_label(pattern: str):
    return lambda task: _match_by_label(task, pattern)


def _find_task(
    tasks: list[dict], match: Callable[[dict], bool], min_or_max: Any
) -> Optional[dict]:
    tasks = [task for task in tasks if match(task)]
    if not tasks:
        return None

    return min_or_max(
        tasks, key=lambda task: datetime.strptime(task["task"]["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
    )


def find_latest_task(tasks: list[dict], match: Callable[[dict], bool]) -> Optional[dict]:
    return _find_task(tasks, match, max)


def find_earliest_task(tasks: list[dict], match: Callable[[dict], bool]):
    return _find_task(tasks, match, min)


def str_to_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")


def task_name(task: dict) -> str:
    """Helper to get the task name"""
    return task["task"]["metadata"]["name"]


def task_url(task_id_or_task: Union[str, dict]) -> str:
    """Helper to get the task url"""
    task_id = (
        task_id_or_task
        if isinstance(task_id_or_task, str)
        else task_id_or_task["status"]["taskId"]
    )
    return f"https://firefox-ci-tc.services.mozilla.com/tasks/{task_id}"


def save_training_run_listing() -> None:
    training_runs = sorted(path.name for path in training_runs_folder.glob("*.json"))
    with open(model_registry_site / "training-runs-listing.json", "w", encoding="utf-8") as f:
        json.dump(training_runs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--cache", action="store_true", help="Cache the remote calls")
    parser.add_argument("--clear_cache", action="store_true", help="Clears the cache")
    args = parser.parse_args()

    cache = None
    if args.cache:
        print("Using a cache")
        cache = shelve.open("data/model_registry.pickle")

    if args.clear_cache:
        print("Clearing the cache")
        if cache is None:
            cache = shelve.open("data/model_registry.pickle")
            cache.clear()
            cache.close()
            cache = None
        else:
            cache.clear()

    training_runs_by_langpair = get_training_runs_by_langpair(cache)
    print_gcs_tree(training_runs_by_langpair)
    collect_training_runs(training_runs_by_langpair, cache)
    save_training_run_listing()

    with open(model_registry_site / "models.json", "w") as outfile:
        json.dump(training_runs_by_langpair, outfile, cls=JsonEncoder, indent=2)

    if cache is not None:
        cache.close()


if __name__ == "__main__":
    main()
