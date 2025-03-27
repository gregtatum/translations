# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
"""

from typing import Any, Iterable, Literal, Optional, TypedDict
from taskgraph.transforms.base import TransformSequence, TransformConfig

transforms = TransformSequence()


class Job(TypedDict):
    """This is minimally typed to just provide some hints for this transform."""

    name: str
    fetches: dict[str, list[dict[str, Any]]]
    dependencies: dict[str, str]


Vocab = TypedDict(
    "Vocab",
    {
        "src": str,
        "trg": str,
    },
)

Model = TypedDict(
    "Model",
    {
        "urls": list[str],
        "mode": Literal["continue"] | Literal["init"] | Literal["use"],
        "type": Literal["default"] | Literal["opusmt"],
    },
)

Models = TypedDict(
    "Models",
    {
        "backwards": Optional[Model],
        "teacher": Optional[Model],
    },
)

Corpus = TypedDict(
    "Corpus",
    {
        "src": str,
        "trg": str,
        "alignments": Optional[str],
        "tok-src": Optional[str],
        "tok-trg": Optional[str],
    },
)

Corpora = TypedDict(
    "Corpora",
    {
        "backtranslations": Optional[Corpus],
        "original-parallel": Optional[Corpus],
        "student-distillation": Optional[Corpus],
    },
)

Continuation = TypedDict(
    "Continuation",
    {
        "models": Optional[Models],
        "corpora": Optional[Corpora],
        "vocab": Optional[Vocab],
    },
)

transforms = TransformSequence()


def rewrite_dependencies(job: Job, old_task: str, new_task: str):
    # Rewrite the dependences
    # For example rewrite:
    #   dependencies:
    #       merge-corpus: merge-corpus-{src_locale}-{trg_locale}
    # To:
    #   dependencies:
    #       corpus-original-parallel: corpus-original-parallel-{src_locale}-{trg_locale}
    dependencies = job.get("dependencies", {})
    task_dependency = dependencies.pop(old_task, None)
    stage = job.get("attributes", {}).get("stage")
    if stage == "score":
        print("!!! old_task, new_task", old_task, new_task)
        print("!!! dependencies", dependencies)
        print("!!! task_dependency", task_dependency)
        
    if task_dependency:
        print("  !!! job, old_task, new_task", old_task, new_task)
        dependencies[new_task] = new_task + "-{src_locale}-{trg_locale}"

    # Rewrite the fetches name to the new task.
    # For example here:
    #   fetches.merge-corpus -> fetches.corpus-original-parallel
    #
    # fetches:
    #     toolchain:
    #         - marian
    #     merge-corpus:
    #         - artifact: corpus.{src_locale}.zst
    #           extract: false
    #         - artifact: corpus.{trg_locale}.zst
    #           extract: false
    fetches = job.get("fetches", {})
    artifacts = fetches.pop(old_task, None)
    if artifacts:
        fetches[new_task] = artifacts

    # Rewrite the fetches for "from-deps", which is the mechanism used for chunking.
    fetches = job.get("from-deps", {}).get("fetches", {})
    artifacts = fetches.pop(old_task, None)
    if artifacts:
        fetches[new_task] = artifacts

    # Replace any substitution fields that mention the old task.
    substitution_fields: list[str] = job.get("task-context", {}).get("substitution-fields", [])
    for key, value in enumerate(substitution_fields):
        substitution_fields[key] = value.replace(old_task, new_task)


@transforms.add
def apply_continuation(config: TransformConfig, jobs: Iterable[Job]):
    """
    When an existing corpus is available, rewriting the task graph to omit the steps
    needed to generate that corpus.

    Rewrites dependencies:
        merge-corpus -> corpus-original-parallel
        merge-mono-trg -> corpus-backtranslations
        cefilter -> corpus-distillation
    """
    training_config: dict = config.params["training_config"]
    continuation: Continuation = training_config.get("continuation", {})

    corpora = continuation.get("corpora")
    backtranslations = validate_corpora_config(corpora, "backtranslations")
    original_parallel = validate_corpora_config(corpora, "original-parallel")
    student_distillation = validate_corpora_config(corpora, "student-distillation")

    vocab: Optional[Vocab] = continuation.get("vocab")
    models: Optional[Models] = continuation.get("models")
    # If the models are in the "use" mode and are the "default" type, they can be used
    # for changing dependencies of tasks.
    teacher_model: Optional[Model] = None
    backwards_model: Optional[Model] = None

    if models:
        teacher_model = models.get("teacher")
        backwards_model = models.get("backwards")

        if (
            not teacher_model
            or teacher_model["mode"] != "use"
            or teacher_model["type"] != "default"
        ):
            teacher_model = None
        if (
            not backwards_model
            or backwards_model["mode"] != "use"
            or backwards_model["type"] != "default"
        ):
            backwards_model = None
    import sys
    og = sys.stdout
    sys.stdout = sys.__stdout__
    
    print("!!! ----------------- ")
    for job in jobs:
        stage = job.get("attributes", {}).get("stage")
        print("!!! ", stage, job["name"])
        if original_parallel:
            if job["name"] == "merge-corpus":
                continue
            
            rewrite_dependencies(
                job, old_task="merge-corpus", new_task="continuation-corpus-original-parallel"
            )
            if original_parallel.get("aln"):
                if job["name"] == "alignments-original":
                    continue
                rewrite_dependencies(
                    job,
                    old_task="alignments-original",
                    new_task="continuation-corpus-original-parallel",
                )

        if backtranslations:
            if job["name"] == "merge-mono-trg":
                continue
            rewrite_dependencies(
                job, old_task="merge-mono-trg", new_task="continuation-corpus-backtranslations"
            )
            if backtranslations.get("alignments"):
                if job["name"] == "alignments-backtranslated":
                    continue
                rewrite_dependencies(
                    job,
                    old_task="alignments-backtranslated",
                    new_task="continuation-corpus-backtranslations",
                )

        if student_distillation:
            if job["name"] == "cefilter":
                continue

            rewrite_dependencies(
                job, old_task="cefilter", new_task="continuation-corpus-student-distillation"
            )
            if student_distillation.get("alignments"):
                if job["name"] == "alignments-student":
                    continue
                rewrite_dependencies(
                    job,
                    old_task="alignments-student",
                    new_task="continuation-corpus-student-distillation",
                )

        if vocab:
            if job["name"] == "train-vocab":
                continue
            rewrite_dependencies(job, old_task="train-vocab", new_task="continuation-vocab")

        if teacher_model:
            if job["name"] == "train-teacher":
                continue
            rewrite_dependencies(
                job, old_task="train-teacher", new_task="continuation-model-teacher"
            )
        if backwards_model:
            if job["name"] == "train-backwards":
                continue
            rewrite_dependencies(
                job, old_task="train-backwards", new_task="continuation-model-backwards"
            )

        yield job
    sys.stdout = og

def validate_corpora_config(corpora: Optional[Corpora], corpus_key: str) -> Optional[Corpus]:
    """
    Ensure that all of the files are defined if using an existing corpus.
    """
    if not corpora:
        print("!!! no corpora", corpora)
        return

    corpus_files: Optional[dict[str, str]] = corpora.get(corpus_key)

    if not corpus_files:
        return

    def raise_error(file_key: str):
        raise ValueError(f'The "{file_key}" key was not found in the "corpora.{corpus_key}"')

    if "src" not in corpus_files:
        raise_error("src")
    if "trg" not in corpus_files:
        raise_error("trg")

    if "tok-src" in corpus_files or "tok-trg" in corpus_files or "alignments" in corpus_files:
        if "tok-src" not in corpus_files:
            raise_error("tok-src")
        if "tok-trg" not in corpus_files:
            raise_error("tok-src")
        if "alignments" not in corpus_files:
            raise_error("alignments")

    return corpus_files  # type: ignore[reportReturnType]
