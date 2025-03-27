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
        
    if task_dependency:
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
    corpus_backtranslations = validate_corpora_config(corpora, "backtranslations")
    corpus_original_parallel = validate_corpora_config(corpora, "original-parallel")
    corpus_student_distillation = validate_corpora_config(corpora, "student-distillation")

    vocab: Optional[Vocab] = continuation.get("vocab")
    models: Optional[Models] = continuation.get("models")
    # If the models are in the "use" mode and are the "default" type, they can be used
    # for changing dependencies of tasks.
    model_teacher: Optional[Model] = None
    model_backwards: Optional[Model] = None

    if models:
        model_teacher = models.get("teacher")
        model_backwards = models.get("backwards")

        if (
            not model_teacher
            or model_teacher["mode"] != "use"
            or model_teacher["type"] != "default"
        ):
            model_teacher = None
        if (
            not model_backwards
            or model_backwards["mode"] != "use"
            or model_backwards["type"] != "default"
        ):
            model_backwards = None
    import sys
    og = sys.stdout
    sys.stdout = sys.__stdout__
    
    for job in jobs:
        stage = job.get("attributes", {}).get("stage")
        name = job["name"]
        
        # Ensure continuation tasks don't get produced unless they are explicitly requested.
        if stage == "continuation":
            if not corpus_backtranslations and name == "backtranslations-{src_locale}-{trg_locale}":
                continue
            if not corpus_original_parallel and name == "original-parallel-{src_locale}-{trg_locale}":
                continue            
            if not corpus_student_distillation and name == "student-distillation-{src_locale}-{trg_locale}":
                continue
            if not vocab and name == "vocab-{src_locale}-{trg_locale}":
                continue
            if not model_teacher and name == "teacher-{src_locale}-{trg_locale}":
                continue            
            if not model_backwards and name == "backwards-{src_locale}-{trg_locale}":
                continue
        
        if corpus_original_parallel:
            if job["name"] == "merge-corpus":
                continue
            
            rewrite_dependencies(
                job, old_task="merge-corpus", new_task="continuation-corpus-original-parallel"
            )
            if corpus_original_parallel.get("aln"):
                if job["name"] == "alignments-original":
                    continue
                rewrite_dependencies(
                    job,
                    old_task="alignments-original",
                    new_task="continuation-corpus-original-parallel",
                )

        if corpus_backtranslations:
            if job["name"] == "merge-mono-trg":
                continue
            rewrite_dependencies(
                job, old_task="merge-mono-trg", new_task="continuation-corpus-backtranslations"
            )
            if corpus_backtranslations.get("alignments"):
                if job["name"] == "alignments-backtranslated":
                    continue
                rewrite_dependencies(
                    job,
                    old_task="alignments-backtranslated",
                    new_task="continuation-corpus-backtranslations",
                )

        if corpus_student_distillation:
            if job["name"] == "cefilter":
                continue

            rewrite_dependencies(
                job, old_task="cefilter", new_task="continuation-corpus-student-distillation"
            )
            if corpus_student_distillation.get("alignments"):
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

        if model_teacher:
            if job["name"] == "train-teacher":
                continue
            rewrite_dependencies(
                job, old_task="train-teacher", new_task="continuation-model-teacher"
            )
        if model_backwards:
            if job["name"] == "train-backwards":
                continue
            rewrite_dependencies(
                job, old_task="train-backwards", new_task="continuation-model-backwards"
            )
            
        # If alignments need to be re-generated, don't attempt to re-use alignment priors.
        if (
            (corpus_student_distillation and not corpus_student_distillation.get("alignments")) or
            (corpus_backtranslations and not corpus_backtranslations.get("alignments"))
        ):
            remove_alignment_priors_dependencies(job)

        yield job
    sys.stdout = og

def remove_alignment_priors_dependencies(job: Job):
    """
    Removes the following in case the corpus.priors are not available.
    
        dependencies:
            alignments-original: alignments-original-{src_locale}-{trg_locale}
        fetches:
            alignments-original:
                - artifact: corpus.priors
    """
    fetches = job.get("fetches")
    dependencies = job.get("dependencies")
    if not dependencies or not fetches:
        return
    alignments = fetches.get("alignments-original", {})
    if len(alignments) == 1 and alignments[0].get("artifact") == "corpus.priors":
        dependencies.pop("alignments-original")
        fetches.pop("alignments-original")
    

def validate_corpora_config(corpora: Optional[Corpora], corpus_key: str) -> Optional[Corpus]:
    """
    Ensure that all of the files are defined if using an existing corpus.
    """
    if not corpora:
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
