# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
"""

from typing import Any, Iterable, Optional, TypedDict
from taskgraph.transforms.base import TransformSequence, TransformConfig

transforms = TransformSequence()


class Job(TypedDict):
    """This is minimally typed to just provide some hints for this transform."""
    fetches: dict[str, list[dict[str, Any]]]
    dependencies: dict[str, str]

Corpus = TypedDict("Corpus", {
    "src": str,
    "trg": str,
    "alignments": Optional[str],
    "tok-src": Optional[str],
    "tok-trg": Optional[str],
})

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
    merge_corpus_dependency = dependencies.pop(old_task, None)
    if merge_corpus_dependency:
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
    continuation: dict = training_config.get("continuation", {})
    
    backtranslations = validate_corpora_config(continuation, "backtranslations")
    original_parallel = validate_corpora_config(continuation, "original-parallel")
    student_distillation = validate_corpora_config(continuation, "student-distillation")
    vocab = continuation.get("vocab")
    
    for job in jobs:
        if original_parallel:
            rewrite_dependencies(job, old_task="merge-corpus", new_task="continuation-corpus-original-parallel")
            if original_parallel.get("aln"):
                rewrite_dependencies(job, old_task="alignments-original", new_task="continuation-corpus-original-parallel")

        if backtranslations:
            rewrite_dependencies(job, old_task="merge-mono-trg", new_task="continuation-corpus-backtranslations")
            if backtranslations.get("alignments"):
                rewrite_dependencies(job, old_task="alignments-backtranslations", new_task="continuation-corpus-backtranslations")
                
        if student_distillation:
            rewrite_dependencies(job, old_task="cefilter", new_task="continuation-corpus-student-distillation")
            if student_distillation.get("alignments"):
                rewrite_dependencies(job, old_task="alignments-student", new_task="continuation-corpus-student-distillation")
                
        if vocab:
            rewrite_dependencies(job, old_task="train-vocab", new_task="continuation-corpus-vocab")
        
        yield job


def validate_corpora_config(continuation: dict[str, Optional[Corpus]], corpus_key: str) -> Optional[Corpus]:
    """
    Ensure that all of the files are defined if using an existing corpus.
    """
    if not continuation:
        return
    
    corpus_files: Optional[dict[str, str]] = continuation.get(corpus_key) # type: ignore[reportAssignmentType]
    
    if not corpus_files:
        return
    
    def raise_error(file_key: str):
        raise ValueError(f"The \"{file_key}\" key was not found in the \"corpora.{corpus_key}\"")
    
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

    return continuation[corpus_key]
