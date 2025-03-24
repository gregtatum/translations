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

class Corpora(TypedDict):
    src: str
    trg: str
    aln: str

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
def apply_corpus_continuation(config: TransformConfig, jobs: Iterable[Job]):
    """
    When an existing corpus is available, rewriting the task graph to omit the steps
    needed to generate that corpus.
    
    Rewrites dependencies:
        merge-corpus -> corpus-original-parallel
        merge-mono-trg -> corpus-backtranslations
        cefilter -> corpus-distillation
    """
    import sys

    sys.stdout = sys.__stdout__
    
    training_config: dict = config.params["training_config"]
    corpora: dict = training_config.get("corpora", {})
    
    backtranslations: Optional[Corpora] = corpora.get("backtranslations")
    original_parallel: Optional[Corpora] = corpora.get("original-parallel")
    student_distillation: Optional[Corpora] = corpora.get("student-distillation")
    
    for job in jobs:
        if original_parallel:
            rewrite_dependencies(job, old_task="merge-corpus", new_task="corpus-original-parallel")

        if backtranslations:
            rewrite_dependencies(job, old_task="merge-mono-trg", new_task="corpus-backtranslations")

        if student_distillation:
            rewrite_dependencies(job, old_task="cefilter", new_task="corpus-student-distillation")
        
        yield job
