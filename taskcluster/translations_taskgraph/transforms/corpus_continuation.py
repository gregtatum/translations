# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
"""

from typing import Any, Iterable, TypedDict
from taskgraph.transforms.base import TransformSequence, TransformConfig
from taskgraph.transforms.run import fetches_schema
from taskgraph.util.schema import Schema
from voluptuous import ALLOW_EXTRA, Optional
import voluptuous

transforms = TransformSequence()

SCHEMA = Schema(
    {
        Optional("corpus-continuation"): {
            Optional("dependencies"): {str: str},
            Optional("replace-fetches"): {str: str},
        },
    },
    extra=ALLOW_EXTRA,
)

CorpusContinuation = TypedDict(
    "CorpusContinuation",
    {"dependencies": dict[str, str], "replace-fetches": dict[str, str]},
    total=False,
)

Job = TypedDict(
    "Job",
    {
        "fetches": dict[str, Any],
        "dependencies": dict[str, str],
        "corpus-continuation": CorpusContinuation,
    },
    total=False,
)

transforms = TransformSequence()
transforms.add_validate(SCHEMA)


@transforms.add
def apply_corpus_continuation(config: TransformConfig, jobs: Iterable[Job]):
    """
    Applies the "corpus-continuation" field from the kind.yml to replace the
    dependencies and fetches.
    """
    import sys

    sys.stdout = sys.__stdout__

    for job in jobs:
        # Remove the corpus continuation properties, as the next transform needs
        # to be cleanly validated.
        corpus_continuation = job.pop("corpus-continuation", None)

        if not corpus_continuation:
            yield job
            continue
        
        # If there are corpus dependencies, use those instead of the original
        # dependencies.
        corpus_dependencies = corpus_continuation.get("dependencies")
        if corpus_dependencies:
            print("!!! corpus_dependencies", corpus_dependencies)
            job["dependencies"] = corpus_dependencies

        # If there are fetches, remap them to another task any of the ones mentioned.
        replace_fetches = corpus_continuation.get("replace-fetches")
        if replace_fetches:
            job_fetches = job.get("fetches")
            assert job_fetches, "Expected to have job dependencies"
            
            for from_name, to_name in replace_fetches.items():
                job_fetches[to_name] = job_fetches.pop(from_name)

        yield job
