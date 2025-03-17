# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This transform is largely of the upstream `cached_task` transform in Taskgraph.
It exists because there are two features that we need that are missing upstream:
 - The ability to influence the cache digest from parameters.
   (https://github.com/taskcluster/taskgraph/issues/391)
"""

import itertools
from typing import Generator

import taskgraph
from taskgraph.transforms.base import TransformSequence, TransformConfig
from taskgraph.transforms.cached_tasks import order_tasks, format_task_digest
from taskgraph.util.cached_tasks import add_optimization
from taskgraph.util.hash import hash_path
from taskgraph.util.schema import Schema, optionally_keyed_by, resolve_keyed_by
from voluptuous import ALLOW_EXTRA, Any, Required, Optional

from translations_taskgraph.task import UnresolvedCache, TaskDescription
from translations_taskgraph.util.dict_helpers import deep_get

transforms = TransformSequence()

SCHEMA = Schema(
    {
        Required("attributes"): {
            Required("cache"): {
                Required("type"): str,
                Optional("resources"): optionally_keyed_by("provider", [str]),
                Optional("from-parameters"): {
                    str: Any([str], str),
                },
            },
        },
    },
    extra=ALLOW_EXTRA,
)

transforms = TransformSequence()
transforms.add_validate(SCHEMA)


@transforms.add
def resolved_keyed_by_fields(config: TransformConfig, tasks: Generator[dict[str, Any], Any, Any]):
    for task_dict in tasks:
        import json

        print("!!! task_dict", json.dumps(task_dict, indent=2))
        task = TaskDescription.from_dict(task_dict)
        resolve_keyed_by(
            task.attributes.cache,
            "resources",
            item_name=task.description,
            **{"provider": task.attributes.provider is not None},
        )

        yield task.to_dict()


@transforms.add
def add_cache(config: TransformConfig, tasks: Generator[dict[str, Any], Any, Any]):
    for task_dict in tasks:
        task = TaskDescription.from_dict(task_dict)
        cache = task.attributes.cache
        assert cache
        cache_type = cache.type
        cache_resources = cache.resources
        cache_parameters = cache.from_parameters or {}
        digest_data: list[str] = []
        # This is untyped
        print('!!! task_dict["worker"]["command"]', task_dict["worker"])
        digest_data.extend(list(itertools.chain.from_iterable(task_dict["worker"]["command"])))

        if cache_resources:
            for r in cache_resources:
                digest_data.append(hash_path(r))

        if cache_parameters:
            for param, path in cache_parameters.items():
                print("!!! path Is this sometimes not a string?", path)
                if isinstance(path, str):
                    value = deep_get(config.params, path)
                    digest_data.append(f"{param}:{value}")
                else:
                    for choice in path:
                        value = deep_get(config.params, choice)
                        if value is not None:
                            digest_data.append(f"{param}:{value}")
                            break

        task.cache = UnresolvedCache(
            type=cache_type,
            # Upstream cached tasks use "/" as a separator for different parts
            # of the digest. If we don't remove them, caches are busted for
            # anything with a "/" in its label.
            name=task.label.replace("/", "_"),
            digest_data=digest_data,
        )

        yield task.to_dict()


@transforms.add
def cache_task(config: TransformConfig, tasks: Generator[dict[str, Any], Any, Any]):
    if taskgraph.fast:
        for task in tasks:
            yield task
        return

    digests: dict[str, str] = {}
    for task in config.kind_dependencies_tasks.values():
        if "cached_task" in task.attributes:
            digests[task.label] = format_task_digest(task.attributes["cached_task"])

    for task_dict in order_tasks(config, tasks):
        task = TaskDescription.from_dict(task_dict)
        cache = task.cache
        if cache is None:
            yield task
            continue

        dependency_digests: list[str] = []
        for p in task.dependencies:
            if p in digests:
                dependency_digests.append(digests[p])
            else:
                raise Exception("Cached task {} has uncached parent task: {}".format(task["label"], p))
        digest_data = cache.digest_data + sorted(dependency_digests)
        add_optimization(
            config,
            task,
            cache_type=cache.type,
            cache_name=cache.name,
            digest_data=digest_data,
        )
        digests[task.label] = format_task_digest(task.attributes.cached_task)

        yield task.to_dict()
