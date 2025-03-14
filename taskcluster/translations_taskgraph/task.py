from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from translations_taskgraph.util.dataclass_helpers import (
    MixedCaseDataclass,
    KebabDataclass,
    StricterDataclass,
    mixed_casing,
)

@dataclass(kw_only=True)
class Cache(KebabDataclass):
  # Invalidates the cache whenever a parameter from the config changes.
  #
  # {
  #   "marian_args": "training_config.marian-args.training-teacher",
  #   "pretrained_teacher": "training_config.experiment.pretrained-models.train-teacher",
  #   "teacher_mode": "training_config.experiment.teacher-mode"
  # }
  from_parameters: Optional[dict[str, str]] = None
  
  # The list of files that will invalidate the cache when they change.
  resources: list[str]
  
  # This is used to form part of the cache key, and organizes the caches by type.
  # e.g. "train-teacher", "merge-mono", "bicleaner-ai"
  # See: https://firefox-ci-tc.services.mozilla.com/tasks/index/translations.cache.level-1/
  type: str

@dataclass(kw_only=True)
class CachedTask:
    # The hash
    digest: str
    # The name, e.g. "evaluate-backward-sacrebleu-wmt09-en-ru"
    name: str
    # e.g. "evaluate-backwards"
    type: str


@dataclass(kw_only=True)
@mixed_casing(
    kebab=[
        "cleaning-type",
        "dataset-category",
        "fetch-artifact",
        "primary-kind-dependency",
        "toolchain-artifact",
    ]
)
class TaskAttributes(MixedCaseDataclass):
    src_locale: Optional[str] = None
    trg_locale: Optional[str] = None
    always_target: Optional[bool] = None
    artifact_prefix: Optional[str] = None
    best_model: Optional[str] = None
    cache: Optional[Cache] = None
    cached_task: Optional[CachedTask] = None
    # For instance, clean-mono, bicleaner-ai, clean-corpus
    cleaning_type: Optional[str] = None # kebab
    # The name of the dataset, e.g. "Neulab-tedtalks_test-1-eng-rus"
    dataset: Optional[str] = None
    # The type of category.
    dataset_category: Optional[
        Literal["train"] | Literal["mono-src"] | Literal["mono-trg"] | Literal["test"] | Literal["devtest"]
    ] = None
    # For fetch tasks, the fetched artifact public/marian-source.tar.zst
    # kebab cased
    fetch_artifact: Optional[str]   = None
    # e.g. "toolchain-build", "train", "base"
    image_name: Optional[str] = None
    # The name of the task kind, e.g. taskcluster/kinds/{name}/kind.yml
    kind: str
    # The locale for the task, generally for monolingual tasks, e.g. "ru", "lt", "en".
    locale: Optional[str] = None
    # Each task has a primary-kind. This is the kind dependency in each grouping
    # that comes first in the list of supported kinds
    # e.g. "translate-mono-src"
    primary_kind_dependency: Optional[str]  = None # kebab
    # The dataset provider, e.g. "opus", "url", "mtdata", etc.
    provider: Optional[str] = None
    # ["all"]
    run_on_projects: list[str]
    # []
    run_on_tasks_for: list[str]
    #
    shipping_phase: None
    # What stage this task is part of, e.g. "translate-mono-trg"
    stage: Optional[str] = None
    # For toolchain tasks, this is the artifact path.
    toolchain_artifact: Optional[str] = None  # kebab
    # When a task is chunked, this is the chunk number.
    this_chunk: Optional[int] = None
    # This is how many chunks this task was split into.
    total_chunks: Optional[int] = None

DateStamp = dict[str, str]  # e.g. { "relative-datestamp": "0 seconds" }


@dataclass(kw_only=True)
class Treeherder(StricterDataclass):
    collection: dict[str, bool]
    groupName: Optional[str] = None
    groupSymbol: Optional[str] = None
    jobKind: str
    machine: dict[str, str]
    symbol: str
    tier: int


@dataclass(kw_only=True)
class TaskExtra(KebabDataclass):
    index: dict[str, int]
    parent: str
    treeherder: Optional[Treeherder] = None
    treeherder_platform: Optional[str] = None
    chainOfTrust: Optional[dict[str, Any]] = None


@dataclass(kw_only=True)
class Task:
    created: DateStamp
    deadline: DateStamp
    expires: DateStamp
    extra: TaskExtra
    metadata: Any
    payload: Any
    priority: Any
    provisionerId: Any
    routes: Any
    scopes: Any
    tags: Any
    workerType: Any


@dataclass
class TaskRepresentation(StricterDataclass):
    """
    When generating the Taskgraph, this is the representation of a single task. Confusingly
    it also contains a task key. This corresponds to the file:

    "artifacts/full-task-graph.json"

    Which is essentially a:
    dict[str: TaskRepresentation]

    Where the str key is the task label.
    """

    # A dictionary of attributes for this task (used for filtering)
    attributes: TaskAttributes
    # tasks this one depends on, in the form {name: label}, for example
    #   {'build': 'build-linux64/opt', 'docker-image': 'docker-image-desktop-test'}
    dependencies: dict[str, str]
    # A short description of the task.
    description: str
    if_dependencies: list[str]
    # The name of the task kind, e.g. taskcluster/kinds/{name}/kind.yml
    kind: str
    # The label for the task, e.g. "translate-mono-src-en-ru-4/10"
    label: str
    # The list of task labels.
    optimization: Optional[dict[str, list[str]]]
    # Tasks this one may depend on if they are available post optimisation. They are set as
    # a list of tasks label.
    soft_dependencies: list[str]
    task: Task
