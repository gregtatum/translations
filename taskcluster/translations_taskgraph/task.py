from typing import Any, Literal, NamedTuple, Optional

from translations_taskgraph.util.serializable import Casing, casing


@casing(casing=Casing.kebab)
class Cache(NamedTuple):
    # The list of files that will invalidate the cache when they change.
    resources: list[str]

    # This is used to form part of the cache key, and organizes the caches by type.
    # e.g. "train-teacher", "merge-mono", "bicleaner-ai"
    # See: https://firefox-ci-tc.services.mozilla.com/tasks/index/translations.cache.level-1/
    type: str
    # Invalidates the cache whenever a parameter from the config changes.
    #
    # {
    #   "marian_args": "training_config.marian-args.training-teacher",
    #   "pretrained_teacher": "training_config.experiment.pretrained-models.train-teacher",
    #   "teacher_mode": "training_config.experiment.teacher-mode"
    # }
    from_parameters: Optional[dict[str, Any]] = None



@casing()
class CachedTask(NamedTuple):
    # The hash
    digest: str
    # The name, e.g. "evaluate-backward-sacrebleu-wmt09-en-ru"
    name: str
    # e.g. "evaluate-backwards"
    type: str


@casing(
    kebab=[
        "cleaning-type",
        "dataset-category",
        "fetch-artifact",
        "primary-kind-dependency",
        "toolchain-artifact",
    ]
)
class TaskAttributes(NamedTuple):
    src_locale: Optional[str] = None
    trg_locale: Optional[str] = None
    always_target: Optional[bool] = None
    artifact_prefix: Optional[str] = None
    best_model: Optional[str] = None
    cache: Optional[Cache] = None
    cached_task: Optional[CachedTask] = None
    # (kebab-cased) For instance, clean-mono, bicleaner-ai, clean-corpus
    cleaning_type: Optional[str] = None
    # The name of the dataset, e.g. "Neulab-tedtalks_test-1-eng-rus"
    dataset: Optional[str] = None
    # The type of category.
    dataset_category: Optional[
        Literal["train"] | Literal["mono-src"] | Literal["mono-trg"] | Literal["test"] | Literal["devtest"]
    ] = None
    # (kebab-cased) For fetch tasks, the fetched artifact public/marian-source.tar.zst
    fetch_artifact: Optional[str] = None
    # e.g. "toolchain-build", "train", "base"
    image_name: Optional[str] = None
    # The name of the task kind, e.g. taskcluster/kinds/{name}/kind.yml
    kind: Optional[str] = None
    # The locale for the task, generally for monolingual tasks, e.g. "ru", "lt", "en".
    locale: Optional[str] = None
    # (kebab-cased) Each task has a primary-kind. This is the kind dependency in each grouping
    # that comes first in the list of supported kinds
    # e.g. "translate-mono-src"
    primary_kind_dependency: Optional[str] = None
    # The dataset provider, e.g. "opus", "url", "mtdata", etc.
    provider: Optional[str] = None
    # ["all"]
    run_on_projects: Optional[list[str]] = None
    # []
    run_on_tasks_for: Optional[list[str]] = None
    #
    shipping_phase: Optional[str] = None
    # What stage this task is part of, e.g. "translate-mono-trg"
    stage: Optional[str] = None
    # (kebab-cased) For toolchain tasks, this is the artifact path.
    toolchain_artifact: Optional[str] = None
    # When a task is chunked, this is the chunk number.
    this_chunk: Optional[int] = None
    # This is how many chunks this task was split into.
    total_chunks: Optional[int] = None


DateStamp = dict[str, str]  # e.g. { "relative-datestamp": "0 seconds" }


@casing()
class Treeherder(NamedTuple):
    collection: dict[str, bool]
    jobKind: str
    machine: dict[str, str]
    symbol: str
    tier: int
    groupName: Optional[str] = None
    groupSymbol: Optional[str] = None


@casing(casing=Casing.kebab)
class TaskExtra(NamedTuple):
    index: dict[str, int]  # type: ignore[reportIncompatibleMethodOverride]
    parent: str
    treeherder: Optional[Treeherder] = None
    treeherder_platform: Optional[str] = None
    chainOfTrust: Optional[dict[str, Any]] = None


@casing()
class Task(NamedTuple):
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


@casing()
class UnresolvedCache(NamedTuple):
    """
    When resolving the cache when building the full taskgraph, this is the additional
    data, along with the dependencies that Taskgraph will use.
    """

    type: str
    name: str
    digest_data: list[str]


@casing()
class Artifact(NamedTuple):
    # type of artifact -- simple file, or recursive directory
    type: Literal["file"] | Literal["directory"]
    # Task image path from which to read artifact
    path: str
    # Name of the produced artifact.
    name: str


@casing(casing=Casing.kebab)
class WorkerCache(NamedTuple):
    # e.g. "persistent"
    type: str
    # The name of the cache, allowing reuse by subsequent tasks naming the same cache.
    name: str
    # Location in the task image where the cache will be mounted.
    mount_point: str
    # Whether the cache is not used in untrusted environments (like the Try repo).
    skip_untrusted: Optional[bool]


@casing(casing=Casing.kebab)
class Worker(NamedTuple):
    # The name of the docker image, or {"in-tree": "train"}
    docker_image: dict[str, str]
    # The time in seconds that this task can run.
    max_run_time: int
    # The environment variables.
    env: dict[str, str]
    artifacts: list[Artifact]
    # The exit status code(s) that indicates the task should be retried.
    retry_exit_status: list[int]
    # e.g. generic-worker, docker-worker, etc.
    implementation: str
    # e.g. "linux"
    os: str
    taskcluster_proxy: bool
    caches: list[WorkerCache]
    # The command to run.
    command: list[str]


@casing(
    kebab=[
        "worker-type",
        "run-on-tasks-for",
        "run-on-projects",
        "run-on-tasks-for",
        "task-from",
        "always-target",
        "soft-dependencies",
    ]
)
class TaskDescription(NamedTuple):
    """
    When generating the Taskgraph, this is the description of a single task. Confusingly
    it also contains a task key. This corresponds to the file:

    "artifacts/full-task-graph.json"

    Which is essentially a:
    dict[str: TaskDescription]

    Where the str key is the task label.

    For up to d
    src/taskgraph/transforms/task.py
    """

    # The label for the task, e.g. "translate-mono-src-en-ru-4/10"
    label: str
    # A short description of the task.
    description: str
    # A dictionary of attributes for this task (used for filtering)
    attributes: Optional[TaskAttributes]
    # tasks this one depends on, in the form {name: label}, for example
    #   {'build': 'build-linux64/opt', 'docker-image': 'docker-image-desktop-test'}
    dependencies: dict[str, str]
    priority: Optional[
        Literal["highest"]
        | Literal["very-high"]
        | Literal["high"]
        | Literal["medium"]
        | Literal["low"]
        | Literal["very-low"]
        | Literal["lowest"]
    ]
    # Tasks this one may depend on if they are available post optimisation. They
    # are set as a list of tasks label.
    soft_dependencies: Optional[list[str]]
    # Dependencies that must be scheduled in order for this task to run.
    if_dependencies: Optional[list[str]]
    requires: Optional[Literal["all-completed"] | Literal["all-resolved"]]
    # expiration and deadline times, relative to task creation, with units
    # (e.g., "14 days").  Defaults are set based on the project.
    expires_after: Optional[str]
    deadline_after: Optional[str]
    # custom routes for this task; the default treeherder routes will be added
    # automatically.
    scopes: Optional[list[str]]
    # Arbitrary tags, e.g.
    # {
    #   "os": "linux",
    #   "worker-implementation": "docker-worker"
    # }
    tags: Optional[dict[str, str]]
    # custom "task.extra" content
    extra: Optional[dict[str, Any]]
    treeherder: Optional[Treeherder]
    index: Optional[Any] # type: ignore[reportIncompatibleMethodOverride]
    # (kebab_cased)
    run_on_projects: Optional[list[str]]
    # (kebab-cased) Generally used in the CI configuration, e.g. "github-push", "github-pull-request"
    run_on_tasks_for: Optional[list[str]]
    # (kebab-cased)
    run_on_git_branches: Optional[list[str]]
    # Used for Firefox releases.
    shipping_phase: Optional[str]
    # (kebab-cased) The `always-target` attribute will cause the task to be included in the
    # target_task_graph regardless of filtering. Tasks included in this manner
    # will be candidates for optimization even when `optimize_target_tasks` is
    # False, unless the task was also explicitly chosen by the target_tasks
    # method.
    always_target: Optional[bool]
    # The list of task labels.
    optimization: Optional[dict[str, list[str]]]
    # (kebab-cased) The type of worker for the task, e.g. "b-linux-large-gcp"
    # These are defined in taskcluster/config.yml under worker.aliases.
    worker_type: Optional[str]

    # The name of the task kind, e.g. taskcluster/kinds/{name}/kind.yml
    kind: Optional[str]
    task: Optional[Task]
    # This gets removed when a task is resolved by Taskgraph.
    cache: Optional[UnresolvedCache]
    worker: Optional[Worker]
    # (kebab-cased) "kind.yml" or a path to the kind.yml
    task_from: Optional[str]
    routes: Optional[list[str]]
