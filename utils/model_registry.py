import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from google.cloud import storage
import taskcluster
import warnings
import shelve

# This script is not integrated into a production environment, so suppress the auth warning.
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth._default")

project_name = "translations-data-prod"
bucket_name = "moz-fx-translations-data--303e-prod-translations-data"
root_dir = Path(__file__).parent.parent


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
class TrainingTaskGroup:
    name: str
    task_group_ids: list[str]


def get_training_runs_by_langpair(
    cache: Optional[shelve.Shelf],
) -> dict[str, list[TrainingTaskGroup]]:
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

    training_runs_by_langpair: dict[str, list[TrainingTaskGroup]] = {}
    training_runs_by_name: dict[str, TrainingTaskGroup] = {}

    for langpair in langpairs:
        training_runs: list[TrainingTaskGroup] = []
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
                training_task_group = TrainingTaskGroup(
                    name=name,
                    task_group_ids=[task_group_id],
                )
                training_runs.append(training_task_group)
                training_runs_by_name[key] = training_task_group

    return training_runs_by_langpair


def print_models_tree(training_runs_by_langpair: dict[str, list[TrainingTaskGroup]]):
    last_langpair_index = len(training_runs_by_langpair) - 1

    print("\nModels")
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


def get_taskgraphs(
    training_runs_by_langpair: dict[str, list[TrainingTaskGroup]], cache: Optional[shelve.Shelf]
):
    options = {"rootUrl": "https://firefox-ci-tc.services.mozilla.com"}
    queue = taskcluster.Queue(options=options)
    for langpair, training_runs in training_runs_by_langpair.items():
        for training_run in training_runs:
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

                print(f"{prefix} {len(tasks)} tasks from {task_group_id}")


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
            cache = shelve.open("model_registry")
            cache.clear()
            cache.close()
            cache = None
        else:
            cache.clear()

    training_runs_by_langpair = get_training_runs_by_langpair(cache)
    print_models_tree(training_runs_by_langpair)
    get_taskgraphs(training_runs_by_langpair, cache)

    if cache is not None:
        cache.close()


if __name__ == "__main__":
    main()
