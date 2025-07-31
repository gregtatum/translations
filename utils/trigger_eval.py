"""
Trigger a training task from the CLI on your current branch.

For example:

  task eval -- --config configs/eval/eval.yml
"""

from pathlib import Path
from taskcluster import Hooks
from taskcluster.helper import TaskclusterConfig
from taskgraph.actions import trigger_action_callback
from taskgraph.util.taskcluster import get_artifact
import taskgraph.parameters
from time import sleep
from typing import Any
from taskgraph.config import load_graph_config
import argparse
import datetime
import jsone
import os
import sys
import yaml

from utils.trigger_training import (
    get_decision_task_push,
    get_decision_task_push_loop,
    get_task_id_from_url,
    run,
    validate_taskcluster_credentials,
)

ROOT_URL = "https://firefox-ci-tc.services.mozilla.com"


def write_to_log(config_path: Path, config: dict, action_task_id: str, branch: str):
    """
    Persist the training log to disk.
    """
    eval_log = Path(__file__).parent / "../trigger-eval.log"
    git_hash = run(["git", "rev-parse", "--short", branch]).strip()

    with open(eval_log, "a") as file:
        lines = [
            "",
            f"config: {config_path}",
            f"langpair: {config['src']}-{config['trg']}",
            f"time: {datetime.datetime.now()}",
            f"action: {ROOT_URL}/tasks/{action_task_id}",
            f"branch: {branch}",
            f"hash: {git_hash}",
        ]
        for line in lines:
            file.write(line + "\n")


def get_eval_action(decision_task_id: str):
    actions_json = get_artifact(decision_task_id, "public/actions.json")

    for action in actions_json["actions"]:
        if action["name"] == "train":
            return action

    print("Could not find the train action.")
    print(actions_json)
    sys.exit(1)


def trigger_eval(decision_task_id: str, config: dict[str, Any]) -> str | None:
    taskcluster = TaskclusterConfig(ROOT_URL)
    taskcluster.auth()
    hooks: Hooks = taskcluster.get_service("hooks")
    eval_action = get_eval_action(decision_task_id)

    # Render the payload using the jsone schema.
    hook_payload = jsone.render(
        eval_action["hookPayload"],
        {
            "input": config,
            "taskId": None,
            "taskGroupId": decision_task_id,
        },
    )

    confirmation = input("\nTrigger evaluation? [Y,n]\n")
    if confirmation and confirmation.lower() != "y":
        return None

    # https://docs.taskcluster.net/docs/reference/core/hooks/api#triggerHook
    response: Any = hooks.triggerHook(
        eval_action["hookGroupId"], eval_action["hookId"], hook_payload
    )

    action_task_id = response["status"]["taskId"]

    print(f"Eval action triggered: {ROOT_URL}/tasks/{action_task_id}")

    return action_task_id


def run_taskgraph(config: dict[str, Any]) -> None:
    print("Running the taskgraph with a --dry_run")

    os.environ["TASK_ID"] = "fake_id"
    os.environ["RUN_ID"] = "0"
    os.environ["TASKCLUSTER_ROOT_URL"] = "https://firefox-ci-tc.services.mozilla.com"

    # Load taskcluster/config.yml
    graph_config = load_graph_config("taskcluster")

    # Add the project's taskgraph directory to the python path, and register
    # any extensions present.
    graph_config.register()

    parameters = taskgraph.parameters.load_parameters_file(None, strict=False)
    parameters.check()

    # This command outputs the stdout. Ignore it here.
    # stdout = sys.stdout
    # devnull = open(os.devnull, "w")
    # sys.stdout = devnull

    # This invokes train_action in taskcluster/translations_taskgraph/actions/train.py
    trigger_action_callback(
        task_group_id=None,
        task_id=None,
        input=config,
        callback="evaluate",
        parameters=parameters,
        root="taskcluster",
        test=True,
    )

    # sys.stdout = stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--config", type=Path, required=True, help="Path the evaluation config")
    parser.add_argument(
        "--branch",
        type=str,
        required=False,
        help="The name of the branch, defaults to the current branch",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Generate the taskgraph, but do not execute it."
    )

    args = parser.parse_args()
    branch: str = args.branch
    config_path: Path = args.config
    dry_run: bool = args.dry_run

    with config_path.open() as file:
        config: dict = yaml.safe_load(file)

    if dry_run:
        run_taskgraph(config)
        return

    validate_taskcluster_credentials()

    timeout = 20
    while True:
        decision_task = get_decision_task_push(branch)

        if decision_task:
            if decision_task.status == "completed" and decision_task.conclusion == "success":
                # The decision task is completed.
                break
            elif decision_task.status == "queued":
                print(f"Decision task is queued, trying again in {timeout} seconds")
            elif decision_task.status == "in_progress":
                print(f"Decision task is in progress, trying again in {timeout} seconds")
            else:
                # The task failed.
                print(
                    f'Decision task is "{decision_task.status}" with the conclusion "{decision_task.conclusion}"'
                )
                sys.exit(1)
        else:
            print(f"Decision task is not available, trying again in {timeout} seconds")

        sleep(timeout)

    decision_task = get_decision_task_push_loop(branch)
    decision_task_id = get_task_id_from_url(decision_task.details_url)

    action_task_id = trigger_evaluation(decision_task_id, config)
    if action_task_id:
        write_to_log(args.config, config, action_task_id, branch)


if __name__ == "__main__":
    main()
