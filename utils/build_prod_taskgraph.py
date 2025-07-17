"""
Build the taskgraph from a config. By default it uses the prod config. This is useful for
debugging and iterating on code or a config file to see what tasks will be produced.
"""

import argparse
from pathlib import Path

import yaml
from taskcluster import Hooks
from taskcluster.helper import TaskclusterConfig

ROOT_PATH = Path(__file__).parent.parent
ROOT_URL = "https://firefox-ci-tc.services.mozilla.com"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "config",
        type=Path,
        help="The path the config",
        default=ROOT_PATH / "taskcluster/configs/config.prod.yml",
    )

    args = parser.parse_args()
    config_path: Path = args.config

    with config_path.open() as file:
        config: dict = yaml.safe_load(file)

        taskcluster = TaskclusterConfig(ROOT_URL)
    taskcluster.auth()
    hooks: Hooks = taskcluster.get_service("hooks")
    train_action = get_train_action(decision_task_id)

    # Render the payload using the jsone schema.
    hook_payload = jsone.render(
        train_action["hookPayload"],
        {
            "input": config,
            "taskId": None,
            "taskGroupId": decision_task_id,
        },
    )


if __name__ == "__main__":
    main()
