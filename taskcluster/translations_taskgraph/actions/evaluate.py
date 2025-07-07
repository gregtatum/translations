# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Any
from taskgraph.actions.registry import register_callback_action
from translations_taskgraph.actions.train import can_train
from taskgraph.config import GraphConfig
from taskgraph.parameters import Parameters
from taskgraph.create import create_tasks


def get_evaluate_schema(evaluate_config):
    return {
        "type": "object",
        "properties": {},
        "required": [],
    }


@register_callback_action(
    name="evaluate",
    title="Evaluate",
    symbol="evaluate",
    description="Evaluate a language model.",
    cb_name="train",
    permission="train",
    order=501,
    context=[],
    available=can_train,
    schema=get_evaluate_schema,
)
def train_action(
    parameters: Parameters,
    graph_config: GraphConfig,
    input: dict[str, Any],
    task_group_id: str,
    task_id: str,
) -> None:
    from taskgraph.decision import taskgraph_decision

    taskgraph_decision({"root": graph_config.root_dir}, parameters=parameters)
    create_tasks(
        graph_config,
        [],
        full_task_graph,
        label_to_task_id,
        parameters,
        decision_task_id,
    )
