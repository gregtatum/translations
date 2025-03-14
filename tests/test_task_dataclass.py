"""
This test goes through the full task graph
"""
from pathlib import Path
import json
from typing import Any

from translations_taskgraph.task import TaskRepresentation
from fixtures import recursively_remove_none_keys, get_full_taskgraph


def test_task_dataclass():
    # Ensure the task graph is generated.
    get_full_taskgraph()
    
    with (Path(__file__).parent / "../artifacts/full-task-graph.json").open() as file:
        full_task_graph: dict[str, dict[str, Any]] = json.load(file)

    assert isinstance(full_task_graph, dict)

    for name, original_task_dict in full_task_graph.items():
        print("Deserializing Task", name)
        task_representation = TaskRepresentation.from_dict(original_task_dict)
        print("  Serializing Task", name)
        serialized_task_dict = task_representation.to_dict()
        assert serialized_task_dict == recursively_remove_none_keys(original_task_dict)
