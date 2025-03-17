from taskgraph.target_tasks import register_target_task
from taskgraph.taskgraph import TaskGraph
from taskgraph.parameters import Parameters
from taskgraph.config import GraphConfig
from taskgraph.task import Task

from translations_taskgraph.training_config import TrainingConfig


@register_target_task("train-target-tasks")
def train_target_tasks(
    full_task_graph: TaskGraph, parameters: Parameters, graph_config: GraphConfig
):
    training_config = TrainingConfig.from_dict(parameters["training_config"])

    def filter(task: Task):
        # These attributes will be present on tasks from all stages
        if task.attributes.get("stage") != training_config.target_stage:
            return False

        if task.attributes.get("src_locale") != training_config.experiment.src:
            return False

        if task.attributes.get("trg_locale") != training_config.experiment.trg:
            return False

        # Datasets are only applicable to dataset-specific tasks. If these
        # attribute isn't present on the task it can be assumed to be included
        # if the above attributes matched, as it will be a task that is either
        # agnostic of datasets, or folds in datasets from earlier tasks.
        # (Pulling in the appropriate datasets for these task must be handled at
        # the task generation level, usually by the `find_upstreams` transform.)
        if "dataset" in task.attributes:
            dataset_category = task.attributes["dataset-category"]
            for dataset in training_config.datasets[dataset_category]:
                provider, dataset = dataset.split("_", 1)
                # If the task is for any of the datasets in the specified category,
                # it's a match, and should be included in the target tasks.
                if (
                    task.attributes["provider"] == provider
                    and task.attributes["dataset"] == dataset
                ):
                    break

        return True

    return [label for label, task in full_task_graph.tasks.items() if filter(task)]
