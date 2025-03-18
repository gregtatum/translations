"""
This file contains the canonical format for the training config. It's defined as a
type-safe dataclass, and then shared in other parts of the codebase.
"""

from pathlib import Path
from typing import Any, TypedDict, Optional, Literal

import yaml
from translations_taskgraph.util import serializable

MarianArgs = TypedDict(
    "MarianArgs",
    {
        "training-backward": dict[str, str],
        "training-teacher": dict[str, str],
        "training-student": dict[str, str],
        "training-student-finetuned": dict[str, str],
        "decoding-backward": dict[str, str],
        "decoding-teacher": dict[str, str],
    },
    total=False,
)

# Limits for monolingual datasets.
MonoMaxSentences = TypedDict(
    "MonoMaxSentences",
    {
        # The limit sentence size for the total dataset.
        "total": int,
        # Limits per downloaded dataset.
        "per-dataset": int
    },
)


CleanerThresholds = TypedDict(
    "CleanerThresholds",
    {
        "default-threshold": float,
        "dataset-thresholds": dict[str, float],
    },
    total=False
)

MonocleanerConfig = TypedDict(
  "MonocleanerConfig",
  {
    "mono-src": CleanerThresholds,
    "mono-trg": CleanerThresholds,
  },
)

HpltMinDocScore = TypedDict(
  "HpltMinDocScore",
  {
    "mono-src": float,
    "mono-trg": float,
  },
)

    # Pre-trained models use URLs as they are flexible to continue training
    # from either long-term bucket storage or from tasks in Taskcluster.
PretrainedModel = TypedDict(
  "PretrainedModel",
  {
    "urls": list[str],
    "mode": Literal["continue", "init", "use"],
    "type": Literal["default", "opusmt"],
  },
)


PretrainedModels = TypedDict(
  "PretrainedModels",
  {
    "train-backwards": Optional[PretrainedModel],
    "train-teacher": Optional[PretrainedModel],
  },
  total=False,
)

Experiment = TypedDict(
  "Experiment",
  {
    # A name for the experiment.
    "name": str,
    # The source locale to train.
    "src": str,
    # The target locale to train.
    "trg": str,
    # Number of teachers to train in an ensemble.
    "teacher-ensemble": int,
    # Teacher training mode.
    "teacher-mode": Literal["one-stage", "two-stage"],
    # Translate with either Marian or CTranslate2.
    "teacher-decoder": Literal["marian", "ctranslate2"],
    # The student model architecture.
    "student-model": Literal["tiny", "base"],

    # Limits for monolingual datasets.
    "mono-max-sentences-trg": MonoMaxSentences,
    "mono-max-sentences-src": MonoMaxSentences,

    # Sentence Piece sample size.
    "spm-sample-size": int,
    # The metric to use for the best model.
    "best-model": Literal[
      "cross-entropy",
      "ce-mean-words",
      "perplexity",
      "valid-script",
      "translation",
      "bleu",
      "bleu-detok",
      "bleu-segmented",  # (deprecated, same as bleu)
      "chrf",
    ],
    # Use OpusCleaner to clean the corpus.
    "use-opuscleaner": Literal["true", "false"],
    # Indicates whether to use dataset specific configs.
    "opuscleaner-mode": Literal["custom", "defaults"],
    # Thresholds for running bicleaner-ai.
    "bicleaner": CleanerThresholds,
    "monocleaner": MonocleanerConfig,
    "hplt-min-doc-score": HpltMinDocScore,
    # Instead of training models from scratch, use pre-trained models.
    "pretrained-models": PretrainedModels,
    "corpus-max-sentences": Optional[int],
    "spm-vocab-size": Optional[int],
  },
  total=False,
)


Taskcluster = TypedDict(
  "Taskcluster",
  {
    "split-chunks": int,
    "worker-classes": dict[str, Literal["gcp-standard", "gcp-spot"]],
  },
  total=False,
)

# Represents the datasets used for training.
Datasets = TypedDict(
  "Datasets",
  {
    # Datasets to merge for validation while training.
    "devtest": list[str],
    # Datasets for evaluation. Each will generate an evaluate-* task for each model type.
    "test": list[str],
    # Parallel training corpora.
    "train": list[str],
    # Monolingual datasets that are translated by the teacher model to generate the
    # data to be used for student distillation.
    "mono-src": list[str],
    # Monolingual datasets that are translated by the back translations model to
    # synthesize data to increase the amount of data available for teacher training.
    "mono-trg": list[str],
  },
  total=False,
)

TrainingConfig = TypedDict(
  "TrainingConfig",
  {
    "datasets": dict[str, list[str]],
    "marian-args": MarianArgs,
    "experiment": Experiment,
    # Taskcluster-specific pipeline configuration, eg: chunking
    "taskcluster": Taskcluster,
    # Enable publication to Weights and Biases.
    "wandb-publication": bool,
    # The stage of the pipeline to run until (any stages this choice depends on will
    # be automatically included).
    "target-stage": str,
    # An array of taskIds of decision or action tasks from the previous group(s) to use
    # to populate our `previous_group_kinds`. Tasks specified here will be used as long
    # as their label matches a needed task, and that task is upstream of `start-stage`.
    # (That is to say: even if a task from one of these groups has a cache digest that
    # doesn't match what the downstream task wants, it will still be used. This can be
    # used for quick iteration of functionality where the quality of the outputs is not
    # important.
    "previous-group-ids": Optional[list[str]],
    # The stage of the pipeline to begin at, provided replacements can be found for tasks
    # upstream of this stage. Usually used in conjunction with `previous-group-ids`
    # which allows for specifying task group ids to fetch existing tasks from.
    "start-stage": Optional[str],
    
    # A mapping of task labels to task IDs that will be re-used in this training run.
    # For example:
    #
    # existing-tasks: {
    #         "build-docker-image-base": "BAvLUilqQ3SYqy6Ck55CUQ",
    #         "build-docker-image-test": "f0gbptvMTDaKODjqL9hlOw",
    #         "build-docker-image-toolchain-build": "LlZa8-L9TRemgyzQcAxuHw",
    #         "build-docker-image-train": "fBMJa9R5SKaXd2wgWeD5yQ",
    #         "fetch-browsermt-marian": "BRviRlEMTie8AUFf5prHvg",
    #     }
    "existing-tasks": Optional[dict[str, str]],
  },
  total=False,
)
@staticmethod
def get_and_validate_training_config(config_dict: dict[str, Any]):
    """
    Creates a TrainingConfig and validates it from the graph config.
    """
    training_config = serializable.from_dict(TrainingConfig, config_dict)

    with open(Path(__file__).parent / "../config.yml", "r") as file:
        valid_stages = yaml.safe_load(file)["valid-stages"]
        
    start_stage: str | None = training_config.get("start-stage")
    if start_stage and start_stage not in valid_stages:
        raise ValueError(
            f'start stage "{start_stage}" is not a valid stage in taskcluster/config.yml'
        )

    target_stage: str | None = training_config.get("target-stage")
    if target_stage and target_stage not in valid_stages:
        raise ValueError(
            f'target stage "{target_stage}" is not a valid stage in taskcluster/config.yml'
        )

    return training_config


@casing()
class Parameters(TypedDict):
    base_repository: str
    base_ref: str
    base_rev: str
    build_date: int
    build_number: int
    do_not_optimize: list[str]
    enable_always_target: bool
    existing_tasks: dict[str, str]
    filters: list[str]
    head_ref: str
    head_repository: str
    head_rev: str
    head_tag: str
    level: str
    moz_build_date: str
    optimize_target_tasks: bool
    owner: str
    project: str
    pushdate: int
    pushlog_id: str
    repository_type: str
    target_tasks_method: str
    tasks_for: str
    training_config: TrainingConfig
    optimize_strategies: Optional[str]
    files_changed: Optional[list[str]]
    next_version: Optional[str]
    version: Optional[str]
