from typing import Literal, TypedDict


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
        "per-dataset": int,
    },
    total=False,
)

CleanerThresholds = TypedDict(
    "CleanerThresholds",
    {
        "default-threshold": float,
        "dataset-thresholds": dict[str, float],
    },
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
    {"mono-src": float, "mono-trg": float},
)


# Pre-trained models use URLs as they are flexible to continue training from either
# long-term bucket storage, or from tasks in Taskcluster.
PretrainedModel = TypedDict(
    "PretrainedModel",
    {
        "urls": list[str],
        "mode": Literal["continue"] | Literal["init"] | Literal["use"],
        "type": Literal["default"] | Literal["opusmt"],
    },
    total=False,
)


PretrainedModels = TypedDict(
    "PretrainedModels",
    {"train-backwards": PretrainedModel, "train-teacher": PretrainedModel},
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
        "teacher-mode": Literal["one-stage"] | Literal["two-stage"],
        # Translate with either Marian or CTranslate2.
        "teacher-decoder": Literal["marian"] | Literal["ctranslate2"],
        # The student model architecture as defined in
        #   pipeline/train/configs/model/student.{model}.yml
        "student-model": Literal["tiny"] | Literal["base"],
        # Limits for the "target" monolingual data, e.g. data used for back translations.
        "mono-max-sentences-trg": MonoMaxSentences,
        # Limits for the "source" monolingual data, e.g. data used for student distillation.
        "mono-max-sentences-src": MonoMaxSentences,
        # How large of a sample to use for the Sentence Piece sample size.
        "spm-sample-size": int,
        #  The metric to use for the best model.
        "best-model": (
            Literal["cross-entropy"]
            | Literal["ce-mean-words"]
            | Literal["perplexity"]
            | Literal["valid-script"]
            | Literal["translation"]
            | Literal["bleu"]
            | Literal["bleu-detok"]
            | Literal["bleu-segmented"]  # (deprecated, same as bleu)
            | Literal["chrf"]
        ),
        # Use OpusCleaner to clean the corpus.
        # https://github.com/hplt-project/OpusCleaner
        "use-opuscleaner": Literal["true"] | Literal["false"],
        # Indicates whether to use dataset specific configs.
        "opuscleaner-mode": Literal["custom"] | Literal["defaults"],
        # Thresholds for running bicleaner-ai.
        # https://github.com/bitextor/bicleaner-ai
        "bicleaner": CleanerThresholds,
        "monocleaner": MonocleanerConfig,
        "hplt-min-doc-score": HpltMinDocScore,
        # Instead of training models from scratch, use pre-trained models.
        "pretrained-models": PretrainedModels,
        "corpus-max-sentences": int,
        "spm-vocab-size": int,
    },
    total=False,
)


Taskcluster = TypedDict(
    "Taskcluster",
    {
        "split-chunks": int,
        "worker-classes": dict[str, Literal["gcp-standard"] | Literal["gcp-spot"]],
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
        # Taskcluster-specific pipeline configuration, e.g/ chunking
        "taskcluster": Taskcluster,
        # Enable publication to Weights and Biases
        "wandb-publication": bool,
        # An array of taskIds of decision or action tasks from the previous group(s) to use
        # to populate our `previous_group_kinds`. Tasks specified here will be used as long
        # as their label matches a needed task, and that task is upstream of `start-stage`.
        # (That is to say, even if a task from one of these groups has a cache digest that
        # doesn't match what the downstream task wants, it will still be used. This can be
        # used for quick iteration of functionality where the quality of the outputs is not
        # important.
        "previous-group-ids": list[str],
        # The stage of the pipeline to begin at, provided replacements can be found for tasks
        # upstream of this stage. Usually used in conjunction with `previous-group-ids`
        # which allows for specifying task group ids to fetch existing tasks from.
        "start-stage": str,
        # The stage of the pipeline to run until (any stages this choice depends on will
        # be automatically included).
        "target-stage": str,
        # A mapping of task labels to task IDs that will be re-used in this training run.
        # For example:
        #
        # existing-tasks- {
        #         "build-docker-image-base": "BAvLUilqQ3SYqy6Ck55CUQ",
        #         "build-docker-image-test": "f0gbptvMTDaKODjqL9hlOw",
        #         "build-docker-image-toolchain-build": "LlZa8-L9TRemgyzQcAxuHw",
        #         "build-docker-image-train": "fBMJa9R5SKaXd2wgWeD5yQ",
        #         "fetch-browsermt-marian": "BRviRlEMTie8AUFf5prHvg",
        #     }
        "existing-tasks": dict[str, str]
        # ------------------------------------------------------------------------------------
        # them to be used by the type system.
    },
    total=False,
)
