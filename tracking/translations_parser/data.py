"""
This file defines data structures and some of the parsing logic for extracting metrics
from logs from in the "train-*" tasks the training pipeline. It provides structured
representations of training epochs, validation results, and evaluation metrics,
ensuring consistency in how data is tracked in Weights and Biases.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from translations_parser.utils import parse_task_label

logger = logging.getLogger(__name__)

METRIC_LOG_RE = re.compile(
    r"|".join(
        [
            r"\+ tee .+\.metrics",
            r"\+ tee .+\.en",
            r"\+ sacrebleu .+",
            r"\+ .+\/marian-decoder .+",
            # Ignore potential comments
            r"^sacreBLEU:",
        ]
    )
)
# Match the task prefix.
# [task 2023-09-16T12:29:33.338Z] ###### Training a model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TC_PREFIX_RE = re.compile(r"^\[task [0-9\-TZ:\.]+\]")


@dataclass
class TrainingEpoch:
    """
    Raw Marian logs are parsed to collect each training epoch.

    See: https://marian-nmt.github.io/examples/training-overview

    For example:
    Ep. 7 : Up. 36000 : Sen. 948,710 : Cost 3.21035433 : Time 317.61s : 77198.20 words/s : gNorm 0.5012
    """

    # The epoch of the training run. Note that with OpusTrainer, that there is only
    # 1 epoch.
    epoch: int

    # How many batches have been processed and optimizer updates run.
    up: int

    # The total sentences, or more accurately training pairs visited.
    sen: int

    # The current value of the cost function based on the training's "cost-type",
    # e.g. "cost-type: ce-mean-words"
    cost: float

    # The time taken to process the last batch of data.
    time: float

    # The tokens per second.
    rate: float

    # Reports the exponential average for the norms of parameter gradients.
    gnorm: float

    # The learning rate can be adjusted over time.
    learning_rate: Optional[float] = None  # Optional


@dataclass
class ValidationEpoch:
    """
    The Marian validator will periodically compute and output its score.

    For example:
    [valid] Ep. 9 : Up. 50000 : bleu : 35.9958 : stalled 2 times (last best: 36.0671)
    [valid] Ep. 9 : Up. 50000 : perplexity : 3.74923 : new best
    """

    # These are documented in the TrainingEpoch's attributes.
    epoch: int
    up: int

    # chrF and bleu are the typical validation metrics.
    chrf: float
    bleu_detok: float

    # Cross-Entropy mean words is the cost function used during training.
    ce_mean_words: float

    # Perplexity is a measure of uncertainty or confidence in the model’s predictions.
    # Lower perplexity indicates a better model. Perplexity measures the average
    # branching factor of the model, or how many choices the model is confused between
    # for predicting the next token.
    perplexity: Optional[float] = None

    # Optional stalled validation metrics
    chrf_stalled: Optional[int] = None
    ce_mean_words_stalled: Optional[int] = None
    bleu_detok_stalled: Optional[int] = None
    perplexity_stalled: Optional[float] = None


@dataclass
class Metric:
    """
    The chrF, bleu, and COMET scores extracted from a .metrics file.

    For example "aug-noise_devtest.metrics":
    43.14
    67.83
    0.9321

    Note that this file is a legacy file, and there is a `*.metrics.json` file that
    provides a more structured format.
    """

    # Evaluation identifiers
    importer: str
    dataset: str
    augmentation: Optional[str]

    # Scores
    chrf: float
    bleu_detok: float
    comet: Optional[float] = None

    @classmethod
    def from_file(
        cls,
        metrics_file: Path,
        importer: Optional[str] = None,
        dataset: Optional[str] = None,
        augmentation: Optional[str] = None,
    ):
        """
        Instanciate a Metric from a `*.metrics` file.
        In case no dataset is set, detects it from the filename.
        """
        logger.debug(f"Reading metrics file {metrics_file.name}")
        values = []
        comet = None
        chrf = None
        bleu_detok = None

        try:
            with metrics_file.open("r") as f:
                lines = f.readlines()
            for line in lines:
                try:
                    values.append(float(line))
                except ValueError:
                    continue
            assert len(values) in (2, 3), "file must contain 2 or 3 lines with a float value"
        except Exception as e:
            raise ValueError(f"Metrics file could not be parsed: {e}")
        if len(values) == 2:
            bleu_detok, chrf = values
        if len(values) == 3:
            bleu_detok, chrf, comet = values
        if importer is None:
            _, importer, dataset, augmentation = parse_task_label(metrics_file.stem)

        # Multiply metric by 100 to match other metrics percentage style
        if comet is not None:
            comet *= 100

        assert importer is not None, "An importer was found."
        assert dataset is not None, "A dataset was found."
        assert chrf is not None, "The chrf metric was found."
        assert bleu_detok is not None, "The bleu_detok metric was found."

        return cls(
            importer=importer,
            dataset=dataset,
            augmentation=augmentation,
            chrf=chrf,
            bleu_detok=bleu_detok,
            comet=comet,
        )

    @classmethod
    def from_tc_context(
        cls, importer: str, dataset: str, lines: Sequence[str], augmentation: str | None = None
    ):
        """
        Try reading the bleu and chrF metrics from Taskcluster logs, looking for two
        successive floats after a line maching METRIC_LOG_RE.
        """
        for index, line in enumerate(lines):
            # Remove eventual Taskcluster prefix
            clean_line = TC_PREFIX_RE.sub("", line).strip()
            if not METRIC_LOG_RE.match(clean_line):
                continue
            try:
                # Attempt to get the next 2 lines.
                next_two_lines = lines[index + 1 : index + 3]
                if len(next_two_lines) != 2:
                    continue

                bleu_detok, chrf = [
                    # Remove the task prefix, and parse the float from the line.
                    #
                    # [task 2023-09-16T12:32:20.995Z] 54.32
                    # ^                               ^
                    # └─ Remove this                  └─ Extract this.
                    float(TC_PREFIX_RE.sub("", val))
                    for val in next_two_lines
                ]
            except ValueError:
                continue

            return cls(
                importer=importer,
                dataset=dataset,
                augmentation=augmentation,
                chrf=chrf,
                bleu_detok=bleu_detok,
            )

        raise ValueError("Metrics logs could not be parsed")


@dataclass
class TrainingLog:
    """
    The final results from the parsing of a training log file.
    """

    # Runtime configuration
    configuration: dict
    training: List[TrainingEpoch]
    validation: List[ValidationEpoch]
    logs: List[str]
    run_date: datetime | None
