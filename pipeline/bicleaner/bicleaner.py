#!/usr/bin/env python3
"""
Cleans corpus using bicleaner-ai. See:
  docs/bicleaner.md
"""

import argparse
from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Generator, Optional
from random import shuffle

from pipeline.common.logging import get_logger
from pipeline.common.marian import assert_gpus_available
from pipeline.common.command_runner import run_command
from pipeline.common.downloads import (
    multiprocess_parallel_lines,
    stream_input,
    write_lines,
    read_lines,
)
from pipeline.common.datasets import (
    FilteringStep,
    Statistics,
)

logger = get_logger(__file__)


@dataclass
class FilteringStatistics(Statistics):
    """
    Gather statistics about the filtering process.
    """

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(dataset_path)
        self.bicleaned = FilteringStep("How many lines were filtered by the bicleaner threshold.")


def get_lang_from_yaml(yaml_path: Path, key: str) -> str:
    with open(yaml_path) as f:
        for line in f:
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    raise ValueError(f"{key} not found in {yaml_path}")


class Config:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--fetches", type=Path, required=True)
        parser.add_argument("--dataset_sanitized", type=str, required=True)
        parser.add_argument("--bicleaner_threshold", type=float, required=True)
        parser.add_argument("--pack_dir", type=Path, required=True)
        parser.add_argument("--cuda_dir", type=Path, required=True)
        parser.add_argument("--src", type=str, required=True, help="The source locale.")
        parser.add_argument("--trg", type=str, required=True, help="The target locale.")
        parser.add_argument(
            "--artifacts", type=Path, required=True, help="Path to the artifacts folder."
        )

        args = parser.parse_args()

        self.bicleaner_threshold: float = args.bicleaner_threshold
        self.pack_dir: Path = args.pack_dir
        self.cuda_dir: Path = args.cuda_dir
        self.src: str = args.src
        self.trg: str = args.trg

        fetches: Path = args.fetches
        dataset_sanitized: str = args.dataset_sanitized
        artifacts: Path = args.artifacts

        self.input_src = fetches / f"{dataset_sanitized}.{self.src}.zst"
        self.input_trg = fetches / f"{dataset_sanitized}.{self.trg}.zst"

        self.output_src = artifacts / f"{dataset_sanitized}.{self.src}.zst"
        self.output_trg = artifacts / f"{dataset_sanitized}.{self.trg}.zst"

        self.output_trg = artifacts / f"{dataset_sanitized}.{self.trg}.zst"

        self.scored = artifacts / f"{dataset_sanitized}.scored.zst"
        self.kept = artifacts / f"{dataset_sanitized}.kept.zst"
        self.filtered = artifacts / f"{dataset_sanitized}.filtered.zst"

        self.filtered_sample = artifacts / f"{dataset_sanitized}.filtered.sample.txt"
        self.kept_sample = artifacts / f"{dataset_sanitized}.kept.sample.txt"


def main():
    assert_gpus_available(logger)
    config = Config()

    stats = FilteringStatistics(config.output_src)

    config.output_src.parent.mkdir(parents=True, exist_ok=True)

    pack_file = next(config.pack_dir.glob("*.yaml"))
    model_src = get_lang_from_yaml(pack_file, "source_lang")
    model_trg = get_lang_from_yaml(pack_file, "target_lang")

    # Determine the ordering of the src/trg columns.
    scol, tcol = 1, 2
    if model_src == config.trg or model_trg == config.src:
        scol, tcol = 2, 1

    if config.bicleaner_threshold == 0.0:
        logger.info("Threshold is 0.0, skipping filtering")
        config.input_src.rename(config.output_src)
        config.input_trg.rename(config.output_trg)
        return

    cuda_visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    assert cuda_visible_devices_str, "GPUs must be available via the CUDA_VISIBLE_DEVICES"
    devices = cuda_visible_devices_str.split(",")

    def on_chunk(
        src_lines: Generator[str, None, None], trg_lines: Generator[str, None, None]
    ) -> list[str]:
        """
        Process a 10M chunk of src and trg lines in parallel. Each GPU will get a chunk
        at a time.
        """
        bicleaner_command = [
            "bicleaner-ai-classify",
                "--disable_hardrules",
                "--scol", str(scol),
                "--tcol", str(tcol),
                "-", "-", config.pack_dir
        ] # fmt: skip

        return stream_input(
            bicleaner_command,
            (
                # Bicleaner expects tab separated parallel sentences.
                f"{src.rstrip()}\t{trg.rstrip()}\n"
                for src, trg in zip(src_lines, trg_lines)
            ),
        )

    with (
        write_lines(config.output_src) as output_src,
        write_lines(config.output_trg) as output_trg,
        write_lines(config.kept) as kept_lines,
        write_lines(config.filtered) as filtered_lines,
    ):

        def on_chunk_done(lines: list[str]):
            for line in lines:
                # Each line has the following format.
                # <src_line>\t<trg_line>\t<score>\n
                src_line, trg_line, score = line.rstrip().split("\t")

                if float(score) > config.bicleaner_threshold:
                    stats.bicleaned.kept += 1
                    output_src.write(src_line + "\n")
                    output_trg.write(trg_line + "\n")
                    kept_lines.write(line)
                else:
                    stats.bicleaned.filtered += 1
                    filtered_lines.write(line)

        multiprocess_parallel_lines(
            config.input_src,
            config.input_trg,
            on_chunk=on_chunk,
            on_chunk_done=on_chunk_done,
            chunk_bytes=10_000_00,  # < 10MB chunks, respecting line boundaries.
            processes=len(devices),
        )

    stats.save_json()

    sample_size = 10_000
    for name, count, corpus_path, sample_path in [
        ("filtered", stats.bicleaned.filtered, config.filtered, config.filtered_sample),
        ("kept", stats.bicleaned.kept, config.kept, config.kept_sample),
    ]:
        logger.info(f"Generating a {sample_size:,} sample of the {name} lines")
        indexes = list(range(count))
        shuffle(indexes)
        sample_indexes = set(indexes[:sample_size])
        with read_lines(corpus_path) as filtered_lines, write_lines(sample_path) as sample_file:
            for index, line in enumerate(filtered_lines):
                if index in sample_indexes:
                    sample_file.write(line)

    logger.info("Done: Bicleaner filtering")


if __name__ == "__main__":
    main()
