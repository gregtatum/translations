"""
Merges multiple corpora into a single "source" language file, and a single "target"
language file, each. For instance:

  dataset1.en.zst dataset1.ru.zst
  dataset2.en.zst dataset2.ru.zst
  dataset3.en.zst dataset3.ru.zst

  Gets merged into:

  corpus.en.zst
  corpus.ru.zst
"""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Generator
from pipeline.common.datasets import (
    FilteringStep,
    Statistics,
    WeakStringSet,
    shuffle_with_max_lines,
)
from pipeline.common.downloads import get_human_readable_file_size, read_lines, write_lines
from pipeline.common.logging import get_logger

logger = get_logger(__file__)

# TODO(CJK) - Issue #424
MAX_WORDS_IN_SENTENCE = 100


@dataclass
class FilteringStatistics(Statistics):
    """
    Gather statistics about the filtering process.
    """

    parallel_corpus: FilteringStep
    datasets: list[FilteringStep]

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(dataset_path)
        self.parallel_corpus = FilteringStep(
            dataset_path,
            "How much of the data was retained across all of the parallel corpora",
        )

    def add_parallel_dataset(self, location: str):
        path = Path(location)
        location_stem = path.parent / Path(path.stem).stem
        step = FilteringStep(location_stem)
        self.datasets.append(step)
        return step


def log_dataset(location: str):
    logger.info(f"Reading dataset {location}")


class DeduplicateCorpus:
    def __init__(
        self,
        datasets_src: list[Path],
        datasets_trg: list[Path],
        src_outpath: Path,
        trg_outpath: Path,
        stats: FilteringStatistics,
        max_lines: int,
        total_corpus_bytes: int,
    ) -> None:
        self.datasets_src: list[Path] = datasets_src
        self.datasets_trg: list[Path] = datasets_trg
        self.src_outpath: Path = src_outpath
        self.trg_outpath: Path = trg_outpath
        self.stats: FilteringStatistics = stats
        self.dataset_stats: FilteringStep = None

        with ExitStack() as stack:
            src_outfile = stack.enter_context(write_lines(self.src_outpath))
            trg_outfile = stack.enter_context(write_lines(self.trg_outpath))

            if max_lines:
                for line in shuffle_with_max_lines(
                    line_stream=self.yield_lines_string(stack),
                    seed=38540735095,
                    max_lines=max_lines,
                    max_words_in_sentence=MAX_WORDS_IN_SENTENCE,
                    total_byte_size=total_corpus_bytes,
                ):
                    src_line, trg_line = line.split("\t")
                    src_outfile.write(src_line)
                    trg_outfile.write(trg_line)
            else:
                for src_line, trg_line in self.yield_lines_tuple(stack):
                    src_outfile.write(src_line)
                    trg_outfile.write(trg_line)

    def yield_lines_tuple(self, stack: ExitStack) -> Generator[tuple[str, str], None, None]:
        strings_seen = WeakStringSet()
        stats = self.stats
        src_lines = stack.enter_context(
            read_lines(self.datasets_src, on_enter_location=self.on_enter_location)
        )
        trg_lines = stack.enter_context(
            read_lines(self.datasets_trg, on_enter_location=log_dataset)
        )

        for src_line, trg_line in zip(src_lines, trg_lines):
            line = src_line + trg_line
            if line in strings_seen:
                stats.parallel_corpus.filtered += 1
                self.dataset_stats.filtered += 1
            else:
                stats.parallel_corpus.kept += 1
                self.dataset_stats.kept += 1

            # No separator is needed as the newline is included.
            strings_seen.add(src_line + trg_line)

            yield src_line, trg_line

    def yield_lines_string(self, stack: ExitStack) -> Generator[str, None, None]:
        for src_line, trg_line in self.yield_lines_tuple(stack):
            if "\t" in src_line or "\t" in trg_line:
                logger.error("A line contained a tab character, skipping:")
                logger.error(f" src: {src_line}")
                logger.error(f" trg: {src_line}")
            else:
                yield f"{src_line}\t{trg_line}"

    def on_enter_location(self, location):
        log_dataset(location)
        self.dataset_stats = self.stats.add_parallel_dataset(location)


def deduplicate_data(
    datasets_src: list[Path],
    datasets_trg: list[Path],
    src_outpath: Path,
    trg_outpath: Path,
    stats: FilteringStatistics,
) -> None:
    strings_seen = WeakStringSet()
    dataset_stats: FilteringStep = None

    def on_enter_src(location: str):
        log_dataset(location)
        nonlocal dataset_stats
        dataset_stats = stats.add_parallel_dataset(location)

    with ExitStack() as stack:
        src_lines = stack.enter_context(read_lines(datasets_src, on_enter_location=on_enter_src))
        trg_lines = stack.enter_context(read_lines(datasets_trg, on_enter_location=log_dataset))
        src_outfile = stack.enter_context(write_lines(src_outpath))
        trg_outfile = stack.enter_context(write_lines(trg_outpath))

        for src_line, trg_line in zip(src_lines, trg_lines):
            line = src_line + trg_line
            if line in strings_seen:
                stats.parallel_corpus.filtered += 1
                dataset_stats.filtered += 1
            else:
                stats.parallel_corpus.kept += 1
                dataset_stats.kept += 1

            # No separator is needed as the newline is included.
            strings_seen.add(src_line + trg_line)

            src_outfile.write(src_line)
            trg_outfile.write(trg_line)


def sample_corpus(
    artifacts: Path, name: str, sample_size: int, src_outpath: Path, trg_outpath: Path
):
    """
    Generate a sample of the corpus data with the following format:

    e.g.
    > cat artifacts/corpus.sample.txt
    Sentence 1 in source language
    Sentence 1 in target language

    Sentence 2 in source language
    Sentence 2 in target language

    Sentence 3 in source language
    Sentence 3 in target language
    ...
    """
    total_byte_size = src_outpath.stat().st_size + trg_outpath.stat().st_size

    with ExitStack() as stack:
        sample_path = artifacts / f"{name}.sample.txt"

        src_lines = stack.enter_context(read_lines(src_outpath))
        trg_lines = stack.enter_context(read_lines(trg_outpath))
        sample_outfile = stack.enter_context(write_lines(sample_path))

        def join_src_trg():
            for src_line, trg_line in zip(src_lines, trg_lines):
                # The src and trg line each have a newline at the end. This means that
                # each sentence pair will be separate by a blank line to make for easy
                # scanning of datasets.
                yield f"{src_line}{trg_line}\n"

        logger.info(f"Write a {sample_size:,} line sample of the merged corpus: {sample_path}")

        for line in shuffle_with_max_lines(
            line_stream=join_src_trg(),
            seed=9834523434,
            max_lines=sample_size,
            max_words_in_sentence=MAX_WORDS_IN_SENTENCE,
            total_byte_size=total_byte_size,
        ):
            sample_outfile.write(line)


def get_datasets(src: str, trg: str, datasets_glob: str):
    dataset_paths: list[str] = glob(datasets_glob)
    datasets_src: list[Path] = []
    datasets_trg: list[Path] = []

    total_corpus_bytes = 0

    for dataset in dataset_paths:
        path = Path(dataset)
        if dataset.endswith(f"{src}.zst"):
            datasets_src.append(path)
        elif dataset.endswith(f"{trg}.zst"):
            datasets_trg.append(path)
        else:
            raise Exception(f"Dataset does not match naming scheme: {dataset}")

        formatted_size, bytes = get_human_readable_file_size(path)
        logger.info(f" - {path} ({formatted_size})")
        total_corpus_bytes += bytes

    return datasets_src, datasets_trg, total_corpus_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--datasets_glob",
        type=str,
        help="A glob-style path to the mono datasets, e.g. /path/to/*.zst",
    )

    parser.add_argument(
        "--max_lines",
        type=str,
        default="None",
        help="The (optionally) maximum number of sentences that will be merged.",
    )

    parser.add_argument(
        "--sample_size", type=int, default=10_000, help="Generate a random sample of sentences."
    )

    parser.add_argument(
        "--artifacts",
        type=Path,
        help="The path to the artifacts directory.",
    )

    parser.add_argument(
        "--name",
        type=str,
        help='The final corpus name, e.g. "corpus" will output a "corpus.en.zst" file.',
    )

    args = parser.parse_args()

    datasets_src, datasets_trg, total_corpus_bytes = get_datasets(
        args.src, args.trg, args.datasets_glob
    )

    logger.info("Parallel datasets:")

    src_outpath = args.artifacts / f"{args.name}.{args.src}.zst"
    trg_outpath = args.artifacts / f"{args.name}.{args.trg}.zst"

    stats = FilteringStatistics()

    deduplicate_data(
        datasets_src=datasets_src,
        datasets_trg=datasets_trg,
        src_outpath=src_outpath,
        trg_outpath=trg_outpath,
        stats=stats,
        total_corpus_bytes=total_corpus_bytes,
    )

    sample_corpus(
        artifacts=args.artifacts,
        name=args.name,
        sample_size=args.sample_size,
        src_outpath=src_outpath,
        trg_outpath=trg_outpath,
    )

    stats.save_json()


if __name__ == "__main__":
    main()
