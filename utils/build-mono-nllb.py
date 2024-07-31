import argparse
import json
import os
import unicodedata
from contextlib import ExitStack
from pathlib import Path
from random import Random

import requests

from pipeline.common.datasets import shuffle_with_max_lines
from pipeline.common.downloads import (
    read_lines,
    stream_download_to_file,
    write_lines,
)

"""
Build a monolingual dataset based off of NLLB. This builds only monolingual data for
the primary language, de-duplicating against the secondary.

    task build-mono-nllb -- --primary sl --secondary en

You can include a sample rate to include less data:

    task build-mono-nllb -- --primary sl --secondary en --sample_rate 0.5

By default if the primary language is English, it only samples 10% of the data. Otherwise
it samples 100% of the data.

The `--cleanup` option can be used to delete some large intermediate files.
"""

DATA_PATH = (Path(__file__).parent / "../data/nllb").resolve()


def compute_hashes_in_parallel_data(
    parallel_archive: Path, lang_pair: str, lang_pair_alt: str, primary_lang: str
):
    """
    In order to de-duplicate sentences we can compute a hash and store it in memory. This makes
    it so that we don't have to store the full sentence in memory
    """
    sentence_hashes: set[int] = set()
    sentences_visited = 0

    print(f"Compute hashes of the parallel data: {parallel_archive}")

    with ExitStack() as stack:
        # Try both lang_pair directions.
        try:
            lines = stack.enter_context(
                read_lines(parallel_archive, path_in_archive=f"NLLB.{lang_pair}.{primary_lang}")
            )
        except Exception:
            lines = stack.enter_context(
                read_lines(
                    parallel_archive, path_in_archive=f"NLLB.{lang_pair_alt}.{primary_lang}"
                )
            )

        for line in lines:
            sentences_visited += 1
            if sentences_visited % 1_000_000 == 0:
                print(f"Hashing sentence {sentences_visited:,}")
            sentence_hashes.add(hash_line(line))

        return sentence_hashes, sentences_visited


def hash_line(line: str) -> int:
    """
    Return a hash of a line. The line has its whitespace stripped and text representation
    normalized to ensure a consistent representation.
    """
    cleaned_line = unicodedata.normalize("NFC", line.strip())
    return hash(cleaned_line)


def filter_and_write_monolingual_data(
    mono_path: Path, output_path: Path, sentence_hashes: set[int]
):
    """
    Filtering is done with a set[int]. Seeing if a line is in the set should be O(1)
    in terms of time complexity. A set[int] was chosen (storing the hash) rather than
    a set[str], as the latter would retain the string in memory.
    """

    with ExitStack() as stack:
        destination = stack.enter_context(write_lines(output_path))
        lines = stack.enter_context(read_lines(mono_path))
        discard_count = 0
        kept_count = 0
        for line in lines:
            if hash_line(line) not in sentence_hashes:
                kept_count += 1
                destination.write(line)
            else:
                discard_count += 1
            if kept_count % 1_000_000 == 0:
                print(f"{kept_count:,} kept, {discard_count:,} discarded")

    return kept_count, discard_count


def build_dataset_sample(source_path: Path, sample_path: Path, dataset_name: str):
    """
    Outputs a sample of 1000 randomly sampled sentences from the dataset
    """
    byte_size = source_path.stat().st_size

    with ExitStack() as stack:
        output = stack.enter_context(write_lines(sample_path))
        lines = stack.enter_context(read_lines(source_path))

        for line in shuffle_with_max_lines(
            line_stream=lines,
            seed=dataset_name,
            max_lines=1000,
            max_words_in_sentence=100,
            total_byte_size=byte_size,
        ):
            output.write(line)


def write_sample_to_file(source: str, destination: str, sample_rate: float):
    """
    Rather than directly download an entire file, sample lines from it. NLLB data is
    not adequately shuffled, so in order to get a properly diverse domain of data, we
    need to pull off a sample rather than the first N lines.
    """

    # Provide a reproducible sample.
    random = Random(123456)

    with ExitStack as stack:
        outfile = stack.enter_context(write_lines(destination))
        lines = stack.enter_context(read_lines(source))

        for line in lines:
            if random.random() < sample_rate:
                outfile.write(line.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--primary", metavar="LANG", type=str, help="The two/three letter langtag")
    parser.add_argument(
        "--secondary", metavar="LANG", type=str, help="The two/three letter langtag"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Delete the intermediate data files"
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        required=False,
        help="How much of the monolingual data to sample, ranged 0.0 to 1.0",
    )

    args = parser.parse_args()
    primary_lang: str = args.primary
    secondary_lang: str = args.secondary

    if args.sample_rate:
        sample_rate: int = args.sample_rate
    elif primary_lang == "en":
        # 52 gigs of data is a bit much, take 1%
        sample_rate = 0.1
    else:
        sample_rate = 1.0

    # The direction of the languages don't matter, but we want a stable ordering.
    if primary_lang < secondary_lang:
        lang_pair = f"{primary_lang}-{secondary_lang}"
        lang_pair_alt = f"{secondary_lang}-{primary_lang}"
    else:
        lang_pair = f"{secondary_lang}-{primary_lang}"
        lang_pair_alt = f"{primary_lang}-{secondary_lang}"

    data_subdir = DATA_PATH / lang_pair
    os.makedirs(data_subdir, exist_ok=True)

    mono_file = f"{primary_lang}.txt.gz"
    mono_path = DATA_PATH / mono_file
    mono_url = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/{mono_file}"

    parallel_file = f"{lang_pair}.txt.zip"
    parallel_file_alt = f"{lang_pair_alt}.txt.zip"
    parallel_path = data_subdir / parallel_file
    parallel_url = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/{parallel_file}"
    parallel_url_alt = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/{parallel_file_alt}"

    output_zst_path = data_subdir / f"nllb-mono-{primary_lang}.txt.zst"
    sample_path = data_subdir / f"nllb-mono-{primary_lang}.sample.txt"
    output_info_path = data_subdir / f"nllb-mono-{primary_lang}.info.json"

    if output_zst_path.exists():
        print(f"{output_zst_path} exists")
    else:
        if mono_path.exists():
            print(f"{mono_file} exists")
        elif sample_rate == 1.0:
            stream_download_to_file(mono_url, mono_path)
        else:
            write_sample_to_file(mono_url, mono_path, sample_rate)

        if parallel_path.exists():
            print(f"{parallel_file} exists")
        else:
            # Decide which URL to use.
            response = requests.head(parallel_url)
            if not response.ok:
                parallel_url = parallel_url_alt

            stream_download_to_file(parallel_url, parallel_path)
            # zip contents:
            # ├── README
            # ├── LICENSE
            # ├── NLLB.en-sl.en
            # ├── NLLB.en-sl.sl
            # └── NLLB.en-sl.scores

        print("Compute a hash of all the sentences in the parallel data.")
        print(f"{parallel_path}")

        sentence_hashes, sentences_visited = compute_hashes_in_parallel_data(
            parallel_path, lang_pair, lang_pair_alt, primary_lang
        )

        print(f"There are {len(sentence_hashes):,} unique sentences out of {sentences_visited:,}")
        print(
            f'{(sentences_visited - len(sentence_hashes)):,} "{primary_lang}" sentences were duplicated'
        )

        print("Identifying and writing out monolingual data.")
        kept_count, discard_count = filter_and_write_monolingual_data(
            mono_path, output_zst_path, sentence_hashes
        )

        print(f"Dataset created {output_zst_path}")
        print(f"{kept_count:,} kept, {discard_count:,} discarded")

        with output_info_path.open("w", encoding="utf-8") as file:
            data = {"sentences_kept": kept_count, "sentences_discarded": discard_count}
            json.dump(data, file, indent=2)

    if sample_path.exists():
        print(f"{sample_path} exists")
    else:
        print(f"Building a sample of the data: {sample_path}")
        build_dataset_sample(output_zst_path, sample_path, f"nllb-mono-{primary_lang}")

    print(f"Data written to: {data_subdir}")
    for file in data_subdir.iterdir():
        print(f" - {data_subdir}/{file.name}")

    if args.cleanup:
        print(f"Cleaning up {mono_path}")
        mono_path.unlink()
        print(f"Cleaning up {parallel_path}")
        parallel_path.unlink()


if __name__ == "__main__":
    main()
