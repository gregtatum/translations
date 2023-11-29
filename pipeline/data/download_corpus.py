#!/usr/bin/env python3
"""
Downloads datasets using an importer. If running this locally, first run
`utils/find_corpus.py` to find the corpus key for downloading. Locally, the datasets
are saved to `data/train-parts`, which is the same directory that OPUS cleaner uses.

Usage:

poetry run pipeline/data/download-corpus.py en ru --corpus opus_ELRC-3075-wikipedia_health/v1

The compression commands can be overidden via the environment variables:
    COMPRESSION_CMD
    ARTIFACT_EXT
"""

import argparse
import os
import re
import subprocess
import sys
from enum import Enum
from os import path
from tempfile import TemporaryDirectory
from typing import Optional

import requests
from tqdm import tqdm

# Allow overriding the compression commands.
COMPRESSION_CMD = os.environ.get("COMPRESSION_CMD", "pigz")
ARTIFACT_EXT = os.environ.get("ARTIFACT_EXT", "gz")

# Get the root of the repo.
ROOT_PATH = path.normpath(path.join(path.dirname(path.abspath(__file__)), "../.."))
DATA_PATH = path.join(ROOT_PATH, "data/train-parts")

Importer = Enum("Importer", ["opus", "flores", "mtdata", "sacrebleu"])


class Corpus:
    """
    Given a corpus key, this class will process and sanitize the results. This class
    raises an exception when a malformed key is provided.

    TODO - Add support for data augmentation keys.

    Example:

    corpus = Corpus("opus_CCMatrix/v1")
    assert corpus.importer.name == "opus"
    assert corpus.dataset == "CCMatrix/v1"

    Corpus("flores_devtest")
    assert corpus.importer.name == "flores"
    assert corpus.dataset == "devtest"

    Corpus("mtdata_Tilde-ema-2016-eng-hun")
    assert corpus.importer.name == "mtdata"
    assert corpus.dataset == "Tilde-ema-2016-eng-hun"
    """

    importer: Importer
    dataset: str

    def __init__(self, key: str):
        # Process the dataset name, e.g.
        #   "opus_CCMatrix/v1"
        #   "flores_devtest"
        #   "mtdata_Tilde-ema-2016-eng-hun"
        #
        # For an example config with datasets see:
        # https://github.com/mozilla/firefox-translations-training/blob/main/configs/tc.prod.yml
        regex = re.compile(
            r"""
                ^(\w+)  # Match the importer, e.g. "opus", or "flores".
                _       # Split on the _ character
                (.+)$   # Match all the rest as the dataset
            """,
            re.X,
        )
        result = re.match(regex, key)
        if not result:
            raise Exception(f"The corpus key was malformed: {key}")

        importer = result.group(1)
        try:
            self.importer = Importer[importer]
        except Exception:
            raise Exception(f"An unknown importer was given: {importer}")

        self.dataset: str = result.group(2)


def sanitize_filename(name: str) -> str:
    """Sanitize a name by replacing non-alphanumeric values"""
    return re.sub(r"[^A-Za-z0-9_\- ]", "_", name)


shell_env: Optional[dict] = None


def set_shell_env(env: dict):
    """Set the shell environment for testing purposes."""
    global shell_env  # noqa: PLW0603
    shell_env = env


def shell_run(cmd: str) -> str:
    """Shell commands must be a string, not an argument list."""
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, env=shell_env)


def download_with_progress_bar(response: requests.Response, download_path: str):
    """Streams the download to a file and output the progress as a download bar."""
    with open(download_path, "wb") as file, tqdm(
        desc="Downloading",
        total=int(response.headers.get("content-length", 0)),
        unit="MiB",
        unit_scale=True,
        unit_divisor=1024 * 1024 * 1024,
    ) as bar:
        for data in response.iter_content(chunk_size=4096):
            file.write(data)
            bar.update(len(data))


def download_sacrebleu_corpus(src: str, trg: str, output_dir: str, corpus: Corpus):
    print("Downloading sacrebleu corpus.")

    # Example dataset name: "wmt08"
    # See: https://github.com/mjpost/sacrebleu/blob/a99299bc098ae35fc6f781934c5d9a4fdfb2fed0/DATASETS.md

    src_output = path.join(output_dir, f"{sanitize_filename(corpus.dataset)}.{src}.{ARTIFACT_EXT}")
    trg_output = path.join(output_dir, f"{sanitize_filename(corpus.dataset)}.{trg}.{ARTIFACT_EXT}")

    shell_run(
        f"sacrebleu --test-set {corpus.dataset} --language-pair {src}-{trg} --echo src | {COMPRESSION_CMD} -c > {src_output}"
    )

    shell_run(
        f"sacrebleu --test-set {corpus.dataset} --language-pair {src}-{trg} --echo ref | {COMPRESSION_CMD} -c > {trg_output}"
    )

    print("Source dataset:", src_output)
    print("Target dataset:", trg_output)
    print("Done: Downloading sacrebleu corpus.")


def download_mtdata_corpus(src: str, trg: str, output_dir: str, corpus: Corpus):
    from mtdata.iso import iso3_code

    print("Downloading mtdata corpus")
    with TemporaryDirectory() as temp_dir:
        src_iso = iso3_code(src, fail_error=True)
        trg_iso = iso3_code(trg, fail_error=True)

        shell_run(f"mtdata get --langs {src}-{trg} --train {corpus.dataset} --out {temp_dir}")

        train_src_path = os.path.join(temp_dir, "train-parts", f"{corpus.dataset}.{src_iso}")
        train_trg_path = os.path.join(temp_dir, "train-parts", f"{corpus.dataset}.{trg_iso}")
        print("train_src_path: ", train_src_path)
        print("train_trg_path: ", train_trg_path)

        name = sanitize_filename(corpus.dataset)

        src_output = path.join(output_dir, f"{name}.{src}.{ARTIFACT_EXT}")
        trg_output = path.join(output_dir, f"{name}.{trg}.{ARTIFACT_EXT}")

        print("Source dataset:", src_output)
        print("Target dataset:", trg_output)

        import time

        if not os.path.exists(train_src_path):
            print("Does not exist train_src_path:", train_src_path)
            time.sleep(1000)

        if not os.path.exists(train_trg_path):
            print("Does not exist train_trg_path:", train_trg_path)
            time.sleep(1000)

        with open(train_src_path, "rb") as src_file:
            subprocess.run(
                [COMPRESSION_CMD, "-c"], stdin=src_file, stdout=open(src_output, "wb"), check=True
            )

        with open(train_trg_path, "rb") as trg_file:
            subprocess.run(
                [COMPRESSION_CMD, "-c"], stdin=trg_file, stdout=open(trg_output, "wb"), check=True
            )

    print("Done: Downloading mtdata corpus")


def download_opus_corpus(src: str, trg: str, output_dir: str, corpus: Corpus):
    print("Downloading opus corpus.")

    # Dataset:          "ELRC-3075-wikipedia_health/v1"
    # Name:             "ELRC-3075-wikipedia_health"
    # Name and version: "ELRC-3075-wikipedia_health_v1"

    name = sanitize_filename(corpus.dataset.split("/")[0])
    name_and_version = sanitize_filename(corpus.dataset)

    url1 = f"https://object.pouta.csc.fi/OPUS-{corpus.dataset}/moses/{src}-{trg}.txt.zip"
    url2 = f"https://object.pouta.csc.fi/OPUS-{corpus.dataset}/moses/{trg}-{src}.txt.zip"
    # Deduce the order of the src_trg
    is_src_trg = True

    # Attempt to get the dataset, trying each language direction.
    print("Attempting: ", url1)
    response = requests.get(url1, stream=True)
    if not response.ok:
        is_src_trg = False
        print("Attempting: ", url2)
        response = requests.get(url2, stream=True)

    # Handle download failures.
    if not response.ok:
        print("Could not download the dataset from OPUS.")
        response.raise_for_status()

    # Inflate and deflate the data using the appropriate compression scheme.
    with TemporaryDirectory() as temp_dir:
        archive_path = path.join(temp_dir, "download.txt.zip")

        download_with_progress_bar(response, archive_path)

        # Expected unzipped file:
        #
        # tempdir
        #   ├── README
        #   ├── LICENSE
        #   ├── ELRC-3075-wikipedia_health.en-ru.en
        #   ├── ELRC-3075-wikipedia_health.en-ru.ru
        #   └── ELRC-3075-wikipedia_health.en-ru.xml
        subprocess.run(["unzip", archive_path, "-d", temp_dir], check=True)

        # Write out the final files, e.g.
        #
        #   output_dir
        #   ├── ELRC-3075-wikipedia_health_v1.en.gz
        #   └── ELRC-3075-wikipedia_health_v1.ru.gz
        for lang in [src, trg]:
            output_file = path.join(output_dir, f"{name_and_version}.{lang}.{ARTIFACT_EXT}")
            if is_src_trg:
                input_file = path.join(temp_dir, f"{name}.{src}-{trg}.{lang}")
            else:
                input_file = path.join(temp_dir, f"{name}.{trg}-{src}.{lang}")

            cmd = f"{COMPRESSION_CMD} -c {input_file} > {output_file}"
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
            print("Saved: ", output_file)

    print("Done: Downloading opus corpus")


def main(argsList: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("src", help="The source language")
    parser.add_argument("trg", help="The target language")
    parser.add_argument(
        "--corpus",
        metavar="KEY",
        required=True,
        help='The corpus key, which includes the importer name and dataset, e.g. "opus_ELRC-2480-Estatuto_dos_Deputad/v1"',
    )
    parser.add_argument(
        "--output_dir",
        metavar="PATH",
        help="The directory where the datasets will be saved.",
        # Default the data directory to the one that OpusCleaner uses.
        default=DATA_PATH,
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(argsList)

    try:
        corpus = Corpus(args.corpus)
    except Exception as exception:
        print(
            'An invalid dataset key was provided. Expected a key of the form: "opus_CCMatrix/v1",'
        )
        print("Run utils/find-corpus.py to find valid corpora")
        print(exception)
        sys.exit(1)

    # Determine the output directory.
    output_dir = args.output_dir
    if output_dir == DATA_PATH and not os.path.exists(output_dir):
        # Create the default data path if it does not exist.
        os.makedirs(output_dir)

    if corpus.importer == Importer.opus:
        download = download_opus_corpus
    elif corpus.importer == Importer.sacrebleu:
        download = download_sacrebleu_corpus
    elif corpus.importer == Importer.mtdata:
        download = download_mtdata_corpus
    else:
        print("Unknown importer", corpus.importer.name)
        sys.exit(1)

    download(args.src, args.trg, output_dir, corpus)


if __name__ == "__main__":
    main()
