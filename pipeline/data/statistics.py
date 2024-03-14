#!/usr/bin/env python3

"""
Get the statistical distribution of a dataset.
"""

import argparse
import gzip
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import zstandard
from matplotlib import ticker

import wandb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the pipeline is available on the path.
sys.path.append(os.path.join(CURRENT_DIR, "../.."))

from pipeline.common.datasets import LogDistribution
from pipeline.common.downloads import (
    RemoteGzipLineStreamer,
    RemoteZstdLineStreamer,
)


def get_line_streamer(file_location: str):
    if file_location.startswith("http://") or file_location.startswith("https://"):
        if file_location.endswith(".zst"):
            return RemoteZstdLineStreamer(file_location)
        # Assume gzip.
        return RemoteGzipLineStreamer(file_location)

    if file_location.endswith(".gz"):
        return gzip.open(file_location, "rt")
    if file_location.endswith(".zst"):
        return zstandard.open(file_location, "rt")
    return open(file_location, "rt")


def log_to_wandb(
    file_location: str,
    name: str,
    language: str,
    files: list[str],
):
    with wandb.init(
        project="fxt-training",
        job_type="datasets",
        entity="gtatum",
    ) as run:
        artifact = wandb.Artifact(
            name=f"{name}-{language}",
            type="dataset",
        )
        artifact.add_reference(uri=file_location)
        for file in files:
            artifact.add_file(local_path=file)

        run.log_artifact(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument(
        "--file_location", type=str, required=True, help="A url or file path for analyzing."
    )
    parser.add_argument("--name", type=str, required=True, help="The name of the dataset")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="The dataset language, as a BCP-47 language tag",
    )

    parsed_args = parser.parse_args()

    # url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v1/mono/cs.txt.gz"
    # url = "https://data.statmt.org/news-crawl/cs/news.2007.cs.shuffled.deduped.gz"
    # file = "data/statistics/opensubtitles-cs.txt.gz"

    str_length_distribution = LogDistribution("string length")
    word_distribution = LogDistribution("words")

    with get_line_streamer(parsed_args.file_location) as lines:
        for line in lines:
            str_length_distribution.count(len(line))
            word_distribution.count(len(line.split()))

    str_length_distribution.report_log_scale()
    word_distribution.report_log_scale()

    words_filename = "distribution-words.png"
    codepoints_filename = "distribution-codepoints.png"

    plot_logarithmic_histogram(
        word_distribution.histogram,
        max_size=5_000,  # words
        title="\n".join(
            [
                "Word Count Distribution",
                f"{parsed_args.name} - {parsed_args.language}",
            ]
        ),
        x_axis_label="Words (log scale)",
        filename=words_filename,
    )

    plot_logarithmic_histogram(
        str_length_distribution.histogram,
        max_size=10_000,  # codepoints
        title="\n".join(
            [
                "Codepoints per Sentence Distribution",
                f"{parsed_args.name} - {parsed_args.language}",
            ]
        ),
        x_axis_label="Codepoints (log scale)",
        filename=codepoints_filename,
    )

    log_to_wandb(
        parsed_args.file_location,
        parsed_args.name,
        parsed_args.language,
        files=[words_filename, codepoints_filename],
    )


def plot_logarithmic_histogram(
    histogram: dict[int, int], max_size: int, title: str, x_axis_label: str, filename: str
):
    # Convert the histogram dictionary to a DataFrame.
    # Start with a few small value buckets, since it's easy to start with some small fractional
    # values on a log scale.
    bin_edges = [1.0, 2.0]
    for edge in np.logspace(np.log10(1), np.log10(max_size), 30):
        if edge > 2.0:
            bin_edges.append(edge)
    bin_edges = np.array(bin_edges)

    # Plot a histogram with logarithmic bins.
    plt.title(title)
    plt.hist(histogram.keys(), bins=bin_edges, weights=histogram.values(), alpha=0.7)

    plt.xlabel(x_axis_label)
    plt.xscale("log")
    plt.xticks(ticks=bin_edges, labels=[f"{int(edge)}" for edge in bin_edges], rotation="vertical")

    plt.ylabel("Frequency (linear)")
    plt.yscale("linear")
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # Ensure no labels are cut off.
    plt.tight_layout()
    print("Saving plot to", filename)
    plt.savefig(filename, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
