#!/usr/bin/env python3
"""
Import data from a Google Cloud storage bucket.

Example usage:

pipeline/data/importers/corpus/url.py                                                       \\
  en                                                                      `# src`           \\
  ru                                                                      `# trg`           \\
  artifacts/releng-translations-dev_data_custom-en-ru_zip                 `# output_prefix` \\
  http://storage.google.com/releng-translations-dev/data/custom-en-ru.zip `# url`
"""

import argparse
import re

import requests

from pipeline.utils.downloads import google_cloud_storage


def parse_name(name: str):
    # releng-translations-dev/data/custom-en-ru.zip
    matches = re.search(r"^([\w-]*)/(.*)$", name)
    if not matches:
        raise Exception(f"Could not parse the name {name}")

    bucket_name = matches.group(1)
    file_path = matches.group(2)

    return bucket_name, file_path


def download(url: str, destination: str):
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024**2):
            f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("src", type=str)
    parser.add_argument("trg", type=str)
    parser.add_argument("output_prefix", type=str)
    parser.add_argument("url", type=str)

    args = parser.parse_args()

    bucket_name, file_path = parse_name(args.name)

    print("src", args.src)
    print("trg", args.trg)
    print("output_prefix", args.output_prefix)
    print("name", args.name)
    print("bucket_name", bucket_name)

    src_file = args.src.replace("{lang}", args.src)
    trg_file = args.trg.replace("{lang}", args.trg)
    src_dest = f"{args.output_prefix}.{args.src}.zst"
    trg_dest = f"{args.output_prefix}.{args.trg}.zst"

    print(f"src_dest: {src_dest}")
    print(f"trg_dest: {trg_dest}")

    print("Downloading:", src_file)
    download(src_file, src_dest)
    print("Downloading:", src_file)
    download(trg_file, trg_dest)


if __name__ == "__main__":
    main()
