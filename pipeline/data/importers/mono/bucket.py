#!/usr/bin/env python3
"""
Import monolingual data from a Google Cloud storage bucket.

Example usage:

pipeline/data/importers/mono/bucket.py                                       \
  en                                                       `# lang`          \
  $artifacts/releng-translations-dev_data_custom-en-ru_zip `# output_prefix` \
  releng-translations-dev/data/custom-en-ru.zip            `# name`
"""

import argparse
import re

from pipeline.utils.google_cloud import storage

# pipeline/data/importers/corpus/bucket.py


def parse_name(name: str):
    # releng-translations-dev/data/custom-en-ru.zip
    matches = re.search(r"^([\w-]*)/(.*)$", name)
    if not matches:
        raise Exception(f"Could not parse the name {name}")

    bucket_name = matches.group(1)
    file_path = matches.group(2)

    return bucket_name, file_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("lang", type=str)
    parser.add_argument("output_prefix", type=str)
    parser.add_argument("name", type=str)

    args = parser.parse_args()

    bucket_name, file_path = parse_name(args.name)

    print("lang", args.lang)
    print("output_prefix", args.output_prefix)
    print("name", args.name)
    print("bucket_name", bucket_name)
    print("file_path", f"{file_path}.{args.lang}.zst")

    lang_file = f"{file_path}.{args.lang}.zst"
    lang_dest = f"{args.output_prefix}.zst"

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    bucket.name

    print("Bucket:", bucket_name)
    print("Downloading:", lang_file)
    bucket.blob(lang_file).download_to_filename(lang_dest)


if __name__ == "__main__":
    main()
