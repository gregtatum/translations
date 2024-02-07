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

from pipeline.utils.downloads import google_cloud_storage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("lang", type=str)
    parser.add_argument("output_prefix", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument(
        "--compression_cmd", default="pigz", help="The name of the compression command to use."
    )
    parser.add_argument(
        "--artifact_ext",
        default="gz",
        help="The artifact extension for the compression",
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
