"""
Continue the training pipeline with an existing vocab.
"""
import argparse
from pathlib import Path
from pipeline.common.downloads import (
    stream_download_to_file
)
from pipeline.common.logging import get_logger
from pipeline.common import arg_utils

logger = get_logger(__file__)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src_locale", type=str, required=True, help="The source language for this corpus")
    parser.add_argument("--trg_locale", type=str, required=True, help="The target language for this file")
    parser.add_argument("--src_url", type=str, required=True, help="The source URL for this corpus")
    parser.add_argument("--trg_url", type=str, required=True, help="The target URL for this corpus")
    
    args = parser.parse_args()
    
    src_locale = arg_utils.ensure_string("--src_locale", args.src_locale)
    trg_locale = arg_utils.ensure_string("--trg_locale", args.trg_locale)
    src_url = arg_utils.ensure_string("--src_url", args.src_url)
    trg_url = arg_utils.ensure_string("--trg_url", args.trg_url)
    artifacts: Path = args.artifacts

    artifacts.mkdir(exist_ok=True)
    
    # TODO - Change to the other "if" branch when split vocab lands:
    # See: https://github.com/mozilla/translations/pull/1051
    if True:
        src_destination = artifacts / "vocab.spm"
        trg_destination = artifacts / "vocab.spm"
    else:
        src_destination = artifacts / f"vocab.{src_locale}.spm"
        trg_destination = artifacts / f"vocab.{trg_locale}.spm"
    
    stream_download_to_file(src_url, src_destination)
    stream_download_to_file(trg_url, trg_destination)

if __name__ == "__main__":
    main()
