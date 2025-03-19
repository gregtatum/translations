import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Optional
from pipeline.common.downloads import (
    read_lines,
    write_lines,
)
from pipeline.common.logging import get_logger

logger = get_logger(__file__)

def download_file(url: str, file_destination: Path):
    logger.info(f"Download {url} to {file_destination}")
    with ExitStack() as stack:
        outfile = stack.enter_context(write_lines(file_destination))
        lines = stack.enter_context(read_lines(url))

        for line in lines:
            outfile.write(line)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src_url", type=str, required=True, help="The source URL for this corpus")
    parser.add_argument("--trg_url", type=str, required=True, help="The target URL for this file")
    parser.add_argument("--src_locale", type=str, required=True, help="The source language for this corpus")
    parser.add_argument("--trg_locale", type=str, required=True, help="The target language for this file")
    parser.add_argument("--corpus", choices=["backtranslations", "original-parallel", "student-distillation"])
    parser.add_argument("--aln_url", type=str, default="", help="The alignments file URL for this file")
    parser.add_argument("--artifacts", type=Path, help="The location where the dataset will be saved")
    
    args = parser.parse_args()
    
    src_url: str = args.src_url
    trg_url: str = args.trg_url
    src_locale: str = args.src_locale
    trg_locale: str = args.trg_locale
    corpus: str = args.corpus
    aln_url: str = args.aln_url
    artifacts: Path = args.artifacts
    
    if corpus == "backtranslations":
        file_name_part = "mono"
    elif corpus == "original-parallel":
        file_name_part = "corpus"
    elif corpus == "student-distillation":
        file_name_part = "corpus"
    else:
        raise ValueError(f"Unexpected corpus name: \"{corpus}\"")
    
    aln_part = ""
    if aln_url:
        aln_part = ".tok-icu"
    
    artifacts.mkdir(exist_ok=True)
    src_destination = artifacts / f"{file_name_part}{aln_part}.{src_locale}.zst"
    trg_destination = artifacts / f"{file_name_part}{aln_part}.{trg_locale}.zst"
    aln_destination = artifacts / f"{file_name_part}.aln.zst"
    
    download_file(src_url, src_destination)
    download_file(trg_url, trg_destination)
    if aln_url:
        download_file(aln_url, aln_destination)
    


    

if __name__ == "__main__":
    main()
