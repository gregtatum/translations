"""
Translate a corpus using either Marian or CTranslate2.
"""

import argparse
from enum import Enum
from glob import glob
import os
from pathlib import Path
import tempfile

from pipeline.common.command_runner import apply_command_args, run_command
from pipeline.common.datasets import compress, decompress
from pipeline.common.downloads import count_lines, is_file_empty, write_lines
from pipeline.common.logging import get_logger
from pipeline.translate.translate_ctranslate2 import translate_with_ctranslate2

logger = get_logger(__file__)


class Decoder(Enum):
    marian = "marian"
    ctranslate2 = "ctranslate2"


class Device(Enum):
    cpu = "cpu"
    gpu = "gpu"


def run_marian(
    marian_dir: Path,
    models: list[Path],
    vocab: str,
    input: Path,
    output: Path,
    gpus: list[str],
    workspace: int,
    is_nbest: bool,
    extra_args: list[str],
):
    config = Path(__file__).parent / "decoder.yml"
    marian_bin = str(marian_dir / "marian-decoder")
    log = input.parent / f"{input.name}.log"
    if is_nbest:
        extra_args = ["--n-best", *extra_args]

    logger.info("Starting Marian to translate")

    run_command(
        [
            marian_bin,
            *apply_command_args(
                {
                    "config": config,
                    "models": models,
                    "vocabs": [vocab, vocab],
                    "input": input,
                    "output": output,
                    "log": log,
                    "devices": gpus,
                    "workspace": workspace,
                }
            ),
            *extra_args,
        ],
        logger=logger,
        env={**os.environ},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input", type=Path, required=True, help="The path to the text to translate."
    )
    parser.add_argument(
        "--models_glob",
        type=str,
        required=True,
        help="A glob pattern to the Marian model(s)",
    )
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="Output path to the artifacts."
    )
    parser.add_argument("--nbest", action="store_true", help="Whether to use the nbest")
    parser.add_argument(
        "--marian_dir", type=Path, required=True, help="The path the Marian binaries"
    )
    parser.add_argument("--vocab", type=Path, help="Path to vocab file")
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help='The indexes of the GPUs to use on a system, e.g. --gpus "0 1 2 3"',
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="The amount of Marian memory (in MB) to preallocate",
    )
    parser.add_argument(
        "--decoder",
        type=Decoder,
        default=Decoder.marian,
        help="Either use the normal marian decoder, or opt for CTranslate2.",
    )
    parser.add_argument(
        "--device",
        type=Device,
        default=Device.gpu,
        help="Either use the normal marian decoder, or opt for CTranslate2.",
    )
    parser.add_argument(
        "extra_marian_args",
        nargs=argparse.REMAINDER,
        help="Additional parameters for the training script",
    )

    args = parser.parse_args()

    # Provide the types for the arguments.
    marian_dir: Path = args.marian_dir
    input_zst: Path = args.input
    artifacts: Path = args.artifacts
    models_glob: str = args.models_glob
    models: list[Path] = [Path(path) for path in glob(models_glob)]
    postfix = "nbest" if args.nbest else "out"
    output_zst = artifacts / f"{input_zst.stem}.{postfix}.zst"
    vocab: Path = args.vocab
    gpus: list[str] = args.gpus.split(" ")
    extra_marian_args: list[str] = args.extra_marian_args
    decoder: Decoder = args.decoder
    is_nbest: bool = args.nbest
    device: Device = args.device

    # Do some light validation of the arguments.
    assert input_zst.exists(), f"The input file exists: {input_zst}"
    assert vocab.exists(), f"The vocab file exists: {vocab}"
    if not artifacts.exists():
        artifacts.mkdir()
    for gpu_index in gpus:
        assert gpu_index.isdigit(), f'GPUs must be list of numbers: "{gpu_index}"'
    for model in models:
        assert model.exists(), f"The model file exists {model}"
    if extra_marian_args and extra_marian_args[0] != "--":
        logger.error(" ".join(extra_marian_args))
        raise Exception("Expected the extra marian args to be after a --")

    logger.info(f"Input file: {input_zst}")
    logger.info(f"Output file: {output_zst}")

    # Taskcluster can produce empty input files when chunking out translation for
    # parallelization. In this case skip translating, and write out an empty file.
    if is_file_empty(input_zst):
        logger.info(f"The input is empty, create a blank output: {output_zst}")
        with write_lines(output_zst) as _outfile:
            # Nothing to write, just create the file.
            pass
        return

    if decoder == Decoder.ctranslate2:
        translate_with_ctranslate2(
            input_zst=input_zst,
            artifacts=artifacts,
            extra_marian_args=extra_marian_args,
            models_glob=models_glob,
            is_nbest=is_nbest,
            vocab=[str(vocab)],
            device=device.value,
        )
        return

    # The device flag is for use with CTranslate, but add some assertions here so that
    # we can be consistent in usage.
    if device == Device.cpu:
        assert (
            "--cpu-threads" in extra_marian_args
        ), "Marian's cpu should be controlled with the flag --cpu-threads"
    else:
        assert (
            "--cpu-threads" not in extra_marian_args
        ), "Requested a GPU device, but --cpu-threads was provided"

    # Run the training.
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_txt = temp_dir / input_zst.stem
        output_txt = temp_dir / output_zst.stem

        decompress(input_zst, destination=input_txt, remove=True, logger=logger)

        run_marian(
            marian_dir=marian_dir,
            models=models,
            vocab=vocab,
            input=input_txt,
            output=output_txt,
            gpus=gpus,
            workspace=args.workspace,
            is_nbest=is_nbest,
            # Take off the initial "--"
            extra_args=extra_marian_args[1:],
        )
        assert count_lines(input_txt) == count_lines(
            output_txt
        ), "The input and output had the same number of lines"

        compress(output_txt, destination=output_zst, remove=True, logger=logger)


if __name__ == "__main__":
    main()
