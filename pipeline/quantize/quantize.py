"""
Take the unquantized student model that has been fine-tuned on a simulated quantization
mode, and perform the intgemm8 quantization.

1. Decode the validation set to get a range of float parameter values via the
   `--dump-quantmult` browsermt/marian-dev command line flag. These values
   will be in the range of float values.
2. Take the absolute values of these floats, and determine the `QuantMultA` which is
   the scaling factor for converting from float values to (-127, 127) quantized integers.
3. Quantize and export the model.

For usage see:
    taskcluster/kinds/distillation-student-model-quantize/kind.yml
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from pipeline.common.logging import get_logger
import numpy as np
from itertools import pairwise
from pipeline.common.command_runner import run_executable

logger = get_logger(__file__)


@dataclass
class Shortlist:
    """
    See the config.experiments.shortlist for more documentation.
    """

    path: Path
    first_number: int
    best_number: int
    threshold: float

    def as_marian_args(self) -> list[Path | int | float]:
        return [self.path, self.first_number, self.best_number, self.threshold]


@dataclass
class QuantizeConfig:
    finetuned_model: Path
    vocab_src: Path
    vocab_trg: Path
    shortlist: Shortlist
    devtest: Path
    artifacts: Path
    bmt_marian: Path
    bin: str
    src_locale: str
    trg_locale: str
    # Derived properties:
    marian_log: Path
    quantmults_log: Path
    optimized_alphas_model: Path
    quantized_model: Path
    translated_output_path: Path
    log_path: Path

    @staticmethod
    def from_args() -> "QuantizeConfig":
        parser = argparse.ArgumentParser(description="Quantize a Marian student model")
        parser.add_argument(
            "--finetuned_model",
            required=True,
            description="The finetuned student model, in the .npz format",
        )
        parser.add_argument("--vocab_src", required=True, type=Path, help="Path to src vocab file")
        parser.add_argument("--vocab_trg", required=True, type=Path, help="Path to trg vocab file")
        parser.add_argument(
            "--shortlist_path",
            required=True,
            type=Path,
            help="Path to the shortlist in text format.",
        )
        parser.add_argument(
            "--shortlist_first_number",
            required=True,
            type=int,
            help="See the config's experiment.shortlist for docs.",
        )
        parser.add_argument(
            "--shortlist_best_number",
            required=True,
            type=int,
            help="See the config's experiment.shortlist for docs.",
        )
        parser.add_argument(
            "--shortlist_threshold",
            required=True,
            type=float,
            help="See the config's experiment.shortlist for docs.",
        )
        parser.add_argument(
            "--devtest",
            required=True,
            type=Path,
            help="The path to the uncompressed devtest corpus.",
        )
        parser.add_argument(
            "--artifacts", required=True, type=Path, help="Output path to the artifacts."
        )
        parser.add_argument(
            "--bmt_marian",
            required=True,
            help="The directory that contains the browsermt marian fork.",
        )
        parser.add_argument("--bin", required=True)
        parser.add_argument("--src", required=True)
        parser.add_argument("--trg", required=True)
        args = parser.parse_args()

        artifacts: Path = args.artifacts

        return QuantizeConfig(
            finetuned_model=args.finetuned_model,
            vocab_src=args.vocab_src,
            vocab_trg=args.vocab_trg,
            shortlist=Shortlist(
                path=args.shortlist_path,
                first_number=args.shortlist_first_number,
                best_number=args.shortlist_best_number,
                threshold=args.shortlist_threshold,
            ),
            devtest=args.devtest,
            artifacts=artifacts,
            bmt_marian=args.bmt_marian,
            bin=args.bin,
            src_locale=args.src,
            trg_locale=args.trg,
            # Derive some more paths
            marian_log=artifacts / "marian.log",
            quantmults_log=artifacts / "quantmults.log",
            optimized_alphas_model=artifacts / "model.optimized-alphas.npz",
            quantized_model=artifacts / "model.intgemm.alphas.bin",
            translated_output_path=artifacts / f"validation.translated.{args.trg}.txt",
            log_path=artifacts / "cpu.output.log",
        )


def create_optimized_alphas_model(
    finetuned_model: Path, quantmults_log: Path, optimized_alphas_model: Path
) -> None:
    """
    After fine-tuning on the student, compute the maximum absolute values of the floats
    so that an ideal "QuantMultA" parameter can be found that converts the float into
    a -127 to 127 integral domain. This puts the layers in the non-quantized model
    as an intermediate step towards the final quantization.
    """

    # The key is the parameter name, e.g. "encoder_l1_self_Wq_QuantMultA". The value
    # is every float value that was reported for a decoding. Only store fields that
    # have the name "QuantMultA" in them. The original model doesn't have these fields,
    # but these are simulated by the fine tuned model.
    max_abs_by_quant_parameters: dict[str, list[float]] = {}

    # Go through the ---dump-quantmult log and parse out the simulated "QuantMultA"
    # values of each field.
    with quantmults_log.open("r") as input_file:
        for line in input_file:
            # We're only interested in the simulated quantization multiplier.
            if "tcmalloc" in line or "QuantMultA" not in line:
                # Skip lines that can't be parsed.
                continue

            # Example line ()
            # "Name: F0::encoder_l1_self_Wq_QuantMultA MeanAbs: 1.28948 stddevAbs: 0.996464 Mean: 0.079264 stddev: 1.6277 MaxAbs: 11.6399"
            #
            # And more readable:
            #   Name: F0::encoder_l1_self_Wq_QuantMultA
            #   MeanAbs: 1.28948
            #   stddevAbs: 0.996464
            #   Mean: 0.079264
            #   stddev: 1.6277
            #   MaxAbs: 11.6399
            values: dict[str, str] = {
                # Use pairwise to convert the line into a dict.
                k: v
                for k, v in pairwise(line.strip().split(" "))
            }
            # Convert something like: "F0::encoder_l1_self_Wq_QuantMultA"
            # to "encoder_l1_self_Wq_QuantMultA"
            name = values["Name"].split("::")[-1]
            if "QuantMultA" in name:
                max_abs_by_quant_parameters.setdefault(name, []).append(float(values["maxAbs"]))

    model_file_dict = dict(np.load(finetuned_model))

    # The QuantMultA parameters were only simulated in the original model, add them
    # on to the model now, e.g. add on a "encoder_l1_self_Wq_QuantMultA" parameter.
    for param_name, max_abs in max_abs_by_quant_parameters.items():
        assert "QuantMultA" in param_name
        quant_mult_a = 127 / (np.mean(max_abs) + 1.1 * np.std(max_abs))
        model_file_dict[param_name] = np.array([quant_mult_a], dtype=np.float32)

    # Make sure that the decoder works whether a shortlist has been generated or not:
    # This is necessary because when there is a shortlist, the matrix names are different.
    # The Wemb matrix becomes "none"
    if "Wemb_QuantMultA" in model_file_dict:
        model_file_dict["none_QuantMultA"] = model_file_dict["Wemb_QuantMultA"]
    elif "none_QuantMultA" in model_file_dict:
        model_file_dict["Wemb_QuantMultA"] = model_file_dict["none_QuantMultA"]

    # Create the intermediate model where the quantization alphas have been added, but
    # the model is not yet quantized.
    np.savez(optimized_alphas_model, **model_file_dict)


def decode_validation_corpus(config: QuantizeConfig) -> None:
    """
    This runs the finetuned student model on the CPU to decode the validation corpus.
    The "--dump-quantmult" will output the parameter values to stderr, where they will
    be collected for processing. This will give a max range for the "QuantMultA" value.
    This value is the scaling factor for turning a float32 value into a
    (-127,127) integral value.
    """
    logger.info("Decode the validation corpus in order to get typical quantization values")

    with config.quantmults_log.open("w") as quantmults_file:
        run_executable(
            config.bmt_marian / "marian-decoder",
            {
                "models": config.finetuned_model,
                "vocabs": [config.vocab_src, config.vocab_trg],
                "config": Path(__file__).parent / "decoder.yml",
                "input": config.devtest,
                "output": config.translated_output_path,
                "shortlist": config.shortlist.as_marian_args(),
                # Do not output anything to stdout, rather output to the logs.
                "quiet": True,
                "quiet-translation": True,
                "log": config.marian_log,
                # This is a browsermt only flag that will output the QuantMultA values
                # for each decoding pass.
                "dump-quantmult": True,
            },
            stderr=quantmults_file,
        )


def convert_to_quantized_intgemm8(config: QuantizeConfig) -> None:
    logger.info("Converting")
    run_executable(
        config.bmt_marian / "marian-conv",
        {
            "from": str(config.optimized_alphas_model),
            "to": str(config.quantized_model),
            "--gemm-type": "intgemm8",
        },
    )


def main() -> None:
    config = QuantizeConfig.from_args()

    # Validate the config.
    assert "LD_LIBRARY_PATH" in os.environ, (
        "LD_LIBRARY_PATH environment variable must be set because the marian-decoder "
        "requires it even though it is running in CPU mode"
    )
    assert config.finetuned_model.exists(), "The fine tuned model exists."
    assert config.vocab_src.exists(), "The vocab src exists."
    assert config.vocab_trg.exists(), "The vocab src exists."
    assert config.shortlist.path.exists(), "The shortlist exists."
    assert config.devtest.exists(), "The devtest exists."
    config.artifacts.mkdir(parents=True, exist_ok=True)

    decode_validation_corpus(config)
    create_optimized_alphas_model(
        finetuned_model=config.finetuned_model,
        quantmults_log=config.quantmults_log,
        optimized_alphas_model=config.optimized_alphas_model,
    )
    convert_to_quantized_intgemm8(config)


if __name__ == "__main__":
    main()
