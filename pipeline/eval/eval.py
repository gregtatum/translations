"""
Evaluate a trained model with both the BLEU and chrF metrics.

Kinds:
   taskcluster/kinds/evaluate/kind.yml
   taskcluster/kinds/evaluate-quantized/kind.yml
   taskcluster/kinds/evaluate-teacher-ensemble/kind.yml

Example usage:

   python eval.py \
     --artifacts_prefix $TASK_WORKDIR/artifacts/wmt09                            \
     --dataset_prefix $MOZ_FETCHES_DIR/wmt09                                     \
     --src en                                                                    \
     --trg ca                                                                    \
     --marian $MARIAN                                                            \
     --decoder_config $MOZ_FETCHES_DIR/final.model.npz.best-chrf.npz.decoder.yml \
     `# Marian args must be at the end:`                                         \
     --                                                                          \
       -w $WORSPACE                                                              \
       --devices $GPUS                                                           \
       --models $MOZ_FETCHES_DIR/final.model.npz.best-chrf.npz

Artifacts:

For instance for a artifacts_prefix of: `artifacts/wmt09`:

  artifacts
  ├── wmt09.en             The source sentences
  ├── wmt09.ca             The target output
  ├── wmt09.ca.ref         The original target sentences
  ├── wmt09.log            The Marian log
  ├── wmt09.metrics        The BLEU and chrF score
  └── wmt09.metrics.json   The BLEU and chrF score in json format

Fetches:

For instance for a value of: `fetches/wmt09`:
  fetches
  ├── wmt09.en.zst
  └── wmt09.ca.zst
"""


import argparse
import json
import os
import subprocess
from textwrap import dedent, indent
from typing import Optional

import sh


def run_bash_oneliner(command: str):
    """
    Runs multi-line bash with comments as a one-line command.
    """
    command_dedented = dedent(command)

    # Remove comments and whitespace.
    lines = [
        line.strip() for line in command_dedented.split("\n") if line and not line.startswith("#")
    ]
    command = " \\\n".join(lines)

    print("-----------------Running bash in one line--------------")
    print(indent(command_dedented, "  "))
    print("-------------------------------------------------------")
    return subprocess.check_call(command, shell=True)


# De-compresses files, and pipes the result as necessary.
def decompress(path: str, compression_cmd: str, artifact_ext: str):
    subprocess.check_call(f'{compression_cmd} -dc "{path}"')


def main(args_list: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument(
        "--artifacts_prefix",
        type=str,
        help="The location where the translated results will be saved",
    )
    parser.add_argument(
        "--dataset_prefix", type=str, help="The evaluation datasets prefix, used in the form."
    )
    parser.add_argument("--src", type=str, help='The source language, e.g "en".')
    parser.add_argument("--trg", type=str, help='The target language, e.g "ca".')
    parser.add_argument("--marian", type=str, help="The path the to marian binaries.")
    parser.add_argument("--decoder_config", type=str, help="The marian yaml config for the model.")
    parser.add_argument(
        "--compression_cmd", default="pigz", help="The name of the compression command to use."
    )
    parser.add_argument(
        "--artifact_ext",
        default="gz",
        help="The artifact extension for the compression",
    )
    parser.add_argument(
        "marian_args", nargs=argparse.REMAINDER, help="Additional arguments to pass to marian."
    )

    args = parser.parse_args(args_list)

    src = args.src
    trg = args.trg
    dataset_prefix = args.dataset_prefix
    artifacts_prefix = args.artifacts_prefix

    artifacts_dir = os.path.dirname(artifacts_prefix)
    source_file_compressed = f"{dataset_prefix}.{src}.{args.artifact_ext}"
    source_file = f"{artifacts_prefix}.{src}"
    target_file_compressed = f"{dataset_prefix}.{trg}.{args.artifact_ext}"
    target_file = f"{artifacts_prefix}.{trg}"
    target_ref_file = f"{artifacts_prefix}.{trg}.ref"
    marian_decoder = f'"{args.marian}"/marian-decoder'
    marian_log_file = f"{artifacts_prefix}.log"
    language_pair = f"{src}-{trg}"
    metrics_file = f"{artifacts_prefix}.metrics"
    metrics_json = f"{artifacts_prefix}.metrics.json"

    if args.marian_args[0] != "--":
        print(args.marian_args)
        raise Exception("Expected to find marian args at the end of the command")
    marian_args = " ".join(args.marian_args[1:])

    print("The eval script is configured with the following:")
    print(" >          artifacts_dir:", artifacts_dir)
    print(" > source_file_compressed:", source_file_compressed)
    print(" >            source_file:", source_file)
    print(" >            target_file:", target_file)
    print(" >        target_ref_file:", target_ref_file)
    print(" >         marian_decoder:", marian_decoder)
    print(" >        marian_log_file:", marian_log_file)
    print(" >          language_pair:", language_pair)
    print(" >           metrics_file:", metrics_file)
    print(" >           metrics_json:", metrics_json)
    print(" >            marian_args:", marian_args)

    print("Ensure that the artifacts directory exists.")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Save the original target sentences to the artifacts")

    run_bash_oneliner(
        f"""
        {args.compression_cmd} -dc "{target_file_compressed}" > "{target_ref_file}"
        """
    )

    run_bash_oneliner(
        f"""
        # Decompress the source file, e.g. $fetches/wmt09.en.gz
        {args.compression_cmd} -dc "{source_file_compressed}"

        # Tee the source file into the artifacts directory, e.g. $artifacts/wmt09.en
        | tee "{source_file}"

        # Take the source and pipe it in to be decoded (translated) by Marian.
        | {marian_decoder}
            --config {args.decoder_config}
            --quiet
            --quiet-translation
            --log {marian_log_file}
            {marian_args}

        # The translations be "tee"ed out to the artifacts, e.g. $artifacts/wmt09.ca
        | tee "{target_file}"

        # Take in the translations via stdin and run the evaluation using sacrebleu.
        # The translations will be compared to the original corpus via the target_ref_file
        | sacrebleu "{target_ref_file}"
            --language-pair {language_pair}
            --metrics bleu chrf

        # Finally, tee out the metrics to the json file.
        | tee "{metrics_json}"
        """
    )

    print("Look up the bleu and chrf from the output files.")
    bleu = None
    chrf = None
    with open(metrics_json) as f:
        for metric in json.load(f):
            # Example json: [
            #   {
            #     "name": "BLEU",
            #     "score": 0.4,
            #     "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
            #     "verbose_score": "13.5/0.2/0.1/0.1 (BP = 0.955 ratio = 0.956 hyp_len = 215 ref_len = 225)",
            #     "nrefs": "1",
            #     "case": "mixed",
            #     "eff": "no",
            #     "tok": "13a",
            #     "smooth": "exp",
            #     "version": "2.0.0"
            #   },
            #   {
            #     "name": "chrF2",
            #     "score": 1.0,
            #     "signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.0.0",
            #     "nrefs": "1",
            #     "case": "mixed",
            #     "eff": "yes",
            #     "nc": "6",
            #     "nw": "0",
            #     "space": "no",
            #     "version": "2.0.0"
            #   }
            # ]
            if metric["name"] == "BLEU":
                bleu = metric["score"]
            elif metric["name"] == "chrF2":
                chrf = metric["score"]
            else:
                raise Exception("Could not find the BLEU or chrF2 in the sacrebleu metrics")

    if not bleu:
        raise Exception("Could not find the BLEU score")
    if not chrf:
        raise Exception("Could not find the chrF score")

    print('Also save the metrics in older "text" format.')
    with open(metrics_file, "w") as file:
        file.write(f"{bleu}\n{chrf}\n")


if __name__ == "__main__":
    main()
