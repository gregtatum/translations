"""
Continue the training pipeline with an existing model.
"""
import argparse
from pathlib import Path
from pipeline.common.downloads import stream_download_to_file, location_exists
from pipeline.common.logging import get_logger
from pipeline.common import arg_utils

logger = get_logger(__file__)

potential_models = (
    "model.npz",
    "model.npz.best-bleu-detok.npz",
    "model.npz.best-ce-mean-words.npz",
    "final.model.npz.best-chrf.npz",
    "model.npz.best-chrf.npz",
    "model.npz.optimizer.npz",
)

potential_decoders = [
    "model.npz.best-bleu-detok.npz.decoder.yml",
    "model.npz.best-ce-mean-words.npz.decoder.yml",
    "final.model.npz.best-chrf.npz.decoder.yml",
    "model.npz.best-chrf.npz.decoder.yml",
    "model.npz.decoder.yml",
    "model.npz.progress.yml",
    "model.npz.yml",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--src_locale", type=str, required=True, help="The source language for this model"
    )
    parser.add_argument(
        "--trg_locale", type=str, required=True, help="The target language for this model"
    )
    parser.add_argument("--url_prefix", type=str, required=True, help="The prefix for the URLs")
    parser.add_argument("--vocab_src", type=str, help="The source vocab file")
    parser.add_argument(
        "--vocab_trg", type=str, help="The target vocab file, potentially the same as the source"
    )
    parser.add_argument(
        "--best_model",
        type=str,
        required=True,
        help="The metric used to determine the best model, e.g. chrf, bleu, etc.",
    )
    parser.add_argument(
        "--artifacts", type=Path, help="The location where the models will be saved"
    )

    args = parser.parse_args()

    src_locale = arg_utils.ensure_string("--src_locale", args.src_locale)
    trg_locale = arg_utils.ensure_string("--trg_locale", args.trg_locale)
    url_prefix = arg_utils.ensure_string("--url_prefix", args.url_prefix)
    best_model = arg_utils.ensure_string("--best_model", args.best_model)
    src_vocab_url = arg_utils.handle_none_value(args.vocab_src)
    trg_vocab_url = arg_utils.handle_none_value(args.vocab_trg)
    artifacts: Path = args.artifacts

    model_out = artifacts / f"/final.model.npz.{best_model}.npz"
    decoder_out = artifacts / f"/final.model.npz.{best_model}.npz.decoder.yml"

    model_found = False
    for potential_model in potential_models:
        url = f"{url_prefix}/{potential_model}"
        print("Checking to see if a model exists:", potential_model)
        if location_exists(url):
            stream_download_to_file(url, model_out)
            model_found = True
    assert model_found

    decoder_found = False
    for potential_decoder in potential_decoders:
        url = f"{url_prefix}/{potential_decoder}"
        print("Checking to see if a decoder.yml exists:", potential_decoder)
        if location_exists(url):
            stream_download_to_file(url, decoder_out)
            decoder_found = True
    assert decoder_found

    # Prefer the vocab near the model.
    if not src_vocab_url or not trg_vocab_url:
        if location_exists(f"{url_prefix}/vocab.spm"):
            src_vocab_url = f"{url_prefix}/vocab.spm"
            trg_vocab_url = f"{url_prefix}/vocab.spm"
        elif location_exists(f"{url_prefix}/vocab.{src_locale}.spm") and location_exists(
            f"{url_prefix}/vocab.{trg_locale}.spm"
        ):
            src_vocab_url = f"{url_prefix}/vocab.{src_locale}.spm"
            trg_vocab_url = f"{url_prefix}/vocab.{trg_locale}.spm"
        else:
            raise ValueError(
                "Attempted to run model continuation but no vocab was found or provided. "
                'Add one to the config at "continuation.vocab.src" and '
                ' "continuation.vocab.trg"'
            )

    # TODO - Change to the other "if" branch when split vocab lands:
    # See: https://github.com/mozilla/translations/pull/1051
    if True:
        src_destination = artifacts / "vocab.spm"
        trg_destination = artifacts / "vocab.spm"
    else:
        src_destination = artifacts / f"vocab.{src_locale}.spm"
        trg_destination = artifacts / f"vocab.{trg_locale}.spm"

    stream_download_to_file(src_vocab_url, src_destination)
    stream_download_to_file(trg_vocab_url, trg_destination)


if __name__ == "__main__":
    main()
