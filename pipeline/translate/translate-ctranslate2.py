import argparse
from enum import Enum
from glob import glob
from pathlib import Path

import ctranslate2
import sentencepiece as spm
from ctranslate2.converters.marian import MarianConverter

from pipeline.common.logging import get_logger


def load_vocab(path: str):
    logger.info("Loading vocab:")
    logger.info(path)
    sp = spm.SentencePieceProcessor(path)

    return [sp.id_to_piece(i) for i in range(sp.vocab_size())]


# The vocab expects a .yml file. Instead directly load the vocab .spm file via a monkey patch.
if not ctranslate2.converters.marian.load_vocab:
    raise Exception("Expected to be able to monkey patch the load_vocab function")
ctranslate2.converters.marian.load_vocab = load_vocab

"""
Translate a corpus with a teacher model (transformer-based) using CTranslate2. This is useful
to quickly synthesize training data for student distillation as CTranslate2 is X (TODO) times faster
than Marian.

https://github.com/OpenNMT/CTranslate2/

python pipeline/translate/translate-ctranslate2.py \\
    --teacher_model en-ca.npz                      \\
    --vocab         vocab.spm                      \\
    --device        cpu
"""

logger = get_logger(__file__)


class Device(Enum):
    gpu = "gpu"
    cpu = "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--teacher_models",
        type=str,
        required=True,
        help="The path (glob) to the Marian (.npz) teacher models (must be transfomer-based)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        nargs="+",
        required=True,
        help="The path to the SentencePiece (.spm) file",
    )
    parser.add_argument(
        "--device",
        type=Device,
        choices=list(Device),
        default=Device.gpu,
        help="Which device to use",
    )
    parser.add_argument("--data", type=str, help="The .zst data file to translate")
    parser.add_argument("--output_dir", type=str, help="Where the translations will be stored")

    args = parser.parse_args()

    marian_teacher_models: list[str] = glob(args.teacher_models)

    if not marian_teacher_models:
        logger.info(f"teacher_models: {args.teacher_models}")
        raise Exception("Expected there to be a marian teacher model.")

    tokenizer_src = spm.SentencePieceProcessor(args.vocab[0])
    if len(args.vocab) == 1:
        tokenizer_trg = tokenizer_src
    else:
        tokenizer_trg = spm.SentencePieceProcessor(args.vocab[1])

    for i, marian_teacher_model in enumerate(marian_teacher_models):
        ctranslate2_model_dir = Path(args.output_dir) / f"{Path(marian_teacher_model).stem}-{i}"
        logger.info("Converting the Marian model to Ctranslate2:")
        logger.info(marian_teacher_model)
        logger.info("Outputing model to:")
        logger.info(ctranslate2_model_dir)

        converter = MarianConverter(marian_teacher_model, args.vocab)
        converter.convert(ctranslate2_model_dir)

        translator = ctranslate2.Translator(str(ctranslate2_model_dir), device="cpu")
        logger.info("Loading model")
        translator.load_model()
        logger.info("Model loaded")

        input_text = "ANTONI En bona fe, no sé esplicarme la raó de la meva tristesa. Me preocupa, y a vosaltres també, segons diheu. Mes no puc endevinar com m'ha vingut, perque de mí s'ha apoderat, d'aon naix, ni de quina fusta es feta. Lo cert es que'm té aturdit fins a tal punt, que quasi jo mateix me desconec."
        input_tokens = tokenizer_src.encode(input_text, out_type=str)

        logger.info("Translating")
        translate_results = translator.translate_batch(
            [input_tokens], beam_size=4, return_scores=True
        )

        logger.info(f"Translate: {input_text}")
        for hypothesis, score in zip(translate_results[0].hypotheses, translate_results[0].scores):
            logger.info(f"> {tokenizer_trg.decode(hypothesis)}")
            logger.info(f"Score: {score}")


if __name__ == "__main__":
    main()
