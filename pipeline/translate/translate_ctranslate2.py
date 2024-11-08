"""
Translate a corpus with a teacher model (transformer-based) using CTranslate2. This is useful
to quickly synthesize training data for student distillation as CTranslate2 is X (TODO) times faster
than Marian.

https://github.com/OpenNMT/CTranslate2/
"""

from typing import Any, Generator, TextIO
from enum import Enum
from glob import glob
from pathlib import Path

import ctranslate2
import sentencepiece as spm
from ctranslate2.converters.marian import MarianConverter

from pipeline.common.downloads import read_lines, write_lines
from pipeline.common.logging import get_logger
from pipeline.common.marian import get_combined_config


def load_vocab(path: str):
    logger.info("Loading vocab:")
    logger.info(path)
    sp = spm.SentencePieceProcessor(path)

    return [sp.id_to_piece(i) for i in range(sp.vocab_size())]


# The vocab expects a .yml file. Instead directly load the vocab .spm file via a monkey patch.
if not ctranslate2.converters.marian.load_vocab:
    raise Exception("Expected to be able to monkey patch the load_vocab function")
ctranslate2.converters.marian.load_vocab = load_vocab

logger = get_logger(__file__)


class Device(Enum):
    gpu = "gpu"
    cpu = "cpu"


class MaxiBatchSort(Enum):
    src = "src"
    none = "none"


def get_model(models_glob: str) -> Path:
    models = glob(models_glob)
    if not models:
        raise ValueError(f'No model was found with the glob "{models_glob}"')
    if len(models) != 1:
        logger.info(f"Found models {models}")
        raise ValueError("Ensemble training is not supported in CTranslate2")
    return Path(models[0])


class DecoderConfig:
    def __init__(self, extra_marian_args: list[str]) -> None:
        super().__init__()
        # Combine the two configs.
        self.config = get_combined_config(Path(__file__).parent / "decoder.yml", extra_marian_args)

        self.mini_batch_words: int = self.get_from_config("mini-batch-words", int)
        self.maxi_batch_sentences: int = self.get_from_config("maxi-batch", int)
        self.beam_size: int = self.get_from_config("beam-size", int)
        self.maxi_batch_sort = MaxiBatchSort(self.get_from_config("maxi-batch-sort", str))
        self.precision = self.get_from_config("precision", str)

    def get_from_config(self, key: str, type: any):
        value = self.config.get(key, None)
        if value is None:
            raise ValueError(f'"{key}" could not be found in the decoder.yml config')
        if isinstance(value, type):
            return value
        if type == int and isinstance(value, str):
            return int(value)
        raise ValueError(f'Expected "{key}" to be of a type "{type}" in the decoder.yml config')


def make_batches(
    input_zst: Path,
    decoder_config: DecoderConfig,
    tokenizer_src: spm.SentencePieceProcessor,
) -> Generator[list[list[str]], None, None]:
    """
    Make batches of translations, sorted by sentence size. matching the methodology of Marian
    translations. This function yields "mini" batches.

    "maxi" and "mini" ares used like "maxi dress" and "mini dress". Each * below represents a token
    in a src sentence. The sentences are sorted from smallest to largest in the maxi batch, and then
    chunked into a mini batch. Note that "mini-batch-words" and "mini-batch" are mutually exclusive
    options. "mini-batch" uses the sentence count, so "mini-batch-words" is prefereable as a more
    stable metric.

            "maxi" batch
    ┌────────────────────────────┐
    │       "mini" batch         │
    │  ┌──────────────────────┐  │
    │  │ *                    │  │
    │  │ **                   │  │
    │  │ ****                 │  │
    │  │ ******               │  │
    │  │ ********             │  │
    │  │ *********            │  │
    │  │ *********            │  │
    │  └──────────────────────┘  │
    │       "mini" batch         │
    │  ┌──────────────────────┐  │
    │  │ *********            │  │
    │  │ **********           │  │
    │  │ **********           │  │
    │  │ ************         │  │
    │  │ *************        │  │
    │  └──────────────────────┘  │
    │       "mini" batch         │
    │  ┌──────────────────────┐  │
    │  │ ***************      │  │
    │  │ *****************    │  │
    │  │ ******************** │  │
    │  └──────────────────────┘  │
    └────────────────────────────┘
    """
    with read_lines(input_zst) as lines:
        while True:
            maxi_batch: list[list[str]] = []
            # Fill up the maxi_batch with maxi_batch_sentences size of tokenized sentences.
            for _ in range(decoder_config.maxi_batch_sentences):
                # Build up the "maxi" batch, which is a bigger collection of batches.
                try:
                    line: str = next(lines)
                    tokens = tokenizer_src.Encode(line.strip(), out_type=str)
                    maxi_batch.append(tokens)
                except StopIteration:
                    if maxi_batch:
                        break
                    else:
                        return

            if decoder_config.maxi_batch_sort == MaxiBatchSort.src:
                # Sort the maxi_batch from smallest to largest based on token length.
                # This should have a time complexity of O(n * log(n)).
                maxi_batch.sort(key=len)

            # Chunk out the maxi batch into mini batches.
            words_in_batch = 0
            batch: list[list[str]] = []
            for sentence in maxi_batch:
                batch.append(sentence)
                words_in_batch += len(sentence)
                if words_in_batch > decoder_config.mini_batch_words:
                    yield batch
                    batch = []
                    words_in_batch = 0

            # Yield the final batch.
            if batch:
                yield batch


def write_single_translation(
    _index: int, tokenizer_trg: spm.SentencePieceProcessor, result: Any, outfile: TextIO
):
    """
    Just write each single translation to a new line. If beam search was used all the other
    beam results are discarded.
    """
    line = tokenizer_trg.decode(result.hypotheses[0])
    outfile.write(line)
    outfile.write("\n")


def write_nbest_translations(
    index: int, tokenizer_trg: spm.SentencePieceProcessor, result: Any, outfile: TextIO
):
    """
    Match Marian's way of writing out nbest translations. For example, with a beam-size of 2 and
    collection nbest translations:

    0 ||| Translation attempt
    0 ||| An attempt at translation
    1 ||| The quick brown fox jumped
    1 ||| The brown fox quickly jumped
    ...
    """
    outfile.write(index)
    for hypothesis in result.hypotheses:
        line = tokenizer_trg.decode(hypothesis)
        outfile.write(f"{index} ||| {line}\n")


def translate_with_ctranslate2(
    input_zst: Path,
    artifacts: Path,
    extra_marian_args: list[str],
    models_glob: str,
    is_nbest: bool,
    vocab: list[str],
    device: str,
) -> None:
    model = get_model(models_glob)
    postfix = "nbest" if is_nbest else "out"
    assert not is_nbest, "TODO - nbest is not supported yet"

    tokenizer_src = spm.SentencePieceProcessor(vocab[0])
    if len(vocab) == 1:
        tokenizer_trg = tokenizer_src
    else:
        tokenizer_trg = spm.SentencePieceProcessor(vocab[1])

    if extra_marian_args and extra_marian_args[0] != "--":
        logger.error(" ".join(extra_marian_args))
        raise Exception("Expected the extra marian args to be after a --")

    decoder_config = DecoderConfig(extra_marian_args[1:])

    ctranslate2_model_dir = model.parent / f"{Path(model).stem}"
    logger.info("Converting the Marian model to Ctranslate2:")
    logger.info(model)
    logger.info("Outputing model to:")
    logger.info(ctranslate2_model_dir)

    converter = MarianConverter(model, vocab)
    converter.convert(ctranslate2_model_dir, quantization=decoder_config.precision)

    translator = ctranslate2.Translator(str(ctranslate2_model_dir), device=device)
    logger.info("Loading model")
    translator.load_model()
    logger.info("Model loaded")

    output_zst = artifacts / f"{input_zst.stem}.{postfix}.zst"

    num_hypotheses = 1
    write_translation = write_single_translation
    if is_nbest:
        num_hypotheses = decoder_config.beam_size
        write_translation = write_nbest_translations

    with write_lines(output_zst) as outfile:
        index = 0
        for batch in make_batches(input_zst, decoder_config, tokenizer_src):
            translate_results = translator.translate_batch(
                batch,
                beam_size=decoder_config.beam_size,
                return_scores=False,
                num_hypotheses=num_hypotheses,
            )
            for result in translate_results:
                write_translation(index, tokenizer_trg, result, outfile)
                index += 1
