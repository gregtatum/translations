from typing import Iterator


def shuffle_with_max_lines(
    line_stream: Iterator[str], dataset: Dataset, file_destination: str, max_sentences: int
):
    lines: list[str] = []

    logger.info("Load the stream into memory, discarding sentences that are too long.")
    for line in line_stream:
        if len(line.split()) < MAX_WORDS_IN_SENTENCE:
            lines.append(line)

    logger.info("Perform an in-memory shuffle of the dataset.")
    random = Random(dataset.key)  # Make this deterministic based on dataset key.
    random.shuffle(lines)

    logger.info("Write out the lines, truncated to the max number of sentences.")
    with open(file_destination, "wb") as compressed_file:
        with zstandard.ZstdCompressor().stream_writer(compressed_file) as writer:
            for line in lines[:max_sentences]:
                # The newline is already included.
                writer.write(line.encode("utf-8"))
