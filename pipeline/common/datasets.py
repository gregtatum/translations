import os
import tempfile
from collections import deque
from io import TextIOWrapper
from random import Random
from typing import Iterator


class Dataset:
    """
    Convert a dataset key into a structured format.

    e.g.

    self.key             "bucket_releng-translations-dev/data/en-ru/pytest-dataset"
    self.importer:       "bucket"
    self.name:           "releng-translations-dev/data/en-ru/pytest-dataset"
    self.file_safe_name: "releng-translations-dev_data_en-ru_pytest-dataset"
    """

    def __init__(self, dataset_key: str) -> None:
        key_parts = dataset_key.split("_")

        self.key = dataset_key
        self.importer = key_parts[0]
        self.name = "_".join(key_parts[1:])
        self.file_safe_name = self.name.replace("/", "_").replace(".", "_")

        if not self.importer:
            raise Exception(f"Could not find the importer in the dataset key {dataset_key}")

        if not self.name:
            raise Exception(f"Could not find the name in the dataset key {dataset_key}")


def shuffle_with_max_lines(
    line_stream: Iterator[str],
    seed: str,
    max_lines: int,
    max_words_in_sentence,
    total_byte_size: int,
) -> Iterator[str]:
    """
    Shuffle a line stream, but only retain up to a maximum number of sentences in memory.
    Note that the final ordering is determined by the seed and the contents of the file. So
    running this multiple times on the same dataset will return the same results, but running
    it with the same seed and different content will create a different ordering.

    Only run for monolingual data or where the parallel sentences are separated by a delimiter.
    """
    lines = deque()

    random = Random(seed)  # Make this deterministic based on dataset key.

    total_bytes = 0

    # Fill up the lines up until the max, and measure the total bytes.
    for line in line_stream:
        # Encoding returns the underlying byte representation which is then measured.
        total_bytes = total_bytes + len(line.encode())

        if len(line.split()) > max_words_in_sentence:
            # TODO(CJK) - Issue #424
            # This sentence is too long.
            continue

        lines.append(line)

        if len(lines) == max_lines:
            break

    random.shuffle(lines)

    # Consume the rest of the line stream, but sample based on the probability that adding
    # something to the collection will be representative.

    i = 0
    for line in line_stream:
        i = i + 1
        # Continuously adjust this estimation in case the first sampled data is not representative.
        total_bytes = total_bytes + len(line.encode())
        average_bytes_per_line = total_bytes / (max_lines + i)
        estimated_lines = total_byte_size / average_bytes_per_line
        line_sampling_probability = max_lines / estimated_lines

        if random.random() < line_sampling_probability:
            # Shift the deque so the oldest line is shifted out, and this new sample is shifted in.
            lines.popleft()
            lines.append(line)

    # Do a final shuffle to ensure that the newly sampled lines are shuffled with the original
    # set of shuffled lines.
    random.shuffle(lines)

    return lines


def shuffle_in_temp_files(
    line_stream: Iterator[str], output: TextIOWrapper, seed: str, chunk_size: int, bucket_size: int
):
    """
    Shuffle large datasets by storing chunks to the file system. The ordering is guaranteed to be
    stable across two datasets as long as they are the same length. For instance it could be used
    to shuffle `dataset.en.zst` and `dataset.ca.zst` the same if the two are parallel sentences.

    Take in a stream of lines (from a download, or stdin) and split it out to chunks.

    tmpdir
    ├── chunk.1
    ├── chunk.2
    ├── chunk.3
    ├── chunk.4
    ├── ...
    └── chunk.100

    After the entire dataset is written to chunks, pick random chunks and put them into a
    bucket. Only one bucket is fully loaded into memory at a time, and the contents
    of the bucket is shuffled in memory.

    Bucket:
    ┌───────────┐
    │ chunk.85  │
    │ chunk.3   │
    │ chunk.52  │
    │ chunk.30  │
    │ chunk.12  │
    │ chunk.18  │
    └───────────┘

    • shuffle bucket lines
    • write to output

    At most 1 bucket will be held in memory. At most the dataset + 1 bucket of file space will be
    needed when running this algorithm.
    """
    tmp_dir = os.path.join(tempfile.gettempdir(), "shuffle")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    random = Random(seed)

    chunk_index = 0
    chunk_file = open(os.path.join(tmp_dir, f"chunk.{chunk_index}"), "wt")
    line_count = 0

    # Write out the chunks to disk.
    for line in line_stream:
        chunk_file.write(line)
        line_count += 1
        if line_count > chunk_size:
            line_count = 0
            chunk_index = chunk_index + 1
            chunk_file.close()
            chunk_file = open(os.path.join(tmp_dir, f"chunk.{chunk_index}"), "wt")
    chunk_file.close()

    # Shuffle the chunk indexes
    shuffled_chunk_indexes = random.shuffle([*range(chunk_index + 1)])

    # Put the chunks into buckets.
    buckets = []
    for i in range(0, len(shuffled_chunk_indexes), bucket_size):
        bucket = shuffled_chunk_indexes[i : i + bucket_size]
        buckets.append(bucket)

    # Load a single bucket into memory, discarding the chunks.
    for bucket in buckets:
        lines = []
        for chunk_index in bucket:
            chunk_name = os.path.join(tmp_dir, f"chunk.{chunk_index}")
            with open(chunk_name, "rt") as file:
                lines.append(file.readline())
            os.remove(chunk_name)

        output.writelines(lines)
