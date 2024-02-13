from typing import Iterator

import pytest

from pipeline.common.datasets import shuffle_with_max_lines

ITEMS = 100_000
PERCENTAGE = 0.2
MAX_LINES = ITEMS * PERCENTAGE


def get_total_byte_size(lines: list[str]) -> int:
    total_byte_size = 0
    for line in lines:
        total_byte_size = total_byte_size + len(line.encode())
    return total_byte_size


def compute_distribution(lines: Iterator[str]) -> list[float]:
    """
    Computes a histogram (list of 10 items) with a percentage value of 0.0 - 100.0 for each item.
    """
    histogram = [0] * 10
    for line in lines:
        # This assumes the content will be a tab separated list, with the first item to be the
        # initial sorted order in the list.
        key = int(int(line.split("\t")[0]) * 10 / ITEMS)
        histogram[key] = histogram[key] + (1 / MAX_LINES)

    # Go from 0 - 1 floats to 0 - 100 ints.
    return [round(value * 1000) / 10 for value in histogram]


# Test the distributions of the different types of datasets. This shuffler estimates the content
# size as it iterates through the line stream.
shuffle_params = [
    (
        # Each line is the same bytes as the next line. This should create an even distribution.
        # [
        #     "000000000 000000000 000000000 ... 000000000",
        #     "000000001 000000001 000000001 ... 000000001",
        #     ...
        # ]
        "even distribution",
        [f"{line:09d}\t" * 10 for line in range(ITEMS)],
        [10.2, 10.1, 9.9, 10.0, 10.0, 10.2, 9.9, 9.7, 10.0, 10.0],
    ),
    (
        # The initial lines are low in byte count, and gradually increase. In this case there
        # will be a bias to over-sample the the initial items, but it will eventually even out as
        # more bytes are read in and the average spreads out.
        # [
        #     "0 0 0 ... 0",
        #     "1 1 1 ... 1",
        #     ...
        #     "99997 99997 99997 ... 99997",
        #     "99998 99998 99998 ... 99998",
        #     "99999 99999 99999 ... 99999",
        # ]
        "small content at start",
        [f"{line}\t" * 10 for line in range(ITEMS)],
        [11.4, 11.6, 9.2, 9.5, 9.6, 9.9, 9.7, 9.5, 9.8, 9.8],
        # |      |    |
        # |      |    ^ Lower sample rate.
        # ^^^^^^^^ Higher sampling rate.
    ),
    (
        # [
        #     "99999 99999 99999 ... 99999",
        #     "99998 99998 99998 ... 99998",
        #     "99997 99997 99997 ... 99997",
        #     ...
        #     "1 1 1 ... 1",
        #     "0 0 0 ... 0",
        # ]
        "large content at start",
        [f"{line}\t" * 10 for line in range(ITEMS)][::-1],
        [10.1, 10.2, 9.9, 10.2, 10.3, 10.2, 10.1, 10.2, 10.2, 8.6],
        #                                   lower sample rate ^^^
    ),
]


@pytest.mark.parametrize("params", shuffle_params, ids=[d[0] for d in shuffle_params])
def test_shuffle_with_max_lines(params):
    description, line_stream, histograph = params
    # [
    #     "0000 0000 0000 ... 0000",
    #     "0001 0001 0001 ... 0001",
    #     "0002 0002 0002 ... 0002",
    #     ...
    # ]

    output = shuffle_with_max_lines(
        line_stream,
        seed="test",
        max_lines=MAX_LINES,
        max_words_in_sentence=100,
        total_byte_size=get_total_byte_size(line_stream),
    )

    assert compute_distribution(output) == histograph, description
