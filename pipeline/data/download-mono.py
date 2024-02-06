#!/usr/bin/env python3
"""
Download a monolingual dataset, shuffle it, and truncate it to a max size.
"""


import random
import subprocess


def decompress_zstd(file_path):
    """Decompress a file using zstd."""
    result = subprocess.run(["zstd", "-dc", file_path], capture_output=True)
    return result.stdout.decode()


def compress_zstd(data, output_path):
    """Compress data using zstd and save to a file."""
    process = subprocess.Popen(["zstd", "-o", output_path], stdin=subprocess.PIPE)
    process.communicate(input=data.encode())


def process_file(filename, output_path, max_sent=1000000, coef=0.1):
    decompressed_data = decompress_zstd(filename)

    # Shuffle lines
    lines = decompressed_data.split("\n")
    random.shuffle(lines)

    # Calculate the number of lines to select
    num_lines_to_select = int((max_sent + max_sent * coef) / 1)

    # Filter lines with less than 100 words and limit to max_sent
    filtered_lines = [line for line in lines if len(line.split()) < 100][:max_sent]

    # Compress and save the output
    compress_zstd("\n".join(filtered_lines), output_path)


# Example usage
process_file("path_to_input_file.zst", "path_to_output_file.zst")
