#!/bin/bash

# Quick truncation script.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file=$1
num_lines=92218925
# num_lines=61479283
# num_lines=30739641

temp_file="${input_file%.zst}_sampled.zst"

# Uncompress, shuffle, truncate, and recompress
zstdcat "$input_file" | shuf -n "$num_lines" | zstd -o "$temp_file"

# Replace the original file with the truncated version
mv "$temp_file" "$input_file"

echo "Sampled $num_lines lines and replaced $input_file with the truncated version"
