import sys
from time import sleep


def read_and_write_to_file(output_file):
    with open(output_file, "w") as f:
        f.write("")

    with open(output_file, "a") as f:  # Open in append mode
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            f.write(line)
            f.flush()


read_and_write_to_file("output.txt")
