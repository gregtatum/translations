import argparse
import subprocess
import sys

import sacrebleu


def download_and_compress(src: str, trg: str, output_prefix: str, dataset: str) -> bool:
    try:
        for lang, echo in [(src, "src"), (trg, "ref")]:
            result = subprocess.run(
                [
                    "sacrebleu",
                    "--test-set",
                    dataset,
                    "--language-pair",
                    f"{src}-{trg}",
                    "--echo",
                    echo,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output_file = f"{output_prefix}.{lang}.zst"
            compress_with_zstd(result.stdout, output_file)
        return True
    except subprocess.CalledProcessError:
        return False


def compress_with_zstd(data: str, output_file: str) -> None:
    process = subprocess.Popen(
        ["zstdmt", "-c"],
        stdin=subprocess.PIPE,
        stdout=open(output_file, "wb"),
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=data.encode())
    if process.returncode != 0:
        raise RuntimeError(f"Compression failed: {stderr.decode()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and compress a sacrebleu test set.")
    parser.add_argument("src", type=str, help="Source language")
    parser.add_argument("trg", type=str, help="Target language")
    parser.add_argument("output_prefix", type=str, help="Prefix for output files")
    parser.add_argument("dataset", type=str, help="Dataset name")

    args = parser.parse_args()

    print("###### Downloading sacrebleu corpus")

    success = download_and_compress(args.src, args.trg, args.output_prefix, args.dataset)

    if not success:
        print("The first import failed, try again by switching the language pair direction.")
        # Retry in reversed direction
        reversed_success = download_and_compress(
            args.trg, args.src, args.output_prefix, args.dataset
        )
        if not reversed_success:
            print("Both attempts failed.", file=sys.stderr)
            sys.exit(1)

    print("###### Done: Downloading sacrebleu corpus")


if __name__ == "__main__":
    main()
