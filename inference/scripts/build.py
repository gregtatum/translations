#!/usr/bin/env python3

import argparse
from glob import glob
import subprocess
from pathlib import Path
import multiprocessing
import tarfile
from typing import Optional
from detect_docker import detect_docker

ROOT_DIR = (Path(__file__).parent / "../..").resolve()
BUILD_DIR = ROOT_DIR / "inference/build"


def main():
    parser = argparse.ArgumentParser(
        description="Run cmake and make from the inference directory."
    )
    parser.add_argument("--test", action="store_true", help="Compile the test code as well")
    parser.add_argument(
        "--archive", type=Path, default=None, help="Save the binary files to a .tar.zst archive"
    )
    args = parser.parse_args()
    test: bool = args.test
    archive: Optional[Path] = args.archive

    detect_docker()

    if not BUILD_DIR.exists():
        print(f"[build] Creating {BUILD_DIR.relative_to(ROOT_DIR)} directory...")
        BUILD_DIR.mkdir()
    else:
        print(f"[build] {BUILD_DIR.name} directory already exists. Skipping creation.")

    print(f"[build] Running cmake for {BUILD_DIR.name}...")
    cmake_args = ["cmake", "../"]
    if test:
        cmake_args.append("-DCOMPILE_TESTS=ON")
    subprocess.run(cmake_args, cwd=BUILD_DIR, check=True)

    cpus = multiprocessing.cpu_count()
    print(f"[build] Running make with {cpus} CPUs...")
    subprocess.run(["make", f"-j{cpus}"], cwd=BUILD_DIR, check=True)

    if archive:
        assert str(archive).endswith(
            ".tar.zst"
        ), f"The archive must end with .tar.zst: {archive.name}"

        if not archive.parent.exists():
            print(f"[build] Creating directory: {archive.parent}")
            archive.mkdir(parents=True)

        print("[build] Collecting build artifacts for compression...")

        files = []
        files.extend(glob(str(BUILD_DIR / "marian*")))
        files.extend(glob(str(BUILD_DIR / "spm*")))
        files = [Path(f) for f in files if Path(f).is_file()]

        # e.g. "marian-fork.tar.zst" -> "marian-fork.tar"
        tar_path = archive.with_suffix("")

        print(f"[build] Creating archive: {tar_path}")
        with tarfile.open(tar_path, "w") as tar:
            for file_path in files:
                tar.add(file_path, arcname=file_path.name)

        print(f"[build] Compressing archive to: {archive}")
        subprocess.run(["zstd", "-f", tar_path, "-o", archive], check=True)

        print(f"[build] Removing uncompressed archive: {tar_path}")
        tar_path.unlink()


if __name__ == "__main__":
    main()
