import json
import os
import subprocess
from io import BufferedReader
from textwrap import dedent, indent

import requests


def run_bash_oneliner(command: str):
    """
    Runs multi-line bash with comments as a one-line command.
    """
    command_dedented = dedent(command)

    # Remove comments and whitespace.
    lines = [
        line.strip() for line in command_dedented.split("\n") if line and not line.startswith("#")
    ]
    command = " \\\n".join(lines)

    print("-----------------Running bash in one line--------------")
    print(indent(command_dedented, "  "))
    print("-------------------------------------------------------")
    return subprocess.check_call(command, shell=True)


def sanitize_dataset_key(dataset_key: str) -> str:
    """Sanitize dataset keys so that they are safe for the file system"""
    return dataset_key.replace("/", "_").replace(".", "_")


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
        self.file_safe_name = sanitize_dataset_key(self.name)

        if not self.importer:
            raise Exception(f"Could not find the importer in the dataset key {dataset_key}")

        if not self.name:
            raise Exception(f"Could not find the name in the dataset key {dataset_key}")


class MockedResponse:
    def __init__(self, file_handle: BufferedReader) -> None:
        self.raw = file_handle

    def close(self) -> None:
        self.raw.close()


def attempt_mocked_request(url: str) -> requests.Response:
    if not os.environ.get("MOCKED_DOWNLOADS"):
        return None

    mocked_downloads = json.loads(os.environ.get("MOCKED_DOWNLOADS"))

    if not isinstance(mocked_downloads, dict):
        raise Exception(
            "Expected the mocked downloads to be a json object mapping the URL to file path"
        )

    source_file = mocked_downloads.get(url)
    if not source_file:
        print("MOCKED_DOWNLOADS:", mocked_downloads)
        raise Exception(f"Received a URL that was not in MOCKED_DOWNLOADS {url}")

    if not os.path.exists(source_file):
        raise Exception(f"The source file specified did not exist {source_file}")

    return MockedResponse(open(source_file, "rb"))
