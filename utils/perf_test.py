import argparse
from pathlib import Path
import subprocess
from typing import Optional
import taskcluster

from pipeline.common.downloads import stream_download_to_file

queue = taskcluster.Queue({"rootUrl": "https://firefox-ci-tc.services.mozilla.com"})

models_dir = Path(__file__).parent / "../data/quantized_models"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Wasm perform test")
    parser.add_argument(
        "--task_group_id",
        type=str,
        required=True,
        help="A task group with an export-{src}-{trg} task.",
    )

    return parser.parse_args()


def get_language_pair_from_task_name(task_name: str):
    return task_name.split("-")[-2:]


def find_model_file(model_dir: Path, part: str) -> Optional[Path]:
    for file in model_dir.glob("**/*"):
        if part in file.name:
            return file
    return None


class ModelFiles:
    def __init__(self, model_path: Path, task_id: str):
        self.model_path = model_path
        self.task_id = task_id
        self.artifacts = queue.listLatestArtifacts(task_id)["artifacts"]

    def download_artifact(self, part: str):
        artifact = self.find_artifact(part)
        name: str = artifact["name"]
        file_name = name.split("/")[-1]
        file_path = self.model_path / file_name
        stream_download_to_file(
            queue.buildUrl("getLatestArtifact", self.task_id, name),
            file_path,
        )
        return file_path

    def get(self, part: str):
        """Returns the file path to a model, and downloads it if necessary"""
        file_path = find_model_file(self.model_path, part)
        if not file_path:
            file_path = self.download_artifact(part)
        if not file_path:
            raise Exception(f'Could not find file for the part "{part}"')
        return file_path

    def find_artifact(self, part: str):
        for artifact in self.artifacts:
            if part in artifact["name"]:
                return artifact
        raise Exception("Could not find artifact")


def main() -> None:
    args = parse_args()
    task_group_id: str = args.task_group_id

    response = queue.listTaskGroup(task_group_id)
    tasks = response["tasks"]

    export_task = None
    for task in tasks:
        name = task["task"]["metadata"]["name"]
        if name.startswith("export-"):
            export_task = task
            break

    if task["status"]["state"] != "completed":
        raise Exception("The export-{src}-{trg} task was not completed")

    if not export_task:
        raise Exception("Could not find the export-{src}-{trg} task")

    task_id = task["status"]["taskId"]
    task_name = task["task"]["metadata"]["name"]

    print(task_name)
    print(f"https://firefox-ci-tc.services.mozilla.com/tasks/{task_id}")

    language_pair = "-".join(task_name.split("-")[-2:])

    model_path = (models_dir / f"{language_pair}-{task_id}").resolve()
    if not model_path.exists():
        print(f"Creating model directory: {model_path}")
        model_path.mkdir(parents=True, exist_ok=True)

    model_files = ModelFiles(model_path, task_id)

    lexical_shortlist = model_files.get(".s2t.bin")
    marian_bin = model_files.get(".intgemm.alphas.bin")
    vocab = model_files.get(".spm.")

    print("\nModel files are available at:")
    print(f"{model_path}")
    print("")
    print(f"  lexical_shortlist: {lexical_shortlist.name}")
    print(f"              model: {marian_bin.name}")
    print(f"              vocab: {vocab.name}")

    wasm_test_dir = (Path(__file__).parent / "../inference/wasm/tests").resolve()
    subprocess.check_call(["npm", "run", "test"], cwd=wasm_test_dir)

    return model_path


if __name__ == "__main__":
    main()
