import json
import os
import shlex
import shutil
import subprocess
import sys
from subprocess import CompletedProcess
from typing import Optional

import zstandard as zstd

from utils.preflight_check import get_taskgraph_parameters, run_taskgraph

FIXTURES_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(FIXTURES_PATH, "../../data"))
TESTS_DATA = os.path.join(DATA_PATH, "tests_data")


en_sample = """The little girl, seeing she had lost one of her pretty shoes, grew angry, and said to the Witch, “Give me back my shoe!”
“I will not,” retorted the Witch, “for it is now my shoe, and not yours.”
“You are a wicked creature!” cried Dorothy. “You have no right to take my shoe from me.”
“I shall keep it, just the same,” said the Witch, laughing at her, “and someday I shall get the other one from you, too.”
This made Dorothy so very angry that she picked up the bucket of water that stood near and dashed it over the Witch, wetting her from head to foot.
Instantly the wicked woman gave a loud cry of fear, and then, as Dorothy looked at her in wonder, the Witch began to shrink and fall away.
“See what you have done!” she screamed. “In a minute I shall melt away.”
“I’m very sorry, indeed,” said Dorothy, who was truly frightened to see the Witch actually melting away like brown sugar before her very eyes.
"""

ru_sample = """Маленькая девочка, увидев, что потеряла одну из своих красивых туфелек, рассердилась и сказала Ведьме: «Верни мне мою туфельку!»
«Я не буду, — парировала Ведьма, — потому что теперь это моя туфля, а не твоя».
«Ты злое существо!» - воскликнула Дороти. «Ты не имеешь права забирать у меня туфлю».
«Я все равно сохраню его, — сказала Ведьма, смеясь над ней, — и когда-нибудь я получу от тебя и другой».
Это так разозлило Дороти, что она взяла стоявшее рядом ведро с водой и облила им Ведьму, обмочив ее с головы до ног.
Мгновенно злая женщина громко вскрикнула от страха, а затем, когда Дороти с удивлением посмотрела на нее, Ведьма начала сжиматься и падать.
«Посмотри, что ты наделал!» она закричала. «Через минуту я растаю».
«Мне действительно очень жаль», — сказала Дороти, которая была по-настоящему напугана, увидев, что Ведьма тает, как коричневый сахар, у нее на глазах.
"""


class DataDir:
    """
    Creates a persistent data directory in data/tests_data/{dir_name} that will be
    cleaned out before a test run. This should help in persisting artifacts between test
    runs to manually verify the results.
    """

    def __init__(self, dir_name: str) -> None:
        self.path = os.path.join(TESTS_DATA, dir_name)

        # Ensure the base /data directory exists.
        os.makedirs(TESTS_DATA, exist_ok=True)

        # Clean up a previous run if this exists.
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

        os.makedirs(self.path)
        print("Tests are using the subdirectory:", self.path)

    def join(self, name: str):
        """Create a folder or file name by joining it to the test directory."""
        return os.path.join(self.path, name)

    def load(self, name: str):
        """Load a text file"""
        with open(self.join(name), "r") as file:
            return file.read()

    def create_zst(self, name: str, contents: str) -> str:
        """
        Creates a compressed zst file and returns the path to it.
        """
        zst_path = os.path.join(self.path, name)
        if not os.path.exists(self.path):
            raise Exception(f"Directory for the compressed file does not exist: {self.path}")
        if os.path.exists(zst_path):
            raise Exception(f"A file already exists and would be overwritten: {zst_path}")

        # Create the compressed file.
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(contents.encode("utf-8"))

        print("Writing a compressed file to: ", zst_path)
        with open(zst_path, "wb") as file:
            file.write(compressed_data)

        return zst_path

    def create_file(self, name: str, contents: str) -> str:
        """
        Creates a text file and returns the path to it.
        """
        text_path = os.path.join(self.path, name)
        if not os.path.exists(self.path):
            raise Exception(f"Directory for the text file does not exist: {self.path}")
        if os.path.exists(text_path):
            raise Exception(f"A file already exists and would be overwritten: {text_path}")

        print("Writing a text file to: ", text_path)
        with open(text_path, "w") as file:
            file.write(contents)

        return text_path


def fail_on_error(result: CompletedProcess[bytes]):
    """When a process fails, surface the stderr."""
    if not result.returncode == 0:
        for line in result.stderr.decode("utf-8").split("\n"):
            print(line, file=sys.stderr)

        raise Exception(f"{result.args[0]} exited with a status code: {result.returncode}")


# Only (lazily) create the full taskgraph once per test suite run.
_full_taskgraph: Optional[dict[str, object]] = None


def get_full_taskgraph():
    """
    Generates the full taskgraph and stores it for re-use. It uses the config.pytest.yml
    in this directory.
    """
    global _full_taskgraph
    if _full_taskgraph:
        return _full_taskgraph
    current_folder = os.path.dirname(os.path.abspath(__file__))
    config = os.path.join(current_folder, "config.pytest.yml")
    task_graph_json = os.path.join(current_folder, "../../artifacts/full-task-graph.json")

    run_taskgraph(config, get_taskgraph_parameters())

    with open(task_graph_json, "rb") as file:
        _full_taskgraph = json.load(file)
    return _full_taskgraph


def get_task_command_and_env(task_name: str, script=None) -> tuple[str, dict[str, str]]:
    """
    Extracts a task's command from the full taskgraph. This allows for testing
    the full taskcluster pipeline and the scripts that it generates.
    See artifacts/full-task-graph.json for the full list of what is generated.

    task_name - The full task name like "split-mono-src-en"
        or "evaluate-backward-sacrebleu-wmt09-en-ru".
    script - If this is provided, then it will return all of the arguments provided
        to a script, and ignore everything that came before it.
    """
    full_taskgraph = get_full_taskgraph()
    task = full_taskgraph[task_name]

    env = task["task"]["payload"]["env"]

    # The commands typically take the form:
    #  [
    #    ['chmod', '+x', 'run-task'],
    #    ['./run-task', '--firefox_translations_training-checkout=./checkouts/vcs/', '--', 'bash', '-c', "full command"]
    #  ]
    commands = task["task"]["payload"]["command"]

    # Get the "full command" from ['./run-task', ..., "full command"]
    command = commands[-1][-1]

    if script:
        # Return the parts after the script name.
        parts = command.split(script)
        if len(parts) != 2:
            raise Exception(f"Could not correctly find {script} in: {command}")
        command = parts[1].strip()

    # Return the full command.
    return command, env


def run_task(
    task_name: str,
    script: str,
    work_dir: str,
    fetches_dir: str,
    env: dict[str, str] = {},
):
    """
    Runs a task from the taskgraph. See artifacts/full-task-graph.json after running a
    test for the full list of task names

    Arguments:

    task_name - The full task name like "split-mono-src-en"
        or "evaluate-backward-sacrebleu-wmt09-en-ru".

    script - The name of the script (bash or python) that is running. This is used
        since the full command generally contains things that shouldn't run in the
        test suite, like pip installs.

    work_dir - This is the TASK_WORKDIR, in tests generally the test's DataDir.

    fetches_dir - The MOZ_FETCHES_DIR, generally set as the test's DataDir.

    env - Any environment variable overrides.
    """
    if not task_name:
        raise Exception("Expected a task_name")
    if not script:
        raise Exception("Expected a script")
    if not work_dir:
        raise Exception("Expected a work_dir")
    if not fetches_dir:
        raise Exception("Expected a fetches_dir")

    command, task_env = get_task_command_and_env(task_name, script=script)

    # There are some non-string environment variables that involve taskcluster references
    # Remove these.
    for key in task_env:
        if not isinstance(task_env[key], str):
            task_env[key] = ""

    current_folder = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_folder, "../.."))

    command_env = {
        **os.environ,
        **task_env,
        "TASK_WORKDIR": work_dir,
        "MOZ_FETCHES_DIR": fetches_dir,
        "VCS_PATH": root_path,
        **env,
    }

    # Manually apply the environment variables, as they don't get added to the args
    # through the subprocess.run
    command = command.replace("$TASK_WORKDIR/$VCS_PATH", root_path)
    command = command.replace("$TASK_WORKDIR", work_dir)
    command = command.replace("$MOZ_FETCHES_DIR", fetches_dir)

    # Log the command in case tests fail.
    print("Running command:", command)

    result = subprocess.run(
        [script, *shlex.split(command)],
        env=command_env,
        cwd=root_path,
        stderr=subprocess.PIPE,
        check=False,
    )
    fail_on_error(result)
