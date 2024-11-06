import json
from pathlib import Path
import shutil

import pytest
from fixtures import DataDir, en_sample
from pipeline.common.command_runner import marian_args_to_dict

fixtures_path = Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir():
    data_dir = DataDir("test_translate")
    shutil.copyfile("tests/data/vocab.spm", data_dir.join("vocab.spm"))
    return data_dir


def sanitize_marian_args(data_dir: DataDir, args_list: list[str]):
    """
    Marian args can have details that reflect the host machine or are unique per run.
    Sanitize those here.
    """
    base_dir = str((Path(__file__).parent / "..").resolve())
    args_dict = marian_args_to_dict(args_list)
    for key, value in args_dict.items():
        if isinstance(value, list):
            for index, value_inner in enumerate(value):
                if value_inner.startswith("/tmp"):
                    args_dict[key][index] = "<tmp>/" + Path(value_inner).name
                if value_inner.startswith(base_dir):
                    args_dict[key][index] = value_inner.replace(base_dir, "<src>")
        else:
            if value.startswith("/tmp"):
                args_dict[key] = "<tmp>/" + Path(value).name
            if value.startswith(base_dir):
                args_dict[key] = value.replace(base_dir, "<src>")

    return args_dict


def test_translate_corpus(data_dir: DataDir):
    data_dir.create_zst("file.1.zst", en_sample)
    data_dir.run_task(
        "translate-corpus-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
        },
    )
    data_dir.print_tree()

    assert (
        data_dir.read_text("artifacts/file.1.nbest.zst") == en_sample.upper()
    ), "The text is pseudo-translated"

    args = json.loads(data_dir.read_text("marian-decoder.args.txt"))
    assert sanitize_marian_args(data_dir, args) == {
        "config": "<src>/pipeline/translate/decoder.yml",
        "vocabs": [
            "<src>/data/tests_data/test_translate/vocab.spm",
            "<src>/data/tests_data/test_translate/vocab.spm",
        ],
        "input": "<tmp>/file.1",
        "output": "<tmp>/file.1.nbest",
        "log": "<tmp>/file.1.log",
        "devices": ["0", "1", "2", "3"],
        "workspace": "12000",
        "mini-batch-words": "4000",
        "precision": "float16",
    }


def test_translate_corpus_empty(data_dir: DataDir):
    """
    Test the case of an empty file.
    """
    data_dir.create_zst("file.1.zst", "")
    data_dir.run_task(
        "translate-corpus-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
        },
    )

    data_dir.print_tree()

    assert data_dir.read_text("artifacts/file.1.nbest.zst") == "", "The text is empty"


@pytest.mark.parametrize("direction", ["src", "trg"])
def test_translate_mono(direction: str, data_dir: DataDir):
    data_dir.create_zst("file.1.zst", en_sample)
    data_dir.print_tree()
    data_dir.run_task(
        f"translate-mono-{direction}-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
        },
    )
    data_dir.print_tree()

    assert (
        data_dir.read_text("artifacts/file.1.out.zst") == en_sample.upper()
    ), "The text is pseudo-translated"

    args = json.loads(data_dir.read_text("marian-decoder.args.txt"))
    assert sanitize_marian_args(data_dir, args) == {
        "config": "<src>/pipeline/translate/decoder.yml",
        "vocabs": [
            "<src>/data/tests_data/test_translate/vocab.spm",
            "<src>/data/tests_data/test_translate/vocab.spm",
        ],
        "input": "<tmp>/file.1",
        "output": "<tmp>/file.1.out",
        "log": "<tmp>/file.1.log",
        "devices": ["0", "1", "2", "3"],
        "workspace": "12000",
        "mini-batch-words": "4000",
        "precision": "float16",
    }
