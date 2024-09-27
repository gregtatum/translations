from pathlib import Path
import shutil

import pytest
from fixtures import DataDir, en_sample

fixtures_path = Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir():
    data_dir = DataDir("test_translate")
    shutil.copyfile("tests/data/vocab.spm", data_dir.join("vocab.spm"))
    return data_dir


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


def test_translate_corpus_empty(data_dir):
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
