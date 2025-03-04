from pathlib import Path
from fixtures import DataDir

config_path = Path(__file__).parent / "../taskcluster/configs/config.ci-corpora.yml"


def test_basic_corpus_import():
    data_dir = DataDir("test_corpora_continuation")
    data_dir.run_task(
        "merge-corpus-en-ru",
        config=str(config_path),
    )
