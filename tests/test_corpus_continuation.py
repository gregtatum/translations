from pathlib import Path
from fixtures import get_full_taskgraph


def test_corpus_continuation():
    config_path = Path(__file__).parent / "../taskcluster/configs/config.ci-corpora.yml"
    get_full_taskgraph(str(config_path.resolve()))
