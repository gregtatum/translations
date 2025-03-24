from pathlib import Path
from fixtures import get_full_taskgraph
import cProfile

def test_corpus_continuation():
    profiler = cProfile.Profile()
    profiler.enable()

    config_path = Path(__file__).parent / "../taskcluster/configs/config.ci-corpora.yml"
    get_full_taskgraph(str(config_path.resolve()))
    
    profiler.disable()
    profiler.dump_stats("profile.prof")

def test_corpus_continuation():
    profiler = cProfile.Profile()
    profiler.enable()

    config_path = Path(__file__).parent / "../taskcluster/configs/config.ci-corpora-aln.yml"
    get_full_taskgraph(str(config_path.resolve()))
    
    profiler.disable()
    profiler.dump_stats("profile.prof")
