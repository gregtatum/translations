import json

import pytest

from fixtures import DataDir
from pipeline.common.downloads import read_lines

corpus_sample = """CORPUS 1
CORPUS 2
CORPUS 3
CORPUS_SHARED 1
CORPUS_SHARED 2
CORPUS 4
CORPUS 5
"""

news_2014_sample = """NEWS_2014 1
NEWS_2014 2
MONO_SHARED 1
MONO_SHARED 2
NEWS_2014 3
CORPUS_SHARED 1
NEWS_2014 4
"""

nllb_sample = """NLLB 1
NLLB 2
MONO_SHARED 1
MONO_SHARED 2
NLLB 3
CORPUS_SHARED 2
NLLB 4
"""


@pytest.mark.parametrize("params", [("src", "en"), ("trg", "ru")])
def test_merge_mono(params: tuple[str, str]):
    side, locale = params
    data_dir = DataDir("test_merge_mono")
    data_dir.mkdir("corpus")
    data_dir.create_zst(f"corpus/corpus.{locale}.zst", corpus_sample)
    data_dir.create_zst(f"news_2014.{locale}.zst", news_2014_sample)
    data_dir.create_zst(f"nllb.{locale}.zst", nllb_sample)
    data_dir.run_task(
        f"merge-mono-{side}-{locale}",
        env={"TEST_ARTIFACTS": data_dir.path},
        extra_args=["--sample_size", "5"],
    )

    data_dir.print_tree()

    assert json.loads(data_dir.load(f"artifacts/mono.{locale}.stats.json")) == {
        "duplicates_of_monolingual_corpus": 2,
        "duplicates_of_parallel_corpus": 2,
        "parallel_corpus_lines": 7,
        "original_monolingual_lines": 14,
        "deduplicated_monolingual_lines": 10,
        "final_truncated_monolingual_lines": 10,
        "final_truncated_monolingual_codepoints": 104,
    }

    with read_lines(data_dir.join(f"artifacts/mono.{locale}.zst")) as lines_iter:
        lines = list(lines_iter)
        lines_sorted = list(lines)
        lines_sorted.sort()

        assert lines_sorted == [
            "MONO_SHARED 1\n",
            "MONO_SHARED 2\n",
            "NEWS_2014 1\n",
            "NEWS_2014 2\n",
            "NEWS_2014 3\n",
            "NEWS_2014 4\n",
            "NLLB 1\n",
            "NLLB 2\n",
            "NLLB 3\n",
            "NLLB 4\n",
        ]

        assert lines != lines_sorted, "The results are shuffled."

    with read_lines(data_dir.join(f"artifacts/mono.{locale}.sample.txt")) as lines_iter:
        lines = list(lines_iter)
        lines_sorted = list(lines)
        lines_sorted.sort()

        assert lines_sorted == [
            "NEWS_2014 2\n",
            "NEWS_2014 4\n",
            "NLLB 1\n",
            "NLLB 3\n",
            "NLLB 4\n",
        ]

        assert lines != lines_sorted, "The results are shuffled."
