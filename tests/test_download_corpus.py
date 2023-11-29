import os
from tempfile import TemporaryDirectory

import pytest

from pipeline.data.download_corpus import main as download_corpus, set_shell_env

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OPUS_DATA_PATH = os.path.join(CURRENT_FOLDER, "fixtures/ca-en.txt.zip")
FIXTURES_PATH = os.path.join(CURRENT_FOLDER, "fixtures")


def assert_file_has_contents(language, *path_parts):
    archive = os.path.join(*path_parts)
    assert os.path.isfile(archive), f"The {language} data exists"
    assert os.stat(archive).st_size > 0, f"The {language} data has file contents"


@pytest.mark.parametrize(
    "language_pairs",
    [("en", "ca"), ("ca", "en")],
)
def test_opus_download_corpus(language_pairs, requests_mock):
    with open(OPUS_DATA_PATH, "rb") as file, TemporaryDirectory() as output_dir:
        corpus_key = "opus_ELRC-2614-Localidades_2007/v1"

        requests_mock.get(
            "https://object.pouta.csc.fi/OPUS-ELRC-2614-Localidades_2007/v1/moses/en-ca.txt.zip",
            text="Not Found",
            status_code=404,
        )
        requests_mock.get(
            "https://object.pouta.csc.fi/OPUS-ELRC-2614-Localidades_2007/v1/moses/ca-en.txt.zip",
            body=file,
        )

        download_corpus(
            [
                language_pairs[0],
                language_pairs[1],
                "--corpus",
                corpus_key,
                "--output_dir",
                output_dir,
            ]
        )

        assert_file_has_contents("English", output_dir, "ELRC-2614-Localidades_2007_v1.en.gz")
        assert_file_has_contents("Catalan", output_dir, "ELRC-2614-Localidades_2007_v1.ca.gz")


class SacrebleuFixtures:
    """
    Prevents downloads from sacrebleu by setting the environment variable to
    the tests/fixtures folder.
    """

    def __enter__(self):
        env = os.environ.copy()
        env["SACREBLEU"] = FIXTURES_PATH
        set_shell_env(env)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        set_shell_env(None)
        return True


def test_sacrebleu_download_corpus():
    """
    This test actually downloads file, since we can't mock out requests. A local
    copy of the data is stored in ~/.sacrebleu
    """
    with SacrebleuFixtures(), TemporaryDirectory() as output_dir:
        download_corpus(
            [
                "en",
                "fr",
                "--corpus",
                "sacrebleu_wmt08",
                "--output_dir",
                output_dir,
            ]
        )
        assert_file_has_contents("English", output_dir, "wmt08.en.gz")
        assert_file_has_contents("French", output_dir, "wmt08.fr.gz")


def test_mtdata_download_corpus():
    with TemporaryDirectory() as output_dir:
        # It's possible to mock out the download directory with the MTDATA environment
        # variable, but locally mtdata must download an index of all the data sources,
        # which comes out to ~80 MBs. It would be too hard to guard against this, so
        # This test will do live downloading, and can fail.
        download_corpus(
            [
                "en",
                "ca",
                "--corpus",
                "mtdata_ELRC-wikipedia_health-1-cat-eng",
                "--output_dir",
                output_dir,
            ]
        )

        assert_file_has_contents("English", output_dir, "ELRC-wikipedia_health-1-cat-eng.en.gz")
        assert_file_has_contents("Catalan", output_dir, "ELRC-wikipedia_health-1-cat-eng.ca.gz")
