import shutil
from pathlib import Path

from fixtures import DataDir

from pipeline.common.downloads import stream_download_to_file


def download_and_cache(data_dir: DataDir, url: str, cached_filename: str, data_dir_name: str):
    src_dir = Path(__file__).parent.parent
    cached_file = src_dir / "data/tests" / cached_filename
    if not cached_file.exists():
        stream_download_to_file(url, cached_file)
    shutil.copy(cached_file, data_dir.join(data_dir_name))


def test_ctranslate2():
    data_dir = DataDir("test_ctranslate2")
    data_dir.mkdir("model0")
    data_dir.mkdir("model1")

    # Download the teacher models.
    for index in [0, 1]:
        download_and_cache(
            data_dir,
            "https://storage.googleapis.com/releng-translations-dev/models/ca-en/dev/teacher-finetuned{index}/final.model.npz.best-chrf.npz",
            cached_filename=f"en-ca-teacher-{index}.npz",
            data_dir_name=f"model{index}/final.model.npz.best-chrf.npz",
        )

    # Download the vocab.
    download_and_cache(
        data_dir,
        "https://storage.googleapis.com/releng-translations-dev/models/ca-en/dev/vocab/vocab.spm",
        cached_filename="en-ca-vocab.spm",
        data_dir_name="vocab.spm",
    )

    env = {}
    data_dir.run_task(
        "translate-mono-src-en-ru-1/10",
        env=env,
        extra_args=["--device", "cpu"],
    )
    data_dir.print_tree()
