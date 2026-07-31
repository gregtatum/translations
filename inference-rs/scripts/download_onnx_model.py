#!/usr/bin/env python3
"""
Download the float student model the ONNX export evaluation reads (inference-rs/onnx/).

Unlike `download_model.py` (which fetches the *quantized* intgemm `.bin` from Remote
Settings), the clean-room ONNX converter needs the *pre-quantization* float checkpoint —
`final.model.npz.best-chrf.npz` from the `student-finetuned/` training stage. That is not
on Remote Settings; it lives on the public prod GCS bucket. This:

  1. Resolves the latest en-fr `student-finetuned` run via the model registry (a GCS
     object listing — the same source `pipeline/eval/translators.py` uses).
  2. Downloads the float `.npz` (+ its decoder.yml) and the split SentencePiece vocabs
     with the shared `pipeline.common.downloads.stream_download_to_file` helper.
  3. Verifies each file's md5 against the GCS object metadata, and skips files already
     present (so it is a cheap no-op once the model is in place).

Files land in `data/models/en-fr/student-finetuned/`, exactly where `onnx/model_npz.py`
looks. Run via poetry so the pipeline helper is importable:
    PYTHONPATH=$(pwd) poetry run python -W ignore inference-rs/scripts/download_onnx_model.py

Or through the task wrapper (also a dependency of the other rs:onnx-* tasks):
    task rs:onnx-download-model
"""

import argparse
import base64
import hashlib
from pathlib import Path

import requests

from pipeline.common.downloads import stream_download_to_file

# Public read bucket for released/production models (see artifacts/marian-mac/model-registry.md).
BUCKET = "moz-fx-translations-data--303e-prod-translations-data"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA_ROOT = f"https://storage.googleapis.com/{BUCKET}"

# The float checkpoint the converter reads (quantization-aware-finetuned; see
# notes/15-onnx-port.md). `.decoder.yml` carries the sinusoidal-position flag etc.
MODEL_NPZ = "final.model.npz.best-chrf.npz"


def list_objects(prefix: str) -> list[dict]:
    """All GCS objects under `prefix`, following pagination."""
    items: list[dict] = []
    page_token = None
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        response = requests.get(LIST_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return items


def resolve_run(objects: list[dict], src: str, trg: str) -> str:
    """The latest `{experiment}_{task_group_id}` run that has a student-finetuned .npz.

    Matches only the model-type dir (`{run}/student-finetuned/<MODEL_NPZ>`), not the
    `{run}/evaluation/student-finetuned/…` metrics subtree.
    """
    prefix = f"models/{src}-{trg}/"
    runs = []
    for obj in objects:
        parts = obj["name"][len(prefix) :].split("/")
        if len(parts) == 3 and parts[1] == "student-finetuned" and parts[2] == MODEL_NPZ:
            runs.append((obj["updated"], parts[0]))
    if not runs:
        raise SystemExit(f"[gcs] no student-finetuned {MODEL_NPZ} found under {prefix}")
    runs.sort(reverse=True)  # newest `updated` first
    if len(runs) > 1:
        print(f"[gcs] {len(runs)} en-fr student-finetuned runs; using latest ({runs[0][1]})")
    return runs[0][1]


def md5_base64(path: Path) -> str:
    """The file's md5 digest, base64-encoded — the form GCS reports in `md5Hash`."""
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def download_one(name: str, url: str, dest: Path, expected_md5: str | None) -> None:
    """Download a single object to `dest` (atomically), verifying md5 when known."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)  # stream_download_to_file refuses to overwrite
    print(f"[dl] {name} <- {url}")
    stream_download_to_file(url, tmp)
    if expected_md5:
        got = md5_base64(tmp)
        if got != expected_md5:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"[verify] MD5 MISMATCH for {name}\n  expected {expected_md5}\n  got      {got}"
            )
        print(f"[verify] OK {name}")
    tmp.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # en-fr is the current ONNX POC pair; positionals let this generalize later.
    parser.add_argument(
        "source", nargs="?", default="en", help="Source language code (default: en)"
    )
    parser.add_argument(
        "target", nargs="?", default="fr", help="Target language code (default: fr)"
    )
    parser.add_argument(
        "--models-dir",
        default="data/models",
        help="Root directory to download into (default: data/models)",
    )
    args = parser.parse_args()

    src, trg = args.source.lower(), args.target.lower()
    dest_dir = Path(args.models_dir) / f"{src}-{trg}" / "student-finetuned"
    file_names = [MODEL_NPZ, f"{MODEL_NPZ}.decoder.yml", f"vocab.{src}.spm", f"vocab.{trg}.spm"]

    # Fast path: fully present → no network. Keeps this cheap as a task dependency.
    if all((dest_dir / name).exists() for name in file_names):
        print(f"[onnx-model] {src}-{trg} student model already present in {dest_dir}")
        return

    print(f"[gcs] resolving latest {src}-{trg} student-finetuned run")
    objects = list_objects(f"models/{src}-{trg}/")
    run = resolve_run(objects, src, trg)
    base = f"models/{src}-{trg}/{run}/student-finetuned"
    md5_by_name = {obj["name"]: obj.get("md5Hash") for obj in objects}

    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in file_names:
        dest = dest_dir / name
        if dest.exists():
            print(f"[skip] {name} already present")
            continue
        gcs_name = f"{base}/{name}"
        download_one(name, f"{MEDIA_ROOT}/{gcs_name}", dest, md5_by_name.get(gcs_name))

    print(
        f"\nDone. Float student model in {dest_dir}\n  Build the ONNX graphs with: task rs:onnx-export"
    )


if __name__ == "__main__":
    main()
