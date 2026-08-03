#!/usr/bin/env python3
"""
Resolve and download the float student model behind a *production* Firefox translation
model, for the clean-room ONNX converter (inference-rs/onnx/).

The converter reads the pre-quantization float checkpoint (`final.model.npz.best-chrf.npz`
from the `student-finetuned/` training stage). That is not on Remote Settings — RS ships
only the quantized intgemm `.bin`. The float `.npz` lives on the public prod GCS bucket. To
stay faithful to what Firefox actually ships, we resolve it by matching hashes rather than
guessing which training run to use (size alone does not disambiguate — several en-ru runs
ship byte-count-identical quantized bins with different contents):

  1. Fetch the production `model` record for the pair from Remote Settings
     (`translations-models-v2`). It carries `decompressedHash` (the sha256 of the
     decompressed intgemm bin) and a `version`. Default to the latest version = current
     production; `--version` pins an explicit one (e.g. a superseded architecture we still
     want to benchmark against).
  2. Every exported GCS run has `models/{pair}/{run}/exported/metadata.json`, which carries
     the same sha256 as `hash`, plus `architecture` and the full `modelConfig`. Match
     `metadata.json.hash == decompressedHash` to identify the run — no large download needed.
  3. Validate the architecture: transformer encoder + SSRU decoder only. Anything else is
     refused with a clear error rather than silently mis-converted.
  4. Download that run's float `.npz` (+ `.decoder.yml`) and SentencePiece vocab(s),
     md5-verified against the object listing, into `{models-dir}/onnx/{pair}/`, and write a
     `resolved.json` that records the provenance and the `modelConfig` the converter reads.

Run via poetry so the RS/download helpers are importable:
    PYTHONPATH=$(pwd) poetry run python -W ignore inference-rs/scripts/download_onnx_model.py en ru

Or through the task wrapper (a dependency of the model-reading rs:onnx-* tasks):
    task rs:onnx-download-model
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import requests

from pipeline.common.downloads import stream_download_to_file
from utils.common.remote_settings import get_prod_records_url

# Public read bucket for released/production models (see artifacts/marian-mac/model-registry.md).
BUCKET = "moz-fx-translations-data--303e-prod-translations-data"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA_ROOT = f"https://storage.googleapis.com/{BUCKET}"

# Production Remote Settings collection (the "-v2" collection; see download_model.py).
DEFAULT_COLLECTION = "translations-models-v2"

# The float checkpoint the converter reads and its decode-config sidecar.
MODEL_NPZ = "final.model.npz.best-chrf.npz"
DECODER_YML = f"{MODEL_NPZ}.decoder.yml"


def fetch_rs_records(collection: str) -> list[dict]:
    url = get_prod_records_url(collection)
    print(f"[rs] fetching records: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()["data"]


def version_key(record: dict) -> tuple:
    try:
        return tuple(int(p) for p in str(record.get("version", "0")).split("."))
    except ValueError:
        return (0,)


def pick_model_record(records: list[dict], src: str, trg: str, version: str | None) -> dict:
    """The production `model` record for the pair — a pinned version, or the latest."""
    matches = [
        r
        for r in records
        if r.get("fileType") == "model"
        and r.get("sourceLanguage") == src
        and r.get("targetLanguage") == trg
    ]
    if not matches:
        raise SystemExit(f"[rs] no 'model' record for {src}-{trg} in the production collection")
    if version is not None:
        pinned = [r for r in matches if str(r.get("version")) == version]
        if not pinned:
            have = ", ".join(sorted(str(r.get("version")) for r in matches))
            raise SystemExit(f"[rs] no {src}-{trg} 'model' at version {version}; have: {have}")
        return pinned[0]
    matches.sort(key=version_key, reverse=True)
    if len(matches) > 1:
        versions = ", ".join(str(r.get("version")) for r in matches)
        print(f"[rs] {len(matches)} '{src}-{trg}' model versions ({versions}); using latest")
    return matches[0]


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


def resolve_run_by_hash(objects: list[dict], prefix: str, decompressed_hash: str) -> dict:
    """The run whose exported bin matches `decompressed_hash`, via its metadata.json.

    Returns the parsed metadata.json (architecture + modelConfig + hash). Fetches only the
    handful of `exported/metadata.json` objects, not the whole run.
    """
    metas = [o["name"] for o in objects if o["name"].endswith("/exported/metadata.json")]
    for name in metas:
        meta = requests.get(f"{MEDIA_ROOT}/{name}", timeout=30).json()
        if meta.get("hash") == decompressed_hash:
            run = name[len(prefix) :].split("/")[0]
            meta["_run"] = run
            arch = meta.get("architecture")
            print(f"[gcs] matched run {run} (architecture: {arch})")
            return meta
    raise SystemExit(
        f"[gcs] no exported run under {prefix} has hash {decompressed_hash}\n"
        f"      (checked {len(metas)} metadata.json files)"
    )


def validate_architecture(model_config: dict) -> None:
    """Refuse anything the converter cannot faithfully represent.

    The converter knows exactly one shape: a transformer encoder with an SSRU
    (autoregressive-rnn) decoder. Different dims/depths are fine — those are read from the
    config — but a different op structure would be silently mis-converted, so reject it.
    """
    problems = []
    if model_config.get("type") != "transformer":
        problems.append(f"type={model_config.get('type')!r} (expected 'transformer')")
    if model_config.get("dec-cell") != "ssru":
        problems.append(f"dec-cell={model_config.get('dec-cell')!r} (expected 'ssru')")
    if model_config.get("transformer-decoder-autoreg") != "rnn":
        problems.append(
            f"transformer-decoder-autoreg={model_config.get('transformer-decoder-autoreg')!r} "
            "(expected 'rnn')"
        )
    if problems:
        raise SystemExit(
            "[arch] unsupported model architecture — the ONNX converter only handles a "
            "transformer encoder + SSRU decoder:\n  " + "\n  ".join(problems)
        )


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
    print(f"[dl] {name}")
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


def resolve_vocab_names(names: set[str], src: str, trg: str) -> tuple[list[str], bool]:
    """The vocab file(s) to fetch from a run's student-finetuned/ dir.

    Shared-vocab pairs ship a single `vocab.spm`; split-vocab pairs ship
    `vocab.{src}.spm` / `vocab.{trg}.spm`. Returns (files, shared).
    """
    if "vocab.spm" in names:
        return ["vocab.spm"], True
    split = [f"vocab.{src}.spm", f"vocab.{trg}.spm"]
    if all(v in names for v in split):
        return split, False
    raise SystemExit(f"[gcs] no vocab found in run: expected vocab.spm or {split[0]}/{split[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", nargs="?", default="en", help="Source language (default: en)")
    parser.add_argument("target", nargs="?", default="ru", help="Target language (default: ru)")
    parser.add_argument(
        "--version",
        default=None,
        help="Pin a Remote Settings model version (default: latest = current production)",
    )
    parser.add_argument(
        "--models-dir",
        default="data/models",
        help="Root directory to download into (default: data/models)",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    src, trg = args.source.lower(), args.target.lower()
    pair = f"{src}-{trg}"
    dest_dir = Path(args.models_dir) / "onnx" / pair

    # 1. Production RS record -> the sha256 of the shipped (decompressed) intgemm bin.
    record = pick_model_record(fetch_rs_records(args.collection), src, trg, args.version)
    decompressed_hash = record.get("decompressedHash")
    rs_version = str(record.get("version"))
    if not decompressed_hash:
        raise SystemExit(f"[rs] {pair} model v{rs_version} record has no decompressedHash")
    print(f"[rs] {pair} production model v{rs_version}, sha256 {decompressed_hash}")

    # 2. Match that hash to a GCS run via its exported/metadata.json.
    prefix = f"models/{pair}/"
    objects = list_objects(prefix)
    meta = resolve_run_by_hash(objects, prefix, decompressed_hash)
    run, arch = meta["_run"], meta.get("architecture")
    model_config = meta["modelConfig"]

    # 3. Refuse architectures the converter cannot represent.
    validate_architecture(model_config)

    # 4. Download the float .npz + decoder.yml + vocab(s) from that run's student-finetuned/.
    base = f"{prefix}{run}/student-finetuned"
    present = {o["name"][len(base) + 1 :] for o in objects if o["name"].startswith(base + "/")}
    vocab_files, shared_vocab = resolve_vocab_names(present, src, trg)
    file_names = [MODEL_NPZ, DECODER_YML, *vocab_files]
    md5_by_name = {o["name"]: o.get("md5Hash") for o in objects}

    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in file_names:
        dest = dest_dir / name
        if dest.exists():
            print(f"[skip] {name} already present")
            continue
        gcs_name = f"{base}/{name}"
        download_one(name, f"{MEDIA_ROOT}/{gcs_name}", dest, md5_by_name.get(gcs_name))

    src_vocab, trg_vocab = (vocab_files * 2)[:2] if shared_vocab else vocab_files
    resolved = {
        "pair": pair,
        "source": src,
        "target": trg,
        "rs_version": rs_version,
        "architecture": arch,
        "run": run,
        "decompressed_hash": decompressed_hash,
        "npz": MODEL_NPZ,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
        "shared_vocab": shared_vocab,
        "model_config": model_config,
    }
    (dest_dir / "resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")

    print(
        f"\nDone. {pair} {arch} (RS v{rs_version}) float model in {dest_dir}\n"
        f"  Build the ONNX graphs with: task rs:onnx-export"
    )


if __name__ == "__main__":
    main()
