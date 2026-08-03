"""Load the float .npz student model and expose config + named weights.

The model is resolved and cached by `scripts/download_onnx_model.py`, which writes a
`resolved.json` next to the `.npz` recording the production provenance and the marian
`modelConfig`. The architecture constants below (DIM, DEC_DEPTH, ...) are derived from that
config rather than hardcoded, so one converter handles every shipped architecture
(base / base-memory / tiny). The public attribute surface is unchanged, so the exporters,
numpy reference, and engine consume `npz.DIM`, `npz.DEC_DEPTH`, ... exactly as before.

Which model is active is chosen by the `ONNX_MODEL_DIR` env var, else the default pair
`data/models/onnx/en-ru` (the current ONNX evaluation target). Run
`task rs:onnx-download-model` first — the config constants can't be known until it has.

Weights in the .npz are stored logically ``[in, out]`` (see SPEC "Weight orientation"), so
a linear layer is a plain ``MatMul(x[.,in], W[in,out])`` with no transpose. Biases are
stored as ``(1, out)`` and are returned flattened to ``(out,)``.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np

# Repo root is three levels up: onnx/ -> inference-rs/ -> translations/
_REPO_ROOT = Path(__file__).resolve().parents[2]

_MODEL_DIR = Path(
    os.environ.get("ONNX_MODEL_DIR", _REPO_ROOT / "data" / "models" / "onnx" / "en-ru")
)
_RESOLVED_PATH = _MODEL_DIR / "resolved.json"


def _load_resolved() -> dict:
    if not _RESOLVED_PATH.exists():
        raise SystemExit(
            f"[model] {_RESOLVED_PATH} not found — run `task rs:onnx-download-model` "
            f"(or set ONNX_MODEL_DIR) to resolve the production float model first."
        )
    return json.loads(_RESOLVED_PATH.read_text())


_RESOLVED = _load_resolved()
_CFG = _RESOLVED["model_config"]

NPZ_PATH = _MODEL_DIR / _RESOLVED["npz"]
SRC_SPM_PATH = _MODEL_DIR / _RESOLVED["src_vocab"]
TGT_SPM_PATH = _MODEL_DIR / _RESOLVED["trg_vocab"]

ARCHITECTURE = _RESOLVED.get("architecture")

# --- config constants, derived from the resolved marian modelConfig ---
# The converter only knows the transformer-encoder + SSRU-decoder shape; the resolver
# already refuses anything else, but assert here too so a stale resolved.json can't slip a
# mismatched graph through.
if _CFG.get("type") != "transformer" or _CFG.get("dec-cell") != "ssru":
    raise SystemExit(
        f"[model] unsupported architecture in {_RESOLVED_PATH}: "
        f"type={_CFG.get('type')!r} dec-cell={_CFG.get('dec-cell')!r}"
    )

DIM = int(_CFG["dim-emb"])
HEADS = int(_CFG["transformer-heads"])
HEAD_DIM = DIM // HEADS
ATTN_SCALE = 1.0 / np.sqrt(HEAD_DIM)
ENC_DEPTH = int(_CFG["enc-depth"])
DEC_DEPTH = int(_CFG["dec-depth"])
FFN_DIM = int(_CFG["transformer-dim-ffn"])
VOCAB = int(_CFG["dim-vocabs"][0])
EPS = 1e-6
EMBED_SCALE = float(np.sqrt(DIM))
EOS_ID = 0


@functools.lru_cache(maxsize=1)
def _npz() -> dict[str, np.ndarray]:
    return dict(np.load(NPZ_PATH))


def weight(name: str) -> np.ndarray:
    """Return a named tensor as float32.

    Matrices come back in logical ``[in, out]`` orientation (as stored). Bias
    vectors stored as ``(1, out)`` are flattened to ``(out,)``.
    """
    arr = np.asarray(_npz()[name], dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    return arr


def wemb() -> np.ndarray:
    """Tied embedding table, shape ``(VOCAB, DIM)``."""
    return np.asarray(_npz()["Wemb"], dtype=np.float32)


def logit_bias() -> np.ndarray:
    """Output projection bias ``decoder_ff_logit_out_b``, shape ``(VOCAB,)``."""
    return np.asarray(_npz()["decoder_ff_logit_out_b"], dtype=np.float32).reshape(-1)
