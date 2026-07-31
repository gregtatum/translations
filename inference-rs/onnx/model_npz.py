"""Load the float .npz student model and expose config + named weights.

Weights in the .npz are stored logically ``[in, out]`` (see SPEC "Weight
orientation"), so a linear layer is a plain ``MatMul(x[.,in], W[in,out])`` with
no transpose. Biases are stored as ``(1, out)`` and are returned flattened to
``(out,)``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

# Repo root is three levels up: onnx/ -> inference-rs/ -> translations/
_REPO_ROOT = Path(__file__).resolve().parents[2]
# The float student model, fetched from GCS by `task rs:onnx-download-model` (a dependency
# of the rs:onnx-* tasks). See scripts/download_onnx_model.py.
_MODEL_DIR = _REPO_ROOT / "data" / "models" / "en-fr" / "student-finetuned"

NPZ_PATH = _MODEL_DIR / "final.model.npz.best-chrf.npz"
SRC_SPM_PATH = _MODEL_DIR / "vocab.en.spm"
TGT_SPM_PATH = _MODEL_DIR / "vocab.fr.spm"

# --- config constants (SPEC "Config constants") ---
DIM = 384
HEADS = 8
HEAD_DIM = 48  # DIM // HEADS
ATTN_SCALE = 1.0 / np.sqrt(HEAD_DIM)  # ~0.144338
ENC_DEPTH = 6
DEC_DEPTH = 4
FFN_DIM = 1536
VOCAB = 32000
EPS = 1e-6
EMBED_SCALE = float(np.sqrt(DIM))  # ~19.5959
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
    """Tied embedding table, shape ``(32000, 384)``."""
    return np.asarray(_npz()["Wemb"], dtype=np.float32)


def logit_bias() -> np.ndarray:
    """Output projection bias ``decoder_ff_logit_out_b``, shape ``(32000,)``."""
    return np.asarray(_npz()["decoder_ff_logit_out_b"], dtype=np.float32).reshape(-1)
