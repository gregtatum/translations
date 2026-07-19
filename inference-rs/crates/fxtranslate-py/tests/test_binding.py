"""Offline packaging tests for the fxtranslate Python binding.

Same stance as the Rust `packaging.rs` and the npm parity harness: prove the
packaging, not the engine. The hermetic suite touches no network and needs no real
model; the translate-parity test is opt-in (guarded on ``FXTRANSLATE_MODEL_DIR``) and
pins to the Rust CLI oracle so it stays anti-cheat without a ~150 MB model in the
hermetic path.
"""

import os
import subprocess
from pathlib import Path

import pytest

import fxtranslate
from fxtranslate import Translator, discovery

FIXTURES = Path(__file__).parent / "fixtures"

# The tiny zstd fixture is a download/cache-plumbing fixture (see the Rust
# `packaging.rs`): it is fed only through verify/decompress, never through
# `Engine::from_bytes` — so it is not a loadable, translatable engine. These are the
# decompressed bytes and their known SHA-256.
TINY_PLAIN = b"fxtranslate tiny model fixture\n"
TINY_HASH = "5a9aaf6b319b6cdb5f3ef4ff520599018f2df654ddc4d9bb73dcb687092c77b8"

# The trimmed Remote Settings snapshot has 7 records: en-es (shared vocab) and en-ja
# (split vocab).
KNOWN_SIMD_BACKENDS = {
    "i8mm+neon64",
    "neon64",
    "avx2",
    "avx512bw",
    "sse2",
    "wasm-simd128",
}


def records_body() -> str:
    return (FIXTURES / "rs-models-v2.json").read_text()


class TestBindingLoads:
    """The wheel loads and the module initializes on the running interpreter."""

    def test_imports(self):
        assert Translator is fxtranslate.Translator
        assert discovery is fxtranslate.discovery

    def test_translator_surface(self):
        # The BYO constructor + load classmethod + the three instance methods are
        # present; the two wasm-only methods are intentionally absent.
        assert hasattr(Translator, "load")
        for name in ("translate", "translate_long", "backend"):
            assert hasattr(Translator, name)
        assert not hasattr(Translator, "linear_memory_bytes")
        assert not hasattr(Translator, "translate_block_phased")


class TestBackend:
    """Guard the silent-scalar-wheel risk.

    ``backend()`` is an instance method and there is no loadable engine in the
    hermetic fixtures (the tiny fixture is cache-plumbing only, not a real model), so
    the *value* is asserted in the opt-in parity suite (which has a real model). Here
    we only assert the method is present on the compiled surface.
    """

    def test_backend_method_present(self):
        assert callable(Translator.backend)


class TestDiscovery:
    """Feed the fixture records through the discovery surface (no network)."""

    def test_parse_records(self):
        recs = discovery.parse_records(records_body())
        assert isinstance(recs, list)
        assert len(recs) == 7
        first = recs[0]
        assert isinstance(first, dict)
        for key in (
            "name",
            "fileType",
            "sourceLanguage",
            "targetLanguage",
            "version",
            "architecture",
            "decompressedHash",
            "location",
        ):
            assert key in first

    def test_model_pairs(self):
        pairs = discovery.model_pairs(records_body())
        assert ("en", "es") in pairs
        assert ("en", "ja") in pairs
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_resolve_route_direct(self):
        route = discovery.resolve_route(records_body(), "en", "es")
        assert route == {"kind": "direct", "src": "en", "trg": "es"}

    def test_resolve_route_error(self):
        # No model and no pivot for this pair in the fixture.
        with pytest.raises(ValueError):
            discovery.resolve_route(records_body(), "zz", "qq")

    def test_catalog(self):
        cat = discovery.catalog(records_body(), "en")
        assert set(cat.keys()) == {"bidirectional", "sourceOnly", "targetOnly"}
        # en → es and en → ja exist (source-only from en's view; es/ja are targets).
        assert "es" in cat["targetOnly"] or "es" in cat["bidirectional"]

    def test_segment_sentences(self):
        out = discovery.segment_sentences("Hello world. How are you?")
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].strip().startswith("Hello")

    def test_verify_and_decompress_ok(self):
        compressed = (FIXTURES / "tiny.bin.zst").read_bytes()
        # Without a hash: just decompress.
        assert discovery.verify_and_decompress(compressed) == TINY_PLAIN
        # With the correct hash: verifies and returns the bytes.
        assert discovery.verify_and_decompress(compressed, TINY_HASH) == TINY_PLAIN

    def test_verify_and_decompress_wrong_hash(self):
        compressed = (FIXTURES / "tiny.bin.zst").read_bytes()
        with pytest.raises(ValueError):
            discovery.verify_and_decompress(compressed, "00" * 32)


class TestErrorMapping:
    """The engine's Err(String) surfaces as a Python ValueError, not a panic/abort."""

    def test_garbage_model_raises_value_error(self):
        with pytest.raises(ValueError):
            Translator(b"garbage", b"", b"")


def _model_triple(model_dir: Path):
    """Locate (model, src_vocab, trg_vocab) under a model dir, no shortlist.

    Matches the Rust `Engine::load` and npm parity test: a shared-vocab pair uses one
    vocab for both sides.
    """
    model = next(iter(model_dir.glob("model.*")), None)
    vocab = next(iter(model_dir.glob("vocab.*")), None)
    src_vocab = next(iter(model_dir.glob("srcvocab.*")), None)
    trg_vocab = next(iter(model_dir.glob("trgvocab.*")), None)
    if model is None:
        return None
    if vocab is not None:
        return model, vocab, vocab
    if src_vocab is not None and trg_vocab is not None:
        return model, src_vocab, trg_vocab
    return None


@pytest.mark.skipif(
    not os.environ.get("FXTRANSLATE_MODEL_DIR"),
    reason="set FXTRANSLATE_MODEL_DIR to a dir with a real en-es model triple",
)
class TestTranslateParity:
    """Opt-in: build a real engine and pin its output to the Rust CLI oracle.

    Mirrors Pass B / conformance-corpus — needs a real model, so it is skipped unless
    FXTRANSLATE_MODEL_DIR is set. The output is pinned to the transitively oracle-
    validated Rust CLI (`target/conformance/debug/fxtranslate`), not a hardcoded
    self-referential string.
    """

    SRC = "en"
    TRG = "es"
    TEXT = "Hello world."

    def _translator(self):
        triple = _model_triple(Path(os.environ["FXTRANSLATE_MODEL_DIR"]))
        if triple is None:
            pytest.skip("no model triple found in FXTRANSLATE_MODEL_DIR")
        model, sv, tv = triple
        # No shortlist — matches Engine::load / the npm translate parity test.
        return Translator(model.read_bytes(), sv.read_bytes(), tv.read_bytes())

    def test_backend_is_known(self):
        t = self._translator()
        b = t.backend()
        assert isinstance(b, str) and b
        assert b == "scalar" or b in KNOWN_SIMD_BACKENDS

    def test_translate_matches_oracle(self):
        rust_bin = (
            Path(__file__).resolve().parents[3]
            / "target"
            / "conformance"
            / "debug"
            / "fxtranslate"
        )
        if not rust_bin.exists():
            pytest.skip(
                "Rust oracle binary missing; build it with `task rs:build-py` "
                "(deps on build-cli) or `cargo build -p fxtranslate-cli "
                "--target-dir target/conformance`"
            )
        t = self._translator()
        got = t.translate(self.TEXT)
        oracle = subprocess.run(
            [str(rust_bin), "translate", self.SRC, self.TRG, self.TEXT],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert got.strip() == oracle
