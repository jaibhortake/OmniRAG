"""Deterministic, dependency-free providers.

Used by the test suite and for running the full app on a machine with no model weights.
Embeddings are hashed bag-of-words, so retrieval still behaves sensibly (keyword overlap).
"""
import hashlib
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import Providers, Segment, normalize

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of and or to in on for is are was were be by with this that it as at from what which who how do does show me tell about".split())


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _hash_vec(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in _tokens(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 64) & 1 else 0.8
    if not v.any():
        v[0] = 1e-3
    return normalize(v)


SIDECAR_DIRS: list[Path] = []  # extra dirs searched by original file name (tests register fixtures here)


def _sidecar(path: str) -> str:
    """Fake models read <file>.txt next to the media file (or in SIDECAR_DIRS), so tests and demos
    can control 'what the model sees'. Upload timestamps ("1712_name.jpg") are ignored."""
    p = Path(path)
    original = re.sub(r"^\d+_", "", p.name)
    cands = [p.with_name(p.name + ".txt"), p.with_suffix(".txt")]
    cands += [d / (original + ".txt") for d in SIDECAR_DIRS]
    for cand in cands:
        if cand.exists():
            return cand.read_text(encoding="utf-8").strip()
    return ""


def _pixel_hash(path: str) -> str:
    """Identical-looking images get the same token (a crude stand-in for CLIP visual similarity)."""
    from PIL import Image
    try:
        return hashlib.md5(Image.open(path).convert("L").resize((16, 16)).tobytes()).hexdigest()[:12]
    except Exception:
        return "none"


class FakeText:
    dim = 256

    def embed(self, texts, is_query=False):
        return np.stack([_hash_vec(t, self.dim) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class FakeClip:
    dim = 128

    def embed_images(self, paths):
        return np.stack([_hash_vec(f"{_sidecar(p)} px{_pixel_hash(p)}", self.dim) for p in paths])

    def embed_texts(self, texts):
        return np.stack([_hash_vec(t, self.dim) for t in texts])


class FakeReranker:
    def score(self, query, docs):
        q = set(_tokens(query))
        if not q:
            return [0.0] * len(docs)
        return [len(q & set(_tokens(d))) / len(q) for d in docs]


class FakeLLM:
    """Extractive 'LLM': quotes the first sentence of each context block with its citation."""

    def complete(self, system, prompt):
        return "".join(self.stream(system, prompt))

    def stream(self, system, prompt) -> Iterator[str]:
        if "Rewrite" in system:
            yield prompt.rsplit("Follow-up question:", 1)[-1].strip()
            return
        blocks = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\nQuestion:|\Z)", prompt, re.S)
        if not blocks:
            yield "Not found in your files."
            return
        for n, body in blocks[:3]:
            sent = re.split(r"(?<=[.!?])\s", body.strip(), maxsplit=1)[0][:240]
            for tok in f"{sent} [{n}] ".split(" "):
                yield tok + " "


class FakeVision:
    def caption(self, image_path):
        return _sidecar(image_path)


class FakeOCR:
    def read(self, image_path):
        return ""


class FakeSTT:
    def transcribe(self, audio_path):
        text = _sidecar(audio_path)
        if not text:
            return []
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        return [Segment(i * 5.0, i * 5.0 + 5.0, s) for i, s in enumerate(sents)]


class FakeTTS:
    def synthesize(self, text):
        return None


class FakeNER:
    _cap = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")

    def entities(self, text):
        out = []
        for m in self._cap.findall(text):
            if m.lower() not in _STOP and len(m) > 2:
                out.append((m, "ENTITY"))
        return out


def build() -> Providers:
    return Providers(FakeText(), FakeClip(), FakeReranker(), FakeLLM(), FakeVision(), FakeOCR(), FakeSTT(), FakeTTS(), FakeNER(), "fake")
