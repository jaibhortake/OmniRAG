"""Model provider interfaces. Pipelines depend only on these, never on a concrete model library."""
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Segment:
    start: float
    end: float
    text: str


class TextEmbedder(Protocol):
    dim: int

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class ClipEmbedder(Protocol):
    """Shared image/text space for cross-modal search."""
    dim: int

    def embed_images(self, paths: list[str]) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...  # 0..1, higher = more relevant


class LLM(Protocol):
    def stream(self, system: str, prompt: str) -> Iterator[str]: ...
    def complete(self, system: str, prompt: str) -> str: ...


class Vision(Protocol):
    def caption(self, image_path: str) -> str: ...


class OCR(Protocol):
    def read(self, image_path: str) -> str: ...


class STT(Protocol):
    def transcribe(self, audio_path: str) -> list[Segment]: ...


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes | None: ...  # wav bytes, or None -> client-side fallback


class EntityExtractor(Protocol):
    def entities(self, text: str) -> list[tuple[str, str]]: ...  # (name, label)


@dataclass
class Providers:
    text: TextEmbedder
    clip: ClipEmbedder
    reranker: Reranker
    llm: LLM
    vision: Vision
    ocr: OCR
    stt: STT
    tts: TTS
    ner: EntityExtractor
    name: str


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)
