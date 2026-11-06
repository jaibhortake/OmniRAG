"""Real local-model providers. Every heavy import and model load is lazy, on first use,
so the API starts instantly and a missing model only fails the feature that needs it."""
import io
import logging
import threading
import wave
from typing import Iterator

import numpy as np

from ..config import Settings
from .base import Providers, Segment, normalize

log = logging.getLogger("omnirag.providers")


def _device(s: Settings) -> str:
    if s.device != "auto":
        return s.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class _Lazy:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None

    def get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self.load()
        return self._model

    def load(self):  # pragma: no cover - overridden
        raise NotImplementedError


class BGEText(_Lazy):
    dim = 384
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, s: Settings):
        super().__init__()
        self.s = s

    def load(self):
        from sentence_transformers import SentenceTransformer
        log.info("loading text embedder %s", self.s.text_embed_model)
        return SentenceTransformer(self.s.text_embed_model, device=_device(self.s))

    def embed(self, texts, is_query=False):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        if is_query:
            texts = [self.QUERY_PREFIX + t for t in texts]
        return normalize(self.get().encode(texts, batch_size=32, normalize_embeddings=True))


class OpenClip(_Lazy):
    dim = 512

    def __init__(self, s: Settings):
        super().__init__()
        self.s = s

    def load(self):
        import open_clip
        import torch
        dev = _device(self.s)
        model, _, preprocess = open_clip.create_model_and_transforms(self.s.clip_model, pretrained=self.s.clip_pretrained, device=dev)
        model.eval()
        return model, preprocess, open_clip.get_tokenizer(self.s.clip_model), dev, torch

    def embed_images(self, paths):
        from PIL import Image
        model, preprocess, _, dev, torch = self.get()
        out = []
        for i in range(0, len(paths), 16):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + 16]]).to(dev)
            with torch.no_grad():
                out.append(model.encode_image(batch).float().cpu().numpy())
        return normalize(np.concatenate(out))

    def embed_texts(self, texts):
        model, _, tok, dev, torch = self.get()
        with torch.no_grad():
            return normalize(model.encode_text(tok(texts).to(dev)).float().cpu().numpy())


class BGEReranker(_Lazy):
    def __init__(self, s: Settings):
        super().__init__()
        self.s = s

    def load(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.s.reranker_model, device=_device(self.s), max_length=512)

    def score(self, query, docs):
        if not docs:
            return []
        logits = self.get().predict([(query, d) for d in docs], batch_size=16)
        return [float(1 / (1 + np.exp(-x))) for x in np.atleast_1d(logits)]


class OllamaLLM:
    def __init__(self, s: Settings):
        import ollama
        self.client = ollama.Client(host=s.ollama_host)
        self.s = s

    def _opts(self):
        return {"temperature": 0.1, "num_ctx": 4096}

    def stream(self, system, prompt) -> Iterator[str]:
        for part in self.client.chat(
            model=self.s.llm_model, stream=True, keep_alive=self.s.ollama_keep_alive, options=self._opts(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        ):
            yield part["message"]["content"]

    def complete(self, system, prompt):
        return "".join(self.stream(system, prompt))


class OllamaVision:
    PROMPT = ("Describe this image in detail for a search index: objects, people, any visible text, "
              "charts or diagrams and what they show. Be factual and concise.")

    def __init__(self, s: Settings):
        import ollama
        self.client = ollama.Client(host=s.ollama_host)
        self.s = s

    def caption(self, image_path):
        r = self.client.generate(model=self.s.vision_model, prompt=self.PROMPT, images=[image_path],
                                 keep_alive=self.s.ollama_keep_alive, options={"temperature": 0})
        return r["response"].strip()


class RapidOCR(_Lazy):
    def load(self):
        from rapidocr_onnxruntime import RapidOCR as R
        return R()

    def read(self, image_path):
        result, _ = self.get()(image_path)
        return " ".join(r[1] for r in (result or []) if r[2] > 0.5)


class Whisper(_Lazy):
    def __init__(self, s: Settings):
        super().__init__()
        self.s = s

    def load(self):
        from faster_whisper import WhisperModel
        dev = _device(self.s)
        return WhisperModel(self.s.whisper_model, device=dev, compute_type=self.s.whisper_compute if dev == "cuda" else "int8")

    def transcribe(self, audio_path):
        segs, _ = self.get().transcribe(audio_path, vad_filter=True, beam_size=1)
        return [Segment(s.start, s.end, s.text.strip()) for s in segs if s.text.strip()]


class PiperTTS(_Lazy):
    def __init__(self, s: Settings):
        super().__init__()
        self.s = s

    def load(self):
        from piper import PiperVoice
        return PiperVoice.load(self.s.piper_voice)

    def synthesize(self, text):
        if not self.s.piper_voice:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self.get().synthesize(text, w)
        return buf.getvalue()


class SpacyNER(_Lazy):
    KEEP = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"}

    def load(self):
        import spacy
        return spacy.load("en_core_web_sm", disable=["lemmatizer"])

    def entities(self, text):
        return [(e.text.strip(), e.label_) for e in self.get()(text[:20000]).ents if e.label_ in self.KEEP and len(e.text.strip()) > 1]


def build(s: Settings) -> Providers:
    return Providers(
        text=BGEText(s), clip=OpenClip(s), reranker=BGEReranker(s), llm=OllamaLLM(s), vision=OllamaVision(s),
        ocr=RapidOCR(), stt=Whisper(s), tts=PiperTTS(s), ner=SpacyNER(), name="real",
    )
