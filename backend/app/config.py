"""Central configuration. Every value can be overridden with an OMNIRAG_* env var or backend/.env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNIRAG_", env_file=BASE_DIR / ".env", extra="ignore")

    # "real" = local models (Ollama, Whisper, BGE, CLIP). "fake" = lightweight deterministic stand-ins
    # used for tests and for running the app on a machine without model weights.
    providers: str = "real"

    data_dir: Path = BASE_DIR / "data"
    qdrant_path: str = ""  # empty -> <data_dir>/qdrant ; ":memory:" for tests

    # Ollama
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    vision_model: str = "moondream"
    ollama_keep_alive: str = "2m"  # 4GB GPUs: let models unload so LLM and vision can swap

    # Local models (HuggingFace ids; downloaded on first use)
    text_embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    whisper_model: str = "small.en"
    whisper_compute: str = "int8"
    piper_voice: str = ""  # path to a piper .onnx voice; empty -> browser TTS fallback
    device: str = "auto"  # auto | cuda | cpu

    # Chunking / retrieval
    chunk_chars: int = 900
    chunk_overlap: int = 150
    top_k: int = 8
    candidate_k: int = 30
    rerank_threshold: float = 0.15  # below this -> "not found in your files"
    max_video_keyframes: int = 40
    history_turns: int = 4

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "omnirag.db"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.thumbs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
