"""Shared ingestion types and helpers."""
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

EXT_MODALITY = {
    **{e: "pdf" for e in (".pdf",)},
    **{e: "image" for e in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")},
    **{e: "audio" for e in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm")},
    **{e: "video" for e in (".mp4", ".mov", ".mkv", ".avi", ".m4v")},
}

Progress = Callable[[float, str], None]


def modality_for(filename: str) -> str | None:
    return EXT_MODALITY.get(Path(filename).suffix.lower())


@dataclass
class Piece:
    """One retrievable unit produced by a loader, before chunking/embedding."""
    kind: str  # text | table | figure | image | transcript | keyframe
    content: str
    page: int | None = None
    t_start: float | None = None
    t_end: float | None = None
    thumb: str = ""
    image_path: str | None = None  # if set, also indexed in the CLIP collection
    meta: dict = field(default_factory=dict)
    splittable: bool = True


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-y", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def make_thumb(src: str, dst: Path, size: int = 512) -> str:
    img = Image.open(src)
    img = img.convert("RGB")
    img.thumbnail((size, size))
    img.save(dst, "JPEG", quality=85)
    return dst.name


_SENT = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-aware windowing with character overlap."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    sents, out, cur = _SENT.split(text), [], ""
    for s in sents:
        while len(s) > size:  # pathological long "sentence"
            out.append((cur + " " + s[:size]).strip()) if cur else out.append(s[:size])
            cur, s = "", s[size - overlap:]
        if len(cur) + len(s) + 1 > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:].split(" ", 1)[-1] if overlap else ""
        cur = f"{cur} {s}".strip()
    if cur.strip():
        out.append(cur.strip())
    return out
