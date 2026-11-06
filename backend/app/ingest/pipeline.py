"""Ingestion orchestration: background worker -> loader -> chunk -> embed -> index."""
import logging
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING

from ..db import new_id
from ..index.store import CLIP, TEXT
from .common import Piece, split_text
from .media import load_audio, load_image, load_video
from .pdf import load_pdf

if TYPE_CHECKING:
    from ..engine import Engine

log = logging.getLogger("omnirag.ingest")


def pieces_to_chunks(pieces: list[Piece], file_id: str, modality: str, size: int, overlap: int) -> list[dict]:
    chunks = []
    for pc in pieces:
        parts = split_text(pc.content, size, overlap) if pc.splittable else [pc.content.strip()]
        for part in parts:
            if not part:
                continue
            chunks.append({
                "id": new_id(), "file_id": file_id, "modality": modality, "kind": pc.kind, "content": part,
                "page": pc.page, "t_start": pc.t_start, "t_end": pc.t_end, "thumb": pc.thumb,
                "meta": {**pc.meta, **({"image_path": pc.image_path} if pc.image_path else {})},
            })
    return chunks


class IngestWorker:
    """Single worker thread: on a 4GB GPU, parallel ingestion would thrash VRAM."""

    def __init__(self, engine: "Engine"):
        self.engine = engine
        self.q: queue.Queue[str] = queue.Queue()
        self._t = threading.Thread(target=self._run, daemon=True, name="ingest")
        self._t.start()

    def submit(self, file_id: str) -> None:
        self.q.put(file_id)

    def wait_idle(self, timeout: float = 60) -> None:
        end = time.time() + timeout
        while (self.q.unfinished_tasks > 0) and time.time() < end:
            time.sleep(0.05)

    def _run(self):
        while True:
            fid = self.q.get()
            try:
                self.process(fid)
            finally:
                self.q.task_done()

    def process(self, fid: str) -> None:
        e = self.engine
        f = e.db.get_file(fid)
        if not f:
            return
        s, p = e.settings, e.providers
        last = [0.0]

        def progress(frac: float, stage: str):
            now = time.time()
            if now - last[0] > 0.25 or frac >= 1:
                last[0] = now
                e.db.update_file(fid, progress=round(min(frac, 1.0), 3), stage=stage)

        t0 = time.time()
        e.db.update_file(fid, status="processing", progress=0.01, stage="Starting", error="")
        try:
            mod, path, th = f["modality"], f["path"], s.thumbs_dir
            if mod == "pdf":
                pieces = load_pdf(path, fid, th, p, progress)
            elif mod == "image":
                pieces = load_image(path, fid, th, p, progress)
            elif mod == "audio":
                pieces = load_audio(path, fid, th, p, progress, s.chunk_chars)
            elif mod == "video":
                pieces = load_video(path, fid, th, p, progress, s.chunk_chars, s.max_video_keyframes)
            else:
                raise ValueError(f"unsupported modality {mod}")

            chunks = pieces_to_chunks(pieces, fid, mod, s.chunk_chars, s.chunk_overlap)
            if not chunks:
                raise ValueError("no extractable content (empty file, silent audio or unreadable scan)")
            progress(0.8, f"Embedding {len(chunks)} chunks")
            e.store.upsert(TEXT, chunks, p.text.embed([c["content"] for c in chunks]))
            visual = [c for c in chunks if c["meta"].get("image_path")]
            if visual:
                progress(0.9, f"CLIP-embedding {len(visual)} visuals")
                e.store.upsert(CLIP, visual, p.clip.embed_images([c["meta"]["image_path"] for c in visual]))
            e.db.add_chunks(chunks)
            thumb = next((c["thumb"] for c in chunks if c["thumb"]), "")
            e.db.update_file(fid, status="indexed", progress=1.0, stage=f"Indexed in {time.time() - t0:.1f}s",
                             n_chunks=len(chunks), thumb=thumb)
            e.on_index_changed()
            log.info("indexed %s (%d chunks) in %.1fs", f["name"], len(chunks), time.time() - t0)
        except Exception as ex:
            log.error("ingest failed for %s: %s\n%s", f["name"], ex, traceback.format_exc())
            e.store.delete_file(fid)
            e.db.update_file(fid, status="failed", error=str(ex)[:500], stage="Failed")
