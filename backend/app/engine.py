"""Engine: owns all long-lived components and the query orchestration."""
import logging
import time
from pathlib import Path
from typing import Iterator

from .config import Settings
from .db import Database
from .graph.kg import KnowledgeGraph
from .index.store import VectorStore
from .ingest.pipeline import IngestWorker
from .llm.answer import NOT_FOUND, citation_payload, rewrite_query, stream_answer, validate_citations
from .providers import build_providers
from .retrieval.hybrid import Retriever

log = logging.getLogger("omnirag")


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(str(settings.db_path))
        self.providers = build_providers(settings)
        qpath = settings.qdrant_path or str(settings.data_dir / "qdrant")
        self.store = VectorStore(qpath, self.providers.text.dim, self.providers.clip.dim)
        self.retriever = Retriever(self)
        self.graph = KnowledgeGraph()
        self.worker = IngestWorker(self)
        log.info("engine ready (providers=%s)", self.providers.name)

    def on_index_changed(self):
        self.retriever.bm25.invalidate()
        self.graph.invalidate()

    def delete_file(self, fid: str) -> bool:
        f = self.db.get_file(fid)
        if not f:
            return False
        self.store.delete_file(fid)
        self.db.delete_file(fid)
        Path(f["path"]).unlink(missing_ok=True)
        for t in self.settings.thumbs_dir.glob(f"{fid}*"):
            t.unlink(missing_ok=True)
        self.on_index_changed()
        return True

    # ---------------- query ----------------
    def answer_stream(self, question: str, history: list[dict], image_path: str | None = None,
                      modalities: list[str] | None = None, file_ids: list[str] | None = None) -> Iterator[dict]:
        """Yields events: {"type": "status"|"sources"|"token"|"done", ...}."""
        t0 = time.perf_counter()
        timings: dict = {}
        standalone = question
        if history and question:
            yield {"type": "status", "text": "Understanding follow-up…"}
            t = time.perf_counter()
            standalone = rewrite_query(self.providers, history[-2 * self.settings.history_turns:], question)
            timings["rewrite_ms"] = round((time.perf_counter() - t) * 1000, 1)
        yield {"type": "status", "text": "Searching text, images, audio and video…"}
        res = self.retriever.retrieve(standalone, image_path=image_path, modalities=modalities, file_ids=file_ids)
        timings.update(res.timings)
        cites = citation_payload(res.hits)
        yield {"type": "sources", "citations": cites, "query": res.query_used, "grounded": res.grounded}

        if not res.grounded:
            # strict grounding: refuse rather than hallucinate, but still surface the closest material
            yield {"type": "token", "text": NOT_FOUND}
            yield {"type": "done", "answer": NOT_FOUND, "citations": cites, "used": [], "grounded": False,
                   "timings": {**timings, "total_ms": round((time.perf_counter() - t0) * 1000, 1)}}
            return

        yield {"type": "status", "text": "Writing answer…"}
        t = time.perf_counter()
        parts = []
        for tok in stream_answer(self.providers, res.query_used or standalone, res.hits):
            parts.append(tok)
            yield {"type": "token", "text": tok}
        timings["generate_ms"] = round((time.perf_counter() - t) * 1000, 1)
        answer, used, invalid = validate_citations("".join(parts).strip(), len(res.hits))
        if not used and not answer.startswith(NOT_FOUND[:9]):
            timings["warning"] = "answer had no valid citations"
        timings["invalid_citations_removed"] = invalid
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        yield {"type": "done", "answer": answer, "citations": cites, "used": used, "grounded": True, "timings": timings}

    def answer(self, question: str, history: list[dict], persist: bool = False, **kw) -> dict:
        final = {}
        for ev in self.answer_stream(question, history, **kw):
            if ev["type"] == "done":
                final = ev
        return final
