"""Hybrid cross-modal retrieval.

  text query  -> bge dense (text index)  ┐
              -> BM25 keyword            ├─ Reciprocal Rank Fusion -> cross-encoder rerank -> top-k
              -> CLIP text encoder (clip index: images / figures / keyframes)
  image query -> CLIP image encoder (clip index) + vision caption fed into the text paths
"""
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from ..index.store import CLIP, TEXT

if TYPE_CHECKING:
    from ..engine import Engine

_TOK = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokenize(t: str) -> list[str]:
    return _TOK.findall(t.lower())


@dataclass
class RetrievalResult:
    hits: list[dict]
    grounded: bool
    timings: dict = field(default_factory=dict)
    query_used: str = ""


class BM25Index:
    def __init__(self):
        self._lock = threading.Lock()
        self._dirty = True
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def invalidate(self):
        self._dirty = True

    def search(self, engine: "Engine", query: str, limit: int, modalities, file_ids) -> list[tuple[str, float]]:
        with self._lock:
            if self._dirty:
                rows = engine.db.all_chunks()
                self._ids = [r["id"] for r in rows]
                self._meta = [{"modality": r["modality"], "file_id": r["file_id"]} for r in rows]
                corpus = [tokenize(r["content"]) or ["_"] for r in rows]
                self._bm25 = BM25Okapi(corpus) if corpus else None
                self._dirty = False
            if not self._bm25:
                return []
            scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            m = self._meta[i]
            if (modalities and m["modality"] not in modalities) or (file_ids and m["file_id"] not in file_ids):
                continue
            out.append((self._ids[i], float(scores[i])))
            if len(out) >= limit:
                break
        return out


def rrf(rankings: dict[str, list[tuple[str, float]]], weights: dict[str, float]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        for rank, (cid, _) in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + w / (RRF_K + rank + 1)
    return fused


class Retriever:
    def __init__(self, engine: "Engine"):
        self.e = engine
        self.bm25 = BM25Index()

    def retrieve(self, query: str, image_path: str | None = None, modalities: list[str] | None = None,
                 file_ids: list[str] | None = None, top_k: int | None = None) -> RetrievalResult:
        e, s, p = self.e, self.e.settings, self.e.providers
        k, ck = top_k or s.top_k, s.candidate_k
        timings, rankings, t = {}, {}, time.perf_counter()

        caption = ""
        if image_path:
            caption = p.vision.caption(image_path)
            rankings["clip_image"] = e.store.search(CLIP, p.clip.embed_images([image_path])[0], ck, modalities, file_ids)
            timings["image_query_ms"] = _ms(t)
        text_q = " ".join(x for x in (query.strip(), caption) if x)
        if text_q:
            t1 = time.perf_counter()
            rankings["dense"] = e.store.search(TEXT, p.text.embed([text_q], is_query=True)[0], ck, modalities, file_ids)
            timings["dense_ms"] = _ms(t1)
            t1 = time.perf_counter()
            rankings["bm25"] = self.bm25.search(e, text_q, ck, modalities, file_ids)
            timings["bm25_ms"] = _ms(t1)
            t1 = time.perf_counter()
            rankings["clip_text"] = e.store.search(CLIP, p.clip.embed_texts([query or caption])[0], ck // 2, modalities, file_ids)
            timings["clip_text_ms"] = _ms(t1)

        fused = rrf(rankings, {"dense": 1.0, "bm25": 0.8, "clip_text": 0.7, "clip_image": 1.2})
        cand_ids = sorted(fused, key=lambda c: -fused[c])[:ck]
        chunks = e.db.get_chunks(cand_ids)
        cands = [chunks[c] for c in cand_ids if c in chunks]
        if not cands:
            return RetrievalResult([], False, timings, text_q)

        t1 = time.perf_counter()
        rr = p.reranker.score(text_q, [c["content"] for c in cands]) if text_q else [0.0] * len(cands)
        timings["rerank_ms"] = _ms(t1)
        clip_img = dict(rankings.get("clip_image", []))
        for c, r in zip(cands, rr):
            c["rerank"] = round(r, 4)
            c["fused"] = round(fused[c["id"]], 5)
            # an image query is grounded by visual similarity even when captions don't overlap
            c["score"] = max(r, clip_img.get(c["id"], 0.0)) if image_path else r
        cands.sort(key=lambda c: -c["score"])
        hits = _diversify(cands, k)
        grounded = bool(hits) and hits[0]["score"] >= s.rerank_threshold
        timings["retrieval_total_ms"] = _ms(t)
        return RetrievalResult(hits, grounded, timings, text_q)


def _diversify(cands: list[dict], k: int, per_file: int = 4) -> list[dict]:
    """Cap hits per file so one long document can't crowd out other modalities."""
    out, counts = [], {}
    for c in cands:
        if counts.get(c["file_id"], 0) >= per_file:
            continue
        counts[c["file_id"]] = counts.get(c["file_id"], 0) + 1
        out.append(c)
        if len(out) >= k:
            break
    return out


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)
