"""Vector index on Qdrant (embedded mode, no server).

Two collections:
  text - bge embeddings of every text-bearing chunk (PDF text, tables, OCR, captions, transcripts)
  clip - CLIP embeddings of images, PDF figures and video keyframes (enables text->image and image->image)
"""
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

TEXT, CLIP = "text", "clip"


def _pid(chunk_id: str) -> str:
    return str(uuid.UUID(hex=chunk_id.ljust(32, "0")[:32]))


class VectorStore:
    def __init__(self, path: str, text_dim: int, clip_dim: int):
        self.client = QdrantClient(location=":memory:") if path == ":memory:" else QdrantClient(path=path)
        for name, dim in ((TEXT, text_dim), (CLIP, clip_dim)):
            if self.client.collection_exists(name):
                existing = self.client.get_collection(name).config.params.vectors.size
                if existing != dim:  # switched provider set -> index is incompatible
                    self.client.delete_collection(name)
            if not self.client.collection_exists(name):
                self.client.create_collection(name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
                for field in ("file_id", "modality"):
                    try:
                        self.client.create_payload_index(name, field, qm.PayloadSchemaType.KEYWORD)
                    except Exception:  # local mode ignores payload indexes
                        pass

    def upsert(self, collection: str, chunks: list[dict], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self.client.upsert(collection, points=[
            qm.PointStruct(id=_pid(c["id"]), vector=v.tolist(),
                           payload={"chunk_id": c["id"], "file_id": c["file_id"], "modality": c["modality"]})
            for c, v in zip(chunks, vectors)
        ])

    def search(self, collection: str, vector: np.ndarray, limit: int, modalities: list[str] | None = None,
               file_ids: list[str] | None = None) -> list[tuple[str, float]]:
        must = []
        if modalities:
            must.append(qm.FieldCondition(key="modality", match=qm.MatchAny(any=modalities)))
        if file_ids:
            must.append(qm.FieldCondition(key="file_id", match=qm.MatchAny(any=file_ids)))
        res = self.client.query_points(collection, query=vector.tolist(), limit=limit,
                                       query_filter=qm.Filter(must=must) if must else None)
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def delete_file(self, file_id: str) -> None:
        flt = qm.FilterSelector(filter=qm.Filter(must=[qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id))]))
        for name in (TEXT, CLIP):
            self.client.delete(name, points_selector=flt)

    def close(self):
        self.client.close()
