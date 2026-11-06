"""Grounded answer generation with enforced, validated citations."""
import re
from typing import Iterator

from ..providers.base import Providers

SYSTEM = """You are OmniRAG, an assistant that answers ONLY from the numbered sources provided.
Rules:
- Every factual sentence must end with one or more citations like [1] or [2][3] referring to the sources.
- Sources may be PDF text, tables, image/figure descriptions, audio transcripts or video scenes; treat image and scene descriptions as what the image shows.
- If the sources do not contain the answer, reply exactly: "Not found in your files." Do not use outside knowledge.
- Be concise and well structured. Use bullet points for lists. Never invent sources."""

REWRITE_SYSTEM = """Rewrite the follow-up question into a standalone search query using the conversation.
Return only the rewritten query, nothing else."""

NOT_FOUND = "Not found in your files."
_CITE = re.compile(r"\[(\d+)\]")


def locator(c: dict) -> str:
    if c.get("page"):
        return f"page {c['page']}"
    if c.get("t_start") is not None:
        return f"{fmt_ts(c['t_start'])}–{fmt_ts(c.get('t_end') or c['t_start'])}"
    return c.get("kind", "")


def fmt_ts(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = [f"[{i}] ({c['file_name']} · {c['kind']} · {locator(c)})\n{c['content']}" for i, c in enumerate(hits, 1)]
    return "Sources:\n" + "\n".join(blocks) + f"\nQuestion: {question}\nAnswer with citations:"


def rewrite_query(p: Providers, history: list[dict], question: str) -> str:
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history)
    out = p.llm.complete(REWRITE_SYSTEM, f"Conversation:\n{convo}\nFollow-up question: {question}").strip().strip('"')
    return out if 3 < len(out) < 400 else question


def stream_answer(p: Providers, question: str, hits: list[dict]) -> Iterator[str]:
    yield from p.llm.stream(SYSTEM, build_prompt(question, hits))


def validate_citations(answer: str, n_sources: int) -> tuple[str, list[int], int]:
    """Strip citations pointing at sources that don't exist; return (clean answer, used ids, n_invalid)."""
    used, invalid = [], 0

    def fix(m):
        nonlocal invalid
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            if n not in used:
                used.append(n)
            return m.group(0)
        invalid += 1
        return ""

    return _CITE.sub(fix, answer), used, invalid


def citation_payload(hits: list[dict]) -> list[dict]:
    return [{
        "n": i, "chunk_id": c["id"], "file_id": c["file_id"], "file_name": c["file_name"], "modality": c["modality"],
        "kind": c["kind"], "page": c["page"], "t_start": c["t_start"], "t_end": c["t_end"], "thumb": c["thumb"],
        "locator": locator(c), "snippet": c["content"][:400], "score": round(c.get("score", 0), 3),
    } for i, c in enumerate(hits, 1)]
