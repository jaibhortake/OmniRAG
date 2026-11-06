"""Retrieval + answer quality evaluation over a golden set.

golden.jsonl rows: {"question": str, "expected_file": str | null, "expected_terms": [str]}
  expected_file - substring of the file name that should be retrieved; null = unanswerable (expects refusal)
Metrics:
  recall@k        expected file appears in top-k hits
  mrr             1 / rank of first hit from expected file
  citation_rate   fraction of answer sentences carrying a valid [n] citation
  term_coverage   fraction of expected_terms present in the answer
  refusal_acc     unanswerable questions correctly answered "Not found"
"""
import json
import re
import time
from statistics import mean
from typing import TYPE_CHECKING

from ..db import new_id
from ..llm.answer import NOT_FOUND

if TYPE_CHECKING:
    from ..engine import Engine

_SENT = re.compile(r"(?<=[.!?])\s+")


def load_golden(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def citation_rate(answer: str) -> float:
    sents = [s for s in _SENT.split(answer.strip()) if len(s) > 15]
    if not sents:
        return 0.0
    return sum(1 for s in sents if re.search(r"\[\d+\]", s)) / len(sents)


def run_eval(engine: "Engine", rows: list[dict], k: int = 5) -> dict:
    details = []
    for row in rows:
        t0 = time.perf_counter()
        res = engine.answer(row["question"], history=[], persist=False)
        latency = (time.perf_counter() - t0) * 1000
        exp = row.get("expected_file")
        names = [c["file_name"].lower() for c in res["citations"]][:k]
        rank = next((i + 1 for i, n in enumerate(names) if exp and exp.lower() in n), None)
        ans = res["answer"]
        terms = row.get("expected_terms") or []
        details.append({
            "question": row["question"], "expected_file": exp, "answer": ans, "latency_ms": round(latency, 1),
            "hit": rank is not None, "rank": rank, "refused": ans.strip().startswith(NOT_FOUND[:9]),
            "citation_rate": round(citation_rate(ans), 3),
            "term_coverage": round(sum(t.lower() in ans.lower() for t in terms) / len(terms), 3) if terms else None,
            "timings": res["timings"],
        })
    pos = [d for d in details if d["expected_file"]]
    neg = [d for d in details if not d["expected_file"]]
    answered = [d for d in pos if not d["refused"]]
    tc = [d["term_coverage"] for d in pos if d["term_coverage"] is not None]
    summary = {
        "n": len(details), "k": k,
        "recall_at_k": round(mean(d["hit"] for d in pos), 3) if pos else None,
        "mrr": round(mean(1 / d["rank"] if d["rank"] else 0 for d in pos), 3) if pos else None,
        "citation_rate": round(mean(d["citation_rate"] for d in answered), 3) if answered else None,
        "term_coverage": round(mean(tc), 3) if tc else None,
        "refusal_acc": round(mean(d["refused"] for d in neg), 3) if neg else None,
        "avg_latency_ms": round(mean(d["latency_ms"] for d in details), 1) if details else None,
        "providers": engine.providers.name,
    }
    run = {"id": new_id(), "created": time.time(), "summary": summary, "details": details}
    with engine.db.tx() as c:
        c.execute("INSERT INTO eval_runs(id,created,summary,details) VALUES(?,?,?,?)",
                  (run["id"], run["created"], json.dumps(summary), json.dumps(details)))
    return run
