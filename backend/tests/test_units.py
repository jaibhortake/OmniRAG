from app.ingest.common import split_text
from app.ingest.media import window_segments
from app.llm.answer import fmt_ts, validate_citations
from app.providers.base import Segment
from app.retrieval.hybrid import rrf
from app.eval.runner import citation_rate


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    parts = split_text(text, 200, 40)
    assert len(parts) > 5
    assert all(len(p) <= 240 for p in parts)
    assert split_text("short", 200, 40) == ["short"]
    assert split_text("   ", 200, 40) == []


def test_window_segments_keeps_timestamps():
    segs = [Segment(i * 2.0, i * 2.0 + 2, "word " * 20) for i in range(10)]
    wins = window_segments(segs, 250)
    assert wins[0].t_start == 0.0
    assert wins[-1].t_end == 20.0
    assert all(w.t_start <= w.t_end for w in wins)


def test_validate_citations_strips_invalid():
    ans, used, invalid = validate_citations("Fact one [1]. Fact two [7]. Fact three [2][1].", 3)
    assert used == [1, 2]
    assert invalid == 1
    assert "[7]" not in ans


def test_rrf_rewards_agreement():
    fused = rrf({"a": [("x", 1), ("y", 1)], "b": [("y", 1), ("z", 1)]}, {})
    assert max(fused, key=fused.get) == "y"


def test_fmt_ts_and_citation_rate():
    assert fmt_ts(75) == "1:15"
    assert fmt_ts(3725) == "1:02:05"
    assert citation_rate("This is cited [1]. This one is not cited at all.") == 0.5
