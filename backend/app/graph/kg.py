"""Cross-file knowledge graph: entities from every chunk, edges = co-occurrence in a chunk."""
import itertools
import re
import threading
from collections import Counter, defaultdict

import networkx as nx


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip(" .,:;'\"()[]")).strip()


class KnowledgeGraph:
    def __init__(self):
        self._lock = threading.Lock()
        self._g: nx.Graph | None = None

    def invalidate(self):
        self._g = None

    def build(self, chunks: list[dict], ner) -> nx.Graph:
        g = nx.Graph()
        mentions: dict[str, set] = defaultdict(set)
        labels: dict[str, Counter] = defaultdict(Counter)
        files: dict[str, set] = defaultdict(set)
        for c in chunks:
            ents = {}
            for name, label in ner.entities(c["content"]):
                n = _norm(name)
                if len(n) < 3 or n.lower() in {"image", "figure", "video", "page", "text"}:
                    continue
                key = n.lower()
                ents[key] = n
                labels[key][label] += 1
                files[key].add(c["file_name"])
                mentions[key].add(c["id"])
            for a, b in itertools.combinations(sorted(ents), 2):
                w = g.get_edge_data(a, b, {"weight": 0})["weight"]
                g.add_edge(a, b, weight=w + 1)
            for key, disp in ents.items():
                g.add_node(key, name=disp)
        for key in g.nodes:
            g.nodes[key].update(count=len(mentions[key]), label=labels[key].most_common(1)[0][0],
                                files=sorted(files[key]), chunks=sorted(mentions[key])[:50])
        return g

    def get(self, engine) -> nx.Graph:
        with self._lock:
            if self._g is None:
                self._g = self.build(engine.db.all_chunks(), engine.providers.ner)
            return self._g

    def payload(self, engine, limit: int = 120) -> dict:
        g = self.get(engine)
        top = sorted(g.nodes, key=lambda n: -g.nodes[n]["count"])[:limit]
        keep = set(top)
        nodes = [{"id": n, **{k: g.nodes[n][k] for k in ("name", "count", "label", "files")}} for n in top]
        links = [{"source": a, "target": b, "weight": d["weight"]} for a, b, d in g.edges(data=True) if a in keep and b in keep]
        return {"nodes": nodes, "links": links}
