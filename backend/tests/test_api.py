import json

from .conftest import wait_indexed


def upload(client, *paths):
    files = [("files", (p.name, p.read_bytes())) for p in paths]
    r = client.post("/api/files", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def chat(client, message, **form):
    with client.stream("POST", "/api/chat", data={"message": message, **form}) as r:
        assert r.status_code == 200
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    return events, next(e for e in events if e["type"] == "done")


def ingest_all(client, fx):
    from app.providers import fake
    fake.SIDECAR_DIRS[:] = [fx]
    return upload(client, fx / "biology_notes.pdf", fx / "eiffel_tower.jpg", fx / "meeting.mp3")


def test_health(client):
    assert client.get("/api/health").json()["providers"] == "fake"


def test_rejects_unsupported(client, tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"MZ")
    assert client.post("/api/files", files=[("files", ("x.exe", b"MZ"))]).json()[0]["error"]


def test_multimodal_ingest_and_grounded_answers(client, fixtures):
    out = ingest_all(client, fixtures)
    files = wait_indexed(client, 3)
    by = {f["name"]: f for f in files}
    assert all(f["status"] == "indexed" for f in files), files
    assert by["biology_notes.pdf"]["n_chunks"] >= 3  # 2 pages + figure

    chunks = client.get(f"/api/files/{by['biology_notes.pdf']['id']}/chunks").json()
    kinds = {c["kind"] for c in chunks}
    assert {"text", "figure"} <= kinds
    assert all(c["page"] for c in chunks)

    # text -> PDF with page citation
    _, done = chat(client, "What does the Calvin cycle fix?")
    assert done["grounded"] and done["used"]
    top = done["citations"][0]
    assert top["file_name"] == "biology_notes.pdf" and top["page"] == 1

    # text -> image (cross-modal)
    _, done = chat(client, "Eiffel Tower photo in Paris")
    assert done["citations"][0]["modality"] == "image"

    # audio with timestamps
    _, done = chat(client, "What happened to revenue in the budget meeting?")
    c = done["citations"][0]
    assert c["modality"] == "audio" and c["t_start"] is not None

    # strict refusal
    _, done = chat(client, "quantum chromodynamics gluon lattice")
    assert done["answer"] == "Not found in your files." and not done["grounded"]

    # modality filter
    _, done = chat(client, "Eiffel Tower photo in Paris", modalities="pdf")
    assert all(c["modality"] == "pdf" for c in done["citations"])


def test_image_query_and_sessions(client, fixtures):
    ingest_all(client, fixtures)
    wait_indexed(client, 3)
    q = fixtures / "query_tower.jpg"
    q.write_bytes((fixtures / "eiffel_tower.jpg").read_bytes())
    with client.stream("POST", "/api/chat", data={"message": ""}, files={"image": ("eiffel_tower.jpg", q.read_bytes())}) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    done = next(e for e in events if e["type"] == "done")
    sid = next(e for e in events if e["type"] == "session")["session_id"]
    assert done["citations"] and done["citations"][0]["modality"] == "image"

    _, done2 = chat(client, "and the meeting revenue?", session_id=sid)
    msgs = client.get(f"/api/sessions/{sid}/messages").json()
    assert len(msgs) == 4 and msgs[-1]["role"] == "assistant"


def test_graph_eval_delete(client, fixtures):
    out = ingest_all(client, fixtures)
    wait_indexed(client, 3)
    g = client.get("/api/graph").json()
    names = {n["name"] for n in g["nodes"]}
    assert "Eiffel Tower" in names or "Paris" in names
    node = client.get(f"/api/graph/node/{g['nodes'][0]['id']}").json()
    assert node["mentions"]

    golden = [{"question": "What does the Calvin cycle fix?", "expected_file": "biology", "expected_terms": ["carbon"]},
              {"question": "quantum chromodynamics gluon lattice", "expected_file": None}]
    assert client.put("/api/eval/golden", json=golden).json()["n"] == 2
    run = client.post("/api/eval/run").json()
    assert run["summary"]["recall_at_k"] == 1.0 and run["summary"]["refusal_acc"] == 1.0
    assert client.get("/api/eval/runs").json()[0]["id"] == run["id"]

    pdf = next(f for f in out if f["name"] == "biology_notes.pdf")
    assert client.delete(f"/api/files/{pdf['id']}").json()["ok"]
    _, done = chat(client, "What does the Calvin cycle fix?")
    assert all(c["file_name"] != "biology_notes.pdf" for c in done["citations"])
