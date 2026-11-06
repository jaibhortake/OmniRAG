import asyncio
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..eval.runner import load_golden, run_eval
from ..ingest.common import modality_for

router = APIRouter(prefix="/api")
MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB
_SAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def eng(req: Request):
    return req.app.state.engine


def sse(gen):
    async def agen():
        loop = asyncio.get_running_loop()
        it = iter(gen)
        sentinel = object()
        while True:
            # model calls block, so pull each event in a worker thread
            ev = await loop.run_in_executor(None, next, it, sentinel)
            if ev is sentinel:
                break
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(agen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _save_upload(f: UploadFile, dst: Path) -> int:
    size = 0
    with dst.open("wb") as out:
        while chunk := await f.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"{f.filename} exceeds 1 GB")
            out.write(chunk)
    return size


# ---------------- health ----------------
@router.get("/health")
def health(req: Request):
    e = eng(req)
    s = e.settings
    ollama_ok, models = False, []
    if e.providers.name == "real":
        try:
            import ollama
            models = [m.model for m in ollama.Client(host=s.ollama_host).list().models]
            ollama_ok = True
        except Exception:
            pass
    return {"status": "ok", "providers": e.providers.name, "ollama": ollama_ok, "ollama_models": models,
            "llm": s.llm_model, "vision": s.vision_model, "files": len(e.db.list_files())}


# ---------------- files ----------------
@router.get("/files")
def list_files(req: Request):
    return eng(req).db.list_files()


@router.post("/files")
async def upload(req: Request, files: list[UploadFile] = File(...)):
    e, out = eng(req), []
    for f in files:
        mod = modality_for(f.filename or "")
        if not mod:
            out.append({"name": f.filename, "error": "unsupported file type"})
            continue
        name = _SAFE.sub("_", Path(f.filename).name)[:150]
        dst = e.settings.uploads_dir / f"{int(time.time() * 1000)}_{name}"
        size = await _save_upload(f, dst)
        fid = e.db.add_file(name, str(dst), mod, size)
        e.worker.submit(fid)
        out.append(e.db.get_file(fid))
    return out


@router.post("/files/{fid}/retry")
def retry(fid: str, req: Request):
    e = eng(req)
    if not e.db.get_file(fid):
        raise HTTPException(404)
    e.store.delete_file(fid)
    with e.db.tx() as c:
        c.execute("DELETE FROM chunks WHERE file_id=?", (fid,))
    e.db.update_file(fid, status="queued", progress=0, stage="Queued", error="")
    e.worker.submit(fid)
    return e.db.get_file(fid)


@router.delete("/files/{fid}")
def delete(fid: str, req: Request):
    if not eng(req).delete_file(fid):
        raise HTTPException(404)
    return {"ok": True}


@router.get("/files/{fid}/raw")
def raw(fid: str, req: Request):
    f = eng(req).db.get_file(fid)
    if not f:
        raise HTTPException(404)
    return FileResponse(f["path"], filename=f["name"], content_disposition_type="inline")


@router.get("/files/{fid}/chunks")
def file_chunks(fid: str, req: Request):
    return eng(req).db.query("SELECT id,kind,content,page,t_start,t_end,thumb FROM chunks WHERE file_id=? ORDER BY page, t_start", (fid,))


@router.get("/files/stream")
async def files_stream(req: Request):
    e = eng(req)

    async def gen():
        last = None
        while not await req.is_disconnected():
            cur = json.dumps(e.db.list_files())
            if cur != last:
                last = cur
                yield f"data: {cur}\n\n"
            await asyncio.sleep(0.7)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.get("/thumbs/{name}")
def thumb(name: str, req: Request):
    d = eng(req).settings.thumbs_dir.resolve()
    p = (d / name).resolve()
    if p.parent != d or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ---------------- chat ----------------
@router.get("/sessions")
def sessions(req: Request):
    return eng(req).db.query("SELECT * FROM sessions ORDER BY created DESC")


@router.post("/sessions")
def new_session(req: Request):
    return eng(req).db.create_session()


@router.get("/sessions/{sid}/messages")
def session_messages(sid: str, req: Request):
    return eng(req).db.messages(sid)


@router.delete("/sessions/{sid}")
def delete_session(sid: str, req: Request):
    with eng(req).db.tx() as c:
        c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    return {"ok": True}


@router.post("/chat")
async def chat(req: Request, message: str = Form(""), session_id: str = Form(""), modalities: str = Form(""),
               file_ids: str = Form(""), image: UploadFile | None = File(None)):
    e = eng(req)
    if not message.strip() and not image:
        raise HTTPException(400, "message or image required")
    if not session_id or not e.db.one("SELECT id FROM sessions WHERE id=?", (session_id,)):
        session_id = e.db.create_session()["id"]
    history = [{"role": m["role"], "content": m["content"]} for m in e.db.messages(session_id)]
    if not history:
        title = (message.strip() or "Image search")[:60]
        with e.db.tx() as c:
            c.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))

    img_path, img_thumb = None, ""
    if image and image.filename:
        if modality_for(image.filename) != "image":
            raise HTTPException(400, "query attachment must be an image")
        tmp = e.settings.thumbs_dir / f"query_{int(time.time() * 1000)}{Path(image.filename).suffix.lower()}"
        await _save_upload(image, tmp)
        img_path, img_thumb = str(tmp), tmp.name
    e.db.add_message(session_id, "user", message.strip() or "(image query)", meta={"image": img_thumb})
    mods = [m for m in modalities.split(",") if m] or None
    fids = [f for f in file_ids.split(",") if f] or None

    def gen():
        yield {"type": "session", "session_id": session_id}
        for ev in e.answer_stream(message.strip(), history, image_path=img_path, modalities=mods, file_ids=fids):
            if ev["type"] == "done":
                ev["message_id"] = e.db.add_message(session_id, "assistant", ev["answer"], ev["citations"],
                                                    {"used": ev["used"], "grounded": ev["grounded"], "timings": ev["timings"]})
            yield ev
    return sse(gen())


# ---------------- voice ----------------
@router.post("/transcribe")
async def transcribe(req: Request, audio: UploadFile = File(...)):
    e = eng(req)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"in{Path(audio.filename or 'a.webm').suffix or '.webm'}"
        await _save_upload(audio, src)
        from ..ingest.common import run_ffmpeg
        wav = Path(td) / "in.wav"
        run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", "16000", str(wav)])
        segs = await asyncio.to_thread(e.providers.stt.transcribe, str(wav if wav.exists() else src))
    return {"text": " ".join(s.text for s in segs).strip()}


class TTSIn(BaseModel):
    text: str


@router.post("/tts")
async def tts(body: TTSIn, req: Request):
    clean = re.sub(r"\[\d+\]", "", body.text)[:3000]
    wav = await asyncio.to_thread(eng(req).providers.tts.synthesize, clean)
    if not wav:
        return Response(status_code=204)  # client falls back to browser speech synthesis
    return Response(wav, media_type="audio/wav")


# ---------------- graph ----------------
@router.get("/graph")
async def graph(req: Request, limit: int = 120):
    e = eng(req)
    return await asyncio.to_thread(e.graph.payload, e, limit)


@router.get("/graph/node/{node_id}")
async def graph_node(node_id: str, req: Request):
    e = eng(req)
    g = await asyncio.to_thread(e.graph.get, e)
    if node_id not in g:
        raise HTTPException(404)
    n = g.nodes[node_id]
    chunks = e.db.get_chunks(n["chunks"][:12])
    nbrs = sorted(g[node_id].items(), key=lambda kv: -kv[1]["weight"])[:15]
    return {"id": node_id, **{k: n[k] for k in ("name", "label", "count", "files")},
            "neighbors": [{"id": k, "name": g.nodes[k]["name"], "weight": v["weight"]} for k, v in nbrs],
            "mentions": [{"chunk_id": c["id"], "file_id": c["file_id"], "file_name": c["file_name"], "kind": c["kind"],
                          "page": c["page"], "t_start": c["t_start"], "snippet": c["content"][:240]} for c in chunks.values()]}


# ---------------- eval ----------------
def _golden_path(e) -> Path:
    p = e.settings.data_dir / "golden.jsonl"
    if not p.exists():
        sample = Path(__file__).resolve().parents[2] / "eval" / "golden.sample.jsonl"
        if sample.exists():
            shutil.copy(sample, p)
    return p


@router.get("/eval/golden")
def get_golden(req: Request):
    return load_golden(_golden_path(eng(req)))


@router.put("/eval/golden")
def put_golden(rows: list[dict], req: Request):
    for r in rows:
        if not isinstance(r.get("question"), str) or not r["question"].strip():
            raise HTTPException(400, "each row needs a question")
    _golden_path(eng(req)).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return {"ok": True, "n": len(rows)}


@router.post("/eval/run")
async def eval_run(req: Request, k: int = 5):
    e = eng(req)
    rows = load_golden(_golden_path(e))
    if not rows:
        raise HTTPException(400, "golden set is empty")
    return await asyncio.to_thread(run_eval, e, rows, k)


@router.get("/eval/runs")
def eval_runs(req: Request):
    rows = eng(req).db.query("SELECT * FROM eval_runs ORDER BY created DESC LIMIT 20")
    for r in rows:
        r["summary"], r["details"] = json.loads(r["summary"]), json.loads(r["details"])
    return rows
