# OmniRAG: Offline Multimodal RAG

Ask questions across **PDFs, images, audio recordings and videos** in plain English, and get answers that cite the **exact page, image or timestamp**. Everything runs **on your own machine**: no cloud APIs and no data leaving the device.

- **Unified multimodal index:** PDFs (text layer, OCR for scanned pages, tables, embedded figures), images (vision captions, OCR, CLIP), audio (Whisper with timestamps) and video (scene keyframes plus transcript, aligned in time).
- **Cross-modal search:** text→image ("show me the circuit diagram"), image→text (upload a photo and find related passages), and image→image.
- **Citation transparency:** every claim carries a `[n]` citation. Clicking it opens the PDF page, the image, or the audio/video player at that second. Citations to sources that don't exist are stripped automatically.
- **Strictly grounded:** when the answer isn't in your files, OmniRAG says *"Not found in your files"* instead of making something up, and shows the closest matches.
- **Voice in/out**, saved chat sessions with follow-up understanding, an **entity knowledge graph** across all files, and an **evaluation dashboard** (Recall@5, MRR, citation rate, refusal accuracy, latency).

## Architecture

```
React + Tailwind UI ──HTTP / SSE──► FastAPI
                                     ├─ Ingestion worker
                                     │   PDF   → PyMuPDF text · RapidOCR fallback · pdfplumber tables · figures → vision caption
                                     │   Image → moondream caption · OCR · CLIP
                                     │   Audio → ffmpeg → faster-whisper (timestamped segments → windows)
                                     │   Video → audio transcript + scene-change keyframes → caption + CLIP, time-aligned
                                     ├─ Index: Qdrant (embedded)  text: bge-small (384d)   clip: ViT-B/32 (512d)
                                     ├─ Retrieval: dense + BM25 + CLIP-text (+ CLIP-image) → RRF fusion
                                     │             → bge-reranker cross-encoder → per-file diversity → grounding gate
                                     ├─ Generation: Ollama qwen2.5:3b, streamed, forced [n] citations + validation
                                     ├─ Knowledge graph: spaCy NER + co-occurrence (NetworkX)
                                     └─ Eval: golden set → recall@k, MRR, citation rate, refusal accuracy
SQLite: files · chunks · sessions · messages · eval runs
```

**Provider pattern:** every model sits behind an interface ([backend/app/providers/base.py](backend/app/providers/base.py)). `OMNIRAG_PROVIDERS=real` uses the local models. `fake` uses deterministic stand-ins, so the entire app and its test suite run on any laptop without downloading weights.

| Role | Model | VRAM |
|---|---|---|
| Answer LLM | qwen2.5:3b (Ollama, Q4) | ~2.5 GB |
| Vision captions | moondream (Ollama) | ~1.7 GB |
| Speech-to-text | faster-whisper small.en int8 | ~0.8 GB |
| Text embeddings / reranker | bge-small-en-v1.5 / bge-reranker-base | ~0.6 GB |
| Cross-modal | OpenCLIP ViT-B-32 | ~0.4 GB |

Ollama unloads idle models (`keep_alive=2m`), so it runs on a **4 GB GPU**. A single ingestion worker avoids VRAM thrashing.

## Quick start (Windows)

Prerequisites: Python 3.10+ (3.12 recommended), Node 18+, [Ollama](https://ollama.com) and an NVIDIA GPU (or use `-Cpu`). ffmpeg and OCR come bundled through pip.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1   # deps, UI build, one-time model download
.\run.ps1                                             # http://localhost:8000
```

After setup, the network cable can be unplugged: `run.ps1` sets `HF_HUB_OFFLINE=1`.

- UI rehearsal without models: `setup.ps1 -SkipModels`, then `.\run.ps1 -Fake`.
- Docker (Linux + NVIDIA Container Toolkit): `docker compose up --build`.
- Dev mode: `cd backend; .venv\Scripts\uvicorn app.main:app --reload` plus `cd frontend; npm run dev` (http://localhost:5173).

## Tests

```powershell
cd backend; .venv\Scripts\python -m pytest -q
```

The tests cover chunking, timestamp windows, RRF, citation validation, the full API flow over all modalities (ingest → cross-modal query → grounded refusal → modality filter → image query → sessions → graph → eval → delete), and real ffmpeg keyframe extraction with time alignment.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/files` | Upload (multi), queued for ingestion |
| GET | `/api/files/stream` | SSE live ingestion progress |
| POST | `/api/chat` | multipart `message`, `session_id`, `modalities`, optional `image` → SSE `session / status / sources / token / done` |
| POST | `/api/transcribe` · `/api/tts` | Voice in / out |
| GET | `/api/graph` · `/api/graph/node/{id}` | Knowledge graph |
| PUT/GET/POST | `/api/eval/golden` · `/api/eval/runs` · `/api/eval/run` | Evaluation |

Interactive docs are at `/docs`.

## Demo script (3 minutes)

1. **Library:** drop in a PDF (one scanned page), two photos, an MP3 and a short MP4, and let the judges watch live per-stage progress.
2. **Chat:** ask *"What were the key decisions in the meeting?"* and click a citation; the audio jumps to that second.
3. Ask *"Show me the chart about revenue"*; the image is returned from a text query.
4. Attach a photo with no text; related passages come back (image→text).
5. Ask something not in the files; the answer is *"Not found in your files"*, with no hallucination.
6. Press the mic to ask by voice, and 🔊 to hear the answer.
7. **Graph:** click an entity, then "Ask about…".
8. **Eval:** run the evaluation and show the metric scores.
