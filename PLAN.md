# OmniRAG — Offline Multimodal RAG · Build Plan

**Constraints:** 1-day build · NVIDIA 4–6 GB VRAM · English · single user, local · fully offline after first model download.

## 1. Architecture

```
React + Tailwind (Vite) ──HTTP/SSE──► FastAPI backend
                                        ├─ Ingestion pipeline (background job queue, progress via SSE)
                                        │    PDF    → PyMuPDF text + Tesseract OCR fallback + pdfplumber tables
                                        │             + embedded images → vision caption
                                        │    Image  → CLIP embedding + vision-LLM caption + OCR
                                        │    Audio  → faster-whisper (segment timestamps)
                                        │    Video  → ffmpeg audio → whisper; scene keyframes → CLIP + caption
                                        ├─ Chunker (page/segment/time aware, keeps source metadata)
                                        ├─ Embeddings: bge-small-en (text, 384d) + CLIP ViT-B/32 (image, 512d)
                                        ├─ Vector store: Qdrant embedded mode (named vectors "text" and "clip")
                                        ├─ Retriever: hybrid (dense + BM25) → RRF fusion → cross-encoder rerank
                                        ├─ Generator: Ollama LLM, streaming, forced [n] citations
                                        ├─ Knowledge graph: spaCy NER + co-occurrence → NetworkX
                                        └─ Eval: golden set → recall@k, MRR, citation faithfulness
SQLite: file registry, chunks, jobs, chat history, eval runs
Files: ./data/uploads, ./data/thumbs (page renders, keyframes)
```

**Models (sized for VRAM; only one Ollama model is loaded at a time, with keep_alive tuned):**
| Role | Model | Approx. VRAM |
|---|---|---|
| Answer LLM | `qwen2.5:3b` (Ollama, Q4) | ~2.5 GB |
| Vision captions | `moondream` (Ollama) | ~1.7 GB |
| Speech-to-text | faster-whisper `small.en` int8 | ~0.8 GB |
| Text embeddings | `BAAI/bge-small-en-v1.5` | ~0.2 GB |
| Cross-modal | `open_clip ViT-B-32` | ~0.4 GB |
| Reranker | `BAAI/bge-reranker-base` | ~0.4 GB |
| TTS | Piper (CPU), with browser speechSynthesis as fallback | 0 |

## 2. Key behaviours
- **Cross-modal search:** text→image uses the CLIP text encoder against image and keyframe vectors; image→text works by uploading a query image, embedding it with CLIP and captioning it, then searching both indexes.
- **Citations:** every chunk carries `{file, type, page | t_start–t_end | bbox}`. The LLM must cite `[n]`, and a post-check drops any claim whose `[n]` isn't among the retrieved chunks. The UI shows a PDF page preview, the image, or an audio/video player that seeks to the timestamp.
- **Voice:** a mic button records in the browser → `/query/voice` → whisper → normal query. A 🔊 button reads the answer aloud via Piper.
- **Knowledge graph:** entities from all chunks, edges weighted by co-occurrence, shown with react-force-graph. Clicking a node filters chat to that entity.
- **Eval dashboard:** a seeded golden set (`eval/golden.jsonl`) and a "Run eval" button. Charts for recall@5, MRR, faithfulness, and latency per stage.

## 3. Repo layout
```
backend/app/{main.py, config.py, api/, ingest/{pdf,image,audio,video}.py, index/, retrieval/, llm/, graph/, eval/, db.py}
backend/tests/   frontend/src/{pages/Chat,Library,Graph,Eval, components/}
docker-compose.yml (ollama + backend + frontend)   Makefile   README.md   sample_data/
```

## 4. Phases (in priority order; each one is demoable at its end)
1. **Skeleton (≈1h):** FastAPI, config, SQLite, Qdrant, health endpoint, Vite+Tailwind shell, model-download script.
2. **Ingestion, all 4 modalities (≈3h):** loaders, chunker, embeddings, job queue with SSE progress, thumbnails. Tests use tiny fixture files.
3. **Retrieval + generation (≈2h):** hybrid search, RRF, rerank, streaming answer with citations and citation validation.
4. **Polished UI (≈2.5h):** chat with citation cards and media previews (seek to timestamp, page preview), library with drag-drop upload and progress, image-query upload, modality filters.
5. **Voice in/out (≈0.5h).**
6. **Video deep indexing (≈1h):** scene-change keyframes with ffmpeg, captions, timestamp alignment.
7. **Knowledge graph (≈1h).**
8. **Eval dashboard + Docker + README + demo dataset (≈1.5h).**

**Cut line if time runs short:** phases 1–5 make a complete demo. Phases 6–8 are added in order, and the plan only ships what passes its tests.

## 5. Prerequisites
Already installed: Python 3.10+, Node 18+, Ollama. **No system installs needed:** ffmpeg comes from `imageio-ffmpeg` (bundled binary) and OCR from `rapidocr-onnxruntime` (instead of Tesseract). I'll run `ollama pull qwen2.5:3b moondream`. Docker Desktop is optional and only needed for the compose demo.

## 7. Product decisions
- Name **OmniRAG**, clean light UI (minimal, Notion-like), with a generic theme and your own files.
- Strictly grounded: if retrieval finds nothing above the rerank threshold, it replies "Not found in your files" and lists the closest sources.
- Saved chat sessions in SQLite. Follow-up questions are rewritten into standalone queries using the last 4 turns.
- Sized for about 20–50 files, videos under 10 minutes and PDFs under 100 pages.

## 6. Done criteria
- Upload one PDF (including a scanned page), one image, one mp3 and one mp4. Each ends in the "indexed" state.
- "Show me the diagram of X" returns the right image (text→image).
- Uploading a photo returns related passages (image→text).
- Every answer has clickable citations that open the exact page or timestamp.
- The pytest suite passes, and `docker compose up` runs the full app.
