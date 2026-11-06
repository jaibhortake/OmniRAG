"""PDF loader: text layer, OCR fallback for scanned pages, tables, and captioned embedded figures."""
import logging
from pathlib import Path

import fitz  # PyMuPDF

from ..providers.base import Providers
from .common import Piece, Progress

log = logging.getLogger("omnirag.ingest.pdf")
MIN_TEXT_CHARS = 40
MIN_FIG_PX = 150
MAX_FIGURES = 25


def _tables(path: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    try:
        import pdfplumber
    except ImportError:
        return out
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                for t in page.extract_tables() or []:
                    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in t if r and any(r)]
                    if len(rows) < 2:
                        continue
                    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                    out.setdefault(i + 1, []).append("\n".join(md))
    except Exception as e:  # tables are best-effort
        log.warning("table extraction failed: %s", e)
    return out


def load_pdf(path: str, file_id: str, thumbs: Path, p: Providers, progress: Progress) -> list[Piece]:
    pieces: list[Piece] = []
    doc = fitz.open(path)
    tables = _tables(path)
    n, figures = len(doc), 0
    for i, page in enumerate(doc):
        pno = i + 1
        progress(0.05 + 0.6 * i / max(n, 1), f"Reading page {pno}/{n}")
        thumb_name = f"{file_id}_p{pno}.jpg"
        page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(str(thumbs / thumb_name))

        text = page.get_text("text").strip()
        ocr_used = False
        if len(text) < MIN_TEXT_CHARS:  # scanned page -> OCR the rendered page
            hi = thumbs / f"{file_id}_p{pno}_ocr.png"
            page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).save(str(hi))
            text = (text + "\n" + p.ocr.read(str(hi))).strip()
            hi.unlink(missing_ok=True)
            ocr_used = True
        if text:
            pieces.append(Piece("text", text, page=pno, thumb=thumb_name, meta={"ocr": ocr_used}))
        for t in tables.get(pno, []):
            pieces.append(Piece("table", t, page=pno, thumb=thumb_name, splittable=False))

        for img in page.get_images(full=True):
            if figures >= MAX_FIGURES:
                break
            try:
                info = doc.extract_image(img[0])
            except Exception:
                continue
            if info["width"] < MIN_FIG_PX or info["height"] < MIN_FIG_PX:
                continue
            figures += 1
            fig = thumbs / f"{file_id}_p{pno}_fig{figures}.{info['ext']}"
            fig.write_bytes(info["image"])
            progress(0.05 + 0.6 * i / max(n, 1), f"Captioning figure {figures} (page {pno})")
            caption = p.vision.caption(str(fig))
            ocr = p.ocr.read(str(fig))
            content = f"Figure on page {pno}: {caption}" + (f"\nText in figure: {ocr}" if ocr else "")
            pieces.append(Piece("figure", content, page=pno, thumb=fig.name, image_path=str(fig), splittable=False))
    doc.close()
    return pieces
