import os
import time

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

os.environ["OMNIRAG_PROVIDERS"] = "fake"

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    s = Settings(providers="fake", data_dir=tmp_path / "data", qdrant_path=":memory:", rerank_threshold=0.2)
    s.ensure_dirs()
    with TestClient(create_app(s)) as c:
        yield c


def wait_indexed(client, n, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        files = client.get("/api/files").json()
        if len(files) >= n and all(f["status"] in ("indexed", "failed") for f in files):
            return files
        time.sleep(0.1)
    raise AssertionError(f"timed out: {client.get('/api/files').json()}")


@pytest.fixture
def fixtures(tmp_path):
    d = tmp_path / "fx"
    d.mkdir()
    # PDF: text page + table-ish text + embedded figure
    pdf = fitz.open()
    pg = pdf.new_page()
    pg.insert_text((72, 72), "Photosynthesis converts sunlight into chemical energy in chloroplasts.", fontsize=11)
    pg.insert_text((72, 100), "The Calvin cycle fixes carbon dioxide into glucose.", fontsize=11)
    fig = d / "fig.png"
    Image.new("RGB", (300, 300), "green").save(fig)
    pg.insert_image(fitz.Rect(72, 150, 272, 350), filename=str(fig))
    pg2 = pdf.new_page()
    pg2.insert_text((72, 72), "Mitochondria are the powerhouse of the cell and produce ATP.", fontsize=11)
    pdf.save(d / "biology_notes.pdf")
    # Image + sidecar "what the vision model sees"
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).rectangle((50, 50, 350, 250), outline="red", width=5)
    img.save(d / "eiffel_tower.jpg")
    (d / "eiffel_tower.jpg.txt").write_text("A photo of the Eiffel Tower in Paris at sunset with tourists.")
    # Audio: bytes don't matter for fake STT; transcript in sidecar
    (d / "meeting.mp3").write_bytes(b"\x00" * 2048)
    (d / "meeting.mp3.txt").write_text("Welcome to the quarterly budget meeting. Revenue grew twelve percent. "
                                        "The marketing budget will be cut next quarter.")
    return d
