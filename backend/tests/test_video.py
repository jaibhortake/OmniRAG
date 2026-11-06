"""Real ffmpeg path: keyframe extraction + timestamp alignment on a generated test video."""
import shutil

import pytest

from app.ingest.common import run_ffmpeg
from app.ingest.media import load_video
from app.providers import fake


def test_video_keyframes_aligned(tmp_path):
    vid = tmp_path / "lecture.mp4"
    # 3 colour scenes x 2s, so scene detection must find the cuts
    r = run_ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=320x240:d=2", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
                    "-f", "lavfi", "-i", "color=c=green:s=320x240:d=2",
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", str(vid)])
    if r.returncode != 0:
        pytest.skip("ffmpeg unavailable: " + r.stderr[-200:])
    (tmp_path / "lecture.mp4.txt").write_text("Today we study red things. Next blue things. Finally green things.")
    p = fake.build()
    pieces = load_video(str(vid), "vid1", tmp_path, p, lambda *_: None)
    frames = [pc for pc in pieces if pc.kind == "keyframe"]
    assert len(frames) >= 3
    starts = [f.t_start for f in frames]
    assert starts == sorted(starts) and starts[0] == 0.0
    assert any(abs(s - 2.0) < 0.3 for s in starts)
    assert all(f.image_path for f in frames)
    trans = [pc for pc in pieces if pc.kind == "transcript"]
    assert trans and trans[0].thumb.startswith("vid1_kf")
