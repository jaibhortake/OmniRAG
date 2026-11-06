"""Image, audio and video loaders."""
import logging
import re
from pathlib import Path

from ..providers.base import Providers, Segment
from .common import Piece, Progress, make_thumb, run_ffmpeg

log = logging.getLogger("omnirag.ingest.media")


def load_image(path: str, file_id: str, thumbs: Path, p: Providers, progress: Progress) -> list[Piece]:
    progress(0.1, "Generating thumbnail")
    thumb = make_thumb(path, thumbs / f"{file_id}.jpg")
    progress(0.3, "Captioning image")
    caption = p.vision.caption(path)
    progress(0.6, "Reading text in image")
    ocr = p.ocr.read(path)
    content = f"Image: {caption}" + (f"\nText in image: {ocr}" if ocr else "")
    return [Piece("image", content, thumb=thumb, image_path=path, splittable=False)]


def _to_wav(src: str, dst: Path) -> bool:
    r = run_ffmpeg(["-i", src, "-vn", "-ac", "1", "-ar", "16000", str(dst)])
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000


def window_segments(segs: list[Segment], max_chars: int) -> list[Piece]:
    """Group whisper segments into retrieval windows that keep exact start/end timestamps."""
    out, buf = [], []
    for s in segs:
        buf.append(s)
        if sum(len(b.text) for b in buf) >= max_chars:
            out.append(buf)
            buf = buf[-1:]  # one-segment overlap keeps context across windows
    if buf and (not out or buf != out[-1][-1:]):
        out.append(buf)
    return [Piece("transcript", " ".join(b.text for b in w), t_start=w[0].start, t_end=w[-1].end, splittable=False)
            for w in out]


def _transcribe(path: str, file_id: str, thumbs: Path, p: Providers, max_chars: int) -> list[Segment]:
    wav = thumbs / f"{file_id}.wav"
    try:
        if not _to_wav(path, wav):
            log.info("no audio stream in %s", path)
            # fake providers read a sidecar next to the original file
            return p.stt.transcribe(path) if p.name == "fake" else []
        return p.stt.transcribe(str(wav) if p.name != "fake" else path)
    finally:
        wav.unlink(missing_ok=True)


def load_audio(path: str, file_id: str, thumbs: Path, p: Providers, progress: Progress, max_chars: int = 900) -> list[Piece]:
    progress(0.1, "Transcribing audio")
    segs = _transcribe(path, file_id, thumbs, p, max_chars)
    progress(0.7, f"Transcribed {len(segs)} segments")
    return window_segments(segs, max_chars)


_PTS = re.compile(r"pts_time:([\d.]+)")


def extract_keyframes(path: str, file_id: str, thumbs: Path, max_frames: int) -> list[tuple[float, Path]]:
    """Scene-change keyframes (plus the first frame), with their timestamps."""
    pattern = thumbs / f"{file_id}_kf%04d.jpg"
    r = run_ffmpeg(["-i", path, "-vf", "select='eq(n\\,0)+gt(scene\\,0.3)',showinfo,scale=640:-2",
                    "-fps_mode", "vfr", "-frames:v", str(max_frames), "-q:v", "4", str(pattern)])
    times = [float(t) for t in _PTS.findall(r.stderr)]
    files = sorted(thumbs.glob(f"{file_id}_kf*.jpg"))
    if not files:  # very short or odd video: grab one frame
        one = thumbs / f"{file_id}_kf0001.jpg"
        run_ffmpeg(["-i", path, "-frames:v", "1", "-vf", "scale=640:-2", str(one)])
        files, times = ([one], [0.0]) if one.exists() else ([], [])
    return list(zip(times + [0.0] * (len(files) - len(times)), files))


def load_video(path: str, file_id: str, thumbs: Path, p: Providers, progress: Progress,
               max_chars: int = 900, max_frames: int = 40) -> list[Piece]:
    progress(0.05, "Transcribing audio track")
    segs = _transcribe(path, file_id, thumbs, p, max_chars)
    pieces = window_segments(segs, max_chars)
    first_kf = None
    progress(0.4, "Detecting scene keyframes")
    frames = extract_keyframes(path, file_id, thumbs, max_frames)
    for i, (t, f) in enumerate(frames):
        progress(0.4 + 0.5 * i / max(len(frames), 1), f"Captioning keyframe {i + 1}/{len(frames)}")
        t_end = frames[i + 1][0] if i + 1 < len(frames) else (segs[-1].end if segs else t + 5)
        # align: attach what is being said while this scene is on screen
        spoken = " ".join(s.text for s in segs if s.end > t and s.start < t_end)[:600]
        caption = p.vision.caption(str(f))
        content = f"Video scene at {t:.0f}s: {caption}" + (f"\nSpoken during scene: {spoken}" if spoken else "")
        pieces.append(Piece("keyframe", content, t_start=t, t_end=max(t_end, t), thumb=f.name, image_path=str(f), splittable=False))
        first_kf = first_kf or f.name
    for pc in pieces:
        if pc.kind == "transcript" and first_kf:
            # nearest keyframe as the visual for transcript windows
            best = min(frames, key=lambda fr: abs(fr[0] - (pc.t_start or 0)))
            pc.thumb = best[1].name
    return pieces
