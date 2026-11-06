import { useEffect, useRef } from "react";
import { ExternalLink, X } from "lucide-react";
import { Citation, fmtTs, rawUrl, thumbUrl } from "../api";
import { ModBadge } from "./ui";

/** Opens a citation at its exact location: PDF page, image, or media seeked to the timestamp. */
export default function SourceViewer({ c, onClose }: { c: Citation; onClose: () => void }) {
  const media = useRef<HTMLMediaElement>(null);

  useEffect(() => {
    const el = media.current;
    if (!el || c.t_start == null) return;
    const seek = () => { el.currentTime = c.t_start!; el.play().catch(() => {}); };
    if (el.readyState >= 1) seek(); else el.addEventListener("loadedmetadata", seek, { once: true });
  }, [c]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const openHref = c.modality === "pdf" && c.page ? `${rawUrl(c.file_id)}#page=${c.page}` : rawUrl(c.file_id);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4 backdrop-blur-[2px]" onClick={onClose}>
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-line px-5 py-3">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-accent text-xs font-semibold text-white">{c.n}</span>
          <div className="min-w-0 flex-1">
            <div className="truncate font-medium">{c.file_name}</div>
            <div className="flex items-center gap-2 text-xs text-ink-soft"><ModBadge m={c.modality} /> {c.kind} · {c.locator}</div>
          </div>
          <a href={openHref} target="_blank" rel="noreferrer" className="btn-ghost"><ExternalLink size={15} /> Open original</a>
          <button onClick={onClose} className="rounded-lg p-1.5 hover:bg-panel" aria-label="Close"><X size={18} /></button>
        </div>
        <div className="overflow-auto p-5">
          {c.modality === "video" ? (
            <video ref={media as any} src={rawUrl(c.file_id)} controls className="w-full rounded-lg bg-black" />
          ) : c.modality === "audio" ? (
            <audio ref={media as any} src={rawUrl(c.file_id)} controls className="w-full" />
          ) : c.thumb ? (
            <img src={thumbUrl(c.thumb)} className="mx-auto max-h-[55vh] rounded-lg border border-line" />
          ) : null}
          {c.t_start != null && (
            <div className="mt-3 text-sm text-ink-soft">Playing from <b className="text-ink">{fmtTs(c.t_start)}</b>{c.t_end != null && <> to {fmtTs(c.t_end)}</>}</div>
          )}
          <div className="mt-4 rounded-xl bg-panel p-4 text-sm leading-relaxed whitespace-pre-wrap">{c.snippet}</div>
          <div className="mt-2 text-xs text-ink-faint">Relevance score {c.score.toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}
