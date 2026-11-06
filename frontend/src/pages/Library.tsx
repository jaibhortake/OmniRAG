import { useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, Layers, RotateCw, Trash2, UploadCloud, X } from "lucide-react";
import { api, fmtSize, fmtTs, FileRow, Modality, thumbUrl } from "../api";
import { Empty, MOD_META, ModBadge, ModIcon, Spinner } from "../components/ui";

const ACCEPT = ".pdf,.png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff,.mp3,.wav,.m4a,.ogg,.flac,.aac,.webm,.mp4,.mov,.mkv,.avi,.m4v";

export default function Library() {
  const [files, setFiles] = useState<FileRow[]>([]);
  const [drag, setDrag] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [open, setOpen] = useState<FileRow | null>(null);
  const [filter, setFilter] = useState<Modality | "all">("all");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.files().then(setFiles).catch(() => {});
    const es = new EventSource("/api/files/stream");
    es.onmessage = (e) => setFiles(JSON.parse(e.data));
    return () => es.close();
  }, []);

  async function upload(list: FileList | File[]) {
    const arr = Array.from(list);
    if (!arr.length) return;
    setUploading(true);
    try {
      const res = await api.upload(arr);
      setErrors(res.filter((r) => r.error).map((r) => `${r.name}: ${r.error}`));
    } catch (e: any) { setErrors([e.message]); }
    finally { setUploading(false); }
  }

  const counts = files.reduce((a, f) => ({ ...a, [f.modality]: (a[f.modality] ?? 0) + 1 }), {} as Record<string, number>);
  const shown = filter === "all" ? files : files.filter((f) => f.modality === filter);
  const indexed = files.filter((f) => f.status === "indexed");

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Library</h1>
            <p className="mt-1 text-sm text-ink-soft">{indexed.length} files indexed · {indexed.reduce((a, f) => a + f.n_chunks, 0)} searchable chunks</p>
          </div>
          <div className="flex gap-2">
            {(["pdf", "image", "audio", "video"] as Modality[]).map((m) => (
              <div key={m} className="flex items-center gap-2 rounded-xl border border-line px-3 py-2 text-sm">
                <ModIcon m={m} size={15} /> <b>{counts[m] ?? 0}</b> <span className="hidden text-ink-soft sm:inline">{MOD_META[m].label}</span>
              </div>
            ))}
          </div>
        </div>

        <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files); }}
          onClick={() => input.current?.click()}
          className={`mt-6 cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition ${drag ? "border-accent bg-accent-soft" : "border-line hover:border-accent/40 hover:bg-panel"}`}>
          <input ref={input} type="file" multiple hidden accept={ACCEPT} onChange={(e) => { e.target.files && upload(e.target.files); e.target.value = ""; }} />
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-soft text-accent">{uploading ? <Spinner /> : <UploadCloud size={22} />}</div>
          <div className="font-medium">Drop files here or click to upload</div>
          <div className="mt-1 text-sm text-ink-soft">PDF (incl. scanned) · PNG / JPG · MP3 / WAV / M4A · MP4 / MOV / MKV</div>
        </div>
        {errors.map((e) => <div key={e} className="mt-2 flex items-center gap-2 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700"><AlertCircle size={14} /> {e}</div>)}

        <div className="mt-8 mb-3 flex gap-1">
          {(["all", "pdf", "image", "audio", "video"] as const).map((m) => (
            <button key={m} onClick={() => setFilter(m)} className={`rounded-lg px-3 py-1 text-sm ${filter === m ? "bg-ink text-white" : "text-ink-soft hover:bg-panel"}`}>
              {m === "all" ? "All" : MOD_META[m].label}
            </button>
          ))}
        </div>

        {shown.length === 0 ? <Empty icon={<Layers size={22} />} title="No files yet">Upload documents, photos, recordings or videos to build your knowledge base.</Empty> : (
          <div className="overflow-hidden rounded-2xl border border-line">
            {shown.map((f) => (
              <div key={f.id} className="flex items-center gap-4 border-b border-line px-4 py-3 last:border-0 hover:bg-panel/60">
                {f.thumb ? <img src={thumbUrl(f.thumb)} className="h-11 w-11 rounded-lg border border-line object-cover" loading="lazy" />
                  : <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-panel"><ModIcon m={f.modality} size={18} /></div>}
                <button className="min-w-0 flex-1 text-left" onClick={() => f.status === "indexed" && setOpen(f)}>
                  <div className="truncate font-medium">{f.name}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-ink-soft"><ModBadge m={f.modality} /> {fmtSize(f.size)}{f.status === "indexed" && <> · {f.n_chunks} chunks</>}</div>
                </button>
                <div className="w-56 text-right text-xs">
                  {f.status === "indexed" && <span className="inline-flex items-center gap-1 text-emerald-600"><CheckCircle2 size={14} /> {f.stage}</span>}
                  {f.status === "failed" && <span className="inline-flex items-center gap-1 text-red-600" title={f.error}><AlertCircle size={14} /> <span className="truncate max-w-[180px]">{f.error || "Failed"}</span></span>}
                  {(f.status === "processing" || f.status === "queued") && (
                    <div>
                      <div className="mb-1 truncate text-ink-soft">{f.status === "queued" ? "Queued" : f.stage}</div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-accent transition-all" style={{ width: `${Math.max(3, f.progress * 100)}%` }} /></div>
                    </div>
                  )}
                </div>
                {f.status === "failed" && <button onClick={() => api.retry(f.id)} className="icon-btn" title="Retry"><RotateCw size={15} /></button>}
                <button onClick={() => confirm(`Delete ${f.name}?`) && api.del(f.id)} className="icon-btn hover:text-red-600" title="Delete"><Trash2 size={15} /></button>
              </div>
            ))}
          </div>
        )}
      </div>
      {open && <ChunksDrawer f={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function ChunksDrawer({ f, onClose }: { f: FileRow; onClose: () => void }) {
  const [chunks, setChunks] = useState<any[] | null>(null);
  useEffect(() => { api.chunks(f.id).then(setChunks); }, [f.id]);
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/20" onClick={onClose}>
      <div className="h-full w-full max-w-xl overflow-auto bg-white shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 flex items-center gap-3 border-b border-line bg-white px-5 py-4">
          <div className="min-w-0 flex-1"><div className="truncate font-semibold">{f.name}</div><div className="text-xs text-ink-soft">What OmniRAG extracted and indexed</div></div>
          <button onClick={onClose} className="icon-btn"><X size={18} /></button>
        </div>
        <div className="space-y-3 p-5">
          {!chunks ? <Spinner /> : chunks.map((c) => (
            <div key={c.id} className="rounded-xl border border-line p-3">
              <div className="mb-2 flex items-center gap-2 text-xs">
                <span className="rounded bg-panel px-1.5 py-0.5 font-medium">{c.kind}</span>
                <span className="text-ink-soft">{c.page ? `Page ${c.page}` : c.t_start != null ? `${fmtTs(c.t_start)} – ${fmtTs(c.t_end)}` : ""}</span>
              </div>
              <div className="flex gap-3">
                {c.thumb && ["figure", "image", "keyframe"].includes(c.kind) && <img src={thumbUrl(c.thumb)} className="h-16 w-16 rounded-lg object-cover" />}
                <p className="whitespace-pre-wrap text-sm text-ink-soft">{c.content}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
