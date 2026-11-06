import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useLocation, useNavigate } from "react-router-dom";
import { Clock, ImagePlus, MessageSquarePlus, Mic, Search, Send, Square, Trash2, Volume2, VolumeX, X } from "lucide-react";
import { api, Citation, fmtTs, Message, Modality, Session, speak, stopSpeaking, streamChat, thumbUrl } from "../api";
import SourceViewer from "../components/SourceViewer";
import { MOD_META, ModBadge, ModIcon, Spinner } from "../components/ui";

const MODS: Modality[] = ["pdf", "image", "audio", "video"];
const EXAMPLES = ["Summarize the key points across all my files", "Show me images of diagrams or charts",
  "What was discussed in the recordings?", "Which video scenes mention the main topic?"];

export default function Chat() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sid, setSid] = useState<string>("");
  const [msgs, setMsgs] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [mods, setMods] = useState<Modality[]>([]);
  const [busy, setBusy] = useState(false);
  const [viewer, setViewer] = useState<Citation | null>(null);
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState<number | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const loc = useLocation();
  const nav = useNavigate();

  const refreshSessions = () => api.sessions().then(setSessions).catch(() => {});
  useEffect(() => { refreshSessions(); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  // the knowledge graph page links here with ?q=
  useEffect(() => {
    const q = new URLSearchParams(loc.search).get("q");
    if (q) { nav("/", { replace: true }); newChat(); setTimeout(() => send(q), 0); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.search]);

  const openSession = async (id: string) => {
    if (busy) return;
    setSid(id);
    setMsgs(await api.messages(id));
  };
  const newChat = () => { if (!busy) { setSid(""); setMsgs([]); } };

  async function send(text = input) {
    if (busy || (!text.trim() && !image)) return;
    const img = image;
    setInput(""); setImage(null); setBusy(true);
    const userMsg: Message = { role: "user", content: text.trim() || "(image search)", citations: [],
      meta: { image: img ? URL.createObjectURL(img) : undefined } };
    const draft: Message = { role: "assistant", content: "", citations: [], meta: { streaming: true, status: "Thinking…" } };
    setMsgs((m) => [...m, userMsg, draft]);
    const patch = (f: (d: Message) => Message) => setMsgs((m) => [...m.slice(0, -1), f(m[m.length - 1])]);
    abortRef.current = new AbortController();
    try {
      await streamChat({ message: text, session_id: sid, modalities: mods, image: img }, (ev) => {
        if (ev.type === "session") setSid(ev.session_id);
        else if (ev.type === "status") patch((d) => ({ ...d, meta: { ...d.meta, status: ev.text } }));
        else if (ev.type === "sources") patch((d) => ({ ...d, citations: ev.citations }));
        else if (ev.type === "token") patch((d) => ({ ...d, content: d.content + ev.text, meta: { ...d.meta, status: undefined } }));
        else if (ev.type === "done") patch((d) => ({ ...d, content: ev.answer, citations: ev.citations,
          meta: { used: ev.used, grounded: ev.grounded, timings: ev.timings } }));
      }, abortRef.current.signal);
    } catch (e: any) {
      if (e.name !== "AbortError") patch((d) => ({ ...d, content: `⚠️ ${e.message || "Request failed"}`, meta: {} }));
      else patch((d) => ({ ...d, meta: {} }));
    } finally {
      setBusy(false);
      refreshSessions();
    }
  }

  async function toggleMic() {
    if (recording) { recRef.current?.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const parts: Blob[] = [];
      rec.ondataavailable = (e) => parts.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        const { text } = await api.transcribe(new Blob(parts, { type: "audio/webm" }));
        if (text) send(text);
      };
      recRef.current = rec;
      rec.start();
      setRecording(true);
    } catch { alert("Microphone not available"); }
  }

  const toggleSpeak = (i: number, text: string) => {
    if (speaking === i) { stopSpeaking(); setSpeaking(null); return; }
    setSpeaking(i); speak(text).finally(() => {});
  };

  return (
    <div className="flex h-full">
      {/* sessions */}
      <aside className="hidden w-60 shrink-0 flex-col border-r border-line bg-panel/60 lg:flex">
        <button onClick={newChat} className="m-3 flex items-center justify-center gap-2 rounded-lg border border-line bg-white px-3 py-2 text-sm font-medium shadow-card hover:border-accent/40">
          <MessageSquarePlus size={15} /> New chat
        </button>
        <div className="flex-1 overflow-auto px-2 pb-3">
          {sessions.map((s) => (
            <div key={s.id} onClick={() => openSession(s.id)}
              className={`group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-sm ${s.id === sid ? "bg-white shadow-card" : "hover:bg-white/70"}`}>
              <span className="flex-1 truncate">{s.title}</span>
              <button className="hidden text-ink-faint hover:text-red-600 group-hover:block" aria-label="Delete chat"
                onClick={async (e) => { e.stopPropagation(); await api.delSession(s.id); if (s.id === sid) newChat(); refreshSessions(); }}>
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-auto">
          <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
            {msgs.length === 0 ? (
              <div className="pt-12 text-center">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-soft text-accent"><Search size={22} /></div>
                <h1 className="text-2xl font-semibold tracking-tight">Ask anything across your files</h1>
                <p className="mt-2 text-ink-soft">PDFs, images, audio and video, searched together and answered with sources. Runs 100% offline.</p>
                <div className="mt-8 grid gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((e) => (
                    <button key={e} onClick={() => send(e)} className="rounded-xl border border-line px-4 py-3 text-left text-sm hover:border-accent/40 hover:bg-accent-soft/40">{e}</button>
                  ))}
                </div>
              </div>
            ) : msgs.map((m, i) => m.role === "user" ? (
              <div key={i} className="mb-6 flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-white">
                  {m.meta?.image && <img src={m.meta.image.startsWith("blob:") ? m.meta.image : thumbUrl(m.meta.image)} className="mb-2 max-h-40 rounded-lg" />}
                  {m.content}
                </div>
              </div>
            ) : (
              <AssistantMsg key={i} m={m} onCite={setViewer} speaking={speaking === i} onSpeak={() => toggleSpeak(i, m.content)} />
            ))}
            <div ref={endRef} />
          </div>
        </div>

        {/* composer */}
        <div className="border-t border-line bg-white px-4 pb-4 pt-3 sm:px-6">
          <div className="mx-auto max-w-3xl">
            <div className="mb-2 flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-xs text-ink-faint">Search in:</span>
              {MODS.map((m) => {
                const on = mods.includes(m);
                return (
                  <button key={m} onClick={() => setMods(on ? mods.filter((x) => x !== m) : [...mods, m])}
                    className={`flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs ${on ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-soft hover:border-ink-faint"}`}>
                    <ModIcon m={m} size={12} /> {MOD_META[m].label}
                  </button>
                );
              })}
              {mods.length === 0 && <span className="text-xs text-ink-faint">all</span>}
            </div>
            <div className="rounded-2xl border border-line bg-white shadow-card focus-within:border-accent/50">
              {image && (
                <div className="flex items-center gap-2 border-b border-line px-3 py-2">
                  <img src={URL.createObjectURL(image)} className="h-10 w-10 rounded object-cover" />
                  <span className="flex-1 truncate text-xs text-ink-soft">Find content related to <b>{image.name}</b></span>
                  <button onClick={() => setImage(null)} aria-label="Remove image"><X size={14} /></button>
                </div>
              )}
              <div className="flex items-end gap-1 p-2">
                <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => { setImage(e.target.files?.[0] ?? null); e.target.value = ""; }} />
                <button onClick={() => fileRef.current?.click()} className="icon-btn" title="Search by image"><ImagePlus size={18} /></button>
                <button onClick={toggleMic} className={`icon-btn ${recording ? "animate-pulse bg-red-50 text-red-600" : ""}`} title="Ask by voice">
                  {recording ? <Square size={16} /> : <Mic size={18} />}
                </button>
                <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={1} placeholder={recording ? "Listening…" : "Ask a question, or attach an image…"}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                  className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-2 py-2 text-[15px] outline-none" />
                {busy ? (
                  <button onClick={() => abortRef.current?.abort()} className="icon-btn bg-ink text-white hover:bg-ink" title="Stop"><Square size={14} /></button>
                ) : (
                  <button onClick={() => send()} disabled={!input.trim() && !image} className="icon-btn bg-accent text-white hover:bg-accent disabled:opacity-40" title="Send"><Send size={16} /></button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      {viewer && <SourceViewer c={viewer} onClose={() => setViewer(null)} />}
    </div>
  );
}

function AssistantMsg({ m, onCite, speaking, onSpeak }: { m: Message; onCite: (c: Citation) => void; speaking: boolean; onSpeak: () => void }) {
  const [showTimings, setShowTimings] = useState(false);
  const used = new Set(m.meta?.used ?? []);
  const byN = Object.fromEntries(m.citations.map((c) => [c.n, c]));
  const md = m.content.replace(/\[(\d+)\]/g, (s, n) => (byN[n] ? `[${n}](#cite-${n})` : s));
  const shown = m.meta?.grounded === false ? m.citations.slice(0, 4) : m.citations.filter((c) => used.size === 0 || used.has(c.n));
  const refused = m.meta?.grounded === false;

  return (
    <div className="mb-8">
      {m.meta?.status && <div className="mb-2 flex items-center gap-2 text-sm text-ink-soft"><Spinner /> {m.meta.status}</div>}
      <div className={`prose-answer text-[15px] leading-relaxed ${refused ? "text-ink-soft" : ""}`}>
        <ReactMarkdown components={{
          a: ({ href, children }) => {
            const n = href?.startsWith("#cite-") ? Number(href.slice(6)) : null;
            if (n && byN[n]) return <button onClick={() => onCite(byN[n])} className="cite" title={`${byN[n].file_name} · ${byN[n].locator}`}>{n}</button>;
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
        }}>{md}</ReactMarkdown>
        {m.meta?.streaming && !m.meta.status && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-accent align-middle" />}
      </div>

      {shown.length > 0 && !m.meta?.streaming && (
        <div className="mt-4">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-faint">{refused ? "Closest matches" : "Sources"}</div>
          <div className="grid gap-2 sm:grid-cols-2">
            {shown.map((c) => (
              <button key={c.n} onClick={() => onCite(c)} className="flex gap-3 rounded-xl border border-line p-2 text-left transition hover:border-accent/40 hover:shadow-card">
                {c.thumb ? <img src={thumbUrl(c.thumb)} className="h-14 w-14 shrink-0 rounded-lg border border-line object-cover" loading="lazy" />
                  : <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-panel"><ModIcon m={c.modality} size={20} /></div>}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="cite !ml-0">{c.n}</span>
                    <span className="truncate text-sm font-medium">{c.file_name}</span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-xs text-ink-soft">
                    <ModBadge m={c.modality} />
                    {c.page ? `Page ${c.page}` : c.t_start != null ? <span className="inline-flex items-center gap-0.5"><Clock size={11} />{fmtTs(c.t_start)}</span> : c.kind}
                  </div>
                  <div className="mt-1 line-clamp-1 text-xs text-ink-faint">{c.snippet}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {!m.meta?.streaming && m.content && (
        <div className="mt-3 flex items-center gap-3 text-xs text-ink-faint">
          {!refused && <button onClick={onSpeak} className="inline-flex items-center gap-1 hover:text-ink">{speaking ? <VolumeX size={13} /> : <Volume2 size={13} />} {speaking ? "Stop" : "Listen"}</button>}
          {m.meta?.timings && <button onClick={() => setShowTimings(!showTimings)} className="hover:text-ink">{(m.meta.timings.total_ms / 1000).toFixed(1)}s · details</button>}
        </div>
      )}
      {showTimings && m.meta?.timings && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {Object.entries(m.meta.timings).filter(([, v]) => typeof v === "number").map(([k, v]) => (
            <span key={k} className="rounded bg-panel px-2 py-0.5 font-mono text-[11px] text-ink-soft">{k.replace(/_ms$/, "")}: {v}{k.endsWith("_ms") ? "ms" : ""}</span>
          ))}
        </div>
      )}
    </div>
  );
}
