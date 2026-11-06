export type Modality = "pdf" | "image" | "audio" | "video";

export interface FileRow {
  id: string; name: string; modality: Modality; size: number; status: "queued" | "processing" | "indexed" | "failed";
  progress: number; stage: string; error: string; n_chunks: number; thumb: string; created: number;
}

export interface Citation {
  n: number; chunk_id: string; file_id: string; file_name: string; modality: Modality; kind: string;
  page: number | null; t_start: number | null; t_end: number | null; thumb: string; locator: string; snippet: string; score: number;
}

export interface Message {
  id?: string; role: "user" | "assistant"; content: string; citations: Citation[];
  meta: { used?: number[]; grounded?: boolean; timings?: Record<string, number>; image?: string; status?: string; streaming?: boolean };
}

export interface Session { id: string; title: string; created: number }

export const thumbUrl = (name: string) => (name ? `/api/thumbs/${encodeURIComponent(name)}` : "");
export const rawUrl = (fid: string) => `/api/files/${fid}/raw`;

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

export const api = {
  health: () => fetch("/api/health").then(j<any>),
  files: () => fetch("/api/files").then(j<FileRow[]>),
  upload: (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return fetch("/api/files", { method: "POST", body: fd }).then(j<any[]>);
  },
  del: (id: string) => fetch(`/api/files/${id}`, { method: "DELETE" }).then(j),
  retry: (id: string) => fetch(`/api/files/${id}/retry`, { method: "POST" }).then(j),
  chunks: (id: string) => fetch(`/api/files/${id}/chunks`).then(j<any[]>),
  sessions: () => fetch("/api/sessions").then(j<Session[]>),
  delSession: (id: string) => fetch(`/api/sessions/${id}`, { method: "DELETE" }).then(j),
  messages: (id: string) => fetch(`/api/sessions/${id}/messages`).then(j<Message[]>),
  graph: () => fetch("/api/graph").then(j<{ nodes: any[]; links: any[] }>),
  graphNode: (id: string) => fetch(`/api/graph/node/${encodeURIComponent(id)}`).then(j<any>),
  golden: () => fetch("/api/eval/golden").then(j<any[]>),
  putGolden: (rows: any[]) => fetch("/api/eval/golden", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(rows) }).then(j),
  runEval: () => fetch("/api/eval/run", { method: "POST" }).then(j<any>),
  evalRuns: () => fetch("/api/eval/runs").then(j<any[]>),
  transcribe: (blob: Blob) => {
    const fd = new FormData();
    fd.append("audio", blob, "voice.webm");
    return fetch("/api/transcribe", { method: "POST", body: fd }).then(j<{ text: string }>);
  },
};

/** POST /api/chat and consume its SSE stream. */
export async function streamChat(
  form: { message: string; session_id?: string; modalities?: string[]; image?: File | null },
  onEvent: (ev: any) => void,
  signal?: AbortSignal,
) {
  const fd = new FormData();
  fd.append("message", form.message);
  fd.append("session_id", form.session_id ?? "");
  fd.append("modalities", (form.modalities ?? []).join(","));
  if (form.image) fd.append("image", form.image);
  const r = await fetch("/api/chat", { method: "POST", body: fd, signal });
  if (!r.ok || !r.body) throw new Error(await r.text());
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 2);
      if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

let ttsAudio: HTMLAudioElement | null = null;
export async function speak(text: string) {
  stopSpeaking();
  const clean = text.replace(/\[\d+\]/g, "");
  try {
    const r = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: clean }) });
    if (r.status === 200) {
      ttsAudio = new Audio(URL.createObjectURL(await r.blob()));
      await ttsAudio.play();
      return;
    }
  } catch { /* fall through to browser TTS */ }
  if ("speechSynthesis" in window) window.speechSynthesis.speak(new SpeechSynthesisUtterance(clean));
}
export function stopSpeaking() {
  ttsAudio?.pause();
  ttsAudio = null;
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}

export const fmtTs = (s: number | null) => {
  if (s == null) return "";
  const t = Math.floor(s), h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = t % 60;
  return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${String(sec).padStart(2, "0")}`;
};
export const fmtSize = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.ceil(b / 1e3)} KB`);
