import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useNavigate } from "react-router-dom";
import { MessageSquare, Network, Search, X } from "lucide-react";
import { api, fmtTs } from "../api";
import { Empty, Spinner } from "../components/ui";

const PALETTE = ["#4f46e5", "#0891b2", "#d97706", "#7c3aed", "#dc2626", "#059669", "#db2777", "#475569"];

export default function Graph() {
  const [data, setData] = useState<{ nodes: any[]; links: any[] } | null>(null);
  const [sel, setSel] = useState<any>(null);
  const [q, setQ] = useState("");
  const [size, setSize] = useState({ w: 800, h: 600 });
  const box = useRef<HTMLDivElement>(null);
  const fg = useRef<any>(null);
  const nav = useNavigate();

  useEffect(() => { api.graph().then(setData).catch(() => setData({ nodes: [], links: [] })); }, []);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    box.current && ro.observe(box.current);
    return () => ro.disconnect();
  }, []);

  const labels = useMemo(() => [...new Set((data?.nodes ?? []).map((n) => n.label))], [data]);
  const color = (l: string) => PALETTE[labels.indexOf(l) % PALETTE.length];
  const maxCount = Math.max(1, ...(data?.nodes ?? []).map((n) => n.count));
  const match = q.trim().toLowerCase();

  const select = async (n: any) => {
    setSel({ ...n, loading: true });
    setSel(await api.graphNode(n.id));
    fg.current?.centerAt(n.x, n.y, 600);
    fg.current?.zoom(2.5, 600);
  };

  return (
    <div className="flex h-full">
      <div ref={box} className="relative min-w-0 flex-1 bg-panel/40">
        <div className="absolute left-4 top-4 z-10 flex items-center gap-2 rounded-xl border border-line bg-white px-3 py-2 shadow-card">
          <Search size={15} className="text-ink-faint" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find entity…" className="w-48 bg-transparent text-sm outline-none" />
        </div>
        <div className="absolute bottom-4 left-4 z-10 flex flex-wrap gap-2 rounded-xl border border-line bg-white px-3 py-2 text-xs shadow-card">
          {labels.map((l) => <span key={l} className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full" style={{ background: color(l) }} />{l}</span>)}
        </div>
        {!data ? <div className="flex h-full items-center justify-center"><Spinner /></div>
          : data.nodes.length === 0 ? <Empty icon={<Network size={22} />} title="Knowledge graph is empty">Index some files: people, organisations, places and concepts are linked automatically across every modality.</Empty>
          : (
            <ForceGraph2D ref={fg} graphData={data} width={size.w} height={size.h} cooldownTicks={120}
              linkColor={() => "rgba(87,96,106,0.18)"} linkWidth={(l: any) => Math.min(4, 0.5 + l.weight * 0.5)}
              onNodeClick={select}
              nodeCanvasObject={(n: any, ctx, scale) => {
                const r = 3 + 9 * Math.sqrt(n.count / maxCount);
                const dim = match && !n.name.toLowerCase().includes(match);
                ctx.globalAlpha = dim ? 0.15 : 1;
                ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 2 * Math.PI); ctx.fillStyle = color(n.label); ctx.fill();
                if (sel?.id === n.id) { ctx.lineWidth = 2 / scale; ctx.strokeStyle = "#1f2328"; ctx.stroke(); }
                if (scale > 1.2 || n.count / maxCount > 0.35 || (match && !dim)) {
                  ctx.font = `${11 / scale}px Inter, system-ui`; ctx.fillStyle = "#1f2328"; ctx.textAlign = "center";
                  ctx.fillText(n.name, n.x, n.y + r + 10 / scale);
                }
                ctx.globalAlpha = 1;
              }}
              nodePointerAreaPaint={(n: any, c, ctx) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(n.x, n.y, 10, 0, 2 * Math.PI); ctx.fill(); }}
            />
          )}
      </div>
      {sel && (
        <aside className="w-96 shrink-0 overflow-auto border-l border-line bg-white">
          <div className="flex items-start gap-3 border-b border-line p-5">
            <div className="min-w-0 flex-1">
              <div className="text-lg font-semibold">{sel.name}</div>
              <div className="text-xs text-ink-soft">{sel.label} · {sel.count} mentions · {sel.files?.length ?? 0} files</div>
            </div>
            <button onClick={() => setSel(null)} className="icon-btn"><X size={16} /></button>
          </div>
          {sel.loading ? <div className="p-5"><Spinner /></div> : (
            <div className="space-y-5 p-5">
              <button onClick={() => nav(`/?q=${encodeURIComponent(`Tell me everything about ${sel.name}`)}`)}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white hover:opacity-90">
                <MessageSquare size={15} /> Ask about {sel.name}
              </button>
              <div>
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-faint">Connected to</div>
                <div className="flex flex-wrap gap-1.5">
                  {sel.neighbors?.map((n: any) => (
                    <button key={n.id} onClick={() => select(data!.nodes.find((x) => x.id === n.id) ?? n)} className="rounded-full border border-line px-2.5 py-0.5 text-xs hover:border-accent/40">{n.name} <span className="text-ink-faint">{n.weight}</span></button>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-faint">Mentions</div>
                <div className="space-y-2">
                  {sel.mentions?.map((m: any) => (
                    <div key={m.chunk_id} className="rounded-xl border border-line p-3 text-sm">
                      <div className="mb-1 truncate text-xs font-medium">{m.file_name} <span className="font-normal text-ink-faint">· {m.page ? `p. ${m.page}` : m.t_start != null ? fmtTs(m.t_start) : m.kind}</span></div>
                      <div className="line-clamp-3 text-ink-soft">{m.snippet}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </aside>
      )}
    </div>
  );
}
