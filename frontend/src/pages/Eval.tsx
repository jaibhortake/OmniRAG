import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3, CheckCircle2, Play, Save, XCircle } from "lucide-react";
import { api } from "../api";
import { Empty, Spinner } from "../components/ui";

const METRICS = [
  { key: "recall_at_k", label: "Recall@5", hint: "expected file retrieved in top 5", color: "#4f46e5" },
  { key: "mrr", label: "MRR", hint: "how high the right source ranks", color: "#0891b2" },
  { key: "citation_rate", label: "Citation rate", hint: "answer sentences with a valid citation", color: "#d97706" },
  { key: "refusal_acc", label: "Refusal accuracy", hint: "unanswerable questions correctly refused", color: "#7c3aed" },
];

export default function Eval() {
  const [runs, setRuns] = useState<any[]>([]);
  const [golden, setGolden] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [saved, setSaved] = useState(false);

  const load = () => api.evalRuns().then(setRuns).catch(() => {});
  useEffect(() => {
    load();
    api.golden().then((rows) => setGolden(rows.map((r) => JSON.stringify(r)).join("\n"))).catch(() => {});
  }, []);

  const parsed = () => golden.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));

  async function save() {
    setErr("");
    try { await api.putGolden(parsed()); setSaved(true); setTimeout(() => setSaved(false), 1500); return true; }
    catch (e: any) { setErr(`Invalid golden set: ${e.message}`); return false; }
  }
  async function run() {
    if (!(await save())) return;
    setBusy(true);
    try { await api.runEval(); await load(); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  const latest = runs[0];
  const history = [...runs].reverse().map((r, i) => ({ name: `#${i + 1}`, ...Object.fromEntries(METRICS.map((m) => [m.key, r.summary[m.key] ?? 0])) }));

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Evaluation</h1>
            <p className="mt-1 text-sm text-ink-soft">Measure retrieval quality and grounding against a golden question set.</p>
          </div>
          <button onClick={run} disabled={busy} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50">
            {busy ? <Spinner className="border-white border-t-transparent" /> : <Play size={15} />} {busy ? "Running…" : "Run evaluation"}
          </button>
        </div>
        {err && <div className="mt-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{err}</div>}

        {latest ? (
          <>
            <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              {METRICS.map((m) => (
                <div key={m.key} className="rounded-2xl border border-line p-4">
                  <div className="text-xs text-ink-soft">{m.label}</div>
                  <div className="mt-1 text-3xl font-semibold tabular-nums">{latest.summary[m.key] == null ? "—" : `${Math.round(latest.summary[m.key] * 100)}%`}</div>
                  <div className="mt-1 text-[11px] text-ink-faint">{m.hint}</div>
                </div>
              ))}
              <div className="rounded-2xl border border-line p-4">
                <div className="text-xs text-ink-soft">Avg latency</div>
                <div className="mt-1 text-3xl font-semibold tabular-nums">{(latest.summary.avg_latency_ms / 1000).toFixed(2)}s</div>
                <div className="mt-1 text-[11px] text-ink-faint">{latest.summary.n} questions · {latest.summary.providers} models</div>
              </div>
            </div>

            {history.length > 1 && (
              <div className="mt-6 rounded-2xl border border-line p-4">
                <div className="mb-3 text-sm font-medium">Run history</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e6e8eb" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v: number) => `${Math.round(v * 100)}%`} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    {METRICS.map((m) => <Bar key={m.key} dataKey={m.key} name={m.label} fill={m.color} radius={[3, 3, 0, 0]} />)}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="mt-6 overflow-hidden rounded-2xl border border-line">
              <table className="w-full text-sm">
                <thead className="bg-panel text-left text-xs text-ink-soft">
                  <tr><th className="px-4 py-2">Question</th><th className="px-4 py-2">Expected</th><th className="px-4 py-2">Result</th><th className="px-4 py-2">Rank</th><th className="px-4 py-2 text-right">Latency</th></tr>
                </thead>
                <tbody>
                  {latest.details.map((d: any, i: number) => {
                    const ok = d.expected_file ? d.hit : d.refused;
                    return (
                      <tr key={i} className="border-t border-line align-top">
                        <td className="px-4 py-2.5"><div className="font-medium">{d.question}</div><div className="mt-0.5 line-clamp-2 text-xs text-ink-faint">{d.answer}</div></td>
                        <td className="px-4 py-2.5 text-ink-soft">{d.expected_file ?? <i>unanswerable</i>}</td>
                        <td className="px-4 py-2.5">{ok ? <CheckCircle2 size={16} className="text-emerald-600" /> : <XCircle size={16} className="text-red-500" />}</td>
                        <td className="px-4 py-2.5 tabular-nums">{d.rank ?? "—"}</td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-ink-soft">{(d.latency_ms / 1000).toFixed(2)}s</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : <Empty icon={<BarChart3 size={22} />} title="No evaluation runs yet">Edit the golden set below to match your files, then run an evaluation.</Empty>}

        <div className="mt-8 rounded-2xl border border-line">
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <div>
              <div className="text-sm font-medium">Golden set</div>
              <div className="text-xs text-ink-faint">One JSON object per line: question, expected_file (part of the file name, or null if unanswerable), expected_terms</div>
            </div>
            <button onClick={save} className="btn-ghost"><Save size={14} /> {saved ? "Saved" : "Save"}</button>
          </div>
          <textarea value={golden} onChange={(e) => setGolden(e.target.value)} spellCheck={false} rows={10}
            className="w-full resize-y bg-white p-4 font-mono text-xs outline-none" />
        </div>
      </div>
    </div>
  );
}
