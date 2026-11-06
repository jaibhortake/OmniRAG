import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { BarChart3, Library as LibIcon, MessageSquare, Network, ShieldCheck } from "lucide-react";
import { api } from "./api";
import Chat from "./pages/Chat";
import Library from "./pages/Library";
import Graph from "./pages/Graph";
import Eval from "./pages/Eval";

const NAV = [
  { to: "/", label: "Chat", Icon: MessageSquare },
  { to: "/library", label: "Library", Icon: LibIcon },
  { to: "/graph", label: "Graph", Icon: Network },
  { to: "/eval", label: "Eval", Icon: BarChart3 },
];

export default function App() {
  const [health, setHealth] = useState<any>(null);
  useEffect(() => {
    const tick = () => api.health().then(setHealth).catch(() => setHealth(null));
    tick();
    const t = setInterval(tick, 15000);
    return () => clearInterval(t);
  }, []);

  const ok = health && (health.providers === "fake" || health.ollama);
  return (
    <div className="flex h-screen flex-col">
      <header className="flex h-14 shrink-0 items-center gap-6 border-b border-line px-4 sm:px-6">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent"><div className="h-3 w-3 rounded-full border-2 border-white" /></div>
          <span className="font-semibold tracking-tight">OmniRAG</span>
        </div>
        <nav className="flex gap-1">
          {NAV.map(({ to, label, Icon }) => (
            <NavLink key={to} to={to} end className={({ isActive }) => `flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm ${isActive ? "bg-panel font-medium text-ink" : "text-ink-soft hover:text-ink"}`}>
              <Icon size={15} /> <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2 text-xs text-ink-soft" title={health ? `LLM ${health.llm} · vision ${health.vision}` : "backend offline"}>
          <ShieldCheck size={14} className="text-emerald-600" />
          <span className="hidden md:inline">Offline · on-device</span>
          <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : health ? "bg-amber-500" : "bg-red-500"}`} />
          <span className="hidden md:inline">{!health ? "Backend offline" : health.providers === "fake" ? "Demo models" : health.ollama ? health.llm : "Ollama not running"}</span>
        </div>
      </header>
      <main className="min-h-0 flex-1">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/library" element={<Library />} />
          <Route path="/graph" element={<Graph />} />
          <Route path="/eval" element={<Eval />} />
        </Routes>
      </main>
    </div>
  );
}
