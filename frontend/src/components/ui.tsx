import { FileText, Image as ImageIcon, Mic, Video } from "lucide-react";
import type { Modality } from "../api";

export const MOD_META: Record<Modality, { label: string; color: string; bg: string; Icon: any }> = {
  pdf: { label: "PDF", color: "text-pdf", bg: "bg-red-50", Icon: FileText },
  image: { label: "Image", color: "text-image", bg: "bg-cyan-50", Icon: ImageIcon },
  audio: { label: "Audio", color: "text-audio", bg: "bg-amber-50", Icon: Mic },
  video: { label: "Video", color: "text-video", bg: "bg-violet-50", Icon: Video },
};

export function ModIcon({ m, size = 14 }: { m: Modality; size?: number }) {
  const { Icon, color } = MOD_META[m] ?? MOD_META.pdf;
  return <Icon size={size} className={color} />;
}

export function ModBadge({ m }: { m: Modality }) {
  const meta = MOD_META[m] ?? MOD_META.pdf;
  return (
    <span className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium ${meta.bg} ${meta.color}`}>
      <meta.Icon size={11} /> {meta.label}
    </span>
  );
}

export function Empty({ icon, title, children }: { icon: any; title: string; children?: any }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="mb-3 rounded-2xl bg-panel p-4 text-ink-faint">{icon}</div>
      <div className="font-medium">{title}</div>
      <div className="mt-1 max-w-sm text-sm text-ink-soft">{children}</div>
    </div>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return <span className={`inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent border-t-transparent ${className}`} />;
}
