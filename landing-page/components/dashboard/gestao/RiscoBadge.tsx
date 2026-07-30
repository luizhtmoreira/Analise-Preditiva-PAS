import type { RiscoStatus } from "@/lib/types";

// Semáforo oficial — docs/identidade-visual.md §3 (tokens --risk-* em globals.css)
const CONFIG: Record<RiscoStatus, { bg: string; color: string; dot: string; label: string }> = {
  green:  { bg: "var(--risk-low-bg)",  color: "var(--risk-low-text)",  dot: "var(--vp-green)", label: "Baixo Risco" },
  yellow: { bg: "var(--risk-mid-bg)",  color: "var(--risk-mid-text)",  dot: "var(--risk-mid-text)", label: "Oportunidade" },
  red:    { bg: "var(--risk-high-bg)", color: "var(--risk-high-text)", dot: "#C62828", label: "Alto Risco" },
};

export function RiscoBadge({ status }: { status: RiscoStatus }) {
  const c = CONFIG[status];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold whitespace-nowrap"
      style={{ background: c.bg, color: c.color }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c.dot }} />
      {c.label}
    </span>
  );
}
