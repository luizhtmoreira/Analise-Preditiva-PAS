import type { RiscoStatus } from "@/lib/types";

// Semáforo oficial — docs/identidade-visual.md §3
const CONFIG: Record<RiscoStatus, { bg: string; color: string; dot: string; label: string }> = {
  green:  { bg: "#C8E6C9", color: "#1B5E20", dot: "#00843D", label: "Baixo Risco" },
  yellow: { bg: "#FFF9C4", color: "#F57F17", dot: "#F57F17", label: "Oportunidade" },
  red:    { bg: "#FFCDD2", color: "#B71C1C", dot: "#C62828", label: "Alto Risco" },
  // Fora do semáforo de propósito: "não sei" não é um nível de risco, e pintá-lo de
  // vermelho afirmaria sobre o Aluno algo que ninguém mediu.
  grey:   { bg: "#E5E7EB", color: "#4B5563", dot: "#9CA3AF", label: "Sem previsão" },
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
