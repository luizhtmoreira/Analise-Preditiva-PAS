import type { RiscoStatus } from "@/lib/types";

const CONFIG: Record<RiscoStatus, { bg: string; color: string; label: string }> = {
  green:  { bg: "#C8E6C9", color: "#1B5E20", label: "Baixo Risco" },
  yellow: { bg: "#FFF9C4", color: "#F57F17", label: "Oportunidade" },
  red:    { bg: "#FFCDD2", color: "#B71C1C", label: "Alto Risco" },
};

export function RiscoBadge({ status }: { status: RiscoStatus }) {
  const c = CONFIG[status];
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold whitespace-nowrap"
      style={{ background: c.bg, color: c.color }}
    >
      {c.label}
    </span>
  );
}
