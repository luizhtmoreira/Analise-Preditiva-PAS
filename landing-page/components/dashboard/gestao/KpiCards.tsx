import type { GestaoKpis } from "@/lib/types";

interface KpiCardProps {
  label: string;
  value: number;
  bg?: string;
  color?: string;
}

function KpiCard({ label, value, bg = "#fff", color = "#1D1D1F" }: KpiCardProps) {
  return (
    <div className="rounded-xl p-4 flex flex-col gap-1" style={{ background: bg }}>
      <p className="text-xs font-medium" style={{ color: bg === "#fff" ? "#6E6E73" : color }}>{label}</p>
      <p className="text-3xl font-bold" style={{ color }}>{value}</p>
    </div>
  );
}

export function KpiCards({ kpis }: { kpis: GestaoKpis }) {
  return (
    <div className="grid grid-cols-4 gap-3">
      <KpiCard label="Total de Ativos" value={kpis.total} />
      <KpiCard label="Alto Risco" value={kpis.n_red} bg="#FFCDD2" color="#B71C1C" />
      <KpiCard label="Oportunidade (2º Sem)" value={kpis.n_yellow} bg="#FFF9C4" color="#F57F17" />
      <KpiCard label="Baixo Risco" value={kpis.n_green} bg="#C8E6C9" color="#1B5E20" />
    </div>
  );
}
