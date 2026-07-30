import type { GestaoKpis } from "@/lib/types";

interface KpiCardProps {
  label: string;
  value: number;
  accent?: string;
  valueColor?: string;
}

function KpiCard({ label, value, accent = "#00AEEF", valueColor = "#002147" }: KpiCardProps) {
  return (
    <div className="vp-card p-4 flex flex-col gap-1.5 overflow-hidden relative">
      <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: accent }} />
      <p className="vp-label">{label}</p>
      <p className="font-mono text-3xl font-bold leading-none" style={{ color: valueColor }}>
        {value}
      </p>
    </div>
  );
}

export function KpiCards({ kpis }: { kpis: GestaoKpis }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <KpiCard label="Total de Ativos" value={kpis.total} />
      <KpiCard label="Alto Risco" value={kpis.n_red} accent="#C62828" valueColor="#B71C1C" />
      <KpiCard label="Oportunidade (2º Sem)" value={kpis.n_yellow} accent="#F57F17" valueColor="#F57F17" />
      <KpiCard label="Baixo Risco" value={kpis.n_green} accent="#00843D" valueColor="#1B5E20" />
    </div>
  );
}
