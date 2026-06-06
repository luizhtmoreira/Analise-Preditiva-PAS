"use client";
import { useState, useMemo } from "react";
import type { GestaoResponse, RiscoStatus } from "@/lib/types";
import { KpiCards } from "./KpiCards";
import { RiscoBadge } from "./RiscoBadge";

const STATUS_OPTIONS = [
  { value: "todos",  label: "Todos" },
  { value: "red",    label: "Alto Risco" },
  { value: "yellow", label: "Oportunidade" },
  { value: "green",  label: "Baixo Risco" },
];

function Select({ value, onChange, options }: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="text-sm rounded-lg px-3 py-1.5 border focus:outline-none"
      style={{ borderColor: "#E6E6E8", background: "#fff", color: "#1D1D1F" }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

function ChanceCell({ prob1, prob2, status }: { prob1: number; prob2: number; status: string }) {
  const best = Math.max(prob1, prob2);

  let color1: string;
  let color2: string;
  let barColor: string;

  if (status === "yellow") {
    color1 = "#F57F17";
    color2 = prob2 > 70 ? "#00843D" : "#F57F17";
    barColor = color2;
  } else {
    color1 = best >= 50 ? "#00843D" : best >= 30 ? "#F57F17" : "#C62828";
    color2 = best >= 50 ? "#00843D" : best >= 30 ? "#F57F17" : "#C62828";
    barColor = color1;
  }

  return (
    <div className="flex flex-col gap-0.5 min-w-[90px]">
      <div className="flex items-baseline gap-1">
        <span className="text-lg font-bold font-mono leading-none" style={{ color: color1 }}>
          {prob1.toFixed(0)}%
        </span>
        <span className="text-[10px]" style={{ color: "#6E6E73" }}>1º sem</span>
      </div>
      {prob2 > 0 && (
        <div className="flex items-center gap-1">
          <span className="text-xs font-mono font-semibold" style={{ color: color2 }}>
            {prob2.toFixed(0)}%
          </span>
          <span className="text-[10px]" style={{ color: "#6E6E73" }}>2º sem</span>
        </div>
      )}
      <div className="w-full h-1 rounded-full mt-0.5 overflow-hidden" style={{ background: "#E6E6E8" }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(best, 100)}%`, background: barColor }} />
      </div>
    </div>
  );
}

function SugestaoCell({ sugestao }: { sugestao: string }) {
  if (!sugestao || sugestao === "—") {
    return <span className="text-xs" style={{ color: "#C7C7CC" }}>—</span>;
  }
  const [curso, resto] = sugestao.includes(" - ") ? sugestao.split(" - ") : [sugestao, ""];
  return (
    <div className="flex flex-col gap-0.5 max-w-[160px]">
      <span className="text-xs font-medium leading-tight" style={{ color: "#003366" }}>{curso}</span>
      {resto && <span className="text-[10px]" style={{ color: "#6E6E73" }}>{resto}</span>}
    </div>
  );
}

export function GestaoPage({ data }: { data: GestaoResponse }) {
  const { results, kpis, trienio_ref, modelo_disponivel } = data;

  const unidades = useMemo(() => {
    const vals = [...new Set(results.map((r) => r.unidade).filter(Boolean))].sort();
    return [{ value: "todas", label: "Todas" }, ...vals.map((v) => ({ value: v, label: v }))];
  }, [results]);

  const turmas = useMemo(() => {
    const vals = [...new Set(results.map((r) => r.turma).filter(Boolean))].sort();
    return [{ value: "todas", label: "Todas" }, ...vals.map((v) => ({ value: v, label: v }))];
  }, [results]);

  const [unidade, setUnidade] = useState("todas");
  const [turma, setTurma] = useState("todas");
  const [status, setStatus] = useState("todos");

  const filtered = useMemo(() =>
    results.filter((r) => {
      if (unidade !== "todas" && r.unidade !== unidade) return false;
      if (turma !== "todas" && r.turma !== turma) return false;
      if (status !== "todos" && r.status !== status) return false;
      return true;
    }),
    [results, unidade, turma, status]
  );

  return (
    <div className="p-6 pb-24 space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold" style={{ color: "#1D1D1F" }}>Gestão de Ativos</h1>
          <p className="text-sm mt-0.5" style={{ color: "#6E6E73" }}>
            Ref.: triênio {trienio_ref}
            {!modelo_disponivel && (
              <span className="ml-2 text-xs px-2 py-0.5 rounded-full" style={{ background: "#FFF9C4", color: "#F57F17" }}>
                Modelo não disponível — probabilidades zeradas
              </span>
            )}
          </p>
        </div>
      </div>

      {/* KPIs */}
      <KpiCards kpis={kpis} />

      {/* Filtros */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs font-medium" style={{ color: "#6E6E73" }}>Filtrar:</span>
        <Select value={unidade} onChange={setUnidade} options={unidades} />
        <Select value={turma} onChange={setTurma} options={turmas} />
        <Select value={status} onChange={setStatus} options={STATUS_OPTIONS} />
        <span className="text-xs ml-auto" style={{ color: "#6E6E73" }}>
          {filtered.length} de {results.length} alunos
        </span>
      </div>

      {/* Tabela */}
      <div className="rounded-xl overflow-hidden border" style={{ borderColor: "#E6E6E8" }}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: "#F5F5F7", borderBottom: "1px solid #E6E6E8" }}>
                {["Status", "Aluno", "Turma", "Sistema", "Curso Alvo", "Arg. Prev.", "Gap", "Chance de Aprovação", "Histórico", "Sugestão de Curso"].map((h) => (
                  <th key={h} className={`text-left px-4 py-3 text-xs font-semibold whitespace-nowrap ${h === "Chance de Aprovação" ? "text-[#003366]" : ""}`}
                    style={{ color: h === "Chance de Aprovação" ? "#003366" : "#6E6E73" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={10} className="text-center py-12 text-sm" style={{ color: "#6E6E73" }}>
                    Nenhum aluno encontrado com os filtros selecionados.
                  </td>
                </tr>
              ) : filtered.map((r, i) => (
                <tr key={i}
                  style={{ borderBottom: "1px solid #F5F5F7", background: "white" }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#F5F5F7")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "white")}>
                  <td className="px-4 py-2.5">
                    <RiscoBadge status={r.status as RiscoStatus} />
                  </td>
                  <td className="px-4 py-2.5 font-medium whitespace-nowrap" style={{ color: "#1D1D1F" }}>
                    {r.nome}
                  </td>
                  <td className="px-4 py-2.5 text-xs" style={{ color: "#6E6E73" }}>{r.turma || "—"}</td>
                  <td className="px-4 py-2.5 text-xs" style={{ color: "#6E6E73" }}>
                    {r.sistema_concorrencia}
                  </td>
                  <td className="px-4 py-2.5 text-xs max-w-[180px] truncate" style={{ color: "#3A3A3C" }}>
                    {r.curso_alvo}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs font-semibold" style={{ color: "#003366" }}>
                    {r.arg_previsto}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs"
                    style={{ color: r.gap >= 0 ? "#1B5E20" : "#B71C1C" }}>
                    {r.gap >= 0 ? "+" : ""}{r.gap}
                  </td>
                  <td className="px-4 py-2.5">
                    <ChanceCell prob1={r.prob_1_sem} prob2={r.prob_2_sem} status={r.status} />
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background: "#E6E6E8" }}>
                        <div className="h-full rounded-full" style={{
                          width: `${Math.min(r.historico_pct, 100)}%`,
                          background: r.historico_pct >= 50 ? "#00843D" : r.historico_pct >= 30 ? "#F57F17" : "#C62828",
                        }} />
                      </div>
                      <span className="text-xs font-mono" style={{ color: "#6E6E73" }}>
                        {r.historico_pct > 0 ? `${r.historico_pct}%` : "—"}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-2.5">
                    <SugestaoCell sugestao={r.sugestao} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Alerta de risco */}
      {kpis.n_red > 0 && (
        <div className="flex items-start gap-2 rounded-lg p-3 text-sm" style={{ background: "#FFCDD2", color: "#B71C1C" }}>
          <span>⚠️</span>
          <span>
            <strong>{kpis.n_red} aluno{kpis.n_red > 1 ? "s" : ""} ({((kpis.n_red / kpis.total) * 100).toFixed(0)}%)</strong> está{kpis.n_red > 1 ? "ão" : ""} na zona vermelha e pode{kpis.n_red > 1 ? "m" : ""} precisar de redirecionamento de curso.
          </span>
        </div>
      )}
    </div>
  );
}
