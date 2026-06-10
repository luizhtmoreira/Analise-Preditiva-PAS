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
      className="text-sm rounded-lg px-3 py-1.5 border border-[#E6E6E8] bg-white text-[#1D1D1F] focus:outline-none focus:border-[#00AEEF] transition-colors cursor-pointer"
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
        <span className="text-[10px] text-[#6E6E73]">1º sem</span>
      </div>
      {prob2 > 0 && (
        <div className="flex items-center gap-1">
          <span className="text-xs font-mono font-semibold" style={{ color: color2 }}>
            {prob2.toFixed(0)}%
          </span>
          <span className="text-[10px] text-[#6E6E73]">2º sem</span>
        </div>
      )}
      <div className="w-full h-1 rounded-full mt-0.5 overflow-hidden bg-[#E6E6E8]">
        <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(best, 100)}%`, background: barColor }} />
      </div>
    </div>
  );
}

function SugestaoCell({ sugestao }: { sugestao: string }) {
  if (!sugestao || sugestao === "—") {
    return <span className="text-xs text-[#C7C7CC]">—</span>;
  }
  const [curso, resto] = sugestao.includes(" - ") ? sugestao.split(" - ") : [sugestao, ""];
  return (
    <div className="flex flex-col gap-0.5 max-w-[160px]">
      <span className="text-xs font-medium leading-tight text-[#003366]">{curso}</span>
      {resto && <span className="text-[10px] text-[#6E6E73]">{resto}</span>}
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
          <p className="font-mono text-[0.68rem] tracking-[0.22em] uppercase text-[#00843D] mb-1">
            Coordenação · triênio {trienio_ref}
          </p>
          <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F]">
            Gestão de Ativos
          </h1>
          {!modelo_disponivel && (
            <span className="inline-block mt-2 text-xs px-2 py-0.5 rounded-full bg-[#FFF9C4] text-[#F57F17]">
              Modelo não disponível — probabilidades zeradas
            </span>
          )}
        </div>
      </div>

      {/* KPIs */}
      <KpiCards kpis={kpis} />

      {/* Filtros */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="font-mono text-[0.65rem] tracking-[0.14em] uppercase text-[#6E6E73]">Filtrar</span>
        <Select value={unidade} onChange={setUnidade} options={unidades} />
        <Select value={turma} onChange={setTurma} options={turmas} />
        <Select value={status} onChange={setStatus} options={STATUS_OPTIONS} />
        <span className="text-xs ml-auto font-mono text-[#6E6E73]">
          {filtered.length} de {results.length} alunos
        </span>
      </div>

      {/* Tabela */}
      <div className="rounded-xl overflow-hidden border border-[#E6E6E8] bg-white">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-[#F5F5F7] border-b border-[#E6E6E8]">
                {["Status", "Aluno", "Turma", "Sistema", "Curso Alvo", "Arg. Prev.", "Gap", "Chance de Aprovação", "Histórico", "Sugestão de Curso"].map((h) => (
                  <th
                    key={h}
                    className="text-left px-4 py-3 font-mono text-[0.62rem] tracking-[0.12em] uppercase font-medium whitespace-nowrap"
                    style={{ color: h === "Chance de Aprovação" ? "#003366" : "#6E6E73" }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={10} className="text-center py-12 text-sm text-[#6E6E73]">
                    Nenhum aluno encontrado com os filtros selecionados.
                  </td>
                </tr>
              ) : filtered.map((r, i) => (
                <tr key={i} className="border-b border-[#F5F5F7] bg-white hover:bg-[#FAFAFC] transition-colors">
                  <td className="px-4 py-2.5">
                    <RiscoBadge status={r.status as RiscoStatus} />
                  </td>
                  <td className="px-4 py-2.5 font-medium whitespace-nowrap text-[#1D1D1F]">
                    {r.nome}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-[#6E6E73]">{r.turma || "—"}</td>
                  <td className="px-4 py-2.5 text-xs text-[#6E6E73]">
                    {r.sistema_concorrencia}
                  </td>
                  <td className="px-4 py-2.5 text-xs max-w-[180px] truncate text-[#3A3A3C]">
                    {r.curso_alvo}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs font-semibold text-[#003366]">
                    {r.arg_previsto}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs" style={{ color: r.gap >= 0 ? "#1B5E20" : "#B71C1C" }}>
                    {r.gap >= 0 ? "+" : ""}{r.gap}
                  </td>
                  <td className="px-4 py-2.5">
                    <ChanceCell prob1={r.prob_1_sem} prob2={r.prob_2_sem} status={r.status} />
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 rounded-full overflow-hidden bg-[#E6E6E8]">
                        <div className="h-full rounded-full" style={{
                          width: `${Math.min(r.historico_pct, 100)}%`,
                          background: r.historico_pct >= 50 ? "#00843D" : r.historico_pct >= 30 ? "#F57F17" : "#C62828",
                        }} />
                      </div>
                      <span className="text-xs font-mono text-[#6E6E73]">
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
        <div className="flex items-start gap-2.5 rounded-xl p-3.5 text-sm bg-[#FFCDD2] text-[#B71C1C] border border-[#C62828]/20">
          <span className="w-2 h-2 rounded-full bg-[#C62828] mt-1.5 flex-shrink-0" />
          <span>
            <strong>{kpis.n_red} aluno{kpis.n_red > 1 ? "s" : ""} ({((kpis.n_red / kpis.total) * 100).toFixed(0)}%)</strong> está{kpis.n_red > 1 ? "ão" : ""} na zona vermelha e pode{kpis.n_red > 1 ? "m" : ""} precisar de redirecionamento de curso.
          </span>
        </div>
      )}
    </div>
  );
}
