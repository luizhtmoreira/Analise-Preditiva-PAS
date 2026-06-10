"use client";
import { useState, useMemo } from "react";
import type { GestaoResponse, StudentResult, RiscoStatus } from "@/lib/types";
import { RiscoBadge } from "@/components/dashboard/gestao/RiscoBadge";

const ARG_MARGEM = 13.49;

function ProbBar({ label, value }: { label: string; value: number }) {
  const color = value >= 50 ? "#00843D" : value >= 30 ? "#F57F17" : "#C62828";
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-[0.72rem] text-[#6E6E73]">{label}</span>
        <span className="font-mono text-sm font-bold" style={{ color }}>{value.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-[#EDEDEF] overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${Math.min(value, 100)}%`, background: color }} />
      </div>
    </div>
  );
}

/* ─── a folha do relatório (A4-like, whitelabel) ────────────────── */

function ReportSheet({ aluno, tenantLabel, trienioRef }: {
  aluno: StudentResult;
  tenantLabel: string;
  trienioRef: string;
}) {
  const hoje = new Date().toLocaleDateString("pt-BR", { day: "2-digit", month: "long", year: "numeric" });

  return (
    <div className="bg-white rounded-2xl border border-[#E6E6E8] shadow-[0_12px_40px_rgba(0,51,102,0.08)] overflow-hidden print:shadow-none print:border-0 print:rounded-none">
      {/* Cabeçalho institucional */}
      <div
        className="px-5 sm:px-8 py-6 text-white flex items-start justify-between flex-wrap gap-3"
        style={{ background: "linear-gradient(135deg, #002147 0%, #003A70 100%)" }}
      >
        <div>
          <div className="flex items-center gap-2.5 mb-1">
            <span
              className="inline-block w-3 h-3 rounded-[3px] rotate-45"
              style={{ background: "linear-gradient(135deg, #00AEEF, #00843D)" }}
            />
            <span className="font-heading font-bold tracking-tight">Vetor PAS</span>
          </div>
          <p className="font-mono text-[0.62rem] tracking-[0.2em] uppercase text-[#7FD8F7]">
            Relatório preditivo individual
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm font-semibold">{tenantLabel}</p>
          <p className="text-[0.7rem] text-white/55 mt-0.5">{hoje} · ref. triênio {trienioRef}</p>
        </div>
      </div>

      <div className="px-5 sm:px-8 py-7 space-y-7">
        {/* Identificação */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h2 className="font-heading text-2xl font-bold tracking-[-0.02em] text-[#1D1D1F]">{aluno.nome}</h2>
            <p className="text-sm text-[#6E6E73] mt-0.5">
              {[aluno.turma && `Turma ${aluno.turma}`, aluno.unidade, aluno.sistema_concorrencia].filter(Boolean).join(" · ")}
            </p>
          </div>
          <RiscoBadge status={aluno.status as RiscoStatus} />
        </div>

        {/* Argumento previsto */}
        <div className="rounded-xl border border-[#E6E6E8] p-5 sm:p-6 flex items-end justify-between gap-6 flex-wrap">
          <div>
            <p className="font-mono text-[0.62rem] tracking-[0.18em] uppercase text-[#6E6E73] mb-1.5">
              Argumento Final previsto
            </p>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-5xl font-bold text-[#003366] leading-none">
                {aluno.arg_previsto.toFixed(1)}
              </span>
              <span className="text-sm text-[#6E6E73]">pts</span>
            </div>
            <p className="font-mono text-[0.7rem] text-[#9CA1A8] mt-2">
              intervalo {(aluno.arg_previsto - ARG_MARGEM).toFixed(1)} — {(aluno.arg_previsto + ARG_MARGEM).toFixed(1)} (±{ARG_MARGEM})
            </p>
          </div>
          <div className="text-right">
            <p className="font-mono text-[0.62rem] tracking-[0.18em] uppercase text-[#6E6E73] mb-1">
              Gap vs. nota de corte
            </p>
            <p className="font-mono text-3xl font-bold leading-none" style={{ color: aluno.gap >= 0 ? "#00843D" : "#C62828" }}>
              {aluno.gap >= 0 ? "+" : ""}{aluno.gap}
            </p>
          </div>
        </div>

        {/* Curso alvo + probabilidades */}
        <div>
          <p className="font-mono text-[0.62rem] tracking-[0.18em] uppercase text-[#6E6E73] mb-3">
            Curso alvo
          </p>
          <p className="font-heading text-lg font-bold text-[#003366] mb-4">{aluno.curso_alvo || "—"}</p>
          <div className="grid sm:grid-cols-2 gap-x-8 gap-y-4">
            <ProbBar label="Chance de aprovação — 1º semestre" value={aluno.prob_1_sem} />
            {aluno.prob_2_sem > 0 && (
              <ProbBar label="Chance de aprovação — 2º semestre" value={aluno.prob_2_sem} />
            )}
            {aluno.historico_pct > 0 && (
              <ProbBar label="Histórico de alunos similares que alcançaram" value={aluno.historico_pct} />
            )}
          </div>
        </div>

        {/* Recomendação */}
        <div className="rounded-xl bg-[#F5F5F7] border border-[#E6E6E8] p-5">
          <p className="font-mono text-[0.62rem] tracking-[0.18em] uppercase text-[#6E6E73] mb-2">
            Leitura pedagógica
          </p>
          <p className="text-sm leading-relaxed text-[#3A3A3C]">
            {aluno.status === "green" && (
              <>O argumento previsto está acima da nota de corte do curso alvo. Recomenda-se manter o ritmo de estudos e usar o PAS 3 para consolidar a margem de segurança.</>
            )}
            {aluno.status === "yellow" && (
              <>A aprovação no 1º semestre está apertada, mas o 2º semestre é uma janela real. {aluno.sugestao !== "—" && <>Alternativa com alta probabilidade: <strong className="text-[#003366]">{aluno.sugestao}</strong>.</>}</>
            )}
            {aluno.status === "red" && (
              <>O argumento previsto está abaixo da nota de corte atual. Vale conversar sobre estratégia: intensificar a preparação para o PAS 3 {aluno.sugestao !== "—" && <>ou considerar <strong className="text-[#003366]">{aluno.sugestao}</strong>, onde a chance é alta</>}.</>
            )}
          </p>
        </div>
      </div>

      {/* Rodapé */}
      <div className="px-5 sm:px-8 py-4 border-t border-[#E6E6E8] flex items-center justify-between gap-3 flex-wrap">
        <p className="text-[0.65rem] text-[#9CA1A8]">
          Previsão estatística baseada em dados históricos do PAS — não constitui garantia de resultado.
        </p>
        <p className="font-mono text-[0.6rem] tracking-[0.14em] uppercase text-[#9CA1A8]">vetorpas</p>
      </div>
    </div>
  );
}

/* ─── página ─────────────────────────────────────────────────────── */

export function RelatoriosPage({ data, tenantLabel }: {
  data: GestaoResponse;
  tenantLabel: string;
}) {
  const alunos = useMemo(
    () => [...data.results].sort((a, b) => a.nome.localeCompare(b.nome, "pt-BR")),
    [data.results],
  );
  const [selected, setSelected] = useState(alunos[0]?.nome ?? "");
  const aluno = alunos.find((a) => a.nome === selected);

  return (
    <div className="p-6 pb-24 space-y-6 max-w-3xl">
      <div className="print:hidden">
        <p className="font-mono text-[0.68rem] tracking-[0.22em] uppercase text-[#00843D] mb-1">
          Coordenação · documentos whitelabel
        </p>
        <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F]">
          Relatórios PDF
        </h1>
        <p className="text-sm text-[#6E6E73] mt-1 max-w-lg">
          Gere o boletim preditivo individual com a marca da escola — pronto para a
          reunião com pais e alunos.
        </p>
      </div>

      {/* Controles */}
      <div className="flex items-center gap-3 flex-wrap print:hidden">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="flex-1 min-w-52 text-sm rounded-lg px-3 py-2.5 border border-[#E6E6E8] bg-white text-[#1D1D1F] focus:outline-none focus:border-[#00AEEF] cursor-pointer"
        >
          {alunos.map((a) => (
            <option key={a.nome} value={a.nome}>
              {a.nome}{a.turma ? ` — ${a.turma}` : ""}
            </option>
          ))}
        </select>
        <button
          onClick={() => window.print()}
          disabled={!aluno}
          className="px-6 py-2.5 rounded-xl font-semibold text-sm text-white bg-[#003366] hover:bg-[#004080] transition-all hover:-translate-y-0.5 disabled:opacity-60"
        >
          Baixar PDF →
        </button>
      </div>

      {/* Preview = documento final */}
      {aluno && <ReportSheet aluno={aluno} tenantLabel={tenantLabel} trienioRef={data.trienio_ref} />}

      <p className="text-[0.7rem] text-[#9CA1A8] print:hidden">
        O botão usa a impressão do navegador — escolha &ldquo;Salvar como PDF&rdquo; no diálogo.
      </p>
    </div>
  );
}
