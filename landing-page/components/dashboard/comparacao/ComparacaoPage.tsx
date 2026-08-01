"use client";
import { useState, useMemo } from "react";
import { compareGroups } from "@/lib/api";
import type { StudentRow, ComparacaoResponse, BoxStats } from "@/lib/types";

/* ─── métricas derivadas da linha crua ──────────────────────────── */

const METRICS: { key: string; label: string; get: (r: StudentRow) => number }[] = [
  { key: "eb_pas2", label: "Escore Bruto — PAS 2", get: (r) => r.p1_pas2 + r.p2_pas2 },
  { key: "eb_pas1", label: "Escore Bruto — PAS 1", get: (r) => r.p1_pas1 + r.p2_pas1 },
  { key: "evolucao", label: "Evolução (EB 2 − EB 1)", get: (r) => r.p1_pas2 + r.p2_pas2 - r.p1_pas1 - r.p2_pas1 },
  { key: "red_pas2", label: "Redação — PAS 2", get: (r) => r.red_pas2 },
  { key: "red_pas1", label: "Redação — PAS 1", get: (r) => r.red_pas1 },
];

interface GroupSel {
  unidade: string;
  turma: string;
}

function groupLabel(sel: GroupSel): string {
  if (sel.unidade === "todas" && sel.turma === "todas") return "Todos os alunos";
  if (sel.turma === "todas") return sel.unidade;
  if (sel.unidade === "todas") return `Turma ${sel.turma}`;
  return `${sel.unidade} · ${sel.turma}`;
}

/* ─── boxplot SVG (horizontal) ──────────────────────────────────── */

/** Uma linha do boxplot. Vive no escopo do módulo, e não dentro de `BoxPlot`: um componente
 *  redefinido a cada render é um tipo novo a cada render, e o React remonta a subárvore inteira
 *  em vez de atualizá-la. A escala horizontal entra por prop (`X`) porque depende dos dados. */
function Row({ s, y, color, name, X }: { s: BoxStats; y: number; color: string; name: string; X: (v: number) => number }) {
  return (
    <g>
      <text x={62} y={y + 4} textAnchor="end" fontSize={11.5} fontWeight={600} fill="#3A3A3C">{name}</text>
      {/* whiskers */}
      <line x1={X(s.min)} x2={X(s.q1)} y1={y} y2={y} stroke="#B8BEC6" strokeWidth={1.5} />
      <line x1={X(s.q3)} x2={X(s.max)} y1={y} y2={y} stroke="#B8BEC6" strokeWidth={1.5} />
      <line x1={X(s.min)} x2={X(s.min)} y1={y - 8} y2={y + 8} stroke="#B8BEC6" strokeWidth={1.5} />
      <line x1={X(s.max)} x2={X(s.max)} y1={y - 8} y2={y + 8} stroke="#B8BEC6" strokeWidth={1.5} />
      {/* caixa q1–q3 */}
      <rect x={X(s.q1)} y={y - 16} width={Math.max(X(s.q3) - X(s.q1), 2)} height={32} rx={6} fill={color} fillOpacity={0.18} stroke={color} strokeWidth={1.5} />
      {/* mediana */}
      <line x1={X(s.med)} x2={X(s.med)} y1={y - 16} y2={y + 16} stroke={color} strokeWidth={3} />
      {/* média */}
      <circle cx={X(s.mean)} cy={y} r={4} fill="#fff" stroke={color} strokeWidth={2} />
      <text x={X(s.med)} y={y - 22} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-geist-mono), monospace" fill={color} fontWeight={700}>
        {s.med}
      </text>
    </g>
  );
}

function BoxPlot({ a, b, nameA, nameB }: { a: BoxStats; b: BoxStats; nameA: string; nameB: string }) {
  const lo = Math.min(a.min, b.min);
  const hi = Math.max(a.max, b.max);
  const range = hi - lo || 1;
  const pad = range * 0.06;
  const X = (v: number) => 70 + ((v - (lo - pad)) / (range + 2 * pad)) * 560;


  const ticks = useMemo(() => {
    const t: number[] = [];
    const step = (range + 2 * pad) / 5;
    for (let i = 0; i <= 5; i++) t.push(+((lo - pad) + i * step).toFixed(1));
    return t;
  }, [lo, range, pad]);

  return (
    <div className="overflow-x-auto">
      <svg viewBox="0 0 660 190" className="w-full h-auto min-w-[520px]">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={28} y2={150} stroke="#F0F0F2" strokeWidth={1} />
            <text x={X(t)} y={166} textAnchor="middle" fontSize={10} fontFamily="var(--font-geist-mono), monospace" fill="#9CA1A8">{t}</text>
          </g>
        ))}
        <Row X={X} s={a} y={60} color="#00AEEF" name={nameA.length > 14 ? nameA.slice(0, 13) + "…" : nameA} />
        <Row X={X} s={b} y={122} color="#00843D" name={nameB.length > 14 ? nameB.slice(0, 13) + "…" : nameB} />
      </svg>
    </div>
  );
}

/* ─── seletor de grupo ──────────────────────────────────────────── */

function GroupPicker({ titulo, accent, sel, onChange, unidades, turmasFor }: {
  titulo: string;
  accent: string;
  sel: GroupSel;
  onChange: (s: GroupSel) => void;
  unidades: string[];
  turmasFor: (unidade: string) => string[];
}) {
  return (
    <div className="rounded-xl bg-white border border-[#E6E6E8] p-5 relative overflow-hidden">
      <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: accent }} />
      <p className="font-mono text-[0.68rem] tracking-[0.18em] uppercase mb-3" style={{ color: accent }}>
        {titulo}
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label className="flex flex-col gap-1 text-[0.65rem] font-mono uppercase tracking-[0.12em] text-[#6E6E73]">
          Unidade
          <select
            value={sel.unidade}
            onChange={(e) => onChange({ unidade: e.target.value, turma: "todas" })}
            className="font-sans normal-case tracking-normal text-sm rounded-lg px-3 py-2 border border-[#E6E6E8] bg-white text-[#1D1D1F] focus:outline-none focus:border-[#00AEEF] cursor-pointer"
          >
            <option value="todas">Todas</option>
            {unidades.map((u) => <option key={u}>{u}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-[0.65rem] font-mono uppercase tracking-[0.12em] text-[#6E6E73]">
          Turma
          <select
            value={sel.turma}
            onChange={(e) => onChange({ ...sel, turma: e.target.value })}
            className="font-sans normal-case tracking-normal text-sm rounded-lg px-3 py-2 border border-[#E6E6E8] bg-white text-[#1D1D1F] focus:outline-none focus:border-[#00AEEF] cursor-pointer"
          >
            <option value="todas">Todas</option>
            {turmasFor(sel.unidade).map((t) => <option key={t}>{t}</option>)}
          </select>
        </label>
      </div>
    </div>
  );
}

/* ─── página ─────────────────────────────────────────────────────── */

export function ComparacaoPage({ students }: { students: StudentRow[] }) {
  const [metricKey, setMetricKey] = useState("eb_pas2");
  const [selA, setSelA] = useState<GroupSel>({ unidade: "todas", turma: "todas" });
  const [selB, setSelB] = useState<GroupSel>({ unidade: "todas", turma: "todas" });
  const [result, setResult] = useState<ComparacaoResponse | null>(null);
  const [names, setNames] = useState<{ a: string; b: string; metric: string }>({ a: "", b: "", metric: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const unidades = useMemo(
    () => [...new Set(students.map((s) => s.unidade).filter(Boolean))].sort(),
    [students],
  );
  const turmasFor = (unidade: string) =>
    [...new Set(
      students
        .filter((s) => unidade === "todas" || s.unidade === unidade)
        .map((s) => s.turma)
        .filter(Boolean),
    )].sort();

  const metric = METRICS.find((m) => m.key === metricKey)!;

  const valuesFor = (sel: GroupSel) =>
    students
      .filter((s) => (sel.unidade === "todas" || s.unidade === sel.unidade) && (sel.turma === "todas" || s.turma === sel.turma))
      .map(metric.get)
      .filter((v) => Number.isFinite(v));

  const nA = valuesFor(selA).length;
  const nB = valuesFor(selB).length;

  async function handleCompare() {
    const a = valuesFor(selA);
    const b = valuesFor(selB);
    if (a.length < 2 || b.length < 2) {
      setError("Cada grupo precisa de pelo menos 2 alunos.");
      return;
    }
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await compareGroups({
        group_a: a, group_b: b,
        name_a: groupLabel(selA), name_b: groupLabel(selB),
        metric: metric.label,
      });
      setResult(res);
      setNames({ a: groupLabel(selA), b: groupLabel(selB), metric: metric.label });
    } catch {
      setError("Serviço de análise indisponível. Verifique a API.");
    } finally {
      setLoading(false);
    }
  }

  const dAbs = result ? Math.abs(result.effect_size) : 0;
  const impacto = dAbs > 0.8
    ? "discrepância crítica entre os grupos"
    : dAbs > 0.5
      ? "diferença considerável de desempenho"
      : dAbs > 0.2
        ? "diferença leve, mas perceptível"
        : "diferença mínima entre os grupos";
  const pDisplay = result
    ? (result.p_value < 0.0001 ? "< 0.0001" : `= ${result.p_value.toFixed(4)}`)
    : "";

  return (
    <div className="p-6 pb-24 space-y-6 max-w-4xl">
      <div>
        <p className="font-mono text-[0.68rem] tracking-[0.22em] uppercase text-[#00843D] mb-1">
          Coordenação · validação estatística
        </p>
        <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F]">
          Comparação Entre Grupos
        </h1>
        <p className="text-sm text-[#6E6E73] mt-1 max-w-lg">
          Teste t de duas amostras: a diferença entre os grupos é real ou ruído estatístico?
        </p>
      </div>

      {/* Métrica */}
      <label className="flex items-center gap-3 text-xs text-[#6E6E73]">
        <span className="font-mono text-[0.65rem] tracking-[0.14em] uppercase">Métrica</span>
        <select
          value={metricKey}
          onChange={(e) => setMetricKey(e.target.value)}
          className="text-sm rounded-lg px-3 py-1.5 border border-[#E6E6E8] bg-white text-[#1D1D1F] focus:outline-none focus:border-[#00AEEF] cursor-pointer"
        >
          {METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
        </select>
      </label>

      {/* Grupos */}
      <div className="grid md:grid-cols-2 gap-4">
        <GroupPicker titulo={`Grupo A · ${nA} alunos`} accent="#00AEEF" sel={selA} onChange={setSelA} unidades={unidades} turmasFor={turmasFor} />
        <GroupPicker titulo={`Grupo B · ${nB} alunos`} accent="#00843D" sel={selB} onChange={setSelB} unidades={unidades} turmasFor={turmasFor} />
      </div>

      <button
        onClick={handleCompare}
        disabled={loading}
        className="px-6 py-3 rounded-xl font-semibold text-sm text-white bg-[#003366] hover:bg-[#004080] transition-all hover:-translate-y-0.5 disabled:opacity-60 disabled:hover:translate-y-0"
      >
        {loading ? "Calculando…" : "Realizar teste estatístico →"}
      </button>

      {error && <div className="rounded-xl p-3.5 text-sm bg-[#FFCDD2] text-[#B71C1C]">{error}</div>}

      {result && (
        <div className="space-y-5">
          {/* Veredito */}
          <div
            className="rounded-2xl p-6 text-white"
            style={{
              background: result.significativo
                ? "linear-gradient(135deg, #003366 0%, #00254A 100%)"
                : "linear-gradient(135deg, #5C6470 0%, #444B55 100%)",
            }}
          >
            <p className="font-mono text-[0.68rem] tracking-[0.22em] uppercase text-white/60 mb-2">
              {names.a} vs. {names.b} · {names.metric}
            </p>
            <p className="font-heading text-2xl font-bold tracking-[-0.02em] mb-1">
              {result.significativo ? "Diferença de performance confirmada" : "Performance equivalente"}
            </p>
            <p className="text-sm text-white/75">
              {result.significativo
                ? <>O teste indica {impacto} <span className="font-mono">(p {pDisplay})</span>.</>
                : <>Não há variação estatisticamente relevante entre os grupos <span className="font-mono">(p {pDisplay})</span>.</>}
            </p>
          </div>

          {/* Médias */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="rounded-xl bg-white border border-[#E6E6E8] p-4">
              <p className="font-mono text-[0.62rem] tracking-[0.14em] uppercase text-[#00AEEF] mb-1 truncate">{names.a}</p>
              <p className="font-mono text-2xl font-bold text-[#1D1D1F]">{result.mean_a.toFixed(2)}</p>
            </div>
            <div className="rounded-xl bg-white border border-[#E6E6E8] p-4">
              <p className="font-mono text-[0.62rem] tracking-[0.14em] uppercase text-[#00843D] mb-1 truncate">{names.b}</p>
              <p className="font-mono text-2xl font-bold text-[#1D1D1F]">{result.mean_b.toFixed(2)}</p>
            </div>
            <div className="rounded-xl bg-white border border-[#E6E6E8] p-4">
              <p className="font-mono text-[0.62rem] tracking-[0.14em] uppercase text-[#6E6E73] mb-1">Diferença</p>
              <p className="font-mono text-2xl font-bold" style={{ color: result.diff >= 0 ? "#00843D" : "#C62828" }}>
                {result.diff >= 0 ? "+" : ""}{result.diff.toFixed(2)}
              </p>
            </div>
          </div>

          {/* Boxplot */}
          <div className="rounded-xl bg-white border border-[#E6E6E8] p-5">
            <p className="font-mono text-[0.68rem] tracking-[0.18em] uppercase text-[#6E6E73] mb-4">
              Distribuição comparada — {names.metric}
            </p>
            <BoxPlot a={result.box_a} b={result.box_b} nameA={names.a} nameB={names.b} />
            <p className="text-[0.7rem] text-[#9CA1A8] mt-2">
              Caixa = quartis (25%–75%) · traço grosso = mediana · círculo = média · hastes = mín/máx
            </p>
          </div>

          {/* Laudo técnico */}
          <details className="rounded-xl bg-white border border-[#E6E6E8] px-5 py-4 group">
            <summary className="cursor-pointer font-mono text-[0.7rem] tracking-[0.14em] uppercase text-[#6E6E73] select-none hover:text-[#003366] transition-colors">
              Laudo estatístico técnico
            </summary>
            <dl className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
              {[
                ["Valor-p", result.p_value < 0.000001 ? "< 1e-6" : result.p_value.toFixed(6)],
                ["Estatística t", String(result.t_statistic)],
                ["Cohen's d", String(result.effect_size)],
                ["Amostras", `nA=${result.box_a.n} · nB=${result.box_b.n}`],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="font-mono text-[0.6rem] tracking-[0.14em] uppercase text-[#9CA1A8]">{k}</dt>
                  <dd className="font-mono font-semibold text-[#1D1D1F] mt-0.5">{v}</dd>
                </div>
              ))}
            </dl>
          </details>
        </div>
      )}
    </div>
  );
}
