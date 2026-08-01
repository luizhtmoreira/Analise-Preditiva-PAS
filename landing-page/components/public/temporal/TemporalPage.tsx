"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import Link from "next/link";
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { BrandMark } from "@/components/brand/BrandMark";
import { PublicHeader } from "@/components/public/PublicHeader";
import { fetchCorteEvolucao } from "@/lib/api";
import type { TemporalResponse, CorteEvolucao } from "@/lib/types";

/* ─── paleta clara (identidade UnB sobre fundo #F8F9FA) ─────────── */
const LIGHT = {
  grid: "rgba(0,33,71,0.08)",
  tick: "#718096",
  cyan: "#00AEEF",
  green: "#00843D",
  amber: "#F57F17",
  dim: "#4A5568",
};

function ChartTooltip({ active, payload, label }: {
  active?: boolean;
  label?: string | number;
  payload?: { name: string; value: number | null; color: string }[];
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-black/5 bg-white px-3.5 py-2.5 shadow-[0_10px_30px_rgba(0,33,71,0.12)]">
      <p className="font-mono text-[0.65rem] tracking-[0.14em] uppercase text-[#718096] font-bold mb-1.5">{label}</p>
      {payload.map((p) => (
        <p key={p.name} className="text-[0.8rem] flex items-center gap-2 text-[#4A5568]">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          {p.name}: <span className="font-mono font-black text-[#002147] tabular-nums">{p.value ?? "—"}</span>
        </p>
      ))}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-4 mb-3">
      <span className="vp-eyebrow shrink-0">{children}</span>
      <span className="flex-1 h-px bg-[#E2E8F0]" />
    </div>
  );
}

function CursoCombobox({ value, onChange, cursos }: {
  value: string; onChange: (v: string) => void; cursos: string[];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    if (!value.trim()) return cursos.slice(0, 12);
    const norm = (s: string) =>
      s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
    const q = norm(value);
    return cursos.filter((c) => norm(c).includes(q)).slice(0, 12);
  }, [value, cursos]);

  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div ref={ref} className="relative max-w-lg z-30">
      <input
        type="text"
        value={value}
        placeholder="Buscar curso (ex.: Medicina, Direito…)"
        onFocus={() => setOpen(true)}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        className="vp-input text-[0.85rem] font-medium"
      />
      {open && filtered.length > 0 && (
        <div className="vp-dropdown absolute top-[calc(100%+6px)] left-0 right-0 z-50 max-h-60 overflow-y-auto shadow-2xl">
          {filtered.map((c) => (
            <button
              key={c}
              type="button"
              onMouseDown={() => { onChange(c); setOpen(false); }}
              className="vp-dropdown-item block w-full text-left truncate"
              title={c}
            >
              {c}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── página ─────────────────────────────────────────────────────── */

export function TemporalPage({ data }: { data: TemporalResponse }) {
  // Pivot: uma linha por ano, colunas por etapa
  const porAno = useMemo(() => {
    const map = new Map<number, { ano: number; e1?: number; e2?: number; e3?: number; r1?: number; r2?: number; r3?: number }>();
    for (const e of data.etapas) {
      const row = map.get(e.ano) ?? { ano: e.ano };
      if (e.etapa === 1) { row.e1 = e.eb_medio; row.r1 = e.red_media; }
      if (e.etapa === 2) { row.e2 = e.eb_medio; row.r2 = e.red_media; }
      if (e.etapa === 3) { row.e3 = e.eb_medio; row.r3 = e.red_media; }
      map.set(e.ano, row);
    }
    return [...map.values()].sort((a, b) => a.ano - b.ano);
  }, [data.etapas]);

  const [curso, setCurso] = useState("");
  const [cortes, setCortes] = useState<CorteEvolucao[]>([]);
  // "Carregando" é derivado de qual curso já respondeu, e não um booleano ligado à mão antes do
  // fetch: ligá-lo dentro do efeito custa um render extra a cada troca de curso. O `vivo` derruba
  // a resposta de um curso que o Aluno já trocou, que antes sobrescrevia a resposta mais nova.
  const [cursoCarregado, setCursoCarregado] = useState<string | null>(null);
  const loadingCorte = Boolean(curso) && cursoCarregado !== curso;

  useEffect(() => {
    if (!curso) return;
    let vivo = true;
    fetchCorteEvolucao(curso)
      .then((dados) => { if (vivo) { setCortes(dados); setCursoCarregado(curso); } })
      .catch(() => { if (vivo) { setCortes([]); setCursoCarregado(curso); } });
    return () => { vivo = false; };
  }, [curso]);

  // Insight automático: etapa 3 mais recente vs. média histórica da etapa 3
  const insight = useMemo(() => {
    const e3s = data.etapas.filter((e) => e.etapa === 3).sort((a, b) => a.ano - b.ano);
    if (e3s.length < 2) return null;
    const last = e3s[e3s.length - 1];
    const histMean = e3s.slice(0, -1).reduce((s, e) => s + e.eb_medio, 0) / (e3s.length - 1);
    return { ano: last.ano, valor: last.eb_medio, delta: last.eb_medio - histMean };
  }, [data.etapas]);

  return (
    <div className="min-h-screen bg-[#F8F9FA] text-[#1D1D1F] selection:bg-[#00843D] selection:text-white antialiased">
      <PublicHeader />

      {/* ── Lavagem radial estendida por toda a página ── */}
      <div className="vp-wash relative bg-white overflow-hidden min-h-[calc(100vh-65px)]">
        {/* Cabeçalho da página */}
        <div className="relative z-10 max-w-3xl mx-auto px-6 pt-14 pb-6">
          <span className="landing-reveal vp-eyebrow">
            Séries históricas oficiais · Cebraspe
          </span>
          <h1
            className="landing-reveal font-heading text-4xl sm:text-5xl font-extrabold tracking-tight leading-[1.08] text-[#002147] mt-6 mb-3"
            style={{ animationDelay: "90ms" }}
          >
            Análise <span className="text-[#00843D]">Temporal</span>
          </h1>
          <p
            className="landing-reveal text-base sm:text-lg text-[#4A5568] leading-relaxed max-w-lg"
            style={{ animationDelay: "180ms" }}
          >
            Como as provas e as notas de corte se moveram ao longo dos anos — e o
            que isso significa para a sua etapa.
          </p>
        </div>

        {/* Conteúdo */}
        <div className="relative z-10 max-w-3xl mx-auto px-6 py-6 pb-24">
          {/* ── 1. Nota de corte por curso ── */}
          <section className="landing-reveal mb-16 relative z-20" style={{ animationDelay: "270ms" }}>
            <SectionLabel>Evolução da nota de corte</SectionLabel>
            <p className="text-sm text-[#4A5568] leading-relaxed mb-5 max-w-lg">
              Escolha um curso e veja como o corte (Sistema Universal) variou
              entre os triênios — 1º e 2º semestre.
            </p>
            <CursoCombobox value={curso} onChange={setCurso} cursos={data.cursos} />

            {curso && (
              <div className="mt-6 vp-card border-t-4 border-t-[#00AEEF] p-5 sm:p-6">
                <p className="font-heading text-sm font-bold text-[#002147] mb-4">{curso}</p>
                {/* O gráfico vai dentro de um wrapper de altura explícita, e não só com a altura na
                    `ResponsiveContainer`: sem ele o Recharts entra em laço de resize ao trocar de
                    curso (fix 5700fde). Vale para os dois gráficos desta página. */}
                {loadingCorte ? (
                  <p className="text-sm text-[#718096] py-16 text-center">Carregando…</p>
                ) : cortes.length === 0 ? (
                  <p className="text-sm text-[#718096] py-16 text-center">Sem dados de corte para este curso.</p>
                ) : (
                  <div className="w-full h-[280px] relative">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={cortes} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
                        <CartesianGrid stroke={LIGHT.grid} vertical={false} />
                        <XAxis dataKey="trienio" tick={{ fill: LIGHT.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: LIGHT.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                        <Tooltip content={<ChartTooltip />} />
                        <Legend wrapperStyle={{ fontSize: 12, color: LIGHT.dim }} iconType="circle" iconSize={8} />
                        <Line type="monotone" dataKey="corte_1sem" name="Corte 1º semestre" stroke={LIGHT.cyan} strokeWidth={2.5} dot={{ r: 4, fill: LIGHT.cyan }} connectNulls />
                        <Line type="monotone" dataKey="corte_2sem" name="Corte 2º semestre" stroke={LIGHT.amber} strokeWidth={2} strokeDasharray="6 4" dot={{ r: 3, fill: LIGHT.amber }} connectNulls />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            <p className="mt-8 text-sm text-[#718096]">
              Quer saber sua chance contra esse corte?{" "}
              <Link href="/predict" className="text-[#00843D] font-semibold hover:underline">
                Calcule sua previsão →
              </Link>
            </p>
          </section>

          {/* ── 2. Dificuldade das provas ── */}
          <section className="landing-reveal mb-14 relative z-10" style={{ animationDelay: "360ms" }}>
            <SectionLabel>Escore Bruto médio por etapa</SectionLabel>
            <p className="text-sm text-[#4A5568] leading-relaxed mb-6 max-w-lg">
              A média de P1 + P2 de todos os candidatos, ano a ano. Quedas indicam
              provas mais difíceis; o PAS 3 costuma puxar a média para cima.
            </p>
            <div className="vp-card border-t-4 border-t-[#00843D] p-5 sm:p-6">
                <div className="w-full h-[320px] relative">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={porAno} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
                      <CartesianGrid stroke={LIGHT.grid} vertical={false} />
                      <XAxis dataKey="ano" tick={{ fill: LIGHT.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: LIGHT.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Legend wrapperStyle={{ fontSize: 12, color: LIGHT.dim }} iconType="circle" iconSize={8} />
                      <Line type="monotone" dataKey="e1" name="PAS 1" stroke={LIGHT.cyan} strokeWidth={2.5} dot={{ r: 3, fill: LIGHT.cyan }} connectNulls />
                      <Line type="monotone" dataKey="e2" name="PAS 2" stroke={LIGHT.green} strokeWidth={2.5} dot={{ r: 3, fill: LIGHT.green }} connectNulls />
                      <Line type="monotone" dataKey="e3" name="PAS 3" stroke={LIGHT.amber} strokeWidth={2.5} dot={{ r: 3, fill: LIGHT.amber }} connectNulls />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
            </div>
            {insight && (
              <p className="mt-4 text-sm text-[#4A5568] leading-relaxed border-l-4 border-[#00AEEF] pl-4">
                No PAS 3 de <span className="font-mono font-bold text-[#002147]">{insight.ano}</span>, o escore bruto médio foi{" "}
                <span className="font-mono font-bold text-[#002147]">{insight.valor.toFixed(1)}</span> —{" "}
                <span className={`font-mono font-bold ${insight.delta >= 0 ? "text-[#00843D]" : "text-[#F57F17]"}`}>
                  {insight.delta >= 0 ? "+" : ""}{insight.delta.toFixed(1)} pts
                </span>{" "}
                em relação à média histórica da etapa.
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
