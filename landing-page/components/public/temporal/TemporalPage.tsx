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

/* ─── paleta dark (identidade UnB sobre Azul profundo) ──────────── */
const DARK = {
  grid: "rgba(255,255,255,0.07)",
  tick: "rgba(255,255,255,0.5)",
  cyan: "#00AEEF",
  green: "#00E08A",
  amber: "#FFC25E",
  dim: "rgba(255,255,255,0.55)",
};

function DarkTooltip({ active, payload, label }: {
  active?: boolean;
  label?: string | number;
  payload?: { name: string; value: number | null; color: string }[];
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-white/15 bg-[#00305F] px-3.5 py-2.5 shadow-2xl">
      <p className="font-mono text-[0.65rem] tracking-[0.14em] uppercase text-white/55 mb-1.5">{label}</p>
      {payload.map((p) => (
        <p key={p.name} className="text-[0.8rem] flex items-center gap-2 text-white/90">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          {p.name}: <span className="font-mono font-bold">{p.value ?? "—"}</span>
        </p>
      ))}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="font-mono text-[0.7rem] tracking-[0.22em] uppercase text-[#7FD8F7] mb-2 flex items-center gap-3">
      {children}
      <span className="flex-1 h-px bg-gradient-to-r from-[#00AEEF]/35 to-transparent" />
    </p>
  );
}

function CursoCombobox({ value, onChange, cursos }: {
  value: string; onChange: (v: string) => void; cursos: string[];
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    if (!query.trim()) return cursos.slice(0, 12);
    const norm = (s: string) =>
      s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
    const q = norm(query);
    return cursos.filter((c) => norm(c).includes(q)).slice(0, 12);
  }, [query, cursos]);

  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div ref={ref} className="relative max-w-lg">
      <input
        type="text"
        value={open ? query : value || query}
        placeholder="Buscar curso (ex.: Medicina, Direito…)"
        onFocus={() => { setOpen(true); setQuery(""); }}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        className="w-full rounded-[10px] border border-white/15 bg-white/5 px-3.5 py-2.5 text-[0.82rem] text-white placeholder:text-white/30 outline-none transition-colors focus:border-[#00AEEF]/60"
      />
      {open && filtered.length > 0 && (
        <div className="absolute top-[calc(100%+6px)] left-0 right-0 z-50 max-h-60 overflow-y-auto rounded-xl border border-[#00AEEF]/30 bg-[#00305F] shadow-2xl">
          {filtered.map((c) => (
            <button
              key={c}
              type="button"
              onMouseDown={() => { onChange(c); setOpen(false); setQuery(""); }}
              className="block w-full text-left px-3.5 py-2.5 text-[0.78rem] text-white/70 border-b border-white/5 last:border-0 hover:bg-[#00AEEF]/15 hover:text-white transition-colors"
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
  const [loadingCorte, setLoadingCorte] = useState(false);

  useEffect(() => {
    if (!curso) return;
    setLoadingCorte(true);
    fetchCorteEvolucao(curso)
      .then(setCortes)
      .catch(() => setCortes([]))
      .finally(() => setLoadingCorte(false));
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
    <div
      className="min-h-screen text-white"
      style={{ background: "linear-gradient(168deg, #002147 0%, #003366 60%, #003A70 100%)" }}
    >
      <PublicHeader />

      <div className="max-w-3xl mx-auto px-5 py-14 pb-24">
        <p className="landing-reveal font-mono text-xs tracking-[0.22em] uppercase text-[#7FD8F7] mb-3.5">
          Séries históricas oficiais · Cebraspe
        </p>
        <h1 className="landing-reveal font-heading text-4xl sm:text-5xl font-extrabold tracking-[-0.03em] mb-3" style={{ animationDelay: "90ms" }}>
          Análise <span className="text-[#00AEEF]">Temporal</span>
        </h1>
        <p className="landing-reveal text-[0.95rem] text-white/55 leading-relaxed max-w-md mb-12" style={{ animationDelay: "180ms" }}>
          Como as provas e as notas de corte se moveram ao longo dos anos — e o
          que isso significa para a sua etapa.
        </p>

        {/* ── 1. Nota de corte por curso ── */}
        <section className="landing-reveal mb-14" style={{ animationDelay: "270ms" }}>
          <SectionLabel>Evolução da nota de corte</SectionLabel>
          <p className="text-[0.82rem] text-white/45 mb-5 max-w-lg">
            Escolha um curso e veja como o corte (Sistema Universal) variou
            entre os triênios — 1º e 2º semestre.
          </p>
          <CursoCombobox value={curso} onChange={setCurso} cursos={data.cursos} />

          {curso && (
            <div className="mt-6 rounded-2xl border border-white/13 bg-white/5 p-5 sm:p-6">
              <p className="font-heading text-sm font-bold text-white mb-4">{curso}</p>
              {loadingCorte ? (
                <p className="text-sm text-white/45 py-16 text-center">Carregando…</p>
              ) : cortes.length === 0 ? (
                <p className="text-sm text-white/45 py-16 text-center">Sem dados de corte para este curso.</p>
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={cortes} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
                    <CartesianGrid stroke={DARK.grid} vertical={false} />
                    <XAxis dataKey="trienio" tick={{ fill: DARK.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: DARK.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<DarkTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" iconSize={8} />
                    <Line type="monotone" dataKey="corte_1sem" name="Corte 1º semestre" stroke={DARK.cyan} strokeWidth={2.5} dot={{ r: 4, fill: DARK.cyan }} connectNulls />
                    <Line type="monotone" dataKey="corte_2sem" name="Corte 2º semestre" stroke={DARK.amber} strokeWidth={2} strokeDasharray="6 4" dot={{ r: 3, fill: DARK.amber }} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          )}

          <p className="mt-8 text-[0.8rem] text-white/40">
            Quer saber sua chance contra esse corte?{" "}
            <Link href="/predict" className="text-[#00AEEF] hover:underline">
              Calcule sua previsão →
            </Link>
          </p>
        </section>

        {/* ── 2. Dificuldade das provas ── */}
        <section className="landing-reveal mb-14" style={{ animationDelay: "360ms" }}>
          <SectionLabel>Escore Bruto médio por etapa</SectionLabel>
          <p className="text-[0.82rem] text-white/45 mb-6 max-w-lg">
            A média de P1 + P2 de todos os candidatos, ano a ano. Quedas indicam
            provas mais difíceis; o PAS 3 costuma puxar a média para cima.
          </p>
          <div className="rounded-2xl border border-white/13 bg-white/5 p-5 sm:p-6">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={porAno} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
                <CartesianGrid stroke={DARK.grid} vertical={false} />
                <XAxis dataKey="ano" tick={{ fill: DARK.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: DARK.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: DARK.dim }} iconType="circle" iconSize={8} />
                <Line type="monotone" dataKey="e1" name="PAS 1" stroke={DARK.cyan} strokeWidth={2.5} dot={{ r: 3, fill: DARK.cyan }} connectNulls />
                <Line type="monotone" dataKey="e2" name="PAS 2" stroke={DARK.green} strokeWidth={2.5} dot={{ r: 3, fill: DARK.green }} connectNulls />
                <Line type="monotone" dataKey="e3" name="PAS 3" stroke={DARK.amber} strokeWidth={2.5} dot={{ r: 3, fill: DARK.amber }} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {insight && (
            <p className="mt-4 text-[0.82rem] text-white/55 border-l-2 border-[#00AEEF] pl-3">
              No PAS 3 de <span className="font-mono text-white">{insight.ano}</span>, o escore bruto médio foi{" "}
              <span className="font-mono text-white">{insight.valor.toFixed(1)}</span> —{" "}
              <span className={insight.delta >= 0 ? "text-[#00E08A]" : "text-[#FFC25E]"}>
                {insight.delta >= 0 ? "+" : ""}{insight.delta.toFixed(1)} pts
              </span>{" "}
              em relação à média histórica da etapa.
            </p>
          )}
        </section>

        {/* ── 3. Redação ── (desativado; descomente para reativar)
        <section className="landing-reveal" style={{ animationDelay: "450ms" }}>
          <SectionLabel>Redação média por ano</SectionLabel>
          <p className="text-[0.82rem] text-white/45 mb-6 max-w-lg">
            Média entre as etapas aplicadas em cada ano (escala 0–10).
          </p>
          <div className="rounded-2xl border border-white/13 bg-white/5 p-5 sm:p-6">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart
                data={porAno.map((r) => {
                  const reds = [r.r1, r.r2, r.r3].filter((v): v is number => v != null);
                  return { ano: r.ano, red: reds.length ? +(reds.reduce((s, v) => s + v, 0) / reds.length).toFixed(2) : null };
                })}
                margin={{ top: 8, right: 12, bottom: 0, left: -24 }}
              >
                <defs>
                  <linearGradient id="redFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={DARK.cyan} stopOpacity={0.35} />
                    <stop offset="100%" stopColor={DARK.cyan} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={DARK.grid} vertical={false} />
                <XAxis dataKey="ano" tick={{ fill: DARK.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 10]} tick={{ fill: DARK.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Area type="monotone" dataKey="red" name="Redação média" stroke={DARK.cyan} strokeWidth={2.5} fill="url(#redFill)" connectNulls />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>
        */}
      </div>
    </div>
  );
}
