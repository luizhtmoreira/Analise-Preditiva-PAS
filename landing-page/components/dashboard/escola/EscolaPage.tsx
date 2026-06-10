"use client";
import { useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { fetchEscola } from "@/lib/api";
import type { EscolaResponse } from "@/lib/types";

const LIGHT = {
  grid: "#EDEDEF",
  tick: "#6E6E73",
  escola: "#00AEEF",
  geral: "#C2C9D1",
  blue: "#003366",
  green: "#00843D",
  red: "#C62828",
};

function LightTooltip({ active, payload, label }: {
  active?: boolean;
  label?: string | number;
  payload?: { name: string; value: number; color: string }[];
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[#E6E6E8] bg-white px-3.5 py-2.5 shadow-lg">
      <p className="font-mono text-[0.65rem] tracking-[0.14em] uppercase text-[#6E6E73] mb-1.5">{label}</p>
      {payload.map((p) => (
        <p key={p.name} className="text-[0.8rem] flex items-center gap-2 text-[#1D1D1F]">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          {p.name}: <span className="font-mono font-bold">{p.value}</span>
        </p>
      ))}
    </div>
  );
}

function StatCard({ label, value, hint, color = "#1D1D1F" }: {
  label: string; value: string; hint?: string; color?: string;
}) {
  return (
    <div className="rounded-xl bg-white border border-[#E6E6E8] p-4 flex flex-col gap-1">
      <p className="font-mono text-[0.62rem] tracking-[0.16em] uppercase text-[#6E6E73]">{label}</p>
      <p className="font-mono text-2xl font-bold leading-none" style={{ color }}>{value}</p>
      {hint && <p className="text-[0.7rem] text-[#6E6E73] mt-0.5">{hint}</p>}
    </div>
  );
}

export function EscolaPage({ initial, inscricoes }: {
  initial: EscolaResponse;
  inscricoes: string[];
}) {
  const [data, setData] = useState(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function changeTrienio(t: string) {
    setLoading(true); setError("");
    try {
      setData(await fetchEscola(inscricoes, t));
    } catch {
      setError("Falha ao recarregar análise. Verifique a API.");
    } finally {
      setLoading(false);
    }
  }

  const acima = data.diff >= 0;

  // Histograma: bin mais próximo das médias para as linhas de referência
  const histData = useMemo(
    () => data.hist_arg.map((b) => ({ ...b, center: +((b.x0 + b.x1) / 2).toFixed(1) })),
    [data.hist_arg],
  );
  const nearestBin = (v: number) =>
    histData.length ? histData.reduce((p, c) => (Math.abs(c.center - v) < Math.abs(p.center - v) ? c : p)).center : 0;

  return (
    <div className={`p-6 pb-24 space-y-6 ${loading ? "opacity-60 pointer-events-none" : ""}`}>
      {/* Header */}
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <p className="font-mono text-[0.68rem] tracking-[0.22em] uppercase text-[#00843D] mb-1">
            Coordenação · benchmark populacional
          </p>
          <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F]">
            Escola vs. População
          </h1>
        </div>
        <label className="flex items-center gap-2 text-xs text-[#6E6E73]">
          Triênio
          <select
            value={data.trienio}
            onChange={(e) => changeTrienio(e.target.value)}
            className="text-sm rounded-lg px-3 py-1.5 border border-[#E6E6E8] bg-white text-[#1D1D1F] focus:outline-none focus:border-[#00AEEF] cursor-pointer"
          >
            {data.trienios_disponiveis.map((t) => <option key={t}>{t}</option>)}
          </select>
        </label>
      </div>

      {error && (
        <div className="rounded-xl p-3.5 text-sm bg-[#FFCDD2] text-[#B71C1C]">{error}</div>
      )}

      {/* Match */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <StatCard label="Inscrições enviadas" value={String(data.n_enviados)} />
        <StatCard label="Encontrados no PAS" value={String(data.n_encontrados)} />
        <StatCard label="Taxa de match" value={`${data.taxa_match.toFixed(1)}%`} />
      </div>

      {data.n_encontrados < 5 ? (
        <div className="rounded-xl p-4 text-sm bg-[#FFF9C4] text-[#F57F17]">
          Poucos alunos encontrados neste triênio ({data.n_encontrados}). Verifique se as
          inscrições da base correspondem ao triênio selecionado.
        </div>
      ) : (
        <>
          {/* Banner */}
          <div
            className="rounded-2xl p-6 text-white relative overflow-hidden"
            style={{
              background: acima
                ? "linear-gradient(135deg, #00843D 0%, #005C2A 100%)"
                : "linear-gradient(135deg, #B71C1C 0%, #7F1313 100%)",
            }}
          >
            <p className="font-mono text-[0.68rem] tracking-[0.22em] uppercase text-white/70 mb-2">
              Argumento Final médio · triênio {data.trienio}
            </p>
            <p className="font-heading text-2xl sm:text-3xl font-bold tracking-[-0.02em] mb-1">
              Sua escola está{" "}
              <span className="font-mono">{Math.abs(data.diff).toFixed(1)}</span> pontos{" "}
              {acima ? "acima" : "abaixo"} da média geral.
            </p>
            <p className="text-sm text-white/75">
              {data.significativo
                ? `Resultado com validade estatística (teste t, p ${data.p_value < 0.0001 ? "< 0.0001" : `= ${data.p_value.toFixed(4)}`}).`
                : "A diferença observada não tem relevância estatística no momento."}
            </p>
          </div>

          {/* KPIs */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard label="Média da escola" value={data.media_escola.toFixed(2)} hint={`desvio padrão ${data.std_escola.toFixed(2)}`} color={LIGHT.blue} />
            <StatCard label="Média geral PAS" value={data.media_geral.toFixed(2)} hint={`desvio padrão ${data.std_geral.toFixed(2)}`} />
            <StatCard label="Diferença" value={`${data.diff >= 0 ? "+" : ""}${data.diff.toFixed(2)}`} color={acima ? LIGHT.green : LIGHT.red} />
            <StatCard label="Percentil" value={`top ${(100 - data.percentil).toFixed(0)}%`} hint={`supera ${data.percentil.toFixed(1)}% dos candidatos`} color={LIGHT.blue} />
          </div>

          {/* Etapas */}
          <div className="rounded-xl bg-white border border-[#E6E6E8] p-5">
            <p className="font-mono text-[0.68rem] tracking-[0.18em] uppercase text-[#6E6E73] mb-4">
              Escore Bruto médio por etapa
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={data.etapas} margin={{ top: 8, right: 8, bottom: 0, left: -16 }} barGap={6}>
                <CartesianGrid stroke={LIGHT.grid} vertical={false} />
                <XAxis dataKey="etapa" tick={{ fill: LIGHT.tick, fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: LIGHT.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<LightTooltip />} cursor={{ fill: "rgba(0,51,102,0.04)" }} />
                <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" iconSize={8} />
                <Bar dataKey="escola" name="Sua escola" fill={LIGHT.escola} radius={[6, 6, 0, 0]} maxBarSize={48} />
                <Bar dataKey="geral" name="Média geral" fill={LIGHT.geral} radius={[6, 6, 0, 0]} maxBarSize={48} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Histograma posicional */}
          <div className="rounded-xl bg-white border border-[#E6E6E8] p-5">
            <p className="font-mono text-[0.68rem] tracking-[0.18em] uppercase text-[#6E6E73] mb-1">
              Onde sua escola se posiciona
            </p>
            <p className="text-xs text-[#6E6E73] mb-4">
              Distribuição do Argumento Final de todos os candidatos. A linha azul é a média da
              sua escola; a tracejada, a média geral.
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={histData} margin={{ top: 24, right: 8, bottom: 0, left: -16 }} barCategoryGap={1}>
                <CartesianGrid stroke={LIGHT.grid} vertical={false} />
                <XAxis dataKey="center" tick={{ fill: LIGHT.tick, fontSize: 10 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={{ fill: LIGHT.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<LightTooltip />} cursor={{ fill: "rgba(0,51,102,0.04)" }} />
                <Bar dataKey="count" name="Candidatos" radius={[3, 3, 0, 0]}>
                  {histData.map((b) => (
                    <Cell key={b.center} fill={LIGHT.geral} />
                  ))}
                </Bar>
                <ReferenceLine
                  x={nearestBin(data.media_escola)}
                  stroke={LIGHT.escola}
                  strokeWidth={3}
                  label={{ value: `Escola ${data.media_escola.toFixed(1)}`, position: "top", fill: LIGHT.blue, fontSize: 11, fontWeight: 700 }}
                />
                <ReferenceLine
                  x={nearestBin(data.media_geral)}
                  stroke="#8A8F98"
                  strokeDasharray="5 4"
                  strokeWidth={2}
                  label={{ value: `Geral ${data.media_geral.toFixed(1)}`, position: "insideTopRight", fill: "#6E6E73", fontSize: 11 }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
