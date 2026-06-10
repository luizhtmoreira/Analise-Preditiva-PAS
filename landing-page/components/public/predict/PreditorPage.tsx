"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import Link from "next/link";
import { BrandMark } from "@/components/brand/BrandMark";
import { fetchPredict, fetchCourses } from "@/lib/api";
import type { PredictResponse, CourseResult } from "@/lib/types";

/* ─── constants ─────────────────────────────────────────────────── */

const TRIENIOS = ["2024-2026", "2023-2025", "2022-2024"];
const COTAS = [
  "Sistema Universal",
  "L1 - Escola Pública + Renda ≤ 1,5 SM + PPI",
  "L2 - Escola Pública + Renda ≤ 1,5 SM",
  "L9 - Escola Pública + PPI",
  "L10 - Escola Pública",
];

// Paleta da identidade (docs/identidade-visual.md) sobre fundo Azul UnB escuro
const C = {
  bg:       "#002147",
  surface:  "rgba(255,255,255,0.05)",
  border:   "rgba(255,255,255,0.13)",
  borderHi: "rgba(0,174,239,0.5)",
  text:     "#FFFFFF",
  dim:      "rgba(255,255,255,0.55)",
  faint:    "rgba(255,255,255,0.3)",
  cyan:     "#00AEEF",
  cyanSoft: "#7FD8F7",
  green:    "#00C26A",
  red:      "#FF6B6B",
  amber:    "#FFC25E",
} as const;

/* ─── global styles injected once ──────────────────────────────── */

const GLOBAL_STYLES = `
  .pred-root * { box-sizing: border-box; }
  .pred-root .mono { font-family: var(--font-geist-mono), monospace; }
  .pred-root .heading { font-family: var(--font-display), sans-serif; }

  @keyframes predFadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes predBarGrow {
    from { width: 0; }
  }

  .pred-result { animation: predFadeUp 0.55s cubic-bezier(.16,1,.3,1) both; }
  .pred-result-1 { animation-delay: 0.05s; }
  .pred-result-2 { animation-delay: 0.15s; }
  .pred-result-3 { animation-delay: 0.25s; }
  .pred-result-4 { animation-delay: 0.35s; }

  .bar-grow { animation: predBarGrow 1.2s cubic-bezier(.16,1,.3,1) both; animation-delay: 0.5s; }

  .pred-stepper { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 12px; transition: border-color .2s; }
  .pred-stepper:focus-within { border-color: rgba(0,174,239,0.6); }
  .pred-stepper:focus-within .pred-stepper-num { color: #00AEEF !important; }

  .pred-btn-adj { background: transparent; border: none; cursor: pointer; padding: 0 18px; color: rgba(255,255,255,0.35); font-size: 22px; font-weight: 300; transition: color .15s; line-height: 1; }
  .pred-btn-adj:hover { color: #00AEEF; }

  .pred-cta { transition: transform .2s, box-shadow .3s, background .2s, opacity .2s; }
  .pred-cta:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,174,239,0.45) !important; background: #33C1F3 !important; }

  .pred-row:hover td { background: rgba(255,255,255,0.04) !important; }

  .pred-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 16px; }

  .pred-select { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 10px; color: #fff; padding: 10px 14px; font-size: 13px; font-family: var(--font-body), sans-serif; width: 100%; outline: none; transition: border-color .2s; appearance: none; cursor: pointer; }
  .pred-select:focus { border-color: rgba(0,174,239,0.6); }

  .pred-combo-input { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 10px; color: #fff; padding: 10px 14px; font-size: 13px; font-family: var(--font-body), sans-serif; width: 100%; outline: none; transition: border-color .2s; }
  .pred-combo-input:focus { border-color: rgba(0,174,239,0.6); }
  .pred-combo-input::placeholder { color: rgba(255,255,255,0.3); }

  .pred-dropdown { background: #00305F; border: 1px solid rgba(0,174,239,0.3); border-radius: 12px; box-shadow: 0 16px 48px rgba(0,10,25,0.6); overflow: hidden; }
  .pred-dropdown-item { padding: 9px 14px; font-size: 12.5px; color: rgba(255,255,255,0.7); cursor: pointer; transition: all .15s; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .pred-dropdown-item:hover { background: rgba(0,174,239,0.15); color: #fff; }

  .pred-grid-bg {
    background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 56px 56px;
  }

  /* grids empilham no mobile, 2 colunas a partir de 640px */
  .pred-grid-2 { display: grid; grid-template-columns: 1fr; gap: 14px; }
  .pred-grid-cfg { display: grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 640px) {
    .pred-grid-2 { grid-template-columns: 1fr 1fr; }
    .pred-grid-cfg { grid-template-columns: 1fr 1fr; }
  }

  .section-label { font-family: var(--font-geist-mono), monospace; font-size: 11px; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: #7FD8F7; display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
  .section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(to right, rgba(0,174,239,0.35), transparent); }

  @media (prefers-reduced-motion: reduce) {
    .pred-result, .bar-grow { animation: none; }
  }
`;

/* ─── subcomponents ─────────────────────────────────────────────── */

function StepperInput({ label, value, onChange, step = 0.5, min = -100, max = 100 }: {
  label: string; value: string; onChange: (v: string) => void;
  step?: number; min?: number; max?: number;
}) {
  function adjust(d: number) {
    const next = Math.min(max, Math.max(min, parseFloat(((parseFloat(value) || 0) + d).toFixed(3))));
    onChange(String(next));
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim }}>
        {label}
      </span>
      <div className="pred-stepper" style={{ display: "flex", alignItems: "stretch", height: 48 }}>
        <button type="button" className="pred-btn-adj" onClick={() => adjust(-step)}>−</button>
        <input
          type="number" step="any" min={min} max={max} value={value}
          onChange={(e) => onChange(e.target.value)}
          className="pred-stepper-num mono"
          style={{
            flex: 1, textAlign: "center", fontSize: 17, fontWeight: 700, background: "transparent",
            border: "none", outline: "none", color: C.text, width: 0,
          }}
        />
        <button type="button" className="pred-btn-adj" onClick={() => adjust(step)}>+</button>
      </div>
    </div>
  );
}

function CourseCombobox({ value, onChange, courses }: {
  value: string; onChange: (v: string) => void; courses: string[];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const filtered = useMemo(() => {
    if (!value.trim()) return courses.slice(0, 10);
    const q = value.toLowerCase();
    return courses.filter((c) => c.toLowerCase().includes(q)).slice(0, 10);
  }, [value, courses]);

  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <input
        type="text" value={value} placeholder="Buscar curso desejado..."
        className="pred-combo-input"
        onFocus={() => setOpen(true)}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
      />
      {open && filtered.length > 0 && (
        <div className="pred-dropdown" style={{ position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, zIndex: 50, maxHeight: 220, overflowY: "auto" }}>
          {filtered.map((c) => (
            <div key={c} className="pred-dropdown-item" onMouseDown={() => { onChange(c); setOpen(false); }}>
              {c}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ArgCard({ result }: { result: PredictResponse }) {
  const range = (result.arg_max - result.arg_min) || 1;
  const pct = Math.min(100, Math.max(0, ((result.arg_previsto - result.arg_min) / range) * 100));

  return (
    <div className="pred-result pred-result-1" style={{ position: "relative", overflow: "hidden" }}>
      <div style={{
        background: "linear-gradient(135deg, rgba(0,58,112,0.85) 0%, rgba(0,23,50,0.95) 70%)",
        border: "1px solid rgba(0,174,239,0.3)", borderRadius: 20, padding: "32px 32px 28px",
      }}>
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(to right, transparent, #00AEEF, transparent)" }} />

        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
          <div>
            <p className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 6 }}>
              Argumento Final Previsto
            </p>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
              <span className="mono" style={{ fontSize: "clamp(48px, 14vw, 72px)", fontWeight: 700, lineHeight: 1, color: C.cyan, letterSpacing: "-0.02em", textShadow: "0 0 40px rgba(0,174,239,0.4)" }}>
                {result.arg_previsto.toFixed(1)}
              </span>
              <span style={{ fontSize: 16, color: C.dim }}>pts</span>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingTop: 4 }}>
            <div style={{ textAlign: "right" }}>
              <p className="mono" style={{ fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "rgba(0,194,106,0.8)" }}>EB PAS 3 previsto</p>
              <p className="mono" style={{ fontSize: 28, fontWeight: 700, color: C.green, lineHeight: 1.2 }}>
                {result.eb_pas3_previsto.toFixed(1)}
              </p>
            </div>
            <p style={{ fontSize: 11, color: C.faint, textAlign: "right" }}>
              ref. {result.trienio_ref}
            </p>
          </div>
        </div>

        {/* Intervalo de confiança */}
        <div style={{ marginTop: 28 }}>
          <div style={{ position: "relative", height: 6, borderRadius: 99, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
            <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to right, #FF6B6B 0%, #FFC25E 35%, #00C26A 100%)" }} />
            <div
              className="bar-grow"
              style={{
                position: "absolute", top: 0, bottom: 0, left: 0,
                width: `${pct}%`, background: "transparent",
                borderRight: "3px solid #fff",
                boxShadow: "2px 0 12px rgba(255,255,255,0.6)",
              }}
            />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
            <span className="mono" style={{ fontSize: 11, color: C.faint }}>{result.arg_min.toFixed(1)}</span>
            <span style={{ fontSize: 11, color: C.faint }}>intervalo ±13,49</span>
            <span className="mono" style={{ fontSize: 11, color: C.faint }}>{result.arg_max.toFixed(1)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function CursoAlvoCard({ c }: { c: CourseResult }) {
  const isGood = c.prob >= 50;
  const isMid  = c.prob >= 30;
  const color  = isGood ? C.green : isMid ? C.amber : C.red;
  const label  = isGood ? "Dentro do alcance" : isMid ? "Possível no 2º semestre" : "Fora do alcance atual";

  return (
    <div className="pred-result pred-result-2 pred-card" style={{ padding: "24px 28px" }}>
      <p className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 12 }}>
        Curso Alvo · {c.semestre} semestre
      </p>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <div style={{ flex: 1 }}>
          <p className="heading" style={{ fontSize: 17, fontWeight: 700, color: C.text, lineHeight: 1.3, letterSpacing: "-0.01em" }}>{c.curso}</p>
          <p style={{ fontSize: 13, color: C.dim, marginTop: 3 }}>
            {[c.turno, c.campus].filter(Boolean).join(" · ")}
          </p>
          <p style={{ fontSize: 12, fontWeight: 600, color, marginTop: 8 }}>{label}</p>
          <p className="mono" style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>
            corte: {c.nota_corte.toFixed(3)}
          </p>
        </div>
        <div style={{ flexShrink: 0, textAlign: "right" }}>
          <span className="mono" style={{ fontSize: "clamp(40px, 11vw, 56px)", fontWeight: 700, lineHeight: 1, color }}>{c.prob.toFixed(0)}</span>
          <span style={{ fontSize: 20, color }}>%</span>
        </div>
      </div>
      <div style={{ marginTop: 16, height: 3, borderRadius: 99, background: "rgba(255,255,255,0.07)", overflow: "hidden" }}>
        <div className="bar-grow" style={{ height: "100%", borderRadius: 99, background: color, width: `${Math.min(c.prob, 100)}%` }} />
      </div>
    </div>
  );
}

function TopCursosTable({ cursos }: { cursos: CourseResult[] }) {
  if (!cursos.length) return null;
  return (
    <div className="pred-result pred-result-4">
      <p className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 14 }}>
        Cursos acessíveis com sua previsão
      </p>
      <div style={{ border: "1px solid rgba(255,255,255,0.13)", borderRadius: 14, overflow: "hidden" }}>
        <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
        <table style={{ width: "100%", minWidth: 560, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "rgba(255,255,255,0.05)" }}>
              {["Curso", "Campus · Turno", "Sem.", "Corte", "Chance"].map((h) => (
                <th key={h} className="mono" style={{ padding: "10px 16px", textAlign: "left", fontSize: 10, fontWeight: 500, letterSpacing: "0.12em", textTransform: "uppercase", color: C.cyanSoft, borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cursos.map((c, i) => {
              const color = c.prob >= 50 ? C.green : c.prob >= 30 ? C.amber : C.red;
              return (
                <tr key={i} className="pred-row" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                  <td style={{ padding: "10px 16px", fontSize: 13, fontWeight: 600, color: C.text }}>{c.curso}</td>
                  <td style={{ padding: "10px 16px", fontSize: 12, color: C.dim }}>
                    {[c.campus, c.turno].filter(Boolean).join(" · ") || "—"}
                  </td>
                  <td style={{ padding: "10px 16px", fontSize: 12, color: C.dim }}>{c.semestre}</td>
                  <td className="mono" style={{ padding: "10px 16px", fontSize: 12, color: C.cyanSoft }}>{c.nota_corte.toFixed(3)}</td>
                  <td style={{ padding: "10px 16px" }}>
                    <span className="mono" style={{ display: "inline-block", padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 700, color, background: color + "20", border: `1px solid ${color}40` }}>
                      {c.prob.toFixed(0)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        </div>
      </div>
    </div>
  );
}

/* ─── main page ─────────────────────────────────────────────────── */

const emptyScores = () => ({ p1: "0", p2: "0", red: "0" });

export function PreditorPage() {
  const [pas1, setPas1] = useState(emptyScores());
  const [pas2, setPas2] = useState(emptyScores());
  const [cota, setCota] = useState("Sistema Universal");
  const [trienio, setTrienio] = useState("2024-2026");
  const [cursoAlvo, setCursoAlvo] = useState("");
  const [courses, setCourses] = useState<string[]>([]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchCourses(cota, trienio).then(setCourses).catch(() => setCourses([]));
  }, [cota, trienio]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setError(""); setResult(null);
    try {
      const data = await fetchPredict({
        p1_pas1: Number(pas1.p1), p2_pas1: Number(pas1.p2), red_pas1: Number(pas1.red),
        p1_pas2: Number(pas2.p1), p2_pas2: Number(pas2.p2), red_pas2: Number(pas2.red),
        cota, trienio, curso_alvo: cursoAlvo.trim() || undefined,
      });
      setResult(data);
    } catch {
      setError("Serviço indisponível. Verifique se a API está rodando.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: GLOBAL_STYLES }} />

      <div
        className="pred-root"
        style={{
          minHeight: "100vh",
          background: "linear-gradient(168deg, #002147 0%, #003366 60%, #003A70 100%)",
          color: C.text,
        }}
      >

        {/* ── Header ── */}
        <div style={{ borderBottom: "1px solid rgba(255,255,255,0.1)", background: "rgba(0,26,53,0.75)", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 40 }}>
          <div style={{ maxWidth: 680, margin: "0 auto", padding: "14px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <BrandMark sublabel="Preditor PAS 3" />
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div className="hidden sm:flex" style={{ alignItems: "center", gap: 8 }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: C.green, boxShadow: `0 0 8px ${C.green}`, display: "inline-block" }} />
                <span style={{ fontSize: 11, color: C.dim }}>Modelo ativo</span>
              </div>
              <Link href="/" style={{ fontSize: 12, color: C.dim, textDecoration: "none" }} className="hover:!text-white transition-colors">
                ← Início
              </Link>
            </div>
          </div>
        </div>

        {/* ── Conteúdo ── */}
        <div className="pred-grid-bg">
          <div style={{ maxWidth: 680, margin: "0 auto", padding: "48px 20px 96px" }}>

            <p className="mono" style={{ fontSize: 12, letterSpacing: "0.22em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 14 }}>
              Análise preditiva · PAS/UnB
            </p>
            <h1 className="heading" style={{ fontSize: 38, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1, marginBottom: 10, color: "#fff" }}>
              Preditor <span style={{ color: C.cyan }}>PAS 3</span>
            </h1>
            <p style={{ fontSize: 15, color: C.dim, marginBottom: 40, maxWidth: 440, lineHeight: 1.6 }}>
              Insira suas notas do PAS 1 e 2 — o modelo prevê seu Argumento Final e suas chances nos cursos da UnB.
            </p>

            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* PAS 1 + PAS 2 grid */}
              <div className="pred-grid-2">
                {[
                  { title: "PAS 1", state: pas1, set: setPas1 },
                  { title: "PAS 2", state: pas2, set: setPas2 },
                ].map(({ title, state, set }) => (
                  <div key={title} className="pred-card" style={{ padding: "22px 20px" }}>
                    <div className="section-label">{title}</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      <StepperInput label="P1 — Língua Estrangeira" value={state.p1}
                        onChange={(v) => set((s) => ({ ...s, p1: v }))} step={0.5} min={-20} max={20} />
                      <StepperInput label="P2 — Conhecimentos" value={state.p2}
                        onChange={(v) => set((s) => ({ ...s, p2: v }))} step={0.5} min={-100} max={100} />
                      <StepperInput label="Redação" value={state.red}
                        onChange={(v) => set((s) => ({ ...s, red: v }))} step={0.1} min={0} max={10} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Config */}
              <div className="pred-card" style={{ padding: "22px 20px" }}>
                <div className="section-label">Configuração do Candidato</div>
                <div className="pred-grid-cfg">
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Triênio</p>
                    <div style={{ position: "relative" }}>
                      <select value={trienio} onChange={(e) => setTrienio(e.target.value)} className="pred-select">
                        {TRIENIOS.map((t) => <option key={t} style={{ background: "#00305F" }}>{t}</option>)}
                      </select>
                      <span style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: C.dim, pointerEvents: "none", fontSize: 12 }}>▾</span>
                    </div>
                  </div>
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Sistema de Cotas</p>
                    <div style={{ position: "relative" }}>
                      <select value={cota} onChange={(e) => setCota(e.target.value)} className="pred-select">
                        {COTAS.map((c) => <option key={c} style={{ background: "#00305F" }}>{c}</option>)}
                      </select>
                      <span style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: C.dim, pointerEvents: "none", fontSize: 12 }}>▾</span>
                    </div>
                  </div>
                  <div style={{ gridColumn: "1 / -1" }}>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>
                      Curso Alvo <span style={{ textTransform: "none", letterSpacing: 0, fontWeight: 400, fontSize: 11, color: C.faint }}>opcional</span>
                    </p>
                    <CourseCombobox value={cursoAlvo} onChange={setCursoAlvo} courses={courses} />
                  </div>
                </div>
              </div>

              {/* CTA */}
              <button type="submit" disabled={loading} className="pred-cta"
                style={{
                  width: "100%", padding: "16px", borderRadius: 14, border: "none",
                  background: C.cyan, color: "#002147", fontSize: 15, fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                  fontFamily: "var(--font-body), sans-serif", opacity: loading ? 0.6 : 1,
                  boxShadow: "0 8px 30px rgba(0,174,239,0.35)",
                }}>
                {loading ? "Calculando previsão…" : "Calcular minha previsão →"}
              </button>
            </form>

            {/* Error */}
            {error && (
              <div style={{ marginTop: 14, padding: "12px 16px", borderRadius: 10, background: "rgba(255,107,107,0.12)", border: "1px solid rgba(255,107,107,0.3)", color: C.red, fontSize: 13 }}>
                {error}
              </div>
            )}

            {/* Results */}
            {result && (
              <div style={{ marginTop: 40, display: "flex", flexDirection: "column", gap: 14 }}>
                <p className="mono" style={{ fontSize: 11, letterSpacing: "0.22em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 4 }}>
                  Diagnóstico gerado
                </p>
                <ArgCard result={result} />
                {result.curso_alvo_result && <CursoAlvoCard c={result.curso_alvo_result} />}
                <TopCursosTable cursos={result.top_cursos} />
              </div>
            )}

          </div>
        </div>
      </div>
    </>
  );
}
