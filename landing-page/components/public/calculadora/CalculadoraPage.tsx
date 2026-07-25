"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import Link from "next/link";
import { BrandMark } from "@/components/brand/BrandMark";
import { PublicHeader } from "@/components/public/PublicHeader";
import { fetchCourses, fetchCoursesCutoff, fetchStrategy } from "@/lib/api";
import type { StrategyResponse } from "@/lib/types";
import { createClient } from "@/lib/supabase/client";

/* ─── constants & palette ────────────────────────────────────────── */

const TRIENIOS = ["2024-2026", "2023-2025", "2022-2024"];
const COTAS = [
  "Sistema Universal",
  "L1 - Escola Pública + Renda ≤ 1,5 SM + PPI",
  "L2 - Escola Pública + Renda ≤ 1,5 SM",
  "L9 - Escola Pública + PPI",
  "L10 - Escola Pública",
];

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
  .calc-root * { box-sizing: border-box; }
  .calc-root .mono { font-family: var(--font-geist-mono), monospace; }
  .calc-root .heading { font-family: var(--font-display), sans-serif; }

  @keyframes calcFadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .calc-result { animation: calcFadeUp 0.55s cubic-bezier(.16,1,.3,1) both; }
  .calc-result-1 { animation-delay: 0.05s; }
  .calc-result-2 { animation-delay: 0.15s; }

  .calc-stepper { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 12px; transition: border-color .2s; }
  .calc-stepper:focus-within { border-color: rgba(0,174,239,0.6); }
  .calc-stepper:focus-within .calc-stepper-num { color: #00AEEF !important; }

  .calc-btn-adj { background: transparent; border: none; cursor: pointer; padding: 0 18px; color: rgba(255,255,255,0.35); font-size: 22px; font-weight: 300; transition: color .15s; line-height: 1; }
  .calc-btn-adj:hover { color: #00AEEF; }

  .calc-cta { transition: transform .2s, box-shadow .3s, background .2s, opacity .2s; }
  .calc-cta:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,174,239,0.45) !important; background: #33C1F3 !important; }

  .calc-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 16px; }

  .calc-select { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 10px; color: #fff; padding: 10px 14px; font-size: 13px; font-family: var(--font-body), sans-serif; width: 100%; outline: none; transition: border-color .2s; appearance: none; cursor: pointer; }
  .calc-select:focus { border-color: rgba(0,174,239,0.6); }

  .calc-combo-input { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.13); border-radius: 10px; color: #fff; padding: 10px 14px; font-size: 13px; font-family: var(--font-body), sans-serif; width: 100%; outline: none; transition: border-color .2s; }
  .calc-combo-input:focus { border-color: rgba(0,174,239,0.6); }
  .calc-combo-input::placeholder { color: rgba(255,255,255,0.3); }

  .calc-dropdown { background: #00305F; border: 1px solid rgba(0,174,239,0.3); border-radius: 12px; box-shadow: 0 16px 48px rgba(0,10,25,0.6); overflow: hidden; }
  .calc-dropdown-item { padding: 9px 14px; font-size: 12.5px; color: rgba(255,255,255,0.7); cursor: pointer; transition: all .15s; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .calc-dropdown-item:hover { background: rgba(0,174,239,0.15); color: #fff; }

  .calc-grid-bg {
    background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 56px 56px;
  }

  .calc-grid-2 { display: grid; grid-template-columns: 1fr; gap: 14px; }
  .calc-grid-cfg { display: grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 640px) {
    .calc-grid-2 { grid-template-columns: 1fr 1fr; }
    .calc-grid-cfg { grid-template-columns: 1fr 1fr; }
  }

  /* ── Simulador de Itens ── */
  .sim-gate { border-radius: 18px; border: 1.5px solid rgba(0,174,239,0.35); background: rgba(0,174,239,0.06); padding: 24px 22px; margin-top: 4px; }
  .sim-gate-blur { filter: blur(3px); pointer-events: none; user-select: none; opacity: 0.5; }
  .sim-slider { width: 100%; accent-color: #00AEEF; cursor: pointer; }
  .sim-slider:disabled { opacity: 0.4; cursor: default; }
  @keyframes simFadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
  .sim-in { animation: simFadeIn 0.4s cubic-bezier(.16,1,.3,1) both; }
  .sim-progress-track { width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 99px; overflow: hidden; }
  .sim-progress-fill { height: 100%; border-radius: 99px; transition: width 0.4s cubic-bezier(.16,1,.3,1); }
  .sim-unlock-btn { width: 100%; padding: 14px 20px; border-radius: 12px; background: #00AEEF; border: none; color: #fff; font-size: 14px; font-weight: 700; cursor: pointer; transition: transform .2s, box-shadow .3s; }
  .sim-unlock-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 36px rgba(0,174,239,0.45); }
`;

/* ─── subcomponents ─────────────────────────────────────────────── */

interface StepperInputProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  step?: number;
  min?: number;
  max?: number;
}

function StepperInput({ label, value, onChange, step = 0.5, min = -100, max = 100 }: StepperInputProps) {
  function adjust(d: number) {
    const next = Math.min(max, Math.max(min, parseFloat(((parseFloat(value) || 0) + d).toFixed(3))));
    onChange(String(next));
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim }}>
        {label}
      </span>
      <div className="calc-stepper" style={{ display: "flex", alignItems: "stretch", height: 48 }}>
        <button type="button" className="calc-btn-adj" onClick={() => adjust(-step)}>−</button>
        <input
          type="number" step="any" min={min} max={max} value={value}
          onChange={(e) => onChange(e.target.value)}
          className="calc-stepper-num mono"
          style={{
            flex: 1, textAlign: "center", fontSize: 17, fontWeight: 700, background: "transparent",
            border: "none", outline: "none", color: C.text, width: 0,
          }}
        />
        <button type="button" className="calc-btn-adj" onClick={() => adjust(step)}>+</button>
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
    const normalize = (s: string) =>
      s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
    const q = normalize(value);
    return courses.filter((c) => normalize(c).includes(q)).slice(0, 10);
  }, [value, courses]);

  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <input
        type="text"
        placeholder="Digite o curso..."
        value={value}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        className="calc-combo-input"
      />
      {open && filtered.length > 0 && (
        <div className="calc-dropdown" style={{ position: "absolute", left: 0, right: 0, top: "calc(100% + 5px)", zIndex: 50, maxHeight: 220, overflowY: "auto" }}>
          {filtered.map((c) => (
            <div
              key={c}
              className="calc-dropdown-item"
              onClick={() => { onChange(c); setOpen(false); }}
            >
              {c}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── main page ────────────────────────────────────────────────── */

const emptyScores = () => ({ p1: "0", p2: "0", red: "0" });

export function CalculadoraPage() {
  const [pas1, setPas1] = useState(emptyScores());
  const [pas2, setPas2] = useState(emptyScores());
  const [cota, setCota] = useState("Sistema Universal");
  const [trienio, setTrienio] = useState("2024-2026");
  const [semestre, setSemestre] = useState("1°");
  const [cursoAlvo, setCursoAlvo] = useState("");
  const [notaAlvo, setNotaAlvo] = useState<string>("0");
  const [courses, setCourses] = useState<string[]>([]);
  const [loadingCourses, setLoadingCourses] = useState(false);

  // Overrides and AI predictions
  const [iaEstimates, setIaEstimates] = useState<{ p1: number; red: number } | null>(null);
  const [p1Override, setP1Override] = useState<number | null>(null);
  const [redOverride, setRedOverride] = useState<number | null>(null);
  const [showOverrides, setShowOverrides] = useState(false);
  const baseProjecao = "Replicar Padrão 2023-2025";

  // Output State
  const [result, setResult] = useState<StrategyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);

  // Simulador de Itens state
  const [simA, setSimA] = useState(60);
  const [simB, setSimB] = useState(2);
  const [simC, setSimC] = useState(3);
  const [simD, setSimD] = useState(1);

  // Check login and load saved grades
  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (user) {
        setIsLoggedIn(true);
        setUserId(user.id);
        
        supabase
          .from("alunos_perfis")
          .select("*")
          .eq("id", user.id)
          .single()
          .then(({ data: profile, error }) => {
            if (profile && !error) {
              setPas1({
                p1: String(profile.p1_pas1 ?? 0),
                p2: String(profile.p2_pas1 ?? 0),
                red: String(profile.red_pas1 ?? 0),
              });
              setPas2({
                p1: String(profile.p1_pas2 ?? 0),
                p2: String(profile.p2_pas2 ?? 0),
                red: String(profile.red_pas2 ?? 0),
              });
              if (profile.cota) setCota(profile.cota);
              if (profile.trienio) setTrienio(profile.trienio);
              if (profile.curso_alvo) setCursoAlvo(profile.curso_alvo);
            }
          });
      }
    });
  }, []);

  // Fetch Courses list
  useEffect(() => {
    setLoadingCourses(true);
    fetchCourses(cota, trienio)
      .then(setCourses)
      .catch(() => setCourses([]))
      .finally(() => setLoadingCourses(false));
  }, [cota, trienio]);

  // Automatically fetch target cutoff
  useEffect(() => {
    if (cursoAlvo && courses.includes(cursoAlvo)) {
      fetchCoursesCutoff(cursoAlvo, cota, trienio, semestre)
        .then((res) => setNotaAlvo(String(res.cutoff)))
        .catch(() => setNotaAlvo("0"));
    }
  }, [cursoAlvo, cota, trienio, semestre, courses]);

  // Fetch AI Estimates whenever grades or trienio changes
  useEffect(() => {
    const p1_pas1 = Number(pas1.p1) || 0;
    const p2_pas1 = Number(pas1.p2) || 0;
    const red_pas1 = Number(pas1.red) || 0;
    const p1_pas2 = Number(pas2.p1) || 0;
    const p2_pas2 = Number(pas2.p2) || 0;
    const red_pas2 = Number(pas2.red) || 0;

    if (p1_pas1 !== 0 || p2_pas1 !== 0 || p1_pas2 !== 0 || p2_pas2 !== 0) {
      fetchStrategy({
        p1_pas1, p2_pas1, red_pas1,
        p1_pas2, p2_pas2, red_pas2,
        nota_alvo: 0,
        ciclo_aluno: trienio,
      })
        .then((res) => {
          setIaEstimates({ p1: res.p1_ia, red: res.red_ia });
          // If overrides are not touched, initialize them with default estimates
          setP1Override((prev) => prev === null ? res.p1_ia : prev);
          setRedOverride((prev) => prev === null ? res.red_ia : prev);
        })
        .catch((err) => console.error("Erro ao buscar estimativas da IA:", err));
    }
  }, [pas1, pas2, trienio]);

  async function handleRouteCalculate(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const body = {
      p1_pas1: Number(pas1.p1) || 0,
      p2_pas1: Number(pas1.p2) || 0,
      red_pas1: Number(pas1.red) || 0,
      p1_pas2: Number(pas2.p1) || 0,
      p2_pas2: Number(pas2.p2) || 0,
      red_pas2: Number(pas2.red) || 0,
      nota_alvo: Number(notaAlvo) || 0,
      ciclo_aluno: trienio,
      p1_override: p1Override !== null ? p1Override : undefined,
      red_override: redOverride !== null ? redOverride : undefined,
      base_projecao: trienio === "2024-2026" ? baseProjecao : undefined,
    };

    try {
      const data = await fetchStrategy(body);
      setResult(data);

      // Save grades to Supabase if logged in
      if (isLoggedIn && userId) {
        const supabase = createClient();
        await supabase.from("alunos_perfis").upsert({
          id: userId,
          p1_pas1: body.p1_pas1,
          p2_pas1: body.p2_pas1,
          red_pas1: body.red_pas1,
          p1_pas2: body.p1_pas2,
          p2_pas2: body.p2_pas2,
          red_pas2: body.red_pas2,
          cota,
          trienio,
          curso_alvo: cursoAlvo.trim() || null,
          updated_at: new Date().toISOString(),
        });
      }
    } catch {
      setError("Serviço indisponível. Verifique se a API está rodando.");
    } finally {
      setLoading(false);
    }
  }

  // Lógica Semáforo (Cor do resultado)
  const semaphor = useMemo(() => {
    if (!result) return null;
    const ph = result.prob_hist;
    if (ph >= 20.0) {
      return {
        color: C.green,
        title: "Meta Alcançável!",
        desc: `Perfil estatisticamente consolidado. Historicamente, ${ph}% dos candidatos com este perfil de nota obtiveram a vaga.`,
        bg: "rgba(0,194,106,0.12)",
        border: "rgba(0,194,106,0.3)",
        icon: "🟢"
      };
    } else if (ph >= 5.0) {
      return {
        color: C.amber,
        title: "Meta Desafiadora",
        desc: `Zona de concorrência acirrada. A taxa de aprovação histórica para este perfil de nota é de ${ph}%. Requer esforço acima da média.`,
        bg: "rgba(255,194,94,0.12)",
        border: "rgba(255,194,94,0.3)",
        icon: "🟡"
      };
    } else {
      return {
        color: C.red,
        title: "Meta de Alto Risco",
        desc: `Cenário estatisticamente atípico. Apenas ${ph}% da base histórica com estas notas atingiu este resultado. Requer desempenho significativamente acima da média.`,
        bg: "rgba(255,107,107,0.12)",
        border: "rgba(255,107,107,0.3)",
        icon: "🔴"
      };
    }
  }, [result]);

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: GLOBAL_STYLES }} />
      <div
        className="calc-root"
        style={{
          minHeight: "100vh",
          background: "linear-gradient(168deg, #002147 0%, #003366 60%, #003A70 100%)",
          color: C.text,
        }}
      >
        <PublicHeader />

        {/* Content */}
        <div className="calc-grid-bg">
          <div style={{ maxWidth: 680, margin: "0 auto", padding: "48px 20px 96px" }}>
            <p className="mono" style={{ fontSize: 12, letterSpacing: "0.22em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 14 }}>
              ESTRATÉGIA DE NORMALIZAÇÃO · PAS 3
            </p>
            <h1 className="heading" style={{ fontSize: 36, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1, marginBottom: 10, color: "#fff" }}>
              Calculadora de <span style={{ color: C.cyan }}>Estratégia</span>
            </h1>
            <p style={{ fontSize: 14.5, color: C.dim, marginBottom: 36, maxWidth: 480, lineHeight: 1.6 }}>
              Defina seu curso objetivo e analise qual a nota de conhecimentos <span style={{ whiteSpace: "nowrap" }}>(Parte 2)</span> você precisa atingir na prova do PAS 3.
            </p>

            <form onSubmit={handleRouteCalculate} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              
              {/* Configuração do Candidato */}
              <div className="calc-card" style={{ padding: "22px 20px" }}>
                <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 16 }}>
                  CONFIGURAÇÃO DO CANDIDATO
                </p>

                <div className="calc-grid-cfg" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Semestre de Entrada</p>
                    <select value={semestre} onChange={(e) => setSemestre(e.target.value)} className="calc-select">
                      <option value="1°">1º Semestre (Última chamada)</option>
                      <option value="2°">2º Semestre (1ª chamada)</option>
                    </select>
                  </div>

                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Triênio</p>
                    <select value={trienio} onChange={(e) => setTrienio(e.target.value)} className="calc-select">
                      {TRIENIOS.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </div>

                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Sistema de Concorrência</p>
                    <select value={cota} onChange={(e) => setCota(e.target.value)} className="calc-select">
                      {COTAS.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </div>
              </div>

              {/* Notas do PAS 1 & 2 */}
              <div className="calc-grid-2">
                {[
                  { title: "Notas do PAS 1", state: pas1, set: setPas1 },
                  { title: "Notas do PAS 2", state: pas2, set: setPas2 },
                ].map(({ title, state, set }) => (
                  <div key={title} className="calc-card" style={{ padding: "22px 20px" }}>
                    <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 16 }}>
                      {title}
                    </p>
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

              {/* Curso Objetivo */}
              <div className="calc-card" style={{ padding: "22px 20px" }}>
                <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 16 }}>
                  CURSO OBJETIVO
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Curso Objetivo</p>
                    <CourseCombobox value={cursoAlvo} onChange={setCursoAlvo} courses={courses} />
                  </div>

                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Argumento de Corte Alvo</p>
                    <input
                      type="number"
                      step="any"
                      value={notaAlvo}
                      readOnly
                      className="calc-combo-input mono"
                      style={{ fontSize: 14, opacity: 0.65, cursor: "not-allowed" }}
                    />
                    <p style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginTop: 4 }}>
                      Nota padrão atualizada automaticamente ao selecionar o curso.
                    </p>
                  </div>
                </div>
              </div>

              {/* Collapsible Overrides */}
              <div className="calc-card" style={{ padding: "16px 20px" }}>
                <button
                  type="button"
                  onClick={() => setShowOverrides(!showOverrides)}
                  style={{ width: "100%", background: "none", border: "none", color: "#fff", display: "flex", justifyContent: "space-between", alignItems: "center", padding: 0, cursor: "pointer", outline: "none" }}
                >
                  <span style={{ fontSize: 13, fontWeight: 600, color: C.cyanSoft }}>⚙️ Customizar Projeções (Parte 1 e Redação)</span>
                  <span style={{ fontSize: 12, color: C.dim }}>{showOverrides ? "Recolher ▲" : "Expandir ▼"}</span>
                </button>

                {showOverrides && (
                  <div style={{ marginTop: 18, borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 18, display: "flex", flexDirection: "column", gap: 16 }}>
                    <p style={{ fontSize: 12, color: C.dim, lineHeight: 1.5 }}>
                      Ajuste as expectativas de nota para a Parte 1 (Língua Estrangeira) e para a Redação no PAS 3 para refazer os cálculos.
                    </p>

                    <div className="calc-grid-cfg">
                      <div>
                        <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>
                          Expectativa P1 PAS 3 {iaEstimates ? `(Estimativa IA: ${iaEstimates.p1.toFixed(2)})` : ""}
                        </p>
                        <input
                          type="range"
                          min="-20"
                          max="20"
                          step="0.5"
                          value={p1Override ?? (iaEstimates?.p1 ?? 0)}
                          onChange={(e) => setP1Override(parseFloat(e.target.value))}
                          style={{ width: "100%", accentColor: C.cyan }}
                        />
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: C.dim, marginTop: 4 }}>
                          <span>-20.0</span>
                          <span className="mono" style={{ color: "#fff", fontWeight: 700 }}>{((p1Override !== null ? p1Override : (iaEstimates?.p1 ?? 0))).toFixed(1)}</span>
                          <span>20.0</span>
                        </div>
                      </div>

                      <div>
                        <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>
                          Expectativa Redação PAS 3 {iaEstimates ? `(Estimativa IA: ${iaEstimates.red.toFixed(2)})` : ""}
                        </p>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          step="0.1"
                          value={redOverride ?? (iaEstimates?.red ?? 0)}
                          onChange={(e) => setRedOverride(parseFloat(e.target.value))}
                          style={{ width: "100%", accentColor: C.cyan }}
                        />
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: C.dim, marginTop: 4 }}>
                          <span>0.0</span>
                          <span className="mono" style={{ color: "#fff", fontWeight: 700 }}>{((redOverride !== null ? redOverride : (iaEstimates?.red ?? 0))).toFixed(1)}</span>
                          <span>10.0</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading || !cursoAlvo}
                className="calc-cta"
                style={{
                  height: 52,
                  borderRadius: 12,
                  background: C.cyan,
                  border: "none",
                  outline: "none",
                  color: "#fff",
                  fontSize: 15,
                  fontWeight: 700,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: `0 8px 30px rgba(0, 174, 239, 0.3)`,
                  opacity: (loading || !cursoAlvo) ? 0.6 : 1,
                }}
              >
                {loading ? "Calculando rota de aprovação..." : "🚀 Traçar Rota de Aprovação"}
              </button>
            </form>

            {error && (
              <p style={{ color: C.red, fontSize: 13, marginTop: 16, textAlign: "center", fontWeight: 500 }}>
                {error}
              </p>
            )}

            {/* Results Block */}
            {result && semaphor && (
              <div className="calc-result" style={{ marginTop: 32, display: "flex", flexDirection: "column", gap: 16 }}>
                
                {/* Cor semáforo / Feedback */}
                <div
                  style={{
                    background: semaphor.bg,
                    border: `1px solid ${semaphor.border}`,
                    borderRadius: 14,
                    padding: "18px 20px",
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 14,
                  }}
                >
                  <span style={{ fontSize: 24, lineHeight: 1 }}>{semaphor.icon}</span>
                  <div>
                    <h3 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 4px", color: semaphor.color }}>
                      {semaphor.title}
                    </h3>
                    <p style={{ fontSize: 13, color: "rgba(255,255,255,0.75)", margin: 0, lineHeight: 1.5 }}>
                      {semaphor.desc}
                    </p>
                  </div>
                </div>

                {/* Grid metrics */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                  <div className="calc-card" style={{ padding: "16px 14px", textAlign: "center" }}>
                    <p style={{ fontSize: 9, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>
                      P1 PAS 3 (Est.)
                    </p>
                    <p className="mono" style={{ fontSize: 20, fontWeight: 800, color: "#fff", margin: 0 }}>
                      {result.p1_estimado.toFixed(2)}
                    </p>
                  </div>

                  <div className="calc-card" style={{ padding: "16px 14px", textAlign: "center" }}>
                    <p style={{ fontSize: 9, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>
                      Redação (Est.)
                    </p>
                    <p className="mono" style={{ fontSize: 20, fontWeight: 800, color: "#fff", margin: 0 }}>
                      {result.red_estimada.toFixed(2)}
                    </p>
                  </div>

                  {/* P2 NECESSÁRIA */}
                  <div
                    className="calc-card"
                    style={{
                      padding: "16px 14px",
                      textAlign: "center",
                      border: `1.5px solid ${semaphor.border}`,
                      background: "rgba(255,255,255,0.03)",
                    }}
                  >
                    <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: C.cyanSoft, marginBottom: 6 }}>
                      P2 NECESSÁRIA
                    </p>
                    <p className="mono" style={{ fontSize: 22, fontWeight: 800, color: C.cyan, margin: 0 }}>
                      {result.p2_necessario.toFixed(2)}
                    </p>
                    <span
                      style={{
                        fontSize: 9,
                        fontWeight: 700,
                        padding: "2px 6px",
                        borderRadius: 20,
                        background: result.p2_necessario < 30 ? "rgba(0,194,106,0.15)" : (result.p2_necessario < 60 ? "rgba(255,194,94,0.15)" : "rgba(255,107,107,0.15)"),
                        color: result.p2_necessario < 30 ? C.green : (result.p2_necessario < 60 ? C.amber : C.red),
                        marginTop: 4,
                        display: "inline-block",
                      }}
                    >
                      {result.p2_necessario < 30 ? "Nota Acessível" : (result.p2_necessario < 60 ? "Nota Desafiadora" : "Nota Muito Alta")}
                    </span>
                  </div>
                </div>

                {/* Reality Check Details */}
                {result.amostra > 0 ? (
                  <div
                    className="calc-card"
                    style={{
                      padding: "16px 20px",
                      background: "rgba(0,174,239,0.06)",
                      border: "1px solid rgba(0,174,239,0.15)",
                      display: "flex",
                      gap: 12,
                      alignItems: "center",
                    }}
                  >
                    <span style={{ fontSize: 18 }}>📊</span>
                    <p style={{ fontSize: 12, color: C.dim, margin: 0, lineHeight: 1.5 }}>
                      <strong>Reality Check:</strong> Encontramos <strong>{result.amostra} alunos</strong> no histórico com perfil similar de escore bruto (PAS 1 + PAS 2). A probabilidade foi estimada analisando a taxa de conversão deles na prova final.
                    </p>
                  </div>
                ) : (
                  <p style={{ fontSize: 11, color: C.dim, textAlign: "center", margin: 0 }}>
                    ℹ️ Amostra histórica insuficiente para gerar análise de coorte de alunos similares.
                  </p>
                )}

                {/* ═══════════════════════════════════════════════════
                    SIMULADOR DE ITENS — Soft Gate
                ═══════════════════════════════════════════════════ */}
                {(() => {
                  const metaP2 = Math.max(0, result.p2_necessario);
                  const redEst = result.red_estimada ?? 0;

                  // Cálculo em tempo real (apenas para logados)
                  const pontos =
                    simA * 1.0 + simB * 2.0 + simC * 2.0 + simD * 3.0 + redEst;
                  const pct = metaP2 > 0 ? Math.min(100, (pontos / metaP2) * 100) : 0;
                  const atingiu = pontos >= metaP2;
                  const diff = Math.abs(pontos - metaP2);

                  const progressColor = atingiu
                    ? C.green
                    : pct >= 70
                    ? C.amber
                    : C.cyan;

                  return (
                    <div className="sim-gate sim-in" style={{ marginTop: 8 }}>
                      {/* Header */}
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                        <span style={{ fontSize: 20 }}>🎯</span>
                        <div>
                          <p style={{ fontSize: 13, fontWeight: 700, color: "#fff", margin: 0 }}>
                            Simulador de Itens (Meta PAS 3)
                          </p>
                          <p style={{ fontSize: 11, color: C.dim, margin: 0 }}>
                            Converta sua meta em acertos por tipo de questão
                          </p>
                        </div>
                        {/* Badge com a meta */}
                        <span
                          className="mono"
                          style={{
                            marginLeft: "auto",
                            fontSize: 13,
                            fontWeight: 800,
                            color: C.cyan,
                            background: "rgba(0,174,239,0.12)",
                            border: "1px solid rgba(0,174,239,0.3)",
                            borderRadius: 8,
                            padding: "4px 10px",
                            whiteSpace: "nowrap",
                          }}
                        >
                          Alvo: {metaP2.toFixed(2)} pts
                        </span>
                      </div>

                      {/* Separator */}
                      <div style={{ height: 1, background: "rgba(255,255,255,0.08)", marginBottom: 16 }} />

                      {!isLoggedIn ? (
                        /* ── VISITANTE: Banner + Prévia Bloqueada ── */
                        <>
                          <div
                            style={{
                              background: "rgba(0,10,25,0.5)",
                              border: "1px solid rgba(0,174,239,0.25)",
                              borderRadius: 12,
                              padding: "16px 18px",
                              marginBottom: 16,
                              display: "flex",
                              gap: 12,
                              alignItems: "flex-start",
                            }}
                          >
                            <span style={{ fontSize: 22, lineHeight: 1 }}>🔒</span>
                            <div>
                              <p style={{ fontSize: 13, fontWeight: 700, color: "#fff", margin: "0 0 4px" }}>
                                Recurso Exclusivo para Alunos Cadastrados
                              </p>
                              <p style={{ fontSize: 12, color: C.dim, margin: 0, lineHeight: 1.5 }}>
                                O Simulador converte sua nota-alvo em metas práticas de acertos por tipo de questão
                                (Tipo A, B, C e D). Cadastre-se gratuitamente para simular em tempo real.
                              </p>
                            </div>
                          </div>

                          {/* Prévia bloqueada */}
                          <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255,255,255,0.3)", marginBottom: 10 }}>
                            👁️ Prévia — Acesso Bloqueado
                          </p>
                          <div className="sim-gate-blur">
                            <div className="calc-grid-2" style={{ marginBottom: 12 }}>
                              {([
                                { label: "Tipo A · Certo/Errado (Peso 1.0×)", max: 120, val: 60 },
                                { label: "Tipo B · Numérica (Peso 2.0×)",     max: 10,  val: 3 },
                                { label: "Tipo C · Múltipla Escolha (Peso 2.0×)", max: 10, val: 4 },
                                { label: "Tipo D · Discursiva (Peso 3.0×)",   max: 5,   val: 2 },
                              ] as { label: string; max: number; val: number }[]).map((item) => (
                                <div key={item.label}>
                                  <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: C.dim, marginBottom: 4 }}>
                                    {item.label}
                                  </p>
                                  <input type="range" min={0} max={item.max} value={item.val} disabled className="sim-slider" readOnly />
                                  <div style={{ textAlign: "right", fontSize: 11, color: C.faint, marginTop: 2 }}>
                                    {item.val} / {item.max} acertos
                                  </div>
                                </div>
                              ))}
                            </div>
                            <div className="sim-progress-track" style={{ marginBottom: 8 }}>
                              <div className="sim-progress-fill" style={{ width: "70%", background: C.cyan }} />
                            </div>
                            <p style={{ fontSize: 11, color: C.dim, textAlign: "center", margin: 0 }}>Exemplo: 70% da meta atingida</p>
                          </div>

                          <Link href="/login" style={{ display: "block", marginTop: 18, textDecoration: "none" }}>
                            <button className="sim-unlock-btn">
                              🔑 Entrar ou Criar Conta Grátis
                            </button>
                          </Link>
                        </>
                      ) : (
                        /* ── LOGADO: Simulador Interativo ── */
                        <>
                          {/* Status */}
                          <div
                            style={{
                              padding: "10px 14px",
                              borderRadius: 10,
                              background: atingiu
                                ? "rgba(0,194,106,0.12)"
                                : "rgba(0,174,239,0.08)",
                              border: `1px solid ${atingiu ? "rgba(0,194,106,0.35)" : "rgba(0,174,239,0.2)"}`,
                              marginBottom: 14,
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                            }}
                          >
                            <span style={{ fontSize: 16 }}>{atingiu ? "🟢" : "🟡"}</span>
                            <p style={{ fontSize: 12, color: "#fff", margin: 0, fontWeight: 600 }}>
                              {atingiu
                                ? `Meta Atingida! P2 Simulado: ${pontos.toFixed(2)} pts (+${diff.toFixed(2)} pts de folga)`
                                : `Em Progresso: ${pontos.toFixed(2)} pts — Faltam ${diff.toFixed(2)} pts para a meta`}
                            </p>
                          </div>

                          {/* Sliders */}
                          <div className="calc-grid-2" style={{ marginBottom: 14 }}>
                            {([
                              { label: "Tipo A · Certo/Errado (Peso 1.0×)", max: 120, val: simA, set: setSimA },
                              { label: "Tipo B · Numérica (Peso 2.0×)",     max: 10,  val: simB, set: setSimB },
                              { label: "Tipo C · Múltipla Escolha (Peso 2.0×)", max: 10, val: simC, set: setSimC },
                              { label: "Tipo D · Discursiva (Peso 3.0×)",   max: 5,   val: simD, set: setSimD },
                            ] as { label: string; max: number; val: number; set: (v: number) => void }[]).map((item) => (
                              <div key={item.label}>
                                <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: C.dim, marginBottom: 4 }}>
                                  {item.label}
                                </p>
                                <input
                                  type="range"
                                  min={0}
                                  max={item.max}
                                  value={item.val}
                                  onChange={(e) => item.set(Number(e.target.value))}
                                  className="sim-slider"
                                />
                                <div style={{ textAlign: "right", fontSize: 11, color: C.faint, marginTop: 2 }}>
                                  {item.val} / {item.max} acertos
                                </div>
                              </div>
                            ))}
                          </div>

                          {/* Progress Bar */}
                          <div style={{ marginBottom: 8 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                              <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: C.dim }}>Progresso da Meta P2</span>
                              <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: progressColor }}>{pct.toFixed(1)}%</span>
                            </div>
                            <div className="sim-progress-track">
                              <div className="sim-progress-fill" style={{ width: `${pct}%`, background: progressColor }} />
                            </div>
                          </div>

                          {/* Mini Cards (breakdown) */}
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginTop: 14 }}>
                            {([
                              { label: "Tipo A", pts: simA * 1.0 },
                              { label: "Tipo B", pts: simB * 2.0 },
                              { label: "Tipo C", pts: simC * 2.0 },
                              { label: "Tipo D", pts: simD * 3.0 },
                            ] as { label: string; pts: number }[]).map((item) => (
                              <div
                                key={item.label}
                                style={{
                                  background: "rgba(255,255,255,0.04)",
                                  border: "1px solid rgba(255,255,255,0.09)",
                                  borderRadius: 10,
                                  padding: "10px 8px",
                                  textAlign: "center",
                                }}
                              >
                                <p style={{ fontSize: 9, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: C.dim, margin: "0 0 4px" }}>{item.label}</p>
                                <p className="mono" style={{ fontSize: 16, fontWeight: 800, color: "#fff", margin: 0 }}>{item.pts.toFixed(1)}</p>
                                <p style={{ fontSize: 9, color: C.faint, margin: "2px 0 0" }}>pts</p>
                              </div>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
