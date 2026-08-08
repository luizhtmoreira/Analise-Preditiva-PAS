"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import Link from "next/link";
import { BrandMark } from "@/components/brand/BrandMark";
import { PublicHeader } from "@/components/public/PublicHeader";
import { fetchPredict, fetchCourses, fetchCorteEvolucao, fetchCourseChamadas } from "@/lib/api";
import type { PredictResponse, CourseResult, CorteEvolucao, ChamadaCorte } from "@/lib/types";
import { COTAS, ehCotaConhecida } from "@/lib/cotas";
import { useWakingUp } from "@/lib/useWakingUp";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";

/* ─── constants ─────────────────────────────────────────────────── */

const TRIENIOS = ["2024-2026", "2023-2025", "2022-2024"];
// O Cebraspe normaliza a Parte 1 por língua estrangeira e não publica qual língua cada
// candidato fez. Perguntar custa um campo e evita viés sistemático contra quem fez espanhol
// ou francês — em (2024, Etapa 2) a diferença entre as médias passou de um desvio inteiro.
const LINGUAS = [
  { valor: "inglesa", rotulo: "Inglês" },
  { valor: "espanhola", rotulo: "Espanhol" },
  { valor: "francesa", rotulo: "Francês" },
];
// Paleta clara da landing (docs/identidade-visual.md) — Azul UnB como tinta,
// não como fundo. `onDark` guarda os tons que só valem dentro do banner navy.
const C = {
  page:     "#F8F9FA",
  surface:  "#FFFFFF",
  border:   "rgba(0,0,0,0.05)",
  hairline: "#E2E8F0",
  text:     "#002147",
  dim:      "#4A5568",
  faint:    "#718096",
  label:    "#00843D",
  cyan:     "#00AEEF",
  cyanSoft: "#7FD8F7",
  green:    "#00843D",
  red:      "#B71C1C",
  amber:    "#F57F17",
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

  .pred-stepper { background: #F8F9FA; border: 1px solid rgba(0,0,0,0.10); border-radius: 12px; transition: border-color .2s, box-shadow .2s; }
  .pred-stepper:focus-within { border-color: #00843D; box-shadow: 0 0 0 3px rgba(0,132,61,0.10); }
  .pred-stepper:focus-within .pred-stepper-num { color: #00843D !important; }

  .pred-btn-adj { background: transparent; border: none; cursor: pointer; padding: 0 18px; color: #718096; font-size: 22px; font-weight: 300; transition: color .15s; line-height: 1; }
  .pred-btn-adj:hover { color: #00843D; }

  .pred-cta { transition: transform .2s, box-shadow .3s, background .2s, opacity .2s; }
  .pred-cta:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,174,239,0.40) !important; background: #33C1F3 !important; }
  .pred-cta:active:not(:disabled) { transform: scale(0.98); }

  .pred-row:hover td { background: #F8F9FA !important; }

  .pred-card { background: #FFFFFF; border: 1px solid rgba(0,0,0,0.05); border-radius: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }

  @keyframes gateBackdrop { from { opacity: 0; } to { opacity: 1; } }
  @keyframes gateSlide { from { opacity: 0; transform: translateY(24px) scale(0.97); } to { opacity: 1; transform: translateY(0) scale(1); } }
  .gate-backdrop { animation: gateBackdrop 0.2s ease both; }
  .gate-modal { animation: gateSlide 0.28s cubic-bezier(.16,1,.3,1) both; }

  .pred-select { background: #FFFFFF; border: 1px solid rgba(0,0,0,0.15); border-radius: 12px; color: #002147; padding: 11px 14px; font-size: 13px; font-weight: 500; font-family: var(--font-body), sans-serif; width: 100%; outline: none; transition: border-color .2s, box-shadow .2s; appearance: none; cursor: pointer; }
  .pred-select:focus { border-color: #00843D; box-shadow: 0 0 0 3px rgba(0,132,61,0.10); }

  .pred-combo-input { background: #FFFFFF; border: 1px solid rgba(0,0,0,0.15); border-radius: 12px; color: #002147; padding: 11px 14px; font-size: 13px; font-weight: 500; font-family: var(--font-body), sans-serif; width: 100%; outline: none; transition: border-color .2s, box-shadow .2s; }
  .pred-combo-input:focus { border-color: #00843D; box-shadow: 0 0 0 3px rgba(0,132,61,0.10); }
  .pred-combo-input::placeholder { color: #718096; }

  /* Dropdown navy — o contraste que a landing usa nos seletores */
  .pred-dropdown { background: #001730; border: 1px solid rgba(0,132,61,0.55); border-radius: 12px; box-shadow: 0 15px 30px rgba(0,0,0,0.35); overflow: hidden; }
  .pred-dropdown-item { padding: 10px 16px; font-size: 12.5px; color: rgba(255,255,255,0.85); cursor: pointer; transition: all .15s; border-bottom: 1px solid rgba(255,255,255,0.05); }
  .pred-dropdown-item:last-child { border-bottom: none; }
  .pred-dropdown-item:hover { background: #00843D; color: #fff; font-weight: 700; }

  /* grids empilham no mobile, 2 colunas a partir de 640px */
  .pred-grid-2 { display: grid; grid-template-columns: 1fr; gap: 14px; }
  .pred-grid-cfg { display: grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 640px) {
    .pred-grid-2 { grid-template-columns: 1fr 1fr; }
    .pred-grid-cfg { grid-template-columns: 1fr 1fr; }
  }

  .section-label { font-family: var(--font-geist-mono), monospace; font-size: 11px; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: #00843D; display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
  .section-label::after { content: ''; flex: 1; height: 1px; background: #E2E8F0; }

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
      <span className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim }}>
        {label}
      </span>
      <div className="pred-stepper" style={{ display: "flex", alignItems: "stretch", height: 48 }}>
        <button type="button" className="pred-btn-adj" onClick={() => adjust(-step)}>−</button>
        <input
          type="number" step="any" min={min} max={max} value={value}
          onChange={(e) => onChange(e.target.value)}
          className="pred-stepper-num mono"
          style={{
            flex: 1, textAlign: "center", fontSize: 17, fontWeight: 800, background: "transparent",
            border: "none", outline: "none", color: C.text, width: 0,
          }}
        />
        <button type="button" className="pred-btn-adj" onClick={() => adjust(step)}>+</button>
      </div>
    </div>
  );
}

/**
 * Trigger + lista navy, o mesmo par visual do CourseCombobox abaixo — usado para os
 * campos de opção fechada (língua, triênio, cota, semestre) que antes eram `<select>`
 * nativo e, ao abrir, quebravam para o menu do sistema operacional.
 */
function FieldSelect({ value, onChange, options }: {
  value: string; onChange: (v: string) => void; options: { value: string; label: string }[];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const selected = options.find((o) => o.value === value);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="pred-select"
        style={{ display: "flex", alignItems: "center", justifyContent: "space-between", textAlign: "left" }}
      >
        <span>{selected?.label ?? value}</span>
        <span style={{ color: C.faint, fontSize: 12, marginLeft: 8 }}>▾</span>
      </button>
      {open && (
        <div className="pred-dropdown" style={{ position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, zIndex: 50, maxHeight: 220, overflowY: "auto" }}>
          {options.map((o) => (
            <div key={o.value} className="pred-dropdown-item" onMouseDown={() => { onChange(o.value); setOpen(false); }}>
              {o.label}
            </div>
          ))}
        </div>
      )}
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

/**
 * O número da previsão é o clímax da página — vai no único bloco escuro que a
 * estética da landing admite (o banner navy→verde), do mesmo jeito que o CTA
 * final. Tudo em volta permanece claro.
 */
function ArgCard({ result }: { result: PredictResponse }) {
  // Nem faixa `±` nem EB PAS 3 aqui, e os dois por motivos diferentes.
  //
  // A faixa: `previsto ± 13,49` aparecia sem rótulo e acertava 63% — 1 em cada 3 candidatos
  // terminava fora do intervalo que a tela lhe mostrou. Rotulá-la a 80% deixaria o número
  // honesto, mas ele responde uma pergunta que o candidato não fez: ele quer saber em que curso
  // entra, não entre que valores a nota dele cai. A Largura de Incerteza continua trabalhando —
  // é ela que produz as probabilidades por curso logo abaixo (ADR-0012 §7).
  //
  // O EB PAS 3: era uma **segunda** previsão, de um modelo independente, que discordava desta
  // sobre "passa ou não passa" em 11% dos candidatos. Voltará derivado do Argumento, por
  // aritmética, quando o Estimador Auxiliar e o Ano-Âncora existirem (ticket 04 §7.1).
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
              <span style={{ fontSize: 16, color: "rgba(255,255,255,0.7)" }}>pts</span>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingTop: 4, textAlign: "right" }}>
            <p style={{ fontSize: 11, color: "rgba(255,255,255,0.6)" }}>ref. {result.trienio_ref}</p>
          </div>
        </div>

        {/* De onde o número vem: as duas Etapas já feitas são conta fechada, só a terceira é
            previsão. Mostrar a decomposição é o que impede o Argumento de parecer um palpite. */}
        <div style={{ marginTop: 24, paddingTop: 18, borderTop: "1px solid rgba(255,255,255,0.12)", display: "flex", gap: 24, flexWrap: "wrap" }}>
          {[
            { rotulo: "PAS 1 (feito)", valor: result.a1, peso: "×1" },
            { rotulo: "PAS 2 (feito)", valor: result.a2, peso: "×2" },
            { rotulo: "PAS 3 (previsto)", valor: result.a3_previsto, peso: "×3" },
          ].map(({ rotulo, valor, peso }) => (
            <div key={rotulo}>
              <p className="mono" style={{ fontSize: 9, letterSpacing: "0.12em", textTransform: "uppercase", color: "rgba(255,255,255,0.75)", marginBottom: 3 }}>
                {rotulo} <span style={{ color: C.cyanSoft }}>{peso}</span>
              </p>
              <p className="mono" style={{ fontSize: 18, fontWeight: 600, color: C.cyan, lineHeight: 1.2 }}>
                {valor.toFixed(2)}
              </p>
            </div>
          ))}
        </div>

        {result.etapa_1_ausente && (
          <p style={{ marginTop: 16, fontSize: 11, color: "rgba(255,255,255,0.7)", lineHeight: 1.5 }}>
            Calculado como candidato <strong style={{ color: "#FFFFFF" }}>sem PAS 1</strong> — o modelo usa
            uma leitura própria para quem entrou no programa na Etapa 2.
          </p>
        )}

        {/* Ticket 07: a Turma viva ainda não tem o Edital de médias e desvios do Cebraspe de
            nenhuma Etapa — A1/A2 saem de uma estimativa (ticket 06), não da conta oficial, e
            vão mudar quando o Edital sair. Sem isso o Aluno descobre a mudança sozinho depois. */}
        {result.usa_estatistica_derivada && (
          <p style={{ marginTop: 16, fontSize: 11, color: "rgba(255,255,255,0.7)", lineHeight: 1.5 }}>
            Sua turma ainda não tem o Edital oficial de médias e desvios do Cebraspe — este
            Argumento usa uma <strong style={{ color: "#FFFFFF" }}>estimativa</strong> para PAS 1
            e/ou PAS 2, que será atualizada quando o Edital sair.
          </p>
        )}
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
    <div
      className="pred-result pred-result-2 pred-card"
      style={{ padding: "24px 28px", borderTop: `4px solid ${color}` }}
    >
      <p className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: C.label, fontWeight: 700, marginBottom: 12 }}>
        Curso Alvo · {c.semestre} semestre
      </p>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <div style={{ flex: 1 }}>
          <p className="heading" style={{ fontSize: 17, fontWeight: 800, color: C.text, lineHeight: 1.3, letterSpacing: "-0.01em" }}>{c.curso}</p>
          <p style={{ fontSize: 13, color: C.dim, marginTop: 3 }}>
            {[c.turno, c.campus].filter(Boolean).join(" · ")}
          </p>
          <p style={{ fontSize: 12, fontWeight: 700, color, marginTop: 8 }}>{label}</p>
          <p className="mono" style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>
            corte: {c.nota_corte.toFixed(3)}
          </p>
        </div>
        <div style={{ flexShrink: 0, textAlign: "right" }}>
          <span className="mono" style={{ fontSize: "clamp(40px, 11vw, 56px)", fontWeight: 800, lineHeight: 1, color }}>{c.prob.toFixed(0)}</span>
          <span style={{ fontSize: 20, fontWeight: 800, color }}>%</span>
        </div>
      </div>
      <div style={{ marginTop: 16, height: 4, borderRadius: 99, background: C.hairline, overflow: "hidden" }}>
        <div className="bar-grow" style={{ height: "100%", borderRadius: 99, background: color, width: `${Math.min(c.prob, 100)}%` }} />
      </div>
    </div>
  );
}

function TopCursosTable({ cursos, isLoggedIn }: { cursos: CourseResult[]; isLoggedIn: boolean }) {
  const router = useRouter();
  if (!cursos.length) return null;
  return (
    <div className="pred-result pred-result-4">
      <p className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: C.label, fontWeight: 700, marginBottom: 14 }}>
        {isLoggedIn ? "Cursos acessíveis com sua previsão" : "Cursos mais próximos da sua previsão"}
      </p>
      <div style={{ border: `1px solid ${C.border}`, borderRadius: 16, overflow: "hidden", background: C.surface, boxShadow: "0 1px 2px rgba(0,0,0,0.04)" }}>
        <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
        <table style={{ width: "100%", minWidth: 560, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: C.page }}>
              {["Curso", "Chance", "Sem.", "Corte", "Campus · Turno"].map((h) => (
                <th key={h} className="mono" style={{ padding: "10px 16px", textAlign: "left", fontSize: 10, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: C.label, borderBottom: `1px solid ${C.hairline}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cursos.map((c, i) => {
              const color = c.prob >= 50 ? C.green : c.prob >= 30 ? C.amber : C.red;
              return (
                <tr key={i} className="pred-row" style={{ borderBottom: `1px solid ${C.hairline}` }}>
                  <td style={{ padding: "10px 16px", fontSize: 13, fontWeight: 700, color: C.text }}>{c.curso}</td>
                  <td style={{ padding: "10px 16px" }}>
                    <span className="mono" style={{ display: "inline-block", padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 800, color, background: color + "1A", border: `1px solid ${color}33` }}>
                      {c.prob.toFixed(0)}%
                    </span>
                  </td>
                  <td style={{ padding: "10px 16px", fontSize: 12, color: C.dim }}>{c.semestre}</td>
                  <td className="mono" style={{ padding: "10px 16px", fontSize: 12, fontWeight: 600, color: C.dim }}>{c.nota_corte.toFixed(3)}</td>
                  <td style={{ padding: "10px 16px", fontSize: 12, color: C.dim }}>
                    {[c.campus, c.turno].filter(Boolean).join(" · ") || "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        </div>
      </div>

      {!isLoggedIn && (
        <div
          onClick={() => router.push("/auth/cadastro?next=/predict")}
          style={{
            marginTop: 16,
            background: C.surface,
            border: `1px solid ${C.border}`,
            borderTop: `4px solid ${C.cyan}`,
            borderRadius: 16,
            padding: "18px 20px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 12,
            boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
          }}
          className="vp-card-lift"
        >
          <div style={{ flex: "1 1 280px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
              <span className="vp-eyebrow vp-eyebrow-cyan" style={{ fontSize: "0.68rem" }}>
                Painel Multi-Curso
              </span>
            </div>
            <h4 className="heading" style={{ fontSize: 15, fontWeight: 800, color: C.text, margin: "0 0 4px 0" }}>
              Quer comparar suas chances em múltiplos cursos da UnB?
            </h4>
            <p style={{ fontSize: 12.5, color: C.dim, margin: 0, lineHeight: 1.5 }}>
              Crie sua conta para simular vários cursos ao mesmo tempo e ver exatamente o <strong style={{ color: C.text }}>Quanto Falta</strong> no PAS 3.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                router.push("/auth/cadastro?next=/predict");
              }}
              className="vp-btn vp-btn-cyan"
              style={{ padding: "10px 16px", fontSize: 12, fontWeight: 700, whiteSpace: "nowrap" }}
            >
              <span>Criar Conta Grátis</span>
              <span>→</span>
            </button>
            <span
              onClick={(e) => {
                e.stopPropagation();
                router.push("/auth/entrar?next=/predict");
              }}
              style={{
                fontSize: 12,
                color: C.green,
                fontWeight: 700,
                cursor: "pointer",
                textDecoration: "underline",
                textUnderlineOffset: 3,
                whiteSpace: "nowrap",
              }}
              className="hover:opacity-80"
            >
              Entrar
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

type CustomLabelProps = {
  x?: number; y?: number;
  value?: number | null;
  index?: number;
  highlightIndex: number;
};

const CustomLabel = (props: CustomLabelProps) => {
  const { x = 0, y = 0, value, index = 0, highlightIndex } = props;
  if (value === null || value === undefined) return null;

  const isHighlighted = index === highlightIndex;

  if (isHighlighted) {
    return (
      <g>
        {/* Etiqueta destacada do ponto mais recente */}
        <rect
          x={x - 42}
          y={y - 34}
          width={84}
          height={20}
          rx={4}
          fill="#002147"
          stroke="rgba(0,174,239,0.6)"
          strokeWidth={1.5}
        />
        <text
          x={x}
          y={y - 20}
          fill="#fff"
          fontSize={10}
          fontWeight="bold"
          textAnchor="middle"
          fontFamily="monospace"
        >
          {`Atual: ${value.toFixed(2)}`}
        </text>
        {/* Setinha para baixo, apontando o ponto */}
        <polygon
          points={`${x-4},${y-14} ${x+4},${y-14} ${x},${y-10}`}
          fill="#002147"
        />
      </g>
    );
  }

  return (
    <text
      x={x}
      y={y - 12}
      fill="#4A5568"
      fontSize={11}
      textAnchor="middle"
      fontFamily="monospace"
    >
      {value.toFixed(2)}
    </text>
  );
};

function CorteTendenciaCard({
  curso,
  campus,
  turno,
  semestre,
  data,
  highlightTrienio,
}: {
  curso: string;
  campus: string;
  turno: string;
  semestre: string;
  data: CorteEvolucao[];
  highlightTrienio?: string;
}) {
  const chartData = useMemo(() => {
    return data
      .map((item) => {
        // Formato triênio: "2023-2025" -> extrai "2025" como ano de término
        const year = item.trienio.includes("-") ? item.trienio.split("-")[1] : item.trienio;
        // Escolhe o corte do semestre ativo do curso_alvo (com segurança de fallback)
        const corte = semestre.startsWith("1") ? item.corte_1sem : item.corte_2sem;
        // Fallback se o semestre selecionado não tiver dados históricos
        const finalCorte = corte !== null ? corte : (item.corte_1sem !== null ? item.corte_1sem : item.corte_2sem);

        return {
          year,
          corte: finalCorte !== null ? Number(finalCorte.toFixed(2)) : null,
          trienio: item.trienio,
        };
      })
      .filter((item) => item.corte !== null);
  }, [data, semestre]);

  if (chartData.length === 0) return null;

  // O ponto rotulado "Atual" tem que ser o mesmo triênio que a tabela "Histórico de
  // Chamadas" usa (`highlightTrienio`, o `trienio_ref` calculado a partir do triênio do
  // formulário) — não sempre o último ponto da série, senão os dois widgets descrevem
  // triênios diferentes sob o mesmo card sem nenhuma indicação visual disso.
  const highlightIndex = useMemo(() => {
    const idx = highlightTrienio ? chartData.findIndex((d) => d.trienio === highlightTrienio) : -1;
    return idx >= 0 ? idx : chartData.length - 1;
  }, [chartData, highlightTrienio]);

  return (
    <div className="pred-result pred-result-trend" style={{ position: "relative", width: "100%", marginTop: 8 }}>
      <p className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: C.label, fontWeight: 700, marginBottom: 14 }}>
        Análise de Tendência de Corte: {curso}
      </p>

      <div
        className="pred-card"
        style={{
          borderTop: `4px solid ${C.cyan}`,
          padding: "28px 20px 16px 12px",
          height: 280,
          position: "relative",
        }}
      >
        {/* Subtítulo discreto no topo */}
        <div style={{ position: "absolute", top: 12, left: 16, fontSize: 11, color: C.dim, fontWeight: 600 }}>
          Tendência: {curso} ({campus} - {turno}) — {semestre.startsWith("1") || semestre.startsWith("2") ? `${semestre} Semestre` : "Semestre Geral"}
        </div>

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 35, right: 30, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,33,71,0.08)" vertical={false} />
            <XAxis
              dataKey="year"
              stroke={C.faint}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              dy={10}
            />
            <YAxis
              stroke={C.faint}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              domain={["dataMin - 10", "dataMax + 10"]}
              dx={-5}
            />
            <Line
              type="monotone"
              dataKey="corte"
              stroke={C.cyan}
              strokeWidth={3}
              dot={{ r: 5, stroke: C.cyan, strokeWidth: 2, fill: "#fff" }}
              activeDot={{ r: 7, stroke: "#fff", strokeWidth: 2, fill: C.cyan }}
              label={<CustomLabel highlightIndex={highlightIndex} />}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ChamadasHistoricoTable({ chamadas }: { chamadas: ChamadaCorte[] }) {
  if (!chamadas || chamadas.length === 0) return null;
  return (
    <div className="pred-result pred-result-chamadas" style={{ width: "100%", marginTop: 8 }}>
      <p className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", color: C.label, fontWeight: 700, marginBottom: 14, display: "flex", alignItems: "center", gap: 6 }}>
        <span>🕥</span> Histórico de Chamadas (Lista de Espera)
      </p>
      <div style={{ border: `1px solid ${C.border}`, borderRadius: 16, overflow: "hidden", background: C.surface, boxShadow: "0 1px 2px rgba(0,0,0,0.04)" }}>
        <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
          <table style={{ width: "100%", minWidth: 500, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: C.page }}>
                {["Chamada", "Campus", "Turno", "Nota de Corte"].map((h) => (
                  <th key={h} className="mono" style={{ padding: "10px 16px", textAlign: "left", fontSize: 10, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: C.label, borderBottom: `1px solid ${C.hairline}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {chamadas.map((c, i) => (
                <tr key={i} className="pred-row" style={{ borderBottom: `1px solid ${C.hairline}` }}>
                  <td style={{ padding: "10px 16px", fontSize: 13, fontWeight: 700, color: C.text }}>{c.chamada}</td>
                  <td style={{ padding: "10px 16px", fontSize: 12, color: C.dim }}>{c.campus.toUpperCase()}</td>
                  <td style={{ padding: "10px 16px", fontSize: 12, color: C.dim }}>{c.turno.toUpperCase()}</td>
                  <td className="mono" style={{ padding: "10px 16px", fontSize: 13, color: C.text, fontWeight: 800 }}>{c.nota_corte.toFixed(3)}</td>
                </tr>
              ))}
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
  const router = useRouter();
  const [pas1, setPas1] = useState(emptyScores());
  const [pas2, setPas2] = useState(emptyScores());
  const [linguaE1, setLinguaE1] = useState<"inglesa" | "francesa" | "espanhola">("inglesa");
  const [linguaE2, setLinguaE2] = useState<"inglesa" | "francesa" | "espanhola">("inglesa");
  // O Cebraspe registra a língua por Etapa, não por Aluno — 13,9% trocam entre a Etapa 1 e a 2.
  // O segundo campo começa pré-preenchido pelo primeiro (86% não trocam) mas fica editável; uma
  // vez que o Aluno mexe nele diretamente, ele para de seguir o primeiro.
  const [linguaE2Tocada, setLinguaE2Tocada] = useState(false);
  function handleLinguaE1(v: "inglesa" | "francesa" | "espanhola") {
    setLinguaE1(v);
    if (!linguaE2Tocada) setLinguaE2(v);
  }
  function handleLinguaE2(v: "inglesa" | "francesa" | "espanhola") {
    setLinguaE2Tocada(true);
    setLinguaE2(v);
  }
  const [cota, setCota] = useState("Sistema Universal");
  const [cotaSalvaDesconhecida, setCotaSalvaDesconhecida] = useState(false);
  const [trienio, setTrienio] = useState("2024-2026");
  const [cursoAlvo, setCursoAlvo] = useState("");
  const [semestre, setSemestre] = useState("Ambos");
  const [courses, setCourses] = useState<string[]>([]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const { waking, start: startWaking, stop: stopWaking } = useWakingUp();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  // Sem estado de "carregando" para estes dois: nenhuma parte da tela o lia, e ligá-lo dentro do
  // efeito custava um render extra por diagnóstico. Os cards só aparecem quando o dado chega.
  const [evolutionData, setEvolutionData] = useState<CorteEvolucao[] | null>(null);
  const [chamadasData, setChamadasData] = useState<ChamadaCorte[] | null>(null);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user }, error }) => {
      console.log("Vetor PAS Auth Check - User:", user, "Error:", error);
      if (user) {
        setIsLoggedIn(true);
        setUserId(user.id);

        // Carrega notas e configuração salvas
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
              if (profile.cota) {
                if (ehCotaConhecida(profile.cota)) {
                  setCota(profile.cota);
                } else {
                  setCotaSalvaDesconhecida(true);
                }
              }
              if (profile.trienio) setTrienio(profile.trienio);
              if (profile.curso_alvo) setCursoAlvo(profile.curso_alvo);
            }
          });
      }
    });
  }, []);

  useEffect(() => {
    fetchCourses(cota, trienio).then(setCourses).catch(() => setCourses([]));
  }, [cota, trienio]);

  useEffect(() => {
    if (result && result.curso_alvo_result) {
      const c = result.curso_alvo_result;
      const courseKey = `${c.curso} - ${c.turno} (${c.campus})`;

      const targetTrienio = result.trienio_ref || trienio;
      const targetSemestre = c.semestre;

      // `vivo` derruba a resposta de um curso alvo que o Aluno já trocou — sem ele, um
      // diagnóstico antigo que demorasse mais sobrescreveria o card do diagnóstico novo.
      let vivo = true;

      fetchCorteEvolucao(courseKey, cota)
        .then((data) => { if (vivo) setEvolutionData(data); })
        .catch((err) => {
          console.error("Erro ao buscar evolução do corte:", err);
          if (vivo) setEvolutionData(null);
        });

      fetchCourseChamadas(courseKey, cota, targetTrienio, targetSemestre)
        .then((data) => { if (vivo) setChamadasData(data); })
        .catch((err) => {
          console.error("Erro ao buscar histórico de chamadas:", err);
          if (vivo) setChamadasData(null);
        });

      return () => { vivo = false; };
    }
  }, [result, cota]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setError(""); setResult(null);
    startWaking();
    try {
      const data = await fetchPredict({
        p1_pas1: Number(pas1.p1), p2_pas1: Number(pas1.p2), red_pas1: Number(pas1.red),
        p1_pas2: Number(pas2.p1), p2_pas2: Number(pas2.p2), red_pas2: Number(pas2.red),
        lingua_e1: linguaE1, lingua_e2: linguaE2, cota, trienio,
        curso_alvo: isLoggedIn ? (cursoAlvo.trim() || undefined) : undefined,
        is_logged_in: isLoggedIn,
        semestre: isLoggedIn ? semestre : "Ambos",
      });
      setResult(data);

      // Salva notas e configuração do aluno se estiver logado
      if (isLoggedIn && userId) {
        const supabase = createClient();
        await supabase.from("alunos_perfis").upsert({
          id: userId,
          p1_pas1: Number(pas1.p1),
          p2_pas1: Number(pas1.p2),
          red_pas1: Number(pas1.red),
          p1_pas2: Number(pas2.p1),
          p2_pas2: Number(pas2.p2),
          red_pas2: Number(pas2.red),
          cota: cota,
          trienio: trienio,
          curso_alvo: cursoAlvo.trim() || null,
          updated_at: new Date().toISOString()
        });
      }
    } catch {
      setError("Serviço indisponível. Verifique se a API está rodando.");
    } finally {
      setLoading(false);
      stopWaking();
    }
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: GLOBAL_STYLES }} />

      <div
        className="pred-root antialiased selection:bg-[#00843D] selection:text-white"
        style={{ minHeight: "100vh", background: C.page, color: C.text }}
      >

        <PublicHeader />

        {/* ── Lavagem radial estendida por toda a página ── */}
        <div className="vp-wash relative bg-white overflow-hidden min-h-[calc(100vh-65px)]">
          {/* Cabeçalho */}
          <div style={{ position: "relative", zIndex: 1, maxWidth: 720, margin: "0 auto", padding: "56px 24px 24px" }}>
            <span className="vp-eyebrow">Análise preditiva · PAS/UnB</span>
            <h1 className="heading" style={{ fontSize: "clamp(34px, 7vw, 44px)", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.08, margin: "24px 0 12px", color: C.text }}>
              Preditor <span style={{ color: C.green }}>PAS 3</span>
            </h1>
            <p style={{ fontSize: 17, color: C.dim, maxWidth: 480, lineHeight: 1.6 }}>
              Insira suas notas do PAS 1 e 2 — o modelo prevê seu Argumento Final e suas chances nos cursos da UnB.
            </p>
          </div>

          {/* ── Conteúdo ── */}
          <div style={{ position: "relative", zIndex: 1, maxWidth: 720, margin: "0 auto", padding: "16px 24px 96px" }}>

            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* PAS 1 + PAS 2 grid */}
              <div className="pred-grid-2">
                {[
                  { title: "PAS 1", state: pas1, set: setPas1, lingua: linguaE1, onLingua: handleLinguaE1 },
                  { title: "PAS 2", state: pas2, set: setPas2, lingua: linguaE2, onLingua: handleLinguaE2 },
                ].map(({ title, state, set, lingua, onLingua }) => (
                  <div key={title} className="pred-card" style={{ padding: "22px 20px" }}>
                    <div className="section-label">{title}</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      <div>
                        <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>
                          Língua Estrangeira — {title}
                        </p>
                        <FieldSelect
                          value={lingua}
                          onChange={(v) => onLingua(v as typeof lingua)}
                          options={LINGUAS.map((l) => ({ value: l.valor, label: l.rotulo }))}
                        />
                      </div>
                      <StepperInput label="P1 — Língua Estrangeira" value={state.p1}
                        onChange={(v) => set((s) => ({ ...s, p1: v }))} step={0.5} min={-20} max={20} />
                      <StepperInput label="P2 — Conhecimentos" value={state.p2}
                        onChange={(v) => set((s) => ({ ...s, p2: v }))} step={0.5} min={-100} max={100} />
                      <StepperInput label="Redação" value={state.red}
                        onChange={(v) => set((s) => ({ ...s, red: v }))} step={0.1} min={0} max={10} />
                      {/* Quem entrou no programa na Etapa 2 é uma população que o modelo trata
                          por conta própria (8,7% da base histórica), não um caso de nota zero.
                          Sem o botão, o candidato teria que adivinhar que "tudo zero" é a
                          codificação — e quem adivinhasse errado receberia a previsão de alguém
                          que fez a prova e zerou. */}
                      {title === "PAS 1" && (
                        <button
                          type="button"
                          onClick={() => set({ p1: "0", p2: "0", red: "0" })}
                          style={{
                            alignSelf: "flex-start", marginTop: 2, padding: 0, border: "none",
                            background: "transparent", color: C.cyanSoft, fontSize: 11,
                            textDecoration: "underline", cursor: "pointer",
                            fontFamily: "var(--font-body), sans-serif",
                          }}
                        >
                          Não fiz o PAS 1
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Config */}
              <div className="pred-card" style={{ padding: "22px 20px" }}>
                <div className="section-label">Configuração do Candidato</div>
                <div className="pred-grid-cfg" style={isLoggedIn ? { gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))" } : undefined}>
                  <div>
                    <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Triênio</p>
                    <FieldSelect value={trienio} onChange={setTrienio} options={TRIENIOS.map((t) => ({ value: t, label: t }))} />
                  </div>
                  <div>
                    <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Sistema de Cotas</p>
                    <FieldSelect
                      value={cota}
                      onChange={(v) => { setCota(v); setCotaSalvaDesconhecida(false); }}
                      options={COTAS.map((c) => ({ value: c, label: c }))}
                    />
                    {cotaSalvaDesconhecida && (
                      <p className="mono" style={{ fontSize: 11, color: C.amber, marginTop: 6 }}>
                        A cota salva no seu perfil não é mais reconhecida — mostrando Sistema Universal. Selecione a sua novamente.
                      </p>
                    )}
                  </div>
                  {isLoggedIn && (
                    <div>
                      <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>Semestre de Entrada</p>
                      <FieldSelect
                        value={semestre}
                        onChange={setSemestre}
                        options={["Ambos", "1°", "2°"].map((s) => ({ value: s, label: s === "Ambos" ? "Ambos" : `${s} Semestre` }))}
                      />
                    </div>
                  )}
                  <div style={{ gridColumn: "1 / -1" }}>
                    <p className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: C.dim, marginBottom: 6 }}>
                      Curso Alvo <span style={{ textTransform: "none", letterSpacing: 0, fontWeight: 400, fontSize: 11, color: C.faint }}>opcional</span>
                    </p>
                    {isLoggedIn ? (
                      <CourseCombobox value={cursoAlvo} onChange={setCursoAlvo} courses={courses} />
                    ) : (
                      <div
                        onClick={() => router.push("/auth/cadastro?next=/predict")}
                        style={{
                          background: C.page,
                          border: `1px solid ${C.border}`,
                          borderTop: `4px solid ${C.cyan}`,
                          borderRadius: 16,
                          padding: "18px",
                          cursor: "pointer",
                          boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
                        }}
                        className="vp-card-lift"
                      >
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10, flexWrap: "wrap", gap: 6 }}>
                          <span className="vp-eyebrow vp-eyebrow-cyan" style={{ fontSize: "0.68rem" }}>
                            Recurso para Aluno Cadastrado
                          </span>
                        </div>

                        <h4 className="heading" style={{ fontSize: 15, fontWeight: 800, color: C.text, margin: "0 0 4px 0", display: "flex", alignItems: "center", gap: 6 }}>
                          <span>🔒</span> Desbloquear seleção de Curso Alvo
                        </h4>

                        <p style={{ fontSize: 12.5, color: C.dim, margin: "0 0 14px 0", lineHeight: 1.5 }}>
                          Selecione seu curso pretendido para calcular a probabilidade exata de aprovação, conferir as notas de corte para cada chamada e ver os 10 cursos que você está mais bem colocado.
                        </p>

                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10, paddingTop: 4 }}>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              router.push("/auth/cadastro?next=/predict");
                            }}
                            className="vp-btn vp-btn-cyan"
                            style={{ padding: "9px 14px", fontSize: 12, fontWeight: 700 }}
                          >
                            <span>Cadastrar-se grátis</span>
                            <span>→</span>
                          </button>

                          <span
                            onClick={(e) => {
                              e.stopPropagation();
                              router.push("/auth/entrar?next=/predict");
                            }}
                            style={{
                              fontSize: 12,
                              color: C.green,
                              fontWeight: 700,
                              cursor: "pointer",
                              textDecoration: "underline",
                              textUnderlineOffset: 3,
                            }}
                            className="hover:opacity-80"
                          >
                            Já tem conta? Entrar
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* CTA */}
              <button type="submit" disabled={loading} className="pred-cta vp-btn vp-btn-cyan"
                style={{ width: "100%", padding: "16px", fontSize: 15, fontWeight: 700 }}>
                {loading ? (waking ? "Acordando o serviço (pode levar até 40s)…" : "Calculando previsão…") : "Calcular minha previsão →"}
              </button>
            </form>

            {/* Error */}
            {error && (
              <div style={{ marginTop: 14, padding: "12px 16px", borderRadius: 12, background: "#FFCDD2", border: "1px solid rgba(183,28,28,0.2)", color: C.red, fontSize: 13, fontWeight: 600 }}>
                {error}
              </div>
            )}

            {/* Sem previsão: acontece enquanto o Edital de média e desvio de uma das Etapas já
                feitas não foi publicado/extraído. A1 e A2 são a parte exata da conta — aproximá-los
                seria pior que não responder. */}
            {result && !result.modelo_disponivel && (
              <div style={{ marginTop: 40, padding: "20px 24px", borderRadius: 16, background: "rgba(255,194,94,0.08)", border: "1px solid rgba(255,194,94,0.28)" }}>
                <p style={{ fontSize: 14, fontWeight: 600, color: C.amber, marginBottom: 6 }}>
                  Ainda não dá para calcular o seu triênio
                </p>
                <p style={{ fontSize: 13, color: C.dim, lineHeight: 1.6 }}>
                  O Argumento das Etapas que você já fez depende da média e do desvio-padrão que o
                  Cebraspe publica no Edital de cada Etapa, e o do seu triênio ainda não saiu.
                  Preferimos não responder a responder por estimativa.
                </p>
              </div>
            )}

            {/* Results */}
            {result && result.modelo_disponivel && (
              <div style={{ marginTop: 40, display: "flex", flexDirection: "column", gap: 16 }}>
                <p className="mono" style={{ fontSize: 11, letterSpacing: "0.22em", textTransform: "uppercase", color: C.label, fontWeight: 700, marginBottom: 4 }}>
                  Diagnóstico gerado
                </p>
                <ArgCard result={result} />
                {result.curso_alvo_result && <CursoAlvoCard c={result.curso_alvo_result} />}
                {result.curso_alvo_sem_dados_cota && (
                  <div
                    style={{
                      background: "#FFF8E1",
                      border: "1px solid rgba(245,127,23,0.25)",
                      borderLeft: `8px solid ${C.amber}`,
                      borderRadius: "0 16px 16px 0",
                      padding: "16px 18px",
                    }}
                  >
                    <p style={{ fontSize: 12.5, color: C.text, margin: 0, lineHeight: 1.5 }}>
                      <strong>Sem dados para esse curso nessa cota:</strong> a cota selecionada não
                      teve nota de corte publicada para o curso escolhido nesse triênio. Tente outra
                      cota ou o Sistema Universal.
                    </p>
                  </div>
                )}

                {result.curso_alvo_result && evolutionData && (
                  <CorteTendenciaCard
                    curso={result.curso_alvo_result.curso}
                    campus={result.curso_alvo_result.campus}
                    turno={result.curso_alvo_result.turno}
                    semestre={result.curso_alvo_result.semestre}
                    data={evolutionData}
                    highlightTrienio={result.trienio_ref || trienio}
                  />
                )}

                {result.curso_alvo_result && chamadasData && (
                  <ChamadasHistoricoTable chamadas={chamadasData} />
                )}

                <TopCursosTable cursos={result.top_cursos} isLoggedIn={isLoggedIn} />
              </div>
            )}

          </div>
        </div>
      </div>
    </>
  );
}
