"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { CurvaGaussiana } from "@/components/brand/CurvaGaussiana";
import { PublicHeader } from "@/components/public/PublicHeader";
import {
  Television,
  Lightbulb,
  Wrench,
  Megaphone,
  YoutubeLogo,
  CaretRight,
} from "@phosphor-icons/react";

// ---------------------------------------------------------------------------
// Seção: Aprovações Reais — dados e subcomponentes
// ---------------------------------------------------------------------------
interface ProofPerson {
  /** Pseudônimo. Nome real de Aluno nunca entra aqui — ver `selecao-casos-2023-2025.md` §7. */
  name: string;
  school: string;
  curso: string;
  pas1: number;
  pas2: number;
  previsto: number;
  real: number;
  chancePercent: number;
  aprovado: boolean;
  /**
   * Nota de Corte usada para calcular `chancePercent`: a do triênio ANTERIOR, que é a
   * última publicada antes da Etapa 3 do Aluno. A do próprio triênio sai no mesmo Edital
   * que o resultado — usá-la seria dar ao modelo uma informação do futuro.
   * Mesma regra do runtime: `gestao_service._build_cutoff_maps`.
   */
  corteRef: number;
  corteRefTrienio: string;
  theme: {
    bgGradient: string;
    headerBadgeBg: string;
    headerBadgeText: string;
    nameColor: string;
    schoolColor: string;
    cardBg: string;
    cardBorder: string;
    labelColor: string;
    numberColor: string;
    splitPrevBg: string;
    splitRealBg: string;
    gaugeTrack: string;
    gaugeStroke: string;
    gaugeText: string;
    statusBg: string;
    statusText: string;
    statusBorder: string;
  };
}

// Aprovados de verdade na ampla concorrência (Sistema Universal) do triênio 2023/2025, que o
// modelo nunca viu no treino. A chance de cada um foi calculada contra a Nota de Corte de
// 2022/2024 — a última publicada antes da Etapa 3 deles. Seleção e critério em
// `data/prova-do-modelo/selecao-casos-2023-2025.md` §2 (gitignored, contém a inscrição real).
const APROVACOES_REAIS: ProofPerson[] = [
  {
    name: "Aluno A",
    school: "PAS 2023-2025",
    curso: "MEDICINA",
    pas1: 70.0,
    pas2: 86.5,
    previsto: 164.02,
    real: 161.93,
    chancePercent: 58,
    aprovado: true,
    corteRef: 161.14,
    corteRefTrienio: "2022-2024",
    theme: {
      bgGradient: "linear-gradient(150deg, #00AEEF 0%, #0072B8 100%)",
      headerBadgeBg: "rgba(0, 33, 71, 0.25)",
      headerBadgeText: "#002147",
      nameColor: "#002147",
      schoolColor: "rgba(0, 33, 71, 0.8)",
      cardBg: "rgba(255, 255, 255, 0.95)",
      cardBorder: "rgba(255, 255, 255, 0.4)",
      labelColor: "#4A5568",
      numberColor: "#002147",
      splitPrevBg: "#F8F9FA",
      splitRealBg: "rgba(0, 174, 239, 0.12)",
      gaugeTrack: "#E2E8F0",
      gaugeStroke: "#0072B8",
      gaugeText: "#002147",
      statusBg: "#002147",
      statusText: "#FFFFFF",
      statusBorder: "rgba(0, 33, 71, 0.2)",
    },
  },
  {
    name: "Aluno B",
    school: "PAS 2023-2025",
    curso: "DIREITO DIURNO",
    pas1: 53.1,
    pas2: 76.1,
    previsto: 117.6,
    real: 118.18,
    chancePercent: 80,
    aprovado: true,
    corteRef: 105.1,
    corteRefTrienio: "2022-2024",
    theme: {
      bgGradient: "linear-gradient(150deg, #002147 0%, #001024 100%)",
      headerBadgeBg: "rgba(0, 174, 239, 0.2)",
      headerBadgeText: "#7FD8F7",
      nameColor: "#FFFFFF",
      schoolColor: "rgba(255, 255, 255, 0.7)",
      cardBg: "rgba(255, 255, 255, 0.08)",
      cardBorder: "rgba(255, 255, 255, 0.12)",
      labelColor: "#94A3B8",
      numberColor: "#FFFFFF",
      splitPrevBg: "rgba(255, 255, 255, 0.04)",
      splitRealBg: "rgba(0, 174, 239, 0.2)",
      gaugeTrack: "rgba(255, 255, 255, 0.15)",
      gaugeStroke: "#00AEEF",
      gaugeText: "#FFFFFF",
      statusBg: "#00AEEF",
      statusText: "#002147",
      statusBorder: "rgba(0, 174, 239, 0.4)",
    },
  },
  {
    name: "Aluno C",
    school: "PAS 2023-2025",
    curso: "DIREITO NOTURNO",
    pas1: 51.0,
    pas2: 63.8,
    previsto: 92.38,
    real: 91.87,
    chancePercent: 88,
    aprovado: true,
    corteRef: 74.72,
    corteRefTrienio: "2022-2024",
    theme: {
      bgGradient: "linear-gradient(150deg, #00843D 0%, #005626 100%)",
      headerBadgeBg: "rgba(0, 0, 0, 0.25)",
      headerBadgeText: "#FFFFFF",
      nameColor: "#FFFFFF",
      schoolColor: "rgba(255, 255, 255, 0.8)",
      cardBg: "rgba(255, 255, 255, 0.95)",
      cardBorder: "rgba(255, 255, 255, 0.3)",
      labelColor: "#4A5568",
      numberColor: "#002147",
      splitPrevBg: "#F8F9FA",
      splitRealBg: "rgba(0, 132, 61, 0.12)",
      gaugeTrack: "#E2E8F0",
      gaugeStroke: "#00843D",
      gaugeText: "#002147",
      statusBg: "#00843D",
      statusText: "#FFFFFF",
      statusBorder: "rgba(255, 255, 255, 0.3)",
    },
  },
  {
    name: "Aluno D",
    school: "PAS 2023-2025",
    curso: "CIÊNCIA DA COMPUTAÇÃO",
    pas1: 62.9,
    pas2: 63.8,
    previsto: 104.11,
    real: 103.49,
    chancePercent: 67,
    aprovado: true,
    corteRef: 97.63,
    corteRefTrienio: "2022-2024",
    theme: {
      bgGradient: "linear-gradient(150deg, #00AEEF 0%, #0072B8 100%)",
      headerBadgeBg: "rgba(0, 33, 71, 0.25)",
      headerBadgeText: "#002147",
      nameColor: "#002147",
      schoolColor: "rgba(0, 33, 71, 0.8)",
      cardBg: "rgba(255, 255, 255, 0.95)",
      cardBorder: "rgba(255, 255, 255, 0.4)",
      labelColor: "#4A5568",
      numberColor: "#002147",
      splitPrevBg: "#F8F9FA",
      splitRealBg: "rgba(0, 174, 239, 0.12)",
      gaugeTrack: "#E2E8F0",
      gaugeStroke: "#0072B8",
      gaugeText: "#002147",
      statusBg: "#002147",
      statusText: "#FFFFFF",
      statusBorder: "rgba(0, 33, 71, 0.2)",
    },
  },
  {
    name: "Aluno E",
    school: "PAS 2023-2025",
    curso: "ENGENHARIA DE COMPUTAÇÃO",
    pas1: 57.2,
    pas2: 65.1,
    previsto: 99.87,
    real: 101.73,
    chancePercent: 85,
    aprovado: true,
    corteRef: 84.48,
    corteRefTrienio: "2022-2024",
    theme: {
      bgGradient: "linear-gradient(150deg, #002147 0%, #001024 100%)",
      headerBadgeBg: "rgba(0, 174, 239, 0.2)",
      headerBadgeText: "#7FD8F7",
      nameColor: "#FFFFFF",
      schoolColor: "rgba(255, 255, 255, 0.7)",
      cardBg: "rgba(255, 255, 255, 0.08)",
      cardBorder: "rgba(255, 255, 255, 0.12)",
      labelColor: "#94A3B8",
      numberColor: "#FFFFFF",
      splitPrevBg: "rgba(255, 255, 255, 0.04)",
      splitRealBg: "rgba(0, 174, 239, 0.2)",
      gaugeTrack: "rgba(255, 255, 255, 0.15)",
      gaugeStroke: "#00AEEF",
      gaugeText: "#FFFFFF",
      statusBg: "#00AEEF",
      statusText: "#002147",
      statusBorder: "rgba(0, 174, 239, 0.4)",
    },
  },
];

/** Quão perto a previsão chegou. Verde/vermelho vai pela MAGNITUDE do erro, não pelo sinal:
 *  prever 92,4 para quem tirou 91,9 é um acerto, e pintar de vermelho por ter passado 0,5 ponto
 *  para cima contradiz o que a seção está tentando mostrar. 3 pontos de Argumento Final é a
 *  faixa que o `selecao-casos-2023-2025.md` §3 usa como "chegou perto". */
const PROOF_DELTA_OK = 3;

function ProofDeltaBadge({ previsto, real }: { previsto: number; real: number }) {
  const delta = (real - previsto).toFixed(1);
  const positive = Math.abs(real - previsto) <= PROOF_DELTA_OK;
  return (
    <span
      className="inline-block font-mono text-[0.7rem] font-bold px-2 py-0.5 rounded-md"
      style={{
        background: positive ? "#00843D22" : "#EF444422",
        color: positive ? "#00843D" : "#EF4444",
      }}
    >
      {positive ? "+" : ""}{delta}
    </span>
  );
}

function ProofChanceGauge({
  percent,
  strokeColor,
  trackColor,
  textColor,
  size = 60,
}: {
  percent: number;
  strokeColor: string;
  trackColor: string;
  textColor: string;
  size?: number;
}) {
  const strokeWidth = size * 0.09;
  const radius = size / 2 - strokeWidth;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (percent / 100) * circ;
  const center = size / 2;

  return (
    <div className="flex items-center gap-3">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={center} cy={center} r={radius} fill="none" stroke={trackColor} strokeWidth={strokeWidth} />
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${center} ${center})`}
          style={{ transition: "stroke-dashoffset 0.9s ease" }}
        />
        <text
          x={center}
          y={center + size * 0.045}
          textAnchor="middle"
          fontSize={size * 0.24}
          fontWeight={800}
          fill={textColor}
          fontFamily="monospace"
        >
          {percent}%
        </text>
      </svg>
      <p className="text-[0.58rem] font-mono uppercase tracking-wider font-bold leading-tight" style={{ color: textColor }}>
        Chance<br />Prevista
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dados das ferramentas (para a seção 4)
// ---------------------------------------------------------------------------
const FERRAMENTAS = [
  {
    numero: "01",
    titulo: "Preditor PAS 3",
    href: "/predict",
    descricao:
      "Insira suas notas do PAS 1 e 2. Nosso modelo prevê seu Argumento Final e sua probabilidade percentual de aprovação no curso que você quer.",
    acento: {
      borderTop: "border-t-[#00AEEF]",
      text: "text-[#00AEEF]",
      chip: "bg-[#00AEEF]/10",
      hoverText: "group-hover:text-[#00AEEF]",
      hoverBorder: "hover:border-[#00AEEF]/20",
    },
  },
  {
    numero: "02",
    titulo: 'Calculadora "Quanto Falta"',
    href: "/calculadora",
    descricao:
      "Cálculo reverso: saiba exatamente qual Escore Bruto você precisa tirar na prova do PAS 3 para alcançar a nota de corte do seu curso, e quantos itens são necessários para essa nota.",
    acento: {
      borderTop: "border-t-[#00843D]",
      text: "text-[#00843D]",
      chip: "bg-[#00843D]/10",
      hoverText: "group-hover:text-[#00843D]",
      hoverBorder: "hover:border-[#00843D]/20",
    },
  },
  {
    numero: "03",
    titulo: "Análise Histórica",
    href: "/temporal",
    descricao:
      "Evolução histórica das notas de corte por curso e estatísticas médias das edições anteriores do PAS/UnB.",
    acento: {
      borderTop: "border-t-[#002147]",
      text: "text-[#002147]",
      chip: "bg-[#002147]/5",
      hoverText: "group-hover:text-[#002147]",
      hoverBorder: "hover:border-[#002147]/20",
    },
  },
];

// ---------------------------------------------------------------------------
// Dados da seção Como Funciona
// ---------------------------------------------------------------------------
const PASSOS = [
  {
    numero: "01",
    titulo: "Aprendeu com quem já fez a prova",
    descricao:
      "Estudamos os boletins de mais de 60 mil alunos que já fizeram o PAS nos últimos 8 anos — quase todo mundo que prestou a prova desde 2016. É nesse histórico real que o sistema aprendeu a prever a terceira etapa, não em teoria.",
    acento: {
      borderTop: "border-t-[#00AEEF]",
      text: "text-[#00AEEF]",
      chip: "bg-[#00AEEF]/10",
      hoverText: "group-hover:text-[#00AEEF]",
    },
  },
  {
    numero: "02",
    titulo: "Testado sem colar",
    descricao:
      "Pegamos os 8.703 alunos que fizeram o PAS 3 mais recente, escondemos o resultado de cada um e pedimos para o sistema prever a nota de todos sem deixar ele ver a resposta. Só depois comparamos a previsão com o que realmente aconteceu.",
    acento: {
      borderTop: "border-t-[#00843D]",
      text: "text-[#00843D]",
      chip: "bg-[#00843D]/10",
      hoverText: "group-hover:text-[#00843D]",
    },
  },
  {
    numero: "03",
    titulo: "O resultado, sem filtro",
    descricao:
      "Nessa prova cega, entre os 7.449 alunos cujo curso-alvo tinha nota de corte publicada, o sistema errou sobre passar ou não passar em 5,4% dos casos — ou seja, acertou o veredito em mais de 94 de cada 100. É esse número, medido contra gente de verdade, que aparece na sua tela.",
    acento: {
      borderTop: "border-t-[#002147]",
      text: "text-[#002147]",
      chip: "bg-[#002147]/5",
      hoverText: "group-hover:text-[#002147]",
    },
  },
];

// ---------------------------------------------------------------------------
// Dados da seção B2B / Coordenação
// ---------------------------------------------------------------------------
const FEATURES_B2B = [
  {
    titulo: "Gestão de Ativos",
    descricao: "Semáforo de risco de cada aluno em relação ao curso-alvo.",
  },
  {
    titulo: "Escola vs. População",
    descricao: "Distribuição dos seus alunos contra todos os candidatos do PAS.",
  },
  {
    titulo: "Comparação entre grupos",
    descricao: "Turma A vs. turma B, manhã vs. tarde — qualquer recorte.",
  },
  {
    titulo: "Relatórios em PDF",
    descricao: "Documentos whitelabel com a marca da sua escola, prontos para a reunião de pais.",
  },
];

// ---------------------------------------------------------------------------
// Componente principal
// ---------------------------------------------------------------------------
export function LandingPage() {
  // Carrossel de Aprovações Reais
  const proofScrollRef = useRef<HTMLDivElement>(null);
  const [proofScrollState, setProofScrollState] = useState({ atStart: true, atEnd: false });

  const updateProofScrollState = () => {
    const el = proofScrollRef.current;
    if (!el) return;
    setProofScrollState({
      atStart: el.scrollLeft <= 4,
      atEnd: el.scrollLeft >= el.scrollWidth - el.clientWidth - 4,
    });
  };

  const scrollProofBy = (direction: 1 | -1) => {
    const el = proofScrollRef.current;
    if (!el) return;
    const cards = Array.from(el.children) as HTMLElement[];
    if (cards.length === 0) return;
    const currentIndex = cards.findIndex((c) => c.offsetLeft >= el.scrollLeft - 4);
    const fromIndex = currentIndex === -1 ? cards.length - 1 : currentIndex;
    const targetIndex = Math.min(Math.max(fromIndex + direction, 0), cards.length - 1);
    el.scrollTo({ left: cards[targetIndex].offsetLeft, behavior: "smooth" });
  };

  // Recalcular limites do carrossel ao montar / redimensionar
  useEffect(() => {
    updateProofScrollState();
    window.addEventListener("resize", updateProofScrollState);
    return () => window.removeEventListener("resize", updateProofScrollState);
  }, []);

  return (
    <div className="landing-root bg-[#F8F9FA] text-[#1D1D1F] min-h-screen selection:bg-[#00843D] selection:text-white font-sans antialiased overflow-x-clip">
      <PublicHeader />

      {/* ══════════════════════════════════════════════════════════════════
           Seção 1 + 2: Hero + Aprovações Reais
           Um único wrapper com gradiente compartilhado cobre as duas seções.
           O mask-image faz o degradê sumir gradualmente conforme desce —
           cheio na seção 1, a ~35% na seção 2, invisível antes da seção 3.
          ══════════════════════════════════════════════════════════════════ */}
      <div className="relative bg-white overflow-hidden">

        {/* Gradiente único — sistema de coordenadas compartilhado entre as 2 seções */}
        <div
          className="absolute inset-0 pointer-events-none z-0"
          style={{
            backgroundImage:
              "radial-gradient(ellipse at 82% 18%, #00843D 0%, transparent 60%), radial-gradient(ellipse at 18% 82%, #00AEEF 0%, transparent 60%)",
            maskImage:
              "linear-gradient(to bottom, black 0%, black 32%, rgba(0,0,0,0.45) 58%, rgba(0,0,0,0.12) 78%, transparent 92%)",
            WebkitMaskImage:
              "linear-gradient(to bottom, black 0%, black 32%, rgba(0,0,0,0.45) 58%, rgba(0,0,0,0.12) 78%, transparent 92%)",
          }}
        />

        {/* ============ SEÇÃO 1: HERO ============ */}
        <header className="relative z-10 text-[#002147] pt-16 pb-10 sm:pb-0 sm:pt-24">
          <div className="relative z-10 max-w-6xl mx-auto px-6">
            <div className="max-w-3xl">
              <span className="landing-reveal vp-eyebrow">
                Análise Preditiva · PAS/UnB
              </span>
              <h1
                className="landing-reveal font-heading text-4xl sm:text-6xl font-extrabold tracking-tight leading-[1.08] text-[#002147] mt-6 mb-6"
                style={{ animationDelay: "90ms" }}
              >
                Sua aprovação na
                <br />
                UnB, calculada
                <br />
                <span className="text-[#00843D] relative inline-block">
                  com precisão.
                  <span className="absolute bottom-1 left-0 w-full h-[4px] bg-[#00AEEF]/40 rounded-full" />
                </span>
              </h1>
              <p
                className="landing-reveal text-lg sm:text-xl text-[#4A5568] leading-relaxed max-w-xl mb-9"
                style={{ animationDelay: "180ms" }}
              >
                O Vetor PAS combina IA e dados oficiais do Cebraspe para prever seu Argumento Final e calcular a chance real de você passar no seu curso no PAS 3.
              </p>
              <div
                className="landing-reveal flex flex-col sm:flex-row gap-3"
                style={{ animationDelay: "270ms" }}
              >
                <Link href="/predict" className="group vp-btn vp-btn-cyan px-7 py-3.5 text-base">
                  Calcular minha previsão
                  <span className="transition-transform group-hover:translate-x-1">→</span>
                </Link>
                <Link href="/temporal" className="vp-btn vp-btn-ghost px-7 py-3.5 text-base">
                  Explorar dados históricos
                </Link>
              </div>


            </div>
          </div>
        </header>

        {/* A curva é o produto: distribuição do argumento previsto vs. nota de corte */}
        <div className="relative z-0 mt-4 sm:-mt-16">
          <CurvaGaussiana />
        </div>

        {/* ============ SEÇÃO 2: APROVAÇÕES REAIS ============ */}
        <section
          id="aprovacoes"
          className="py-20 sm:py-24 scroll-mt-20 relative z-10"
        >
          <div className="relative z-10 max-w-6xl mx-auto px-6">
            {/* Cabeçalho */}
            <div className="max-w-3xl mb-14 space-y-4">
              <span className="inline-flex items-center gap-1.5 font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00843D] bg-[#00843D]/8 border border-[#00843D]/20 px-3 py-1.5 rounded-lg font-bold">
                Aprovações Reais
              </span>
              <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-[#002147]">
                Antes do gabarito, a previsão já estava feita.
              </h2>
              <p className="text-base sm:text-lg text-[#4A5568] leading-relaxed">
                Antes do PAS 3, o Vetor PAS calculou o argumento final e a chance de aprovação de cada aluno.
                Veja a previsão lado a lado com o resultado oficial.
              </p>
              <p className="text-sm text-[#4A5568]/85 leading-relaxed">
                Todos aprovados na ampla concorrência, no triênio 2023-2025 — que o modelo nunca
                viu. A chance de cada um foi calculada contra a nota de corte de 2022-2024, a
                última publicada antes da prova. Nomes preservados por privacidade.
              </p>
            </div>
          </div>

          {/* Carrossel de cards */}
          <div className="relative z-10 max-w-6xl mx-auto px-6">
            <div className="relative rounded-2xl overflow-hidden border border-black/5 shadow-lg">
              <div
                ref={proofScrollRef}
                onScroll={updateProofScrollState}
                className="flex overflow-x-auto snap-x snap-mandatory scroll-smooth no-scrollbar"
              >
                {APROVACOES_REAIS.map((p, i) => (
                  <div
                    key={p.name}
                    className={`snap-start shrink-0 w-full sm:w-1/2 lg:w-1/3 relative flex flex-col justify-between p-8 sm:p-9 overflow-hidden ${
                      i > 0 ? "border-l border-white/10" : ""
                    }`}
                    style={{ background: p.theme.bgGradient }}
                  >
                    <div className="relative z-10 space-y-5">
                      {/* Cabeçalho editorial: curso (topo-esquerda) / aluno & escola (topo-direita) */}
                      <div
                        className="flex items-start justify-between gap-3 border-b pb-4"
                        style={{ borderColor: p.theme.cardBorder }}
                      >
                        <span
                          className="font-mono text-[0.7rem] font-black px-3 py-1 rounded-full uppercase tracking-wider shadow-sm"
                          style={{ background: p.theme.headerBadgeBg, color: p.theme.headerBadgeText }}
                        >
                          {p.curso}
                        </span>
                        <div className="text-right min-w-0">
                          <p className="font-bold text-base leading-tight truncate" style={{ color: p.theme.nameColor }}>
                            {p.name}
                          </p>
                          <p className="text-xs truncate" style={{ color: p.theme.schoolColor }}>
                            {p.school}
                          </p>
                        </div>
                      </div>

                      {/* Histórico PAS 1 + PAS 2 */}
                      <div>
                        <p
                          className="text-[0.62rem] font-mono uppercase tracking-wider font-bold mb-2"
                          style={{ color: p.theme.nameColor, opacity: 0.9 }}
                        >
                          Histórico de Notas
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                          <div
                            className="p-3 rounded-2xl text-center backdrop-blur-sm shadow-sm"
                            style={{ background: p.theme.cardBg, border: `1px solid ${p.theme.cardBorder}` }}
                          >
                            <p className="text-[0.58rem] font-mono uppercase font-bold" style={{ color: p.theme.labelColor }}>
                              PAS 1
                            </p>
                            <p className="text-xl font-black mt-0.5" style={{ color: p.theme.numberColor }}>
                              {p.pas1.toFixed(1)}
                            </p>
                          </div>
                          <div
                            className="p-3 rounded-2xl text-center backdrop-blur-sm shadow-sm"
                            style={{ background: p.theme.cardBg, border: `1px solid ${p.theme.cardBorder}` }}
                          >
                            <p className="text-[0.58rem] font-mono uppercase font-bold" style={{ color: p.theme.labelColor }}>
                              PAS 2
                            </p>
                            <p className="text-xl font-black mt-0.5" style={{ color: p.theme.numberColor }}>
                              {p.pas2.toFixed(1)}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Argumento Final (PAS 3) — Previsto vs Real Split */}
                      <div>
                        <p
                          className="text-[0.62rem] font-mono uppercase tracking-wider font-bold mb-2"
                          style={{ color: p.theme.nameColor, opacity: 0.9 }}
                        >
                          Argumento Final (PAS 3)
                        </p>
                        <div
                          className="rounded-2xl overflow-hidden backdrop-blur-sm shadow-sm"
                          style={{ background: p.theme.cardBg, border: `1px solid ${p.theme.cardBorder}` }}
                        >
                          <div className="grid grid-cols-2 divide-x" style={{ borderColor: p.theme.cardBorder }}>
                            <div className="p-3 text-center" style={{ background: p.theme.splitPrevBg }}>
                              <p className="text-[0.58rem] font-mono uppercase font-bold" style={{ color: p.theme.labelColor }}>
                                Previsto
                              </p>
                              <p className="text-2xl font-black mt-0.5 opacity-70" style={{ color: p.theme.numberColor }}>
                                {p.previsto.toFixed(1)}
                              </p>
                            </div>
                            <div className="p-3 text-center" style={{ background: p.theme.splitRealBg }}>
                              <p className="text-[0.58rem] font-mono uppercase font-bold" style={{ color: p.theme.labelColor }}>
                                Real
                              </p>
                              <p className="text-2xl font-black mt-0.5" style={{ color: p.theme.numberColor }}>
                                {p.real.toFixed(1)}
                              </p>
                            </div>
                          </div>
                          <div
                            className="px-3.5 py-1.5 border-t flex items-center justify-between"
                            style={{ borderColor: p.theme.cardBorder }}
                          >
                            <p className="text-[0.58rem] font-mono" style={{ color: p.theme.labelColor }}>
                              Δ erro do modelo
                            </p>
                            <ProofDeltaBadge previsto={p.previsto} real={p.real} />
                          </div>
                          {/* A chance foi calculada contra este corte — o do triênio anterior, o
                              único publicado antes da Etapa 3 do Aluno. Sem exibi-lo, o percentual
                              do medidor abaixo fica sem lastro. */}
                          <div
                            className="px-3.5 py-1.5 border-t flex items-center justify-between gap-2"
                            style={{ borderColor: p.theme.cardBorder }}
                          >
                            <p className="text-[0.58rem] font-mono" style={{ color: p.theme.labelColor }}>
                              Corte de referência{" "}
                              <span className="opacity-70">({p.corteRefTrienio})</span>
                            </p>
                            <p
                              className="text-[0.7rem] font-mono font-black tabular-nums"
                              style={{ color: p.theme.numberColor }}
                            >
                              {p.corteRef.toFixed(1)}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Barra Inferior: Chance Gauge (em destaque) & Badge de Aprovado */}
                    <div
                      className="relative z-10 pt-5 mt-5 border-t flex items-center justify-between gap-3"
                      style={{ borderColor: p.theme.cardBorder }}
                    >
                      <ProofChanceGauge
                        percent={p.chancePercent}
                        strokeColor={p.theme.gaugeStroke}
                        trackColor={p.theme.gaugeTrack}
                        textColor={p.theme.gaugeText}
                        size={92}
                      />
                      <div>
                        {p.aprovado && (
                          <span
                            className="inline-block font-bold text-sm px-3.5 py-1.5 rounded-xl shadow-sm"
                            style={{
                              background: p.theme.statusBg,
                              color: p.theme.statusText,
                              border: `1px solid ${p.theme.statusBorder}`,
                            }}
                          >
                            ✓ Aprovada(o)
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Setas de navegação do carrossel */}
              <button
                type="button"
                onClick={() => scrollProofBy(-1)}
                disabled={proofScrollState.atStart}
                aria-label="Aluno anterior"
                className="absolute left-3 top-1/2 -translate-y-1/2 z-20 w-9 h-9 rounded-full bg-black/40 hover:bg-black/60 backdrop-blur-sm flex items-center justify-center text-white transition-all disabled:opacity-0 disabled:pointer-events-none"
              >
                <CaretRight size={16} weight="bold" className="rotate-180" />
              </button>
              <button
                type="button"
                onClick={() => scrollProofBy(1)}
                disabled={proofScrollState.atEnd}
                aria-label="Próximo aluno"
                className="absolute right-3 top-1/2 -translate-y-1/2 z-20 w-9 h-9 rounded-full bg-black/40 hover:bg-black/60 backdrop-blur-sm flex items-center justify-center text-white transition-all disabled:opacity-0 disabled:pointer-events-none"
              >
                <CaretRight size={16} weight="bold" />
              </button>
            </div>
          </div>
        </section>

      </div>


      {/* ============ SEÇÃO 3: DE ALUNO PARA ALUNO ============ */}
      <section id="historia" className="py-24 bg-[#F8F9FA] border-b border-[#E2E8F0] scroll-mt-20">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-white border-l-8 border-[#00843D] rounded-r-3xl p-8 sm:p-12 relative shadow-[0_10px_35px_rgba(0,0,0,0.03)] border-y border-r border-black/5 hover:scale-[1.005] hover:shadow-[0_15px_45px_rgba(0,0,0,0.05)] transition-all duration-300">
            <div className="flex items-center gap-2 mb-4">
              <span className="font-mono text-xs text-[#00843D] uppercase tracking-[0.2em] font-bold">
                De Aluno para Aluno
              </span>
            </div>

            <h2 className="font-heading text-2xl sm:text-4xl font-extrabold mb-6 text-[#002147] leading-snug">
              &ldquo;Eu já estive no seu lugar no PAS 3, e sei como é a sensação de não saber se vai dar pra chegar na nota necessária.&rdquo;
            </h2>

            <div className="space-y-4 text-[#4A5568] leading-relaxed text-sm sm:text-base">
              <p>
                Quando eu estava estudando para o PAS 3, passei meses tentando adivinhar se era possível passar no curso que eu queria. A nota necessária eu sabia, mas qual era a chance de eu tirar aquela nota? Procurava na internet, mas nada me dava uma resposta real sobre a probabilidade de entrar no meu curso na UnB, se valia a pena tentar ele ou mudar a rota para garantir minha aprovação.
              </p>
              <p>
                Foi por isso que criei o <strong className="text-[#002147]">Vetor PAS</strong>: para que você não precise estudar no escuro. Usamos algoritmos de machine learning sobre os boletins oficiais do Cebraspe para te mostrar exatamente onde você está e o que precisa fazer na terceira etapa. Assim, você não precisa esperar a nota do PAS 3 sair para saber se tinha chance ou não.
              </p>
            </div>

            <div className="mt-10 pt-8 border-t border-black/5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full border-2 border-[#00843D]/40 shadow-inner overflow-hidden relative bg-[#F8F9FA] shrink-0">
                  <img
                    src="/luiz.jpeg"
                    alt="Luiz Moreira"
                    className="w-full h-full object-cover object-top"
                  />
                </div>
                <div>
                  <p className="text-sm font-bold text-[#002147]">Luiz Moreira</p>
                  <p className="text-xs text-[#718096]">Criador do Vetor PAS · Engenharia de Software (UnB)</p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <a
                  href="https://www.linkedin.com/in/luiz-henrique-tomaz-moreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-[#F8F9FA] hover:bg-[#00AEEF] hover:text-[#002147] active:bg-[#00AEEF] active:text-[#002147] text-[#4A5568] text-xs font-semibold transition-all border border-black/5 shadow-sm active:scale-95"
                >
                  <span>LinkedIn</span>
                  <span>↗</span>
                </a>
                <a
                  href="mailto:lhtmoreira@gmail.com"
                  className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-[#F8F9FA] hover:bg-[#00843D] hover:text-white active:bg-[#00843D] active:text-white text-[#4A5568] text-xs font-semibold transition-all border border-black/5 shadow-sm active:scale-95"
                >
                  <span>E-mail</span>
                  <span>✉</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ SEÇÃO 4: FERRAMENTAS ============ */}
      {/* Textos da main, mas com botão de link em cada card (como nextjs-frontend) */}
      <section id="ferramentas" className="bg-white py-20 sm:py-28 border-b border-[#E2E8F0] scroll-mt-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-14">
            <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-[#002147]">
              Três ferramentas criadas sob medida para o seu PAS 3.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {FERRAMENTAS.map((f) => (
              <div
                key={f.numero}
                className={`group relative bg-[#F8F9FA] border border-black/5 rounded-2xl p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all duration-300 border-t-4 ${f.acento.borderTop} ${f.acento.hoverBorder} flex flex-col`}
              >
                <span
                  className={`inline-block font-mono text-xs font-bold ${f.acento.text} ${f.acento.chip} px-3 py-1 rounded-md mb-6 transition-transform group-hover:scale-110 self-start`}
                >
                  {f.numero}
                </span>
                <h3 className={`font-heading text-xl font-bold mb-3 text-[#002147] ${f.acento.hoverText} transition-colors`}>
                  {f.titulo}
                </h3>
                <p className="text-sm text-[#4A5568] leading-relaxed flex-1">
                  {f.descricao}
                </p>
                <div className="mt-6 pt-4 border-t border-black/5">
                  <Link
                    href={f.href}
                    className={`inline-flex items-center gap-1.5 text-sm font-semibold ${f.acento.text} hover:underline transition-all group-hover:gap-2.5`}
                  >
                    Acessar ferramenta
                    <CaretRight size={14} weight="bold" />
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ============ SEÇÃO 5: COMO FUNCIONA ============ */}
      <section className="bg-[#F8F9FA] py-20 sm:py-24 border-b border-[#E2E8F0]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-14 space-y-4">
            <span className="vp-eyebrow">Como funciona</span>
            <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-[#002147]">
              Por que dá pra confiar nesse número.
            </h2>
            <p className="text-base sm:text-lg text-[#4A5568] leading-relaxed">
              Não pedimos para você acreditar de graça. Mostramos como testamos, com alunos reais,
              antes de colocar esse número na sua tela.
            </p>
          </div>
          <ol className="grid md:grid-cols-3 gap-6">
            {PASSOS.map((passo) => (
              <li
                key={passo.numero}
                className={`group vp-card vp-card-lift border-t-4 ${passo.acento.borderTop} p-8`}
              >
                <span
                  className={`inline-block font-mono text-xs font-bold ${passo.acento.text} ${passo.acento.chip} px-3 py-1 rounded-md mb-6 transition-transform group-hover:scale-110`}
                >
                  {passo.numero}
                </span>
                <h3
                  className={`font-heading text-xl font-bold mb-3 text-[#002147] transition-colors ${passo.acento.hoverText}`}
                >
                  {passo.titulo}
                </h3>
                <p className="text-sm text-[#4A5568] leading-relaxed">{passo.descricao}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ============ SEÇÃO 6: BUILD IN PUBLIC ============ */}
      <section id="build-in-public" className="bg-[#F8F9FA] py-20 sm:py-24 border-b border-[#E2E8F0] scroll-mt-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            {/* Texto */}
            <div className="lg:col-span-6 space-y-6">
              <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00AEEF] bg-[#00AEEF]/5 border border-[#00AEEF]/20 px-3.5 py-1.5 rounded-lg font-bold animate-unb-breathe-badge">
                Build in Public
              </span>
              <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-tight leading-[1.1] text-[#002147]">
                Criado em público. <br />
                Desenhado por <span className="text-[#00843D]">você</span>.
              </h2>
              <p className="text-base sm:text-lg text-[#4A5568] leading-relaxed">
                O Vetor PAS não está sendo criado a portas fechadas. Acreditamos que a melhor ferramenta para o PAS 3 é aquela construída em parceria com quem realmente vai usá-la em sua preparação.
              </p>
              <p className="text-sm sm:text-base text-[#718096] leading-relaxed">
                Estou desenvolvendo o produto em <strong>lives abertas de código</strong>, onde você pode acompanhar cada linha de programação, sugerir ideias de design, propor novas simulações de nota ou criticar o que não ficou legal.
              </p>
              <div className="pt-2">
                <a
                  href="https://www.youtube.com/@luizhtmoreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2.5 px-5 py-3 rounded-xl font-semibold text-white bg-[#002147] hover:bg-[#FF0000] hover:text-white transition-all shadow-sm text-sm group active:scale-95"
                >
                  <YoutubeLogo size={20} weight="fill" className="text-[#FF0000] group-hover:text-white transition-colors" />
                  <span>Acompanhar lives no YouTube</span>
                </a>
              </div>
            </div>

            {/* Grid de Cards */}
            <div className="lg:col-span-6 grid sm:grid-cols-2 gap-6">
              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div className="w-10 h-10 rounded-xl bg-[#00AEEF]/10 text-[#00AEEF] flex items-center justify-center">
                  <Television size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-[#002147]">Lives de Código</h3>
                <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
                  Acompanhe o desenvolvimento da interface ao vivo.
                </p>
              </div>

              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div className="w-10 h-10 rounded-xl bg-[#00843D]/10 text-[#00843D] flex items-center justify-center">
                  <Lightbulb size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-[#002147]">Opine & Sugira</h3>
                <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
                  Features, designs, cores: você ajuda decidir o rumo das próximas telas.
                </p>
              </div>

              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div className="w-10 h-10 rounded-xl bg-[#002147]/5 text-[#002147] flex items-center justify-center">
                  <Wrench size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-[#002147]">Co-Criação</h3>
                <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
                  Ideal para quem quer aprender arquitetura web e ver decisões de engenharia na prática.
                </p>
              </div>

              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div className="w-10 h-10 rounded-xl bg-[#7FD8F7]/20 text-[#00AEEF] flex items-center justify-center">
                  <Megaphone size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-[#002147]">Feito para Você</h3>
                <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
                  Ao participar das sugestões, você garante que a ferramenta resolverá suas reais dúvidas do PAS 3.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ SEÇÃO 7: COORDENAÇÃO / B2B ============ */}
      <section className="bg-[#F8F9FA] py-20 sm:py-24 border-b border-[#E2E8F0]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="vp-banner py-14 px-8 sm:px-12 text-white">
            <div className="relative z-10 grid lg:grid-cols-[1fr_1.2fr] gap-12 lg:gap-16 items-start">
              <div className="space-y-5">
                <span className="vp-eyebrow vp-eyebrow-on-dark">Para escolas parceiras</span>
                <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight leading-tight">
                  A coordenação enxerga a turma inteira de uma vez.
                </h2>
                <p className="text-base text-white/80 leading-relaxed">
                  O painel B2B transforma os escores dos seus alunos em um mapa de
                  risco acionável — quem está seguro, quem precisa de reforço e
                  onde a escola se posiciona contra toda a população do PAS.
                </p>
                <div className="pt-2">
                  <Link href="/auth/login" className="vp-btn vp-btn-cyan px-7 py-3.5 text-base">
                    Acessar painel da escola →
                  </Link>
                </div>
              </div>
              <ul className="grid sm:grid-cols-2 gap-4">
                {FEATURES_B2B.map((feature) => (
                  <li
                    key={feature.titulo}
                    className="rounded-2xl bg-white/8 border border-white/12 backdrop-blur-sm p-6 space-y-1.5 transition-all duration-300 hover:bg-white/12 hover:scale-[1.02]"
                  >
                    <h3 className="font-heading font-bold text-lg tracking-tight">
                      {feature.titulo}
                    </h3>
                    <p className="text-[0.9rem] leading-relaxed text-white/70">
                      {feature.descricao}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="bg-[#001730] text-white/50 py-12 text-xs">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-6 text-center sm:text-left">
          <span>
            © {new Date().getFullYear()} Vetor PAS — projeto independente, sem
            vínculo com a UnB ou o Cebraspe.
          </span>
          <div className="flex items-center gap-5">
            <Link href="/predict" className="text-[#00AEEF] hover:underline font-medium transition-all">
              Preditor
            </Link>
            <Link href="/temporal" className="text-[#7FD8F7] hover:underline font-medium transition-all">
              Análise Temporal
            </Link>
            <Link href="/auth/login" className="text-white/70 hover:text-white transition-all">
              Coordenadores
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
