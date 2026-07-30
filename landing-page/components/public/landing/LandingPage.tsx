"use client";

import { useState, useRef, useEffect } from "react";
import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "./WaitlistForm";
import { Television, Lightbulb, Wrench, Megaphone, YoutubeLogo, CaretRight, ChartBar, Info, List, X } from "@phosphor-icons/react";

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


// Dados reais do último vestibular UnB (Triênio 2023-2025) extraídos do CSV oficial
// Os 10 cursos com maior nota de corte (na ampla) na 1ª chamada, 2º semestre (único do triênio no arquivo)
interface CorteData {
  curso: string;
  universal: { min: number; max: number; media: number };
  negros: { min: number; max: number; media: number } | null;
}

const NOTAS_CORTE_REAIS: CorteData[] = [
  {
    curso: "MEDICINA (BACHARELADO)",
    universal: { min: 144.827, max: 160.522, media: 146.192 },
    negros: { min: 119.781, max: 119.781, media: 119.781 },
  },
  {
    curso: "DIREITO (BACHARELADO)",
    universal: { min: 118.498, max: 128.709, media: 123.441 },
    negros: { min: 79.032, max: 80.426, media: 79.729 },
  },
  {
    curso: "CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)",
    universal: { min: 106.717, max: 119.192, media: 112.014 },
    negros: { min: 98.223, max: 98.223, media: 98.223 },
  },
  {
    curso: "ENGENHARIA DE COMPUTAÇÃO (BACHARELADO)",
    universal: { min: 94.432, max: 107.201, media: 100.656 },
    negros: { min: 26.875, max: 26.875, media: 26.875 },
  },
  {
    curso: "DIREITO (BACHARELADO) (NOTURNO)",
    universal: { min: 88.544, max: 100.803, media: 93.77 },
    negros: { min: 59.461, max: 62.301, media: 60.881 },
  },
  {
    curso: "PSICOLOGIA (BACHARELADO/LICENCIATURA/PSICÓLOGO)",
    universal: { min: 74.392, max: 86.809, media: 80.491 },
    negros: { min: 62.655, max: 62.655, media: 62.655 },
  },
  {
    curso: "CIÊNCIAS ECONÔMICAS (BACHARELADO)",
    universal: { min: 69.907, max: 85.06, media: 77.25 },
    negros: { min: 56.92, max: 56.92, media: 56.92 },
  },
  {
    curso: "ODONTOLOGIA (BACHARELADO)",
    universal: { min: 69.79, max: 77.937, media: 73.918 },
    negros: { min: 22.06, max: 22.06, media: 22.06 },
  },
  {
    curso: "ARQUITETURA E URBANISMO (BACHARELADO)",
    universal: { min: 63.904, max: 79.454, media: 68.239 },
    negros: { min: 22.14, max: 22.14, media: 22.14 },
  },
  {
    curso: "RELAÇÕES INTERNACIONAIS (BACHARELADO)",
    universal: { min: 63.729, max: 81.165, media: 74.524 },
    negros: { min: 33.835, max: 33.835, media: 33.835 },
  },
];

export function LandingPage() {
  // --- Estados do Preview de Notas de Corte ---
  const [selectedCourse, setSelectedCourse] = useState<CorteData>(NOTAS_CORTE_REAIS[0]);
  const [systemType, setSystemType] = useState<"universal" | "negros">("universal");

  // Menu Hambúrguer Responsivo
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Dropdown customizado para Escolha o Curso
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [dropdownSearch, setDropdownSearch] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  const filteredCourses = NOTAS_CORTE_REAIS.filter(c =>
    c.curso.toLowerCase().includes(dropdownSearch.toLowerCase())
  );

  // Fechar dropdown ao clicar fora
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Recalcular limites do carrossel de Aprovações Reais ao montar / redimensionar
  useEffect(() => {
    updateProofScrollState();
    window.addEventListener("resize", updateProofScrollState);
    return () => window.removeEventListener("resize", updateProofScrollState);
  }, []);

  const handleCourseSelect = (course: CorteData) => {
    setSelectedCourse(course);
    setIsDropdownOpen(false);
    setDropdownSearch("");
    if (!course.negros && systemType === "negros") {
      setSystemType("universal");
    }
  };

  // Acessar dados da opção atual
  const activeMetrics = systemType === "universal" ? selectedCourse.universal : selectedCourse.negros;

  const scrollToForm = (e: React.MouseEvent) => {
    e.preventDefault();
    const formElement = document.getElementById("lista-espera");
    if (formElement) {
      formElement.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="landing-root bg-[#F8F9FA] text-[#1D1D1F] min-h-screen selection:bg-[#00843D] selection:text-white font-sans antialiased overflow-x-clip">
      
      {/* ============ STICKY NAVBAR ============ */}
      <nav className="sticky top-0 z-50 w-full backdrop-blur-md bg-white/95 border-b border-[#E2E8F0] transition-all duration-300">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4 relative">
          <BrandMark light={false} />
          
          {/* Desktop Menu Links */}
          <div className="hidden md:flex items-center gap-8 text-sm font-semibold text-[#002147]">
            <a
              href="#aprovacoes"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("aprovacoes")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00843D] transition-colors cursor-pointer"
            >
              Aprovações
            </a>
            {/* Notas de Corte — seção desativada temporariamente, ver bloco {false && (...)} abaixo */}
            {false && (
              <a
                href="#corte-preview"
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById("corte-preview")?.scrollIntoView({ behavior: "smooth" });
                }}
                className="hover:text-[#00843D] transition-colors cursor-pointer"
              >
                Notas de Corte
              </a>
            )}
            <a
              href="#historia"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("historia")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00843D] transition-colors cursor-pointer"
            >
              Minha História
            </a>
            <a
              href="#ferramentas"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("ferramentas")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00843D] transition-colors cursor-pointer"
            >
              Ferramentas
            </a>
            <a
              href="#build-in-public"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("build-in-public")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00843D] transition-colors cursor-pointer"
            >
              Build in Public
            </a>
          </div>

          {/* Mobile Hamburger Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 text-[#002147] hover:bg-black/5 rounded-xl active:scale-95 transition-all cursor-pointer"
            aria-label="Alternar menu"
          >
            {isMobileMenuOpen ? <X size={24} weight="bold" /> : <List size={24} weight="bold" />}
          </button>

          {/* Mobile Menu Dropdown Panel */}
          {isMobileMenuOpen && (
            <div className="md:hidden absolute top-full left-0 right-0 bg-white border-b border-[#E2E8F0] shadow-lg py-4 px-6 flex flex-col font-semibold text-[#002147] animate-in fade-in slide-in-from-top-1 duration-150 z-50">
              <a
                href="#aprovacoes"
                onClick={(e) => {
                  e.preventDefault();
                  setIsMobileMenuOpen(false);
                  document.getElementById("aprovacoes")?.scrollIntoView({ behavior: "smooth" });
                }}
                className="hover:text-[#00843D] transition-colors py-3 border-b border-black/5 cursor-pointer text-sm"
              >
                Aprovações
              </a>
              {false && (
                <a
                  href="#corte-preview"
                  onClick={(e) => {
                    e.preventDefault();
                    setIsMobileMenuOpen(false);
                    document.getElementById("corte-preview")?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className="hover:text-[#00843D] transition-colors py-3 border-b border-black/5 cursor-pointer text-sm"
                >
                  Notas de Corte
                </a>
              )}
              <a
                href="#historia"
                onClick={(e) => {
                  e.preventDefault();
                  setIsMobileMenuOpen(false);
                  document.getElementById("historia")?.scrollIntoView({ behavior: "smooth" });
                }}
                className="hover:text-[#00843D] transition-colors py-3 border-b border-black/5 cursor-pointer text-sm"
              >
                Minha História
              </a>
              <a
                href="#ferramentas"
                onClick={(e) => {
                  e.preventDefault();
                  setIsMobileMenuOpen(false);
                  document.getElementById("ferramentas")?.scrollIntoView({ behavior: "smooth" });
                }}
                className="hover:text-[#00843D] transition-colors py-3 border-b border-black/5 cursor-pointer text-sm"
              >
                Ferramentas
              </a>
              <a
                href="#build-in-public"
                onClick={(e) => {
                  e.preventDefault();
                  setIsMobileMenuOpen(false);
                  document.getElementById("build-in-public")?.scrollIntoView({ behavior: "smooth" });
                }}
                className="hover:text-[#00843D] transition-colors py-3 cursor-pointer text-sm"
              >
                Build in Public
              </a>
            </div>
          )}
        </div>
      </nav>

      {/* ══════════════════════════════════════════════════════════════════
           Wrapper compartilhado: Hero + Aprovações
           Um único elemento de gradiente cobre ambas as seções.
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

        {/* ============ HERO & HEADER ============ */}
        <header className="relative z-20 text-[#002147] pt-16 pb-24">
          <div className="relative z-10 max-w-6xl mx-auto px-6">
            <div className="grid lg:grid-cols-12 gap-12 items-center">
              {/* Texto de Impacto */}
              <div className="lg:col-span-7 space-y-6">
                <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00843D] bg-[#00843D]/5 border border-[#00843D]/25 px-3.5 py-1.5 rounded-lg font-bold">
                  Análise Preditiva · PAS/UnB
                </span>
                <h1 className="font-heading text-4xl sm:text-6xl font-extrabold tracking-tight leading-[1.08] text-[#002147]">
                  Sua aprovação na UnB, calculada{" "}
                  <br />
                  <span className="text-[#00843D] relative inline-block">
                    com precisão.
                    <span className="absolute bottom-1 left-0 w-full h-[4px] bg-[#00AEEF]/40 rounded-full" />
                  </span>
                </h1>
                <p className="text-lg sm:text-xl text-[#4A5568] leading-relaxed max-w-xl">
                  O Vetor PAS combina IA e dados oficiais do Cebraspe para prever seu Argumento Final e calcular a chance real de você passar no seu curso no PAS 3.
                </p>
              </div>

              {/* Card do Formulário da Lista de Espera */}
              <div id="lista-espera" className="lg:col-span-5 scroll-mt-24">
                <div className="bg-[#002147] border border-black/10 p-1.5 rounded-3xl shadow-[0_20px_50px_rgba(0,33,71,0.15)] text-white hover:scale-[1.01] transition-transform duration-300">
                  <WaitlistForm variantStyle="card" buttonText="Garantir Meu Acesso Antecipado" />
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* ============ APROVAÇÕES REAIS ============ */}
        <section
          id="aprovacoes"
          className="py-20 sm:py-24 scroll-mt-20 relative z-10"
        >
          <div className="relative z-10 max-w-6xl mx-auto px-6">
            {/* Cabeçalho — mesmo padrão tipográfico de todas as seções */}
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

          {/* Carrossel de cards — bloco de cor sólido por aluno, borda a borda entre si, contido na mesma largura das demais seções */}
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

        {/* Borda de separação para seção 3 */}
        <div className="border-t border-[#E2E8F0]" />
      </div>


      {/* ============ NOTAS DE CORTE PREVIEW (desativada temporariamente — pode voltar) ============ */}
      {false && (
      <section id="corte-preview" className="py-20 sm:py-24 bg-white border-b border-[#E2E8F0] scroll-mt-20 relative">
        <div className="max-w-6xl mx-auto px-6 relative z-10">
          
          <div className="max-w-3xl mb-14 space-y-4">
            <span className="inline-flex items-center gap-1.5 font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00AEEF] bg-[#00AEEF]/10 border border-[#00AEEF]/20 px-3 py-1.5 rounded-lg font-bold">
              <ChartBar size={14} />
              Notas de Corte PAS/UnB (Triênio 2023-2025)
            </span>
            <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-[#002147]">
              Veja a nota real de cada curso.
            </h2>
            <p className="text-base sm:text-lg text-[#4A5568] leading-relaxed">
              Explore abaixo as notas máximas, médias e os cortes reais no último vestibular. Selecione o curso e o sistema de cotas para ver a distribuição gráfica.
            </p>
          </div>

          <div className="grid lg:grid-cols-12 gap-8 items-start">
            
            {/* Coluna Esquerda: Filtros (Compacta) */}
            <div className="lg:col-span-5 bg-[#F8F9FA] border border-black/5 p-5 sm:p-6 rounded-3xl flex flex-col space-y-5">
              
              {/* Curso Selecionado (Dropdown Customizado Searchable) */}
              <div className="space-y-1.5 relative" ref={dropdownRef}>
                <label className="block text-xs font-bold uppercase tracking-wider text-[#002147] font-mono">1. Escolha o Curso</label>
                
                <div
                  onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                  className="w-full px-4 py-3 rounded-xl border border-black/15 bg-white text-[#002147] cursor-pointer flex items-center justify-between hover:border-[#00843D] transition-all text-sm select-none font-medium min-w-0"
                >
                  <span className="truncate min-w-0">
                    {selectedCourse.curso.replace(" (BACHARELADO)", "").replace(" (BACHARELADO/LICENCIATURA/PSICÓLOGO)", "")}
                  </span>
                  <span className="text-[#002147]/50 text-xs transition-transform duration-200 shrink-0" style={{ transform: isDropdownOpen ? "rotate(180deg)" : "rotate(0deg)" }}>
                    ▼
                  </span>
                </div>

                {isDropdownOpen && (
                  <div className="absolute left-0 right-0 top-full mt-1.5 bg-[#001730] border border-[#00843D]/55 rounded-xl shadow-[0_15px_30px_rgba(0,0,0,0.35)] z-50 overflow-hidden">
                    <div className="p-2 border-b border-white/10 bg-[#001024]">
                      <input
                        type="text"
                        autoFocus
                        value={dropdownSearch}
                        onChange={(e) => setDropdownSearch(e.target.value)}
                        placeholder="Buscar curso (ex: Medicina, Direito)..."
                        className="w-full px-3 py-2 rounded-lg bg-white/10 border border-white/15 text-white placeholder-white/40 text-base focus:outline-none focus:border-[#00843D]"
                      />
                    </div>
                    <div className="max-h-56 overflow-y-auto divide-y divide-white/5 text-sm overscroll-contain">
                      {filteredCourses.length > 0 ? (
                        filteredCourses.map((c) => (
                          <div
                            key={c.curso}
                            onClick={() => handleCourseSelect(c)}
                            className={`px-4 py-2.5 cursor-pointer transition-colors text-xs sm:text-sm leading-snug ${
                              selectedCourse.curso === c.curso
                                ? "bg-[#00843D] text-white font-bold"
                                : "text-white/85 hover:bg-white/10 hover:text-white"
                            }`}
                          >
                            {c.curso.replace(" (BACHARELADO)", "").replace(" (BACHARELADO/LICENCIATURA/PSICÓLOGO)", "")}
                          </div>
                        ))
                      ) : (
                        <div className="px-4 py-3 text-xs text-white/45 text-center">
                          Nenhum curso encontrado.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Seletor de Cota */}
              <div className="space-y-1.5">
                <label className="block text-xs font-bold uppercase tracking-wider text-[#002147] font-mono">2. Escolha o Sistema</label>
                <div className="grid grid-cols-2 gap-2.5">
                  <button
                    onClick={() => setSystemType("universal")}
                    className={`px-3 py-2.5 rounded-xl border text-xs sm:text-sm font-semibold transition-all active:scale-95 ${systemType === "universal" ? "bg-[#002147] border-[#002147] text-white" : "bg-white border-black/10 text-[#4A5568] hover:bg-gray-50"}`}
                  >
                    Universal
                  </button>
                  <button
                    onClick={() => {
                      if (selectedCourse.negros) {
                        setSystemType("negros");
                      }
                    }}
                    disabled={!selectedCourse.negros}
                    className={`px-3 py-2.5 rounded-xl border text-xs sm:text-sm font-semibold transition-all active:scale-95 ${!selectedCourse.negros ? "opacity-30 cursor-not-allowed border-dashed" : ""} ${systemType === "negros" ? "bg-[#002147] border-[#002147] text-white" : "bg-white border-black/10 text-[#4A5568] hover:bg-gray-50"}`}
                    title={!selectedCourse.negros ? "Sem cota declarada para negros no último vestibular" : ""}
                  >
                    Cota Racial
                  </button>
                </div>
              </div>

              {/* Informação Adicional */}
              <div className="pt-4 border-t border-black/5 flex items-start gap-2 text-[10px] text-gray-500 leading-relaxed">
                <Info size={14} className="text-[#00843D] shrink-0 mt-0.5" />
                <p>
                  Estes dados representam a 1ª chamada oficial do triênio 2023-2025. O banco de dados completo do Vetor PAS cobre todos os 52 cursos e chamadas subsequentes desde 2018.
                </p>
              </div>

            </div>

            {/* Coluna Direita: Dashboard de Métricas (Padronizado e Integrado Estilo História) */}
            <div className="lg:col-span-7 min-w-0 bg-[#002147]/5 border-l-8 border-l-[#00AEEF] rounded-r-3xl border-y border-r border-[#002147]/10 p-6 sm:p-8 flex flex-col justify-between shadow-[0_10px_35px_rgba(0,33,71,0.03)] transition-all">
              
              {activeMetrics ? (
                <div className="space-y-6 w-full min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="inline-block font-mono text-[0.65rem] tracking-[0.25em] uppercase text-[#002147]/70 font-extrabold bg-[#002147]/5 px-2.5 py-1 rounded">
                      MÉTRICAS OFICIAIS CEBRASPE
                    </span>
                  </div>

                  {/* Nome do curso no painel */}
                  <div className="min-w-0">
                    <h4 className="text-[#002147]/60 text-xs font-mono uppercase tracking-wider font-bold">Curso Selecionado</h4>
                    <p className="text-lg sm:text-xl font-extrabold tracking-tight text-[#002147] mt-0.5 truncate">
                      {selectedCourse.curso}
                    </p>
                  </div>

                  {/* Grande Exibição da Nota de Corte */}
                  <div className="pt-2">
                    <p className="text-xs text-[#00843D] font-mono uppercase tracking-wider font-bold">Nota de Corte (Mínimo)</p>
                    <p className="text-5xl sm:text-6xl font-black tracking-tight text-[#002147] mt-1">
                      {activeMetrics!.min >= 0 ? "+" : ""}
                      {activeMetrics!.min.toFixed(3)}
                    </p>
                    <p className="text-xs text-[#4A5568] mt-1.5 font-medium">
                      Nota do último classificado convocado para matrícula.
                    </p>
                  </div>

                  {/* Grid de Máxima e Média */}
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t border-black/10">
                    <div>
                      <p className="text-[10px] text-[#4A5568] uppercase font-mono tracking-wider font-bold">Nota Média</p>
                      <p className="text-lg font-extrabold text-[#002147] mt-0.5">
                        {activeMetrics!.media >= 0 ? "+" : ""}
                        {activeMetrics!.media.toFixed(3)}
                      </p>
                    </div>
                    <div>
                      <p className="text-[10px] text-[#4A5568] uppercase font-mono tracking-wider font-bold">Nota Máxima (1º Lugar)</p>
                      <p className="text-lg font-extrabold text-[#00843D] mt-0.5">
                        {activeMetrics!.max >= 0 ? "+" : ""}
                        {activeMetrics!.max.toFixed(3)}
                      </p>
                    </div>
                  </div>

                  {/* Distribuição Gráfica */}
                  <div className="space-y-3 pt-4">
                    <p className="text-[10px] text-[#4A5568] uppercase font-mono tracking-wider font-bold">Intervalo de Notas dos Aprovados</p>
                    
                    <div className="h-4 bg-black/10 rounded-full w-full relative">
                      {/* Indicador de Nota de Corte (Min) - Usando Azul UnB da Paleta */}
                      <div 
                        className="absolute w-3.5 h-3.5 rounded-full bg-[#002147] border-2 border-white -top-[3px] -translate-x-1/2 z-20 cursor-help"
                        style={{ left: "10%" }}
                        title={`Nota de Corte: ${activeMetrics!.min.toFixed(3)}`}
                      />
                      {/* Indicador de Média */}
                      <div 
                        className="absolute w-3.5 h-3.5 rounded-full bg-[#00AEEF] border-2 border-white -top-[3px] -translate-x-1/2 z-20 cursor-help"
                        style={{ left: "50%" }}
                        title={`Média: ${activeMetrics!.media.toFixed(3)}`}
                      />
                      {/* Indicador de Nota Máxima */}
                      <div 
                        className="absolute w-3.5 h-3.5 rounded-full bg-[#00843D] border-2 border-white -top-[3px] -translate-x-1/2 z-20 cursor-help"
                        style={{ left: "90%" }}
                        title={`Nota Máxima: ${activeMetrics!.max.toFixed(3)}`}
                      />
                      {/* Track de preenchimento */}
                      <div className="absolute top-0 bottom-0 left-[10%] right-[10%] bg-gradient-to-r from-[#002147] via-[#00AEEF] to-[#00843D] opacity-60 rounded-full" />
                    </div>

                    <div className="flex items-center justify-between text-[9px] text-[#4A5568]/70 font-mono font-bold">
                      <span>Corte: {activeMetrics!.min.toFixed(3)}</span>
                      <span>Média: {activeMetrics!.media.toFixed(3)}</span>
                      <span>Máx: {activeMetrics!.max.toFixed(3)}</span>
                    </div>
                  </div>

                </div>
              ) : (
                <div className="text-center py-20 text-gray-500 text-sm">
                  Dados indisponíveis para a categoria selecionada.
                </div>
              )}

              {/* Botão de Chamada para a Lista de Espera */}
              <div className="mt-8 pt-6 border-t border-black/10 text-center">
                <a
                  href="#lista-espera"
                  onClick={scrollToForm}
                  className="inline-flex items-center gap-1 text-xs font-bold uppercase text-[#00AEEF] hover:text-[#33C1F3] transition-colors active:scale-95"
                >
                  <span>Simular minha nota contra estas médias</span>
                  <CaretRight size={12} weight="bold" />
                </a>
              </div>
            </div>

          </div>

        </div>
      </section>
      )}

      {/* ============ HISTÓRIA DO FUNDADOR (CLEAN WHITE CARD WITH GREEN LEFT BORDER) ============ */}
      <section id="historia" className="py-24 bg-[#F8F9FA] border-b border-[#E2E8F0] scroll-mt-20">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-white border-l-8 border-[#00843D] rounded-r-3xl p-8 sm:p-12 relative shadow-[0_10px_35px_rgba(0,0,0,0.03)] border-y border-r border-black/5 hover:scale-[1.005] hover:shadow-[0_15px_45px_rgba(0,0,0,0.05)] transition-all duration-300">
            <div className="flex items-center gap-2 mb-4">
              <span className="font-mono text-xs text-[#00843D] uppercase tracking-[0.2em] font-bold">
                De Aluno para Aluno
              </span>
            </div>

            <h2 className="font-heading text-2xl sm:text-4xl font-extrabold mb-6 text-[#002147] leading-snug">
              "Eu já estive no seu lugar no PAS 3, e sei como é a sensação de não saber se vai dar pra chegar na nota necessária."
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

      {/* ============ FERRAMENTAS (FUNDO LIMPO BRANCO / CARDS COM BORDAS COLORIDAS) ============ */}
      <section id="ferramentas" className="bg-white py-20 sm:py-28 border-b border-[#E2E8F0] scroll-mt-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-14">
            <p className="font-mono text-[0.78rem] tracking-[0.22em] uppercase text-[#00843D] font-bold mb-3">
              No lançamento
            </p>
            <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-[#002147]">
              Três ferramentas criadas sob medida para o seu PAS 3.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="group relative bg-[#F8F9FA] border border-black/5 rounded-2xl p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all duration-300 border-t-4 border-t-[#00AEEF] hover:border-r hover:border-b hover:border-l hover:border-[#00AEEF]/20">
              <span className="inline-block font-mono text-xs font-bold text-[#00AEEF] bg-[#00AEEF]/10 px-3 py-1 rounded-md mb-6 transition-transform group-hover:scale-110">
                01
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147] group-hover:text-[#00AEEF] transition-colors">Preditor PAS 3</h3>
              <p className="text-sm text-[#4A5568] leading-relaxed">
                Insira suas notas do PAS 1 e 2. Nosso modelo prevê seu Argumento Final e sua probabilidade percentual de aprovação no curso que você quer.
              </p>
            </div>

            <div className="group relative bg-[#F8F9FA] border border-black/5 rounded-2xl p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all duration-300 border-t-4 border-t-[#00843D] hover:border-r hover:border-b hover:border-l hover:border-[#00843D]/20">
              <span className="inline-block font-mono text-xs font-bold text-[#00843D] bg-[#00843D]/10 px-3 py-1 rounded-md mb-6 transition-transform group-hover:scale-110">
                02
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147] group-hover:text-[#00843D] transition-colors">Calculadora "Quanto Falta"</h3>
              <p className="text-sm text-[#4A5568] leading-relaxed">
                Cálculo reverso: saiba exatamente qual Escore Bruto você precisa tirar na prova do PAS 3 para alcançar a nota de corte do seu curso.
              </p>
            </div>

            <div className="group relative bg-[#F8F9FA] border border-black/5 rounded-2xl p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all duration-300 border-t-4 border-t-[#002147] hover:border-r hover:border-b hover:border-l hover:border-[#002147]/20">
              <span className="inline-block font-mono text-xs font-bold text-[#002147] bg-[#002147]/5 px-3 py-1 rounded-md mb-6 transition-transform group-hover:scale-110">
                03
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147] group-hover:text-[#002147] transition-colors">Análise Histórica</h3>
              <p className="text-sm text-[#4A5568] leading-relaxed">
                Evolução histórica das notas de corte por curso e estatísticas médias das edições anteriores do PAS/UnB.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============ BUILD IN PUBLIC (LIGHT BG / WHITE CARDS) ============ */}
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

            {/* Grid de Cards (Brancos, Alta Definição) */}
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

      {/* ============ CTA FINAL (VIBRANT DARK BLUE-GREEN GRADIENT BANNER BOX) ============ */}
      <section className="py-20 bg-white text-white border-b border-[#E2E8F0]">
        <div className="max-w-5xl mx-auto px-6">
          <div
            className="rounded-3xl py-16 px-8 sm:px-16 text-center space-y-6 relative overflow-hidden shadow-[0_20px_40px_rgba(0,33,71,0.12)] border border-black/5 hover:scale-[1.005] transition-transform duration-500"
            style={{
              background: "linear-gradient(135deg, #002147 0%, #004723 50%, #001730 100%)",
            }}
          >
            {/* Efeito luminoso interno */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,rgba(0,174,239,0.15),transparent_50%)] pointer-events-none" />

            <span className="inline-block font-mono text-xs text-[#7FD8F7] uppercase tracking-[0.2em] bg-white/10 border border-white/15 px-3.5 py-1.5 rounded-full relative z-10">
              Acesso Antecipado Gratuito
            </span>
            <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-tight relative z-10 leading-tight">
              Não faça a 3ª etapa no escuro.
            </h2>
            <p className="text-base sm:text-lg text-white/80 max-w-xl mx-auto leading-relaxed relative z-10">
              Faça seu cadastro agora na lista de espera para receber o link de acesso em primeira mão assim que a plataforma for liberada.
            </p>
            <div className="pt-4 relative z-10">
              <button
                onClick={scrollToForm}
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-[#002147] bg-[#00AEEF] hover:bg-[#33C1F3] transition-all hover:-translate-y-0.5 shadow-[0_8px_30px_rgba(0,174,239,0.3)] text-base sm:text-lg active:scale-95"
              >
                <span>Quero Garantir Meu Acesso</span>
                <CaretRight size={18} weight="bold" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ============ FOOTER (NAVY BLUE DARK) ============ */}
      <footer className="bg-[#001730] text-white/50 py-12 text-xs">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-6 text-center sm:text-left">
          <p>© {new Date().getFullYear()} Vetor PAS — Projeto independente sem vínculo oficial com a UnB ou o Cebraspe.</p>
          <div className="flex items-center gap-5">
            <a
              href="https://www.linkedin.com/in/luiz-henrique-tomaz-moreira"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#00AEEF] hover:underline transition-all flex items-center gap-1 font-medium"
            >
              <span>LinkedIn</span>
              <span>↗</span>
            </a>
            <a
              href="mailto:lhtmoreira@gmail.com"
              className="text-[#7FD8F7] hover:underline transition-all flex items-center gap-1 font-medium"
            >
              <span>lhtmoreira@gmail.com</span>
            </a>
            <button
              onClick={scrollToForm}
              className="text-white/70 hover:text-white transition-all"
            >
              Ir para o cadastro ↑
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
