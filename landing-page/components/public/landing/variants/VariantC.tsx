"use client";

import { useState } from "react";
import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "../WaitlistForm";

export function VariantC() {
  const [activeTab, setActiveTab] = useState<"quanto-falta" | "preditor" | "relatorios">("quanto-falta");

  return (
    <div className="landing-root bg-[#001730] text-white min-h-screen selection:bg-[#00AEEF] selection:text-[#001730]">
      {/* ============ HEADER MOBILE-FIRST ============ */}
      <header className="border-b border-white/10 bg-[#002147]/80 backdrop-blur-md sticky top-0 z-30">
        <nav className="max-w-5xl mx-auto flex items-center justify-between px-5 py-4">
          <BrandMark />
          <a
            href="#form-mobile"
            className="px-3.5 py-1.5 rounded-lg bg-[#00AEEF] text-[#002147] font-bold text-xs uppercase tracking-wider shadow-md active:scale-95 transition-all"
          >
            Lista VIP
          </a>
        </nav>
      </header>

      {/* ============ HERO (COMPACT & DIRECT) ============ */}
      <section className="py-10 sm:py-16 max-w-3xl mx-auto px-5 text-center space-y-4">
        <div className="inline-flex items-center gap-2 font-mono text-[0.72rem] bg-white/10 border border-white/15 px-3 py-1 rounded-full text-[#7FD8F7]">
          <span className="w-2 h-2 rounded-full bg-[#00AEEF] animate-pulse" />
          <span>Focado no PAS 3 da UnB</span>
        </div>
        <h1 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">
          Saiba o que você precisa tirar no <span className="text-[#00AEEF]">PAS 3</span> para entrar na UnB.
        </h1>
        <p className="text-sm sm:text-base text-white/75 leading-relaxed max-w-xl mx-auto">
          Previsões baseadas no histórico oficial do Cebraspe. Cadastre-se para ser avisado no lançamento.
        </p>
      </section>

      {/* ============ TEASER INTERATIVO DE FUNCIONALIDADES ============ */}
      <section className="pb-12 max-w-3xl mx-auto px-5">
        <div className="bg-[#002147] border border-white/15 rounded-2xl p-5 sm:p-8 space-y-6">
          <div className="flex items-center justify-between border-b border-white/10 pb-4">
            <span className="font-mono text-xs text-[#00AEEF] uppercase tracking-wider">
              Preview do que vem aí
            </span>
            <span className="text-xs text-white/50">Clique nas abas</span>
          </div>

          {/* Abas */}
          <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-none">
            <button
              onClick={() => setActiveTab("quanto-falta")}
              className={`px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold whitespace-nowrap transition-all ${
                activeTab === "quanto-falta"
                  ? "bg-[#00AEEF] text-[#002147]"
                  : "bg-white/5 text-white/70 hover:bg-white/10"
              }`}
            >
              🎯 Quanto Falta
            </button>
            <button
              onClick={() => setActiveTab("preditor")}
              className={`px-4 py-2 rounded-xl text-xs sm:text-sm font-[#002147] font-semibold whitespace-nowrap transition-all ${
                activeTab === "preditor"
                  ? "bg-[#00AEEF] text-[#002147]"
                  : "bg-white/5 text-white/70 hover:bg-white/10"
              }`}
            >
              🤖 Preditor PAS 3
            </button>
            <button
              onClick={() => setActiveTab("relatorios")}
              className={`px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold whitespace-nowrap transition-all ${
                activeTab === "relatorios"
                  ? "bg-[#00AEEF] text-[#002147]"
                  : "bg-white/5 text-white/70 hover:bg-white/10"
              }`}
            >
              📊 Análise por Escola
            </button>
          </div>

          {/* Conteúdo da Aba */}
          <div className="bg-[#001730] border border-white/10 rounded-xl p-5 text-sm text-white/80 space-y-3 min-h-[120px]">
            {activeTab === "quanto-falta" && (
              <>
                <p className="font-semibold text-white">Como funciona o "Quanto Falta":</p>
                <p className="text-xs leading-relaxed text-white/70">
                  Insira seus Escore Brutos do PAS 1 e PAS 2. O sistema faz o cálculo reverso indicando a nota mínima exata que você precisa tirar na prova objetiva do PAS 3 para alcançar a Nota de Corte do seu curso desejado.
                </p>
              </>
            )}
            {activeTab === "preditor" && (
              <>
                <p className="font-semibold text-white">Como funciona o Preditor IA:</p>
                <p className="text-xs leading-relaxed text-white/70">
                  Um conjunto de 4 modelos de Inteligência Artificial calcula seu Argumento Final estimado e te dá uma porcentagem de chance de aprovação na UnB com margem de erro oficial (RMSE 13,49).
                </p>
              </>
            )}
            {activeTab === "relatorios" && (
              <>
                <p className="font-semibold text-white">Relatórios B2B para Escolas:</p>
                <p className="text-xs leading-relaxed text-white/70">
                  Coordenadores pedagógicos podem acompanhar o semáforo de risco da turma inteira e emitir relatórios em PDF com a marca da escola para reuniões de pais.
                </p>
              </>
            )}
          </div>
        </div>
      </section>

      {/* ============ HISTÓRIA COMPACTA ============ */}
      <section className="pb-12 max-w-3xl mx-auto px-5">
        <div className="bg-[#002147]/60 border border-white/10 rounded-2xl p-6 space-y-3">
          <span className="font-mono text-[0.7rem] text-[#00843D] uppercase tracking-wider font-bold">
            Por que criamos o Vetor PAS?
          </span>
          <p className="text-xs sm:text-sm text-white/80 leading-relaxed italic">
            "Passei pelo PAS 3 sem saber se minhas notas dariam conta. Desenvolvi o Vetor PAS para que nenhum estudante de Brasília precise passar pela terceira etapa no escuro."
          </p>
          <p className="text-[0.75rem] text-[#00AEEF] font-semibold">— Fundador do Vetor PAS (Ex-Aluno UnB)</p>
        </div>
      </section>

      {/* ============ FORMULÁRIO MOBILE-FIRST ============ */}
      <section id="form-mobile" className="py-10 bg-[#002147] border-t border-white/10">
        <div className="max-w-md mx-auto px-5 space-y-5">
          <div className="text-center space-y-1">
            <h2 className="font-heading text-xl font-bold text-white">Entrar na Lista VIP</h2>
            <p className="text-xs text-white/60">Apenas para alunos prestando o PAS 3</p>
          </div>
          <WaitlistForm variantStyle="card" buttonText="Garantir Minha Vaga" />
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="border-t border-white/10 py-6 text-[0.75rem] text-white/50 bg-[#001024] text-center px-4">
        <p>© {new Date().getFullYear()} Vetor PAS — Projeto independente sem vínculo oficial com a UnB ou o Cebraspe.</p>
        <p className="font-mono text-[#00843D] mt-1">Variante C — Mobile-First Interactive</p>
      </footer>
    </div>
  );
}
