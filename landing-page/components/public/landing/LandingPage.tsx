"use client";

import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "./WaitlistForm";

export function LandingPage() {
  const scrollToForm = (e: React.MouseEvent) => {
    e.preventDefault();
    const formElement = document.getElementById("lista-espera");
    if (formElement) {
      formElement.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="landing-root bg-white text-[#1D1D1F] min-h-screen selection:bg-[#00AEEF] selection:text-white">
      {/* ============ HEADER & HERO (AZUL UNB #002147) ============ */}
      <header
        className="relative overflow-hidden text-white"
        style={{
          background:
            "linear-gradient(168deg, #002147 0%, #003366 52%, #003A70 100%)",
        }}
      >
        {/* Grade de fundo sutil */}
        <div
          className="absolute inset-0 pointer-events-none opacity-30"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
            backgroundSize: "56px 56px",
            maskImage: "radial-gradient(ellipse 90% 70% at 50% 30%, black, transparent)",
          }}
        />

        {/* Navbar */}
        <nav className="relative z-10 max-w-6xl mx-auto flex items-center justify-between px-6 pt-6 pb-4">
          <BrandMark />
          <button
            onClick={scrollToForm}
            className="px-5 py-2.5 rounded-xl font-semibold text-xs sm:text-sm bg-[#00AEEF] text-[#002147] hover:bg-[#33C1F3] transition-all shadow-[0_4px_15px_rgba(0,174,239,0.35)] active:scale-95"
          >
            Garantir Vaga
          </button>
        </nav>

        {/* Hero Body */}
        <div className="relative z-10 max-w-6xl mx-auto px-6 pt-10 pb-20 sm:pt-16 sm:pb-28">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            {/* Texto de Impacto */}
            <div className="lg:col-span-7 space-y-6">
              <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#7FD8F7] bg-white/10 border border-white/15 px-3.5 py-1.5 rounded-lg">
                Análise Preditiva · PAS/UnB
              </span>
              <h1 className="font-heading text-4xl sm:text-6xl font-extrabold tracking-[-0.03em] leading-[1.06]">
                Sua aprovação na UnB,{" "}
                <br />
                <span className="text-[#00AEEF]">calculada</span> com precisão.
              </h1>
              <p className="text-lg sm:text-xl text-white/75 leading-relaxed max-w-xl">
                O Vetor PAS combina estatística avançada e dados oficiais do Cebraspe para prever seu Argumento Final e calcular exatamente o quanto falta para a nota de corte do seu curso no PAS 3.
              </p>
            </div>

            {/* Card do Formulário da Lista de Espera */}
            <div id="lista-espera" className="lg:col-span-5 scroll-mt-24">
              <WaitlistForm variantStyle="card" buttonText="Garantir Meu Acesso Antecipado" />
            </div>
          </div>
        </div>
      </header>

      {/* ============ HISTÓRIA DO FUNDADOR (VERDE UNB #00843D) ============ */}
      <section className="relative overflow-hidden bg-[#00843D] py-20 sm:py-24 text-white">
        <div
          className="absolute inset-0 pointer-events-none opacity-20"
          style={{
            background:
              "radial-gradient(ellipse 60% 80% at 85% 20%, rgba(0,33,71,0.5), transparent)",
          }}
        />
        <div className="relative max-w-4xl mx-auto px-6">
          <div className="bg-[#002147]/40 border border-white/25 rounded-3xl p-8 sm:p-12 backdrop-blur-md shadow-2xl">
            <span className="font-mono text-xs text-[#7FD8F7] uppercase tracking-[0.2em] block mb-3 font-semibold">
              De Aluno para Aluno
            </span>
            <h2 className="font-heading text-2xl sm:text-4xl font-bold mb-6 text-white leading-snug">
              "Eu já estive no seu lugar no PAS 3 — e sei a angústia que é não saber onde você está."
            </h2>
            <div className="space-y-4 text-white/90 leading-relaxed text-sm sm:text-base">
              <p>
                Quando eu estava prestando o PAS 3, passei meses tentando adivinhar se as minhas notas do PAS 1 e 2 seriam suficientes. Procurava planilhas e cálculos antigos na internet, mas nenhum me dava uma resposta real sobre a probabilidade de entrar no meu curso na UnB.
              </p>
              <p>
                Foi por isso que criei o <strong>Vetor PAS</strong>: para que você não precise estudar no escuro. Usamos algoritmos de machine learning sobre os boletins oficiais do Cebraspe para te mostrar exatamente onde você está e o que precisa fazer na terceira etapa.
              </p>
            </div>
            <div className="mt-8 pt-6 border-t border-white/20 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-white text-[#00843D] font-bold flex items-center justify-center font-mono text-sm shadow-md">
                  VP
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Criador do Vetor PAS</p>
                  <p className="text-xs text-white/70">Ex-estudante do PAS/UnB</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ O QUE VAI ACESSAR (CINZA CLARO #F5F5F7 + CARDS BRANCOS) ============ */}
      <section className="bg-[#F5F5F7] py-20 sm:py-28 border-y border-[#E6E6E8]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-14">
            <p className="font-mono text-[0.78rem] tracking-[0.22em] uppercase text-[#00843D] font-semibold mb-3">
              No lançamento
            </p>
            <h2 className="font-heading text-3xl sm:text-4xl font-bold tracking-[-0.025em] text-[#1D1D1F]">
              Três ferramentas criadas sob medida para o seu PAS 3.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="group relative bg-white border border-[#E6E6E8] rounded-2xl p-8 shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 overflow-hidden">
              <div
                className="absolute top-0 left-0 right-0 h-1.5"
                style={{ background: "linear-gradient(90deg, #00AEEF, #003366)" }}
              />
              <span className="inline-block font-mono text-xs font-bold text-[#00AEEF] bg-[#00AEEF]/10 px-3 py-1 rounded-md mb-6">
                01
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147]">Preditor PAS 3</h3>
              <p className="text-sm text-[#6E6E73] leading-relaxed">
                Insira suas notas do PAS 1 e 2. Nosso modelo prevê seu Argumento Final e sua probabilidade percentual de aprovação no curso que você quer.
              </p>
            </div>

            <div className="group relative bg-white border border-[#E6E6E8] rounded-2xl p-8 shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 overflow-hidden">
              <div
                className="absolute top-0 left-0 right-0 h-1.5"
                style={{ background: "linear-gradient(90deg, #00843D, #00AEEF)" }}
              />
              <span className="inline-block font-mono text-xs font-bold text-[#00843D] bg-[#00843D]/10 px-3 py-1 rounded-md mb-6">
                02
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147]">Calculadora "Quanto Falta"</h3>
              <p className="text-sm text-[#6E6E73] leading-relaxed">
                Cálculo reverso: saiba exatamente qual Escore Bruto você precisa tirar na prova do PAS 3 para alcançar a nota de corte do seu curso.
              </p>
            </div>

            <div className="group relative bg-white border border-[#E6E6E8] rounded-2xl p-8 shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1 overflow-hidden">
              <div
                className="absolute top-0 left-0 right-0 h-1.5"
                style={{ background: "linear-gradient(90deg, #002147, #00843D)" }}
              />
              <span className="inline-block font-mono text-xs font-bold text-[#002147] bg-[#002147]/10 px-3 py-1 rounded-md mb-6">
                03
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147]">Análise Histórica Cebraspe</h3>
              <p className="text-sm text-[#6E6E73] leading-relaxed">
                Evolução histórica das notas de corte por curso e estatísticas médias das edições anteriores do PAS/UnB.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============ CTA FINAL (AZUL UNB FECHANDO O RITMO DE CORES) ============ */}
      <section className="bg-[#002147] text-white py-20 sm:py-24 relative overflow-hidden">
        <div className="max-w-4xl mx-auto px-6 text-center space-y-6 relative z-10">
          <span className="font-mono text-xs text-[#7FD8F7] uppercase tracking-[0.2em] bg-white/10 border border-white/15 px-3.5 py-1.5 rounded-full">
            Acesso Antecipado Gratuito
          </span>
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-tight">
            Não faça a 3ª etapa no escuro.
          </h2>
          <p className="text-base sm:text-lg text-white/75 max-w-xl mx-auto leading-relaxed">
            Faça seu cadastro agora na lista de espera para receber o link de acesso em primeira mão assim que a plataforma for liberada.
          </p>
          <div className="pt-4">
            <button
              onClick={scrollToForm}
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-[#002147] bg-[#00AEEF] hover:bg-[#33C1F3] transition-all hover:-translate-y-0.5 shadow-[0_8px_30px_rgba(0,174,239,0.35)] text-base sm:text-lg"
            >
              <span>Quero Garantir Meu Acesso</span>
              <span>↑</span>
            </button>
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="bg-[#001730] text-white/50 border-t border-white/10 py-8 text-xs">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <p>© {new Date().getFullYear()} Vetor PAS — Projeto independente sem vínculo oficial com a UnB ou o Cebraspe.</p>
          <button
            onClick={scrollToForm}
            className="text-[#00AEEF] hover:underline transition-all"
          >
            Ir para o cadastro ↑
          </button>
        </div>
      </footer>
    </div>
  );
}
