"use client";

import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "./WaitlistForm";
import { Television, Lightbulb, Wrench, Megaphone, YoutubeLogo } from "@phosphor-icons/react";

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
      {/* ============ STICKY NAVBAR ============ */}
      <nav className="sticky top-0 z-50 w-full bg-[#002147] border-b border-white/10 transition-all duration-300">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4">
          <BrandMark />
          <div className="flex items-center gap-5 sm:gap-8 text-xs sm:text-sm font-semibold text-white/90">
            <a
              href="#historia"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("historia")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00AEEF] transition-colors cursor-pointer"
            >
              Minha História
            </a>
            <a
              href="#ferramentas"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("ferramentas")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00AEEF] transition-colors cursor-pointer"
            >
              Ferramentas
            </a>
            <a
              href="#build-in-public"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("build-in-public")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00AEEF] transition-colors cursor-pointer"
            >
              Build in Public
            </a>
          </div>
        </div>
      </nav>

      {/* ============ HEADER & HERO (AZUL UNB DEEP GRADIENT) ============ */}
      <header
        className="relative z-20 text-white"
        style={{
          background:
            "linear-gradient(168deg, #002147 0%, #002D58 45%, #003666 80%, #002B54 100%)",
        }}
      >
        {/* Grade de fundo sutil */}
        <div
          className="absolute inset-0 pointer-events-none opacity-25 overflow-hidden"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
            backgroundSize: "56px 56px",
            maskImage: "radial-gradient(ellipse 90% 70% at 50% 30%, black, transparent)",
          }}
        />

        {/* Hero Body */}
        <div className="relative z-10 max-w-6xl mx-auto px-6 pt-10 pb-20 sm:pt-16 sm:pb-24">
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
                O Vetor PAS combina IA e dados oficiais do Cebraspe para prever seu Argumento Final e calcular a chance real de você passar no seu curso no PAS 3.
              </p>
            </div>

            {/* Card do Formulário da Lista de Espera */}
            <div id="lista-espera" className="lg:col-span-5 scroll-mt-24">
              <WaitlistForm variantStyle="card" buttonText="Garantir Meu Acesso Antecipado" />
            </div>
          </div>
        </div>
      </header>

      {/* ============ HISTÓRIA DO FUNDADOR (TRANSIÇÃO SUAVE DE AZUL) ============ */}
      <section
        id="historia"
        className="relative z-10 text-white py-20 sm:py-24 scroll-mt-20"
        style={{
          background:
            "linear-gradient(180deg, #002B54 0%, #002347 40%, #001A38 100%)",
        }}
      >
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-[#002850]/90 border border-white/15 rounded-3xl p-8 sm:p-12 relative overflow-hidden shadow-2xl backdrop-blur-md">
            {/* Detalhe discreto em Verde UnB no topo do card */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-[#00843D]" />

            <div className="flex items-center gap-2 mb-4">
              <span className="w-2 h-2 rounded-full bg-[#00843D]" />
              <span className="font-mono text-xs text-[#00843D] uppercase tracking-[0.2em] font-bold">
                De Aluno para Aluno
              </span>
            </div>

            <h2 className="font-heading text-2xl sm:text-4xl font-bold mb-6 text-white leading-snug">
              "Eu já estive no seu lugar no PAS 3, e sei como é a sensação de não saber se vai dar pra chegar na nota necessária."
            </h2>
            <div className="space-y-4 text-white/80 leading-relaxed text-sm sm:text-base">
              <p>
                Quando eu estava estudando para o PAS 3, passei meses tentando adivinhar se era possível passar no curso que eu queria. A nota necessária eu sabia, mas qual era a chance de eu tirar aquela nota? Procurava na internet, mas nada me dava uma resposta real sobre a probabilidade de entrar no meu curso na UnB, se valia a pena tentar ele ou mudar a rota para garantir minha aprovação.
              </p>
              <p>
                Foi por isso que criei o <strong>Vetor PAS</strong>: para que você não precise estudar no escuro. Usamos algoritmos de machine learning sobre os boletins oficiais do Cebraspe para te mostrar exatamente onde você está e o que precisa fazer na terceira etapa. Assim, você não precisa esperar a nota do PAS 3 sair para saber se tinha chance ou não.
              </p>
            </div>
            <div className="mt-8 pt-6 border-t border-white/10 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full border border-[#00843D]/40 shadow-inner overflow-hidden relative bg-[#001D3A]">
                  <img
                    src="/luiz.jpeg"
                    alt="Luiz Moreira"
                    className="w-full h-full object-cover object-top"
                    onError={(e) => {
                      // Fallback case
                      e.currentTarget.style.display = 'none';
                      const parent = e.currentTarget.parentElement;
                      if (parent) {
                        const fallbackText = document.createElement('div');
                        fallbackText.className = 'w-full h-full flex items-center justify-center font-mono text-[#00843D] text-sm font-bold';
                        fallbackText.innerText = 'LM';
                        parent.appendChild(fallbackText);
                      }
                    }}
                  />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Luiz Moreira</p>
                  <p className="text-xs text-white/60">Criador do Vetor PAS · Engenharia de Software (UnB)</p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <a
                  href="https://www.linkedin.com/in/luiz-henrique-tomaz-moreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/10 hover:bg-[#00AEEF] hover:text-[#002147] text-white text-xs font-semibold transition-all border border-white/20"
                >
                  <span>LinkedIn</span>
                  <span>↗</span>
                </a>
                <a
                  href="mailto:lhtmoreira@gmail.com"
                  className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/10 hover:bg-[#00843D] text-white text-xs font-semibold transition-all border border-white/20"
                >
                  <span>E-mail</span>
                  <span>✉</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ O QUE VAI ACESSAR (CINZA CLARO #F5F5F7 + CARDS BRANCOS) ============ */}
      <section id="ferramentas" className="bg-[#F5F5F7] py-20 sm:py-28 border-y border-[#E6E6E8] scroll-mt-20">
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

      {/* ============ SEÇÃO BUILD IN PUBLIC (AZUL ESCURO UNB) ============ */}
      <section id="build-in-public" className="bg-[#001D3D] text-white py-20 sm:py-24 relative overflow-hidden border-b border-white/5 scroll-mt-20">
        <div
          className="absolute inset-0 pointer-events-none opacity-20"
          style={{
            backgroundImage:
              "radial-gradient(circle at 80% 20%, #00843D 0%, transparent 50%), radial-gradient(circle at 20% 80%, #00AEEF 0%, transparent 50%)",
          }}
        />
        <div className="max-w-6xl mx-auto px-6 relative z-10">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00AEEF] bg-[#00AEEF]/10 border border-[#00AEEF]/20 px-3.5 py-1.5 rounded-lg">
                Build in Public
              </span>
              <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-tight leading-[1.1]">
                Criado em público. <br />
                Desenhado por <span className="text-[#00AEEF]">você</span>.
              </h2>
              <p className="text-base sm:text-lg text-white/80 leading-relaxed">
                O Vetor PAS não está sendo criado a portas fechadas. Acreditamos que a melhor ferramenta para o PAS 3 é aquela construída em parceria com quem realmente vai usá-la em sua preparação.
              </p>
              <p className="text-sm sm:text-base text-white/70">
                Estou desenvolvendo o produto em <strong>lives abertas de código</strong>, onde você pode acompanhar cada linha de programação, sugerir ideias de design, propor novas simulações de nota ou criticar o que não ficou legal.
              </p>
              <div className="pt-2">
                <a
                  href="https://www.youtube.com/@luizhtmoreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2.5 px-5 py-3 rounded-xl font-semibold text-white bg-white/10 hover:bg-[#FF0000] hover:text-white border border-white/15 transition-all shadow-[0_4px_15px_rgba(0,0,0,0.15)] active:scale-[0.98] text-sm group"
                >
                  <YoutubeLogo size={20} weight="fill" className="text-[#FF0000] group-hover:text-white transition-colors" />
                  <span>Acompanhar lives no YouTube</span>
                </a>
              </div>
            </div>

            <div className="lg:col-span-6 grid sm:grid-cols-2 gap-6">
              <div className="bg-[#002850]/60 border border-white/10 p-6 rounded-2xl space-y-3">
                <div className="w-10 h-10 rounded-xl bg-[#00AEEF]/15 text-[#00AEEF] flex items-center justify-center">
                  <Television size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-white">Lives de Código</h3>
                <p className="text-xs sm:text-sm text-white/70 leading-relaxed">
                  Acompanhe o desenvolvimento da interface ao vivo.
                </p>
              </div>

              <div className="bg-[#002850]/60 border border-white/10 p-6 rounded-2xl space-y-3">
                <div className="w-10 h-10 rounded-xl bg-[#00843D]/15 text-[#00843D] flex items-center justify-center">
                  <Lightbulb size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-white">Opine & Sugira</h3>
                <p className="text-xs sm:text-sm text-white/70 leading-relaxed">
                  Features, designs, cores: você ajuda decidir o rumo das próximas telas.
                </p>
              </div>

              <div className="bg-[#002850]/60 border border-white/10 p-6 rounded-2xl space-y-3">
                <div className="w-10 h-10 rounded-xl bg-white/10 text-white flex items-center justify-center">
                  <Wrench size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-white">Co-Criação</h3>
                <p className="text-xs sm:text-sm text-white/70 leading-relaxed">
                  Ideal para quem quer aprender arquitetura web e ver decisões de engenharia na prática.
                </p>
              </div>

              <div className="bg-[#002850]/60 border border-white/10 p-6 rounded-2xl space-y-3">
                <div className="w-10 h-10 rounded-xl bg-[#7FD8F7]/15 text-[#7FD8F7] flex items-center justify-center">
                  <Megaphone size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-white">Feito para Você</h3>
                <p className="text-xs sm:text-sm text-white/70 leading-relaxed">
                  Ao participar das sugestões, você garante que a ferramenta resolverá suas reais dúvidas do PAS 3.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ CTA FINAL (TRANSIÇÃO GRADUAL DE AZUL PARA O FOOTER) ============ */}
      <section
        className="text-white py-20 sm:py-24 relative overflow-hidden"
        style={{
          background:
            "linear-gradient(180deg, #002B54 0%, #002147 60%, #001A38 100%)",
        }}
      >
        <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
          <span className="inline-block font-mono text-xs text-[#7FD8F7] uppercase tracking-[0.2em] bg-white/10 border border-white/15 px-3.5 py-1.5 rounded-full mb-6">
            Acesso Antecipado Gratuito
          </span>
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-tight mb-4">
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

      {/* ============ FOOTER (AZUL NOTURNO CONTINUADO) ============ */}
      <footer className="bg-[#001730] text-white/50 border-t border-white/10 py-8 text-xs">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
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
