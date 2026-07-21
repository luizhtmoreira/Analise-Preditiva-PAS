"use client";

import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "./WaitlistForm";
import { Television, Lightbulb, Wrench, Megaphone, YoutubeLogo, CaretRight } from "@phosphor-icons/react";

export function LandingPage() {
  const scrollToForm = (e: React.MouseEvent) => {
    e.preventDefault();
    const formElement = document.getElementById("lista-espera");
    if (formElement) {
      formElement.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="landing-root bg-[#F8F9FA] text-[#1D1D1F] min-h-screen selection:bg-[#00843D] selection:text-white font-sans antialiased">
      
      {/* ============ STICKY NAVBAR ============ */}
      <nav className="sticky top-0 z-50 w-full backdrop-blur-md bg-white/95 border-b border-[#E2E8F0] transition-all duration-300">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4">
          <BrandMark />
          <div className="flex items-center gap-5 sm:gap-8 text-xs sm:text-sm font-semibold text-[#002147]">
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
        </div>
      </nav>

      {/* ============ HERO & HEADER (FUNDO LIMPO BRANCO / HIGH CONTRAST) ============ */}
      <header className="relative z-20 bg-white text-[#002147] pt-16 pb-24 border-b border-[#E2E8F0] overflow-hidden">
        {/* Elemento decorativo sutil de fundo */}
        <div
          className="absolute inset-0 pointer-events-none opacity-10"
          style={{
            backgroundImage:
              "radial-gradient(ellipse at 80% 20%, #00843D 0%, transparent 60%), radial-gradient(ellipse at 20% 80%, #00AEEF 0%, transparent 60%)",
          }}
        />

        <div className="relative z-10 max-w-6xl mx-auto px-6">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            {/* Texto de Impacto */}
            <div className="lg:col-span-7 space-y-6">
              <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00843D] bg-[#00843D]/5 border border-[#00843D]/25 px-3.5 py-1.5 rounded-lg font-bold">
                Análise Preditiva · PAS/UnB
              </span>
              <h1 className="font-heading text-4xl sm:text-6xl font-extrabold tracking-tight leading-[1.08] text-[#002147]">
                Sua aprovação na UnB,{" "}
                <br />
                <span className="text-[#00843D] relative inline-block">
                  calculada com precisão.
                  <span className="absolute bottom-1 left-0 w-full h-[4px] bg-[#00AEEF]/40 rounded-full" />
                </span>
              </h1>
              <p className="text-lg sm:text-xl text-[#4A5568] leading-relaxed max-w-xl">
                O Vetor PAS combina IA e dados oficiais do Cebraspe para prever seu Argumento Final e calcular a chance real de você passar no seu curso no PAS 3.
              </p>
            </div>

            {/* Card do Formulário da Lista de Espera */}
            <div id="lista-espera" className="lg:col-span-5 scroll-mt-24">
              <div className="bg-[#002147] border border-black/10 p-1.5 rounded-3xl shadow-[0_20px_50px_rgba(0,33,71,0.15)] text-white">
                <WaitlistForm variantStyle="card" buttonText="Garantir Meu Acesso Antecipado" />
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ============ HISTÓRIA DO FUNDADOR (CLEAN WHITE CARD WITH GREEN LEFT BORDER) ============ */}
      <section id="historia" className="py-24 bg-[#F8F9FA] border-b border-[#E2E8F0] scroll-mt-20">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-white border-l-8 border-[#00843D] rounded-r-3xl p-8 sm:p-12 relative shadow-[0_10px_35px_rgba(0,0,0,0.03)] border-y border-r border-black/5">
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
                <div className="w-10 h-10 rounded-full border-2 border-[#00843D]/40 shadow-inner overflow-hidden relative bg-[#F8F9FA]">
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
                  className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-[#F8F9FA] hover:bg-[#00AEEF] hover:text-[#002147] text-[#4A5568] text-xs font-semibold transition-all border border-black/5 shadow-sm"
                >
                  <span>LinkedIn</span>
                  <span>↗</span>
                </a>
                <a
                  href="mailto:lhtmoreira@gmail.com"
                  className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-[#F8F9FA] hover:bg-[#00843D] hover:text-white text-[#4A5568] text-xs font-semibold transition-all border border-black/5 shadow-sm"
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
            <div className="group relative bg-[#F8F9FA] border border-black/5 rounded-2xl p-8 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-0.5 border-t-4 border-t-[#00AEEF]">
              <span className="inline-block font-mono text-xs font-bold text-[#00AEEF] bg-[#00AEEF]/10 px-3 py-1 rounded-md mb-6">
                01
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147]">Preditor PAS 3</h3>
              <p className="text-sm text-[#4A5568] leading-relaxed">
                Insira suas notas do PAS 1 e 2. Nosso modelo prevê seu Argumento Final e sua probabilidade percentual de aprovação no curso que você quer.
              </p>
            </div>

            <div className="group relative bg-[#F8F9FA] border border-black/5 rounded-2xl p-8 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-0.5 border-t-4 border-t-[#00843D]">
              <span className="inline-block font-mono text-xs font-bold text-[#00843D] bg-[#00843D]/10 px-3 py-1 rounded-md mb-6">
                02
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147]">Calculadora "Quanto Falta"</h3>
              <p className="text-sm text-[#4A5568] leading-relaxed">
                Cálculo reverso: saiba exatamente qual Escore Bruto você precisa tirar na prova do PAS 3 para alcançar a nota de corte do seu curso.
              </p>
            </div>

            <div className="group relative bg-[#F8F9FA] border border-black/5 rounded-2xl p-8 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-0.5 border-t-4 border-t-[#002147]">
              <span className="inline-block font-mono text-xs font-bold text-[#002147] bg-[#002147]/5 px-3 py-1 rounded-md mb-6">
                03
              </span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#002147]">Análise Histórica</h3>
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
              <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00AEEF] bg-[#00AEEF]/5 border border-[#00AEEF]/20 px-3.5 py-1.5 rounded-lg font-bold">
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
                  className="inline-flex items-center gap-2.5 px-5 py-3 rounded-xl font-semibold text-white bg-[#002147] hover:bg-[#FF0000] hover:text-white transition-all shadow-sm text-sm group"
                >
                  <YoutubeLogo size={20} weight="fill" className="text-[#FF0000] group-hover:text-white transition-colors" />
                  <span>Acompanhar lives no YouTube</span>
                </a>
              </div>
            </div>

            {/* Grid de Cards (Brancos, Alta Definição) */}
            <div className="lg:col-span-6 grid sm:grid-cols-2 gap-6">
              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md transition-shadow">
                <div className="w-10 h-10 rounded-xl bg-[#00AEEF]/10 text-[#00AEEF] flex items-center justify-center">
                  <Television size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-[#002147]">Lives de Código</h3>
                <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
                  Acompanhe o desenvolvimento da interface ao vivo.
                </p>
              </div>

              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md transition-shadow">
                <div className="w-10 h-10 rounded-xl bg-[#00843D]/10 text-[#00843D] flex items-center justify-center">
                  <Lightbulb size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-[#002147]">Opine & Sugira</h3>
                <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
                  Features, designs, cores: você ajuda decidir o rumo das próximas telas.
                </p>
              </div>

              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md transition-shadow">
                <div className="w-10 h-10 rounded-xl bg-[#002147]/5 text-[#002147] flex items-center justify-center">
                  <Wrench size={22} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-lg text-[#002147]">Co-Criação</h3>
                <p className="text-xs sm:text-sm text-[#4A5568] leading-relaxed">
                  Ideal para quem quer aprender arquitetura web e ver decisões de engenharia na prática.
                </p>
              </div>

              <div className="bg-white border border-black/5 p-6 rounded-2xl space-y-3 shadow-sm hover:shadow-md transition-shadow">
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
            className="rounded-3xl py-16 px-8 sm:px-16 text-center space-y-6 relative overflow-hidden shadow-[0_20px_40px_rgba(0,33,71,0.12)] border border-black/5"
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
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-[#002147] bg-[#00AEEF] hover:bg-[#33C1F3] transition-all hover:-translate-y-0.5 shadow-[0_8px_30px_rgba(0,174,239,0.3)] text-base sm:text-lg"
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
