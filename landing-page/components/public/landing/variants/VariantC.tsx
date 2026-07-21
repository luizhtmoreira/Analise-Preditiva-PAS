"use client";

import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "../WaitlistForm";
import { Television, Lightbulb, Wrench, Megaphone, YoutubeLogo, ArrowRight } from "@phosphor-icons/react";

export function VariantC() {
  const scrollToForm = (e: React.MouseEvent) => {
    e.preventDefault();
    const formElement = document.getElementById("lista-espera");
    if (formElement) {
      formElement.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="landing-root bg-[#F5F5F7] text-[#121212] min-h-screen selection:bg-[#00843D] selection:text-white font-sans antialiased overflow-x-hidden">
      
      {/* ============ HEADER ============ */}
      <nav className="sticky top-0 z-50 w-full bg-white border-b-4 border-[#121212] transition-all">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4">
          <BrandMark />
          <div className="flex items-center gap-5 sm:gap-8 text-xs sm:text-sm font-bold text-[#121212]">
            <a
              href="#historia"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("historia")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:underline cursor-pointer decoration-2 decoration-[#00AEEF]"
            >
              História
            </a>
            <a
              href="#ferramentas"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("ferramentas")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:underline cursor-pointer decoration-2 decoration-[#00843D]"
            >
              Ferramentas
            </a>
            <a
              href="#build-in-public"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("build-in-public")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:underline cursor-pointer decoration-2 decoration-[#00AEEF]"
            >
              Build in Public
            </a>
          </div>
        </div>
      </nav>

      {/* ============ HERO SECTION ============ */}
      <header className="relative z-20 py-20 bg-[#002147] text-white border-b-4 border-[#121212]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            {/* Texto */}
            <div className="lg:col-span-7 space-y-6">
              <span className="inline-block font-mono text-[0.75rem] tracking-[0.25em] uppercase text-white bg-[#00843D] border-2 border-[#121212] px-3.5 py-1.5 rounded-none shadow-[2px_2px_0px_#121212]">
                ANÁLISE PREDITIVA
              </span>
              <h1 className="font-heading text-4xl sm:text-6xl font-black tracking-tight leading-[1.08]">
                Sua aprovação na UnB, <br />
                <span className="bg-[#00AEEF] text-[#002147] px-2 py-0.5 inline-block border-2 border-[#121212] transform -rotate-1">calculada</span> com precisão.
              </h1>
              <p className="text-lg sm:text-xl text-white/90 leading-relaxed max-w-xl">
                O Vetor PAS combina IA e dados oficiais do Cebraspe para prever seu Argumento Final e calcular a chance real de você passar no seu curso no PAS 3.
              </p>
            </div>

            {/* Form Card (Neo-brutalist) */}
            <div id="lista-espera" className="lg:col-span-5 scroll-mt-24">
              <div className="border-4 border-[#121212] bg-[#001D3D] p-1 rounded-none shadow-[8px_8px_0px_#121212]">
                <WaitlistForm variantStyle="card" buttonText="Garantir Meu Acesso Antecipado" />
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ============ FOUNDER STORY (NEO BRUTALIST BOX) ============ */}
      <section id="historia" className="py-24 border-b-4 border-[#121212] bg-white">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-[#F5F5F7] border-4 border-[#121212] p-8 sm:p-12 rounded-none shadow-[8px_8px_0px_#121212] relative overflow-hidden">
            <div className="flex items-center gap-2 mb-4">
              <span className="font-mono text-xs text-white bg-[#00843D] border border-[#121212] px-2.5 py-1 rounded-none">
                De Aluno para Aluno
              </span>
            </div>

            <h2 className="font-heading text-2xl sm:text-4xl font-extrabold mb-6 text-[#121212] leading-snug">
              "Eu já estive no seu lugar no PAS 3, e sei como é a sensação de não saber se vai dar pra chegar na nota necessária."
            </h2>
            <div className="space-y-4 text-[#121212]/80 leading-relaxed text-sm sm:text-base">
              <p>
                Quando eu estava estudando para o PAS 3, passei meses tentando adivinhar se era possível passar no curso que eu queria. A nota necessária eu sabia, mas qual era a chance de eu tirar aquela nota? Procurava na internet, mas nada me dava uma resposta real sobre a probabilidade de entrar no meu curso na UnB, se valia a pena tentar ele ou mudar a rota para garantir minha aprovação.
              </p>
              <p>
                Foi por isso que criei o <strong>Vetor PAS</strong>: para que você não precise estudar no escuro. Usamos algoritmos de machine learning sobre os boletins oficiais do Cebraspe para te mostrar exatamente onde você está e o que precisa fazer na terceira etapa. Assim, você não precisa esperar a nota do PAS 3 sair para saber se tinha chance ou não.
              </p>
            </div>
            <div className="mt-8 pt-6 border-t-2 border-[#121212] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full border-2 border-[#121212] overflow-hidden relative">
                  <img
                    src="/luiz.jpeg"
                    alt="Luiz Moreira"
                    className="w-full h-full object-cover object-top"
                  />
                </div>
                <div>
                  <p className="text-sm font-bold text-[#121212]">Luiz Moreira</p>
                  <p className="text-xs text-[#121212]/60">Criador do Vetor PAS · Engenharia de Software (UnB)</p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <a
                  href="https://www.linkedin.com/in/luiz-henrique-tomaz-moreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-1.5 px-4 py-2 bg-white hover:bg-[#00AEEF] hover:text-[#002147] text-[#121212] text-xs font-bold transition-all border-2 border-[#121212] shadow-[2px_2px_0px_#121212] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none"
                >
                  <span>LinkedIn</span>
                  <span>↗</span>
                </a>
                <a
                  href="mailto:lhtmoreira@gmail.com"
                  className="inline-flex items-center justify-center gap-1.5 px-4 py-2 bg-white hover:bg-[#00843D] hover:text-white text-[#121212] text-xs font-bold transition-all border-2 border-[#121212] shadow-[2px_2px_0px_#121212] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none"
                >
                  <span>E-mail</span>
                  <span>✉</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ FEATURES ============ */}
      <section id="ferramentas" className="py-24 border-b-4 border-[#121212] bg-[#F5F5F7] scroll-mt-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-16">
            <span className="font-mono text-xs text-[#00843D] uppercase tracking-[0.2em] font-bold block mb-3">No lançamento</span>
            <h2 className="font-heading text-3xl sm:text-4xl font-black text-[#121212] leading-tight">
              Três ferramentas criadas sob medida para o seu PAS 3.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white border-4 border-[#121212] p-8 shadow-[4px_4px_0px_#121212]">
              <span className="inline-block font-mono text-xs font-bold text-[#002147] bg-[#00AEEF] border border-[#121212] px-3 py-1 mb-6">01</span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#121212]">Preditor PAS 3</h3>
              <p className="text-sm text-[#121212]/75 leading-relaxed">
                Insira suas notas do PAS 1 e 2. Nosso modelo prevê seu Argumento Final e sua probabilidade percentual de aprovação no curso que você quer.
              </p>
            </div>

            <div className="bg-white border-4 border-[#121212] p-8 shadow-[4px_4px_0px_#121212]">
              <span className="inline-block font-mono text-xs font-bold text-white bg-[#00843D] border border-[#121212] px-3 py-1 mb-6">02</span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#121212]">Calculadora "Quanto Falta"</h3>
              <p className="text-sm text-[#121212]/75 leading-relaxed">
                Cálculo reverso: saiba exatamente qual Escore Bruto você precisa tirar na prova do PAS 3 para alcançar a nota de corte do seu curso.
              </p>
            </div>

            <div className="bg-white border-4 border-[#121212] p-8 shadow-[4px_4px_0px_#121212]">
              <span className="inline-block font-mono text-xs font-bold text-[#121212] bg-[#E6E6E8] border border-[#121212] px-3 py-1 mb-6">03</span>
              <h3 className="font-heading text-xl font-bold mb-3 text-[#121212]">Análise Histórica</h3>
              <p className="text-sm text-[#121212]/75 leading-relaxed">
                Evolução histórica das notas de corte por curso e estatísticas médias das edições anteriores do PAS/UnB.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============ BUILD IN PUBLIC ============ */}
      <section id="build-in-public" className="bg-white py-24 border-b-4 border-[#121212] scroll-mt-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#121212] bg-[#00AEEF] border border-[#121212] px-3.5 py-1.5 font-bold">
                Build in Public
              </span>
              <h2 className="font-heading text-3xl sm:text-5xl font-black tracking-tight leading-[1.1] text-[#121212]">
                Criado em público. <br />
                Desenhado por <span className="underline decoration-4 decoration-[#00843D]">você</span>.
              </h2>
              <p className="text-base text-[#121212]/80 leading-relaxed">
                O Vetor PAS não está sendo criado a portas fechadas. Acreditamos que a melhor ferramenta para o PAS 3 é aquela construída em parceria com quem realmente vai usá-la em sua preparação.
              </p>
              <p className="text-sm text-[#121212]/70 leading-relaxed">
                Estou desenvolvendo o produto em <strong>lives abertas de código</strong>, onde você pode acompanhar cada linha de programação, sugerir ideias de design, propor novas simulações de nota ou criticar o que não ficou legal.
              </p>
              <div className="pt-2">
                <a
                  href="https://www.youtube.com/@luizhtmoreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2.5 px-5 py-3 rounded-none font-bold text-white bg-[#FF0000] border-2 border-[#121212] transition-all shadow-[4px_4px_0px_#121212] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none text-sm group"
                >
                  <YoutubeLogo size={20} weight="fill" className="text-white" />
                  <span>Acompanhar lives no YouTube ↗</span>
                </a>
              </div>
            </div>

            <div className="lg:col-span-6 grid sm:grid-cols-2 gap-6">
              <div className="bg-[#F5F5F7] border-4 border-[#121212] p-6 shadow-[4px_4px_0px_#121212] space-y-3">
                <div className="w-10 h-10 rounded-none bg-[#00AEEF] border border-[#121212] text-[#002147] flex items-center justify-center">
                  <Television size={22} weight="bold" />
                </div>
                <h3 className="font-heading font-black text-lg text-[#121212]">Lives de Código</h3>
                <p className="text-xs sm:text-sm text-[#121212]/75 leading-relaxed">
                  Acompanhe o desenvolvimento da interface ao vivo.
                </p>
              </div>

              <div className="bg-[#F5F5F7] border-4 border-[#121212] p-6 shadow-[4px_4px_0px_#121212] space-y-3">
                <div className="w-10 h-10 rounded-none bg-[#00843D] border border-[#121212] text-white flex items-center justify-center">
                  <Lightbulb size={22} weight="bold" />
                </div>
                <h3 className="font-heading font-black text-lg text-[#121212]">Opine & Sugira</h3>
                <p className="text-xs sm:text-sm text-[#121212]/75 leading-relaxed">
                  Features, designs, cores: você ajuda decidir o rumo das próximas telas.
                </p>
              </div>

              <div className="bg-[#F5F5F7] border-4 border-[#121212] p-6 shadow-[4px_4px_0px_#121212] space-y-3">
                <div className="w-10 h-10 rounded-none bg-white border border-[#121212] text-[#121212] flex items-center justify-center">
                  <Wrench size={22} weight="bold" />
                </div>
                <h3 className="font-heading font-black text-lg text-[#121212]">Co-Criação</h3>
                <p className="text-xs sm:text-sm text-[#121212]/75 leading-relaxed">
                  Ideal para quem quer aprender arquitetura web e ver decisões de engenharia na prática.
                </p>
              </div>

              <div className="bg-[#F5F5F7] border-4 border-[#121212] p-6 shadow-[4px_4px_0px_#121212] space-y-3">
                <div className="w-10 h-10 rounded-none bg-[#E6E6E8] border border-[#121212] text-[#121212] flex items-center justify-center">
                  <Megaphone size={22} weight="bold" />
                </div>
                <h3 className="font-heading font-black text-lg text-[#121212]">Feito para Você</h3>
                <p className="text-xs sm:text-sm text-[#121212]/75 leading-relaxed">
                  Ao participar das sugestões, você garante que a ferramenta resolverá suas reais dúvidas do PAS 3.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ CTA FINAL ============ */}
      <section className="py-24 bg-[#00843D] text-white border-b-4 border-[#121212] text-center">
        <div className="max-w-4xl mx-auto px-6 space-y-6">
          <span className="inline-block font-mono text-xs text-white bg-[#002147] border border-[#121212] px-3.5 py-1.5 rounded-none shadow-[2px_2px_0px_#121212] mb-6">
            Acesso Antecipado Gratuito
          </span>
          <h2 className="font-heading text-3xl sm:text-5xl font-black tracking-tight leading-tight text-white">
            Não faça a 3ª etapa no escuro.
          </h2>
          <p className="text-base sm:text-lg text-white/90 max-w-xl mx-auto leading-relaxed">
            Faça seu cadastro agora na lista de espera para receber o link de acesso em primeira mão assim que a plataforma for liberada.
          </p>
          <div className="pt-4">
            <button
              onClick={scrollToForm}
              className="inline-flex items-center gap-2 px-8 py-4 rounded-none font-bold text-[#002147] bg-[#00AEEF] hover:bg-[#33C1F3] hover:text-[#002147] border-4 border-[#121212] transition-all shadow-[6px_6px_0px_#121212] active:translate-x-[3px] active:translate-y-[3px] active:shadow-none text-base sm:text-lg"
            >
              <span>Quero Garantir Meu Acesso</span>
              <ArrowRight size={18} />
            </button>
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="bg-white text-[#121212]/60 py-10 text-xs border-t-2 border-[#121212]">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <p>© {new Date().getFullYear()} Vetor PAS — Projeto independente sem vínculo oficial com a UnB ou o Cebraspe.</p>
          <div className="flex items-center gap-5">
            <a
              href="https://www.linkedin.com/in/luiz-henrique-tomaz-moreira"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#002147] font-bold hover:underline"
            >
              LinkedIn ↗
            </a>
            <a href="mailto:lhtmoreira@gmail.com" className="text-[#121212] font-bold hover:underline">
              lhtmoreira@gmail.com
            </a>
            <span className="font-bold text-[#00843D]">Variante C</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
