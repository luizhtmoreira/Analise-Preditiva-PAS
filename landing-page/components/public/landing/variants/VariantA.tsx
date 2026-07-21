"use client";

import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "../WaitlistForm";
import { Television, Lightbulb, Wrench, Megaphone, YoutubeLogo, CaretRight, ArrowUpRight } from "@phosphor-icons/react";

export function VariantA() {
  const scrollToForm = (e: React.MouseEvent) => {
    e.preventDefault();
    const formElement = document.getElementById("lista-espera");
    if (formElement) {
      formElement.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="landing-root bg-[#080B11] text-[#E2E8F0] min-h-screen selection:bg-[#00AEEF] selection:text-black font-sans antialiased overflow-x-hidden">
      {/* Glowes de Fundo Ciberpunk */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-[#00AEEF]/5 blur-[120px] pointer-events-none" />
      <div className="absolute top-[20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-[#00843D]/5 blur-[150px] pointer-events-none" />

      {/* ============ STICKY NAVBAR ============ */}
      <nav className="sticky top-0 z-50 w-full backdrop-blur-lg bg-[#080B11]/75 border-b border-white/5 transition-all duration-300">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4">
          <BrandMark />
          <div className="flex items-center gap-5 sm:gap-8 text-xs sm:text-sm font-mono text-white/60">
            <a
              href="#historia"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("historia")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00AEEF] hover:text-glow transition-colors cursor-pointer"
            >
              [historia]
            </a>
            <a
              href="#ferramentas"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("ferramentas")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00AEEF] hover:text-glow transition-colors cursor-pointer"
            >
              [ferramentas]
            </a>
            <a
              href="#build-in-public"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("build-in-public")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="hover:text-[#00AEEF] hover:text-glow transition-colors cursor-pointer"
            >
              [build_in_public]
            </a>
          </div>
        </div>
      </nav>

      {/* ============ HERO SECTION ============ */}
      <header className="relative z-20 pt-16 pb-24 border-b border-white/5">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            {/* Texto */}
            <div className="lg:col-span-7 space-y-8">
              <div className="inline-flex items-center gap-2 font-mono text-xs text-[#00AEEF] bg-[#00AEEF]/5 border border-[#00AEEF]/20 px-3 py-1.5 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00AEEF] animate-ping" />
                <span>ANÁLISE PREDITIVA ATIVA</span>
              </div>
              <h1 className="font-heading text-4xl sm:text-6xl font-black tracking-tight leading-[1.05] text-white">
                Sua aprovação na UnB, <br />
                <span className="bg-gradient-to-r from-[#00AEEF] via-[#7FD8F7] to-[#00843D] bg-clip-text text-transparent">
                  calculada com precisão.
                </span>
              </h1>
              <p className="text-lg text-white/70 leading-relaxed max-w-xl">
                O Vetor PAS combina IA e dados oficiais do Cebraspe para prever seu Argumento Final e calcular a chance real de você passar no seu curso no PAS 3.
              </p>
            </div>

            {/* Form */}
            <div id="lista-espera" className="lg:col-span-5 scroll-mt-24">
              <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-[#00AEEF] to-[#00843D] rounded-2xl blur opacity-30 group-hover:opacity-40 transition duration-300" />
                <div className="relative">
                  <WaitlistForm variantStyle="card" buttonText="Garantir Meu Acesso Antecipado" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ============ FOUNDER STORY ============ */}
      <section id="historia" className="py-24 border-b border-white/5 scroll-mt-20 relative">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-[#0D131F]/80 border border-white/10 rounded-2xl p-8 sm:p-12 relative overflow-hidden shadow-2xl">
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-[#00843D] to-transparent" />
            <div className="flex items-center gap-2 mb-6">
              <span className="font-mono text-xs text-[#00843D] uppercase tracking-[0.2em] font-bold bg-[#00843D]/10 px-2.5 py-1 rounded">
                // De Aluno para Aluno
              </span>
            </div>

            <h2 className="font-heading text-2xl sm:text-3xl font-bold mb-6 text-white leading-snug">
              "Eu já estive no seu lugar no PAS 3, e sei como é a sensação de não saber se vai dar pra chegar na nota necessária."
            </h2>
            <div className="space-y-6 text-white/70 leading-relaxed text-sm sm:text-base">
              <p>
                Quando eu estava estudando para o PAS 3, passei meses tentando adivinhar se era possível passar no curso que eu queria. A nota necessária eu sabia, mas qual era a chance de eu tirar aquela nota? Procurava na internet, mas nada me dava uma resposta real sobre a probabilidade de entrar no meu curso na UnB, se valia a pena tentar ele ou mudar a rota para garantir minha aprovação.
              </p>
              <p>
                Foi por isso que criei o <strong className="text-white">Vetor PAS</strong>: para que você não precise estudar no escuro. Usamos algoritmos de machine learning sobre os boletins oficiais do Cebraspe para te mostrar exatamente onde você está e o que precisa fazer na terceira etapa. Assim, você não precisa esperar a nota do PAS 3 sair para saber se tinha chance ou não.
              </p>
            </div>
            
            <div className="mt-10 pt-8 border-t border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full border border-[#00843D]/40 shadow-inner overflow-hidden relative bg-[#001D3A]">
                  <img
                    src="/luiz.jpeg"
                    alt="Luiz Moreira"
                    className="w-full h-full object-cover object-top"
                  />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white font-mono">Luiz Moreira</p>
                  <p className="text-xs text-white/50">Criador do Vetor PAS · Software Engineering @ UnB</p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <a
                  href="https://www.linkedin.com/in/luiz-henrique-tomaz-moreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-[#00AEEF] hover:text-black text-white text-xs font-mono transition-all border border-white/10"
                >
                  <span>linkedin</span>
                  <ArrowUpRight size={12} />
                </a>
                <a
                  href="mailto:lhtmoreira@gmail.com"
                  className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-[#00843D] text-white text-xs font-mono transition-all border border-white/10"
                >
                  <span>email</span>
                  <ArrowUpRight size={12} />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ FEATURES ============ */}
      <section id="ferramentas" className="py-24 border-b border-white/5 bg-[#0A0F17] scroll-mt-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-16">
            <span className="font-mono text-xs text-[#00843D] uppercase tracking-[0.2em] block mb-3">// no_lancamento</span>
            <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
              Três ferramentas criadas sob medida para o seu PAS 3.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="group relative bg-[#0D131F]/60 border border-white/5 rounded-xl p-8 hover:border-[#00AEEF]/30 hover:bg-[#0D131F]/90 transition-all duration-300">
              <span className="font-mono text-xs font-bold text-[#00AEEF] block mb-4">[01]</span>
              <h3 className="font-heading text-lg font-bold mb-3 text-white">Preditor PAS 3</h3>
              <p className="text-sm text-white/60 leading-relaxed">
                Insira suas notas do PAS 1 e 2. Nosso modelo prevê seu Argumento Final e sua probabilidade percentual de aprovação no curso que você quer.
              </p>
            </div>

            <div className="group relative bg-[#0D131F]/60 border border-white/5 rounded-xl p-8 hover:border-[#00843D]/30 hover:bg-[#0D131F]/90 transition-all duration-300">
              <span className="font-mono text-xs font-bold text-[#00843D] block mb-4">[02]</span>
              <h3 className="font-heading text-lg font-bold mb-3 text-white">Calculadora "Quanto Falta"</h3>
              <p className="text-sm text-white/60 leading-relaxed">
                Cálculo reverso: saiba exatamente qual Escore Bruto você precisa tirar na prova do PAS 3 para alcançar a nota de corte do seu curso.
              </p>
            </div>

            <div className="group relative bg-[#0D131F]/60 border border-white/5 rounded-xl p-8 hover:border-white/20 hover:bg-[#0D131F]/90 transition-all duration-300">
              <span className="font-mono text-xs font-bold text-white/40 block mb-4">[03]</span>
              <h3 className="font-heading text-lg font-bold mb-3 text-white">Análise Histórica</h3>
              <p className="text-sm text-white/60 leading-relaxed">
                Evolução histórica das notas de corte por curso e estatísticas médias das edições anteriores do PAS/UnB.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============ BUILD IN PUBLIC ============ */}
      <section id="build-in-public" className="bg-[#080B11] py-24 border-b border-white/5 scroll-mt-20 relative">
        <div className="max-w-6xl mx-auto px-6 z-10 relative">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-6 space-y-6">
              <span className="inline-block font-mono text-[0.78rem] tracking-[0.2em] uppercase text-[#00AEEF] bg-[#00AEEF]/5 border border-[#00AEEF]/20 px-3 py-1.5 rounded-lg">
                // build_in_public
              </span>
              <h2 className="font-heading text-3xl sm:text-5xl font-black tracking-tight leading-[1.1] text-white">
                Criado em público. <br />
                Desenhado por <span className="text-[#00AEEF]">você</span>.
              </h2>
              <p className="text-base text-white/70 leading-relaxed">
                O Vetor PAS não está sendo criado a portas fechadas. Acreditamos que a melhor ferramenta para o PAS 3 é aquela construída em parceria com quem realmente vai usá-la em sua preparação.
              </p>
              <p className="text-sm text-white/60 leading-relaxed">
                Estou desenvolvendo o produto em <strong>lives abertas de código</strong>, onde você pode acompanhar cada linha de programação, sugerir ideias de design, propor novas simulações de nota ou criticar o que não ficou legal.
              </p>
              <div className="pt-2">
                <a
                  href="https://www.youtube.com/@luizhtmoreira"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-3 rounded-lg font-mono font-bold text-white bg-white/5 hover:bg-[#FF0000] hover:text-white border border-white/10 transition-all text-xs uppercase tracking-wider group"
                >
                  <YoutubeLogo size={18} weight="fill" className="text-[#FF0000] group-hover:text-white transition-colors" />
                  <span>Acompanhar lives no youtube ↗</span>
                </a>
              </div>
            </div>

            <div className="lg:col-span-6 grid sm:grid-cols-2 gap-6">
              <div className="bg-[#0D131F]/40 border border-white/5 p-6 rounded-xl space-y-3">
                <div className="w-10 h-10 rounded-lg bg-[#00AEEF]/10 text-[#00AEEF] flex items-center justify-center">
                  <Television size={20} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-white">Lives de Código</h3>
                <p className="text-xs text-white/60 leading-relaxed">
                  Acompanhe o desenvolvimento da interface ao vivo.
                </p>
              </div>

              <div className="bg-[#0D131F]/40 border border-white/5 p-6 rounded-xl space-y-3">
                <div className="w-10 h-10 rounded-lg bg-[#00843D]/10 text-[#00843D] flex items-center justify-center">
                  <Lightbulb size={20} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-white">Opine & Sugira</h3>
                <p className="text-xs text-white/60 leading-relaxed">
                  Features, designs, cores: você ajuda decidir o rumo das próximas telas.
                </p>
              </div>

              <div className="bg-[#0D131F]/40 border border-white/5 p-6 rounded-xl space-y-3">
                <div className="w-10 h-10 rounded-lg bg-white/5 text-white/80 flex items-center justify-center">
                  <Wrench size={20} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-white">Co-Criação</h3>
                <p className="text-xs text-white/60 leading-relaxed">
                  Ideal para quem quer aprender arquitetura web e ver decisões de engenharia na prática.
                </p>
              </div>

              <div className="bg-[#0D131F]/40 border border-white/5 p-6 rounded-xl space-y-3">
                <div className="w-10 h-10 rounded-lg bg-[#7FD8F7]/10 text-[#7FD8F7] flex items-center justify-center">
                  <Megaphone size={20} weight="duotone" />
                </div>
                <h3 className="font-heading font-bold text-white">Feito para Você</h3>
                <p className="text-xs text-white/60 leading-relaxed">
                  Ao participar das sugestões, você garante que a ferramenta resolverá suas reais dúvidas do PAS 3.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ CTA FINAL ============ */}
      <section className="py-24 bg-[#0A0F17] relative overflow-hidden text-center">
        <div className="absolute top-[80%] left-[50%] -translate-x-1/2 w-[70%] h-[70%] rounded-full bg-[#00AEEF]/5 blur-[120px] pointer-events-none" />
        <div className="max-w-3xl mx-auto px-6 space-y-8 relative z-10">
          <span className="inline-block font-mono text-xs text-[#7FD8F7] uppercase tracking-[0.2em] bg-white/5 border border-white/10 px-4 py-1.5 rounded-full mb-4">
            [acesso_antecipado_gratuito]
          </span>
          <h2 className="font-heading text-3xl sm:text-5xl font-black tracking-tight text-white leading-tight">
            Não faça a 3ª etapa no escuro.
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-xl mx-auto leading-relaxed">
            Faça seu cadastro agora na lista de espera para receber o link de acesso em primeira mão assim que a plataforma for liberada.
          </p>
          <div className="pt-4">
            <button
              onClick={scrollToForm}
              className="inline-flex items-center gap-2 px-8 py-4 rounded-lg font-mono font-bold text-black bg-[#00AEEF] hover:bg-[#33C1F3] transition-all hover:shadow-[0_0_25px_rgba(0,174,239,0.4)] active:scale-95 text-sm uppercase tracking-wider"
            >
              <span>GARANTIR ACESSO</span>
              <CaretRight size={16} weight="bold" />
            </button>
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="bg-[#080B11] text-white/40 border-t border-white/5 py-12 text-xs">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-6 text-center sm:text-left">
          <p>© {new Date().getFullYear()} Vetor PAS — Projeto independente sem vínculo oficial com a UnB ou o Cebraspe.</p>
          <div className="flex items-center gap-6 font-mono">
            <a
              href="https://www.linkedin.com/in/luiz-henrique-tomaz-moreira"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#00AEEF] hover:underline"
            >
              linkedin ↗
            </a>
            <a href="mailto:lhtmoreira@gmail.com" className="text-white/60 hover:underline">
              lhtmoreira@gmail.com
            </a>
            <span className="text-white/20">|</span>
            <span className="text-[#00AEEF]/80">variante_a</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
