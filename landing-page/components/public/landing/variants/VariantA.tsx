import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "../WaitlistForm";

export function VariantA() {
  return (
    <div className="landing-root bg-[#002147] text-white min-h-screen selection:bg-[#00AEEF] selection:text-[#002147]">
      {/* ============ HEADER ============ */}
      <header className="relative z-20 border-b border-white/10">
        <nav className="max-w-6xl mx-auto flex items-center justify-between px-6 py-5">
          <BrandMark />
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-block font-mono text-xs bg-[#00AEEF]/15 border border-[#00AEEF]/40 text-[#7FD8F7] px-3 py-1 rounded-full uppercase tracking-wider">
              🚀 Pré-Lançamento PAS 3
            </span>
            <a
              href="#lista-espera"
              className="px-4 py-2 rounded-lg font-medium text-xs sm:text-sm bg-white/10 hover:bg-white/20 border border-white/20 transition-all"
            >
              Garantir vaga
            </a>
          </div>
        </nav>
      </header>

      {/* ============ HERO (SPLIT FORM) ============ */}
      <section className="relative overflow-hidden pt-12 pb-20 sm:pt-20 sm:pb-28">
        {/* Background Grid */}
        <div
          className="absolute inset-0 pointer-events-none opacity-40"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />

        <div className="relative z-10 max-w-6xl mx-auto px-6">
          <div className="grid lg:grid-cols-12 gap-12 items-center">
            {/* Esquerda: Texto */}
            <div className="lg:col-span-7 space-y-6">
              <span className="inline-block font-mono text-xs text-[#7FD8F7] tracking-[0.2em] uppercase bg-white/5 border border-white/10 px-3 py-1 rounded-md">
                Inteligência Preditiva para o PAS 3
              </span>
              <h1 className="font-heading text-4xl sm:text-6xl font-extrabold tracking-tight leading-[1.08]">
                Sua aprovação na UnB,{" "}
                <span className="text-[#00AEEF]">calculada com precisão.</span>
              </h1>
              <p className="text-lg sm:text-xl text-white/75 leading-relaxed max-w-2xl">
                O Vetor PAS combina machine learning e histórico oficial do Cebraspe para prever seu Argumento Final e o quanto falta para a nota de corte do seu curso.
              </p>

              {/* Destaques rápidos */}
              <div className="grid grid-cols-3 gap-4 pt-4 border-t border-white/10">
                <div>
                  <p className="font-mono text-xl sm:text-2xl font-bold text-[#00AEEF]">4 Modelos</p>
                  <p className="text-xs text-white/60">Ensemble preditivo</p>
                </div>
                <div>
                  <p className="font-mono text-xl sm:text-2xl font-bold text-[#00843D]">100% Cebraspe</p>
                  <p className="text-xs text-white/60">Pesos oficiais</p>
                </div>
                <div>
                  <p className="font-mono text-xl sm:text-2xl font-bold text-white">Quanto Falta</p>
                  <p className="text-xs text-white/60">Calculadora reversa</p>
                </div>
              </div>
            </div>

            {/* Direita: Formulário */}
            <div id="lista-espera" className="lg:col-span-5 scroll-mt-10">
              <WaitlistForm variantStyle="card" buttonText="Quero Acesso Antecipado" />
            </div>
          </div>
        </div>
      </section>

      {/* ============ HISTÓRIA DO FUNDADOR ============ */}
      <section className="bg-[#001730] py-20 border-y border-white/10">
        <div className="max-w-4xl mx-auto px-6">
          <div className="bg-[#002147] border border-white/15 rounded-3xl p-8 sm:p-12 relative overflow-hidden shadow-xl">
            <span className="font-mono text-xs text-[#00AEEF] uppercase tracking-[0.2em] block mb-3">
              De Aluno para Aluno
            </span>
            <h2 className="font-heading text-2xl sm:text-4xl font-bold mb-6 text-white">
              "Eu já estive no seu lugar no PAS 3 — e sei a angústia que é não saber onde você está."
            </h2>
            <div className="space-y-4 text-white/80 leading-relaxed text-sm sm:text-base">
              <p>
                Quando eu estava prestando o PAS 3, passei meses tentando adivinhar se as minhas notas do PAS 1 e 2 seriam suficientes. Procurava planilhas e cálculos antigos na internet, mas nenhum me dava uma resposta real sobre a probabilidade de entrar no meu curso na UnB.
              </p>
              <p>
                Foi por isso que criei o <strong>Vetor PAS</strong>: para que você não precise estudar no escuro. Usamos algoritmos de machine learning sobre os boletins oficiais do Cebraspe para te mostrar exatamente onde você está e o que precisa fazer na terceira etapa.
              </p>
            </div>
            <div className="mt-8 pt-6 border-t border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-[#00AEEF] text-[#002147] font-bold flex items-center justify-center font-mono">
                  VP
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Criador do Vetor PAS</p>
                  <p className="text-xs text-white/50">Ex-estudante do PAS/UnB</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ RECURSOS TEASER ============ */}
      <section className="py-20 sm:py-28 bg-[#002147]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <span className="font-mono text-xs text-[#00843D] uppercase tracking-[0.2em]">
              O que você vai acessar no lançamento
            </span>
            <h2 className="font-heading text-3xl sm:text-4xl font-bold">
              Três ferramentas criadas para o seu PAS 3
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-[#001D3D] border border-white/10 rounded-2xl p-8 hover:border-[#00AEEF]/50 transition-all">
              <div className="w-12 h-12 bg-[#00AEEF]/20 text-[#00AEEF] rounded-xl flex items-center justify-center font-mono text-xl font-bold mb-6">
                01
              </div>
              <h3 className="font-heading text-xl font-bold mb-3 text-white">Preditor PAS 3</h3>
              <p className="text-sm text-white/70 leading-relaxed">
                Insira suas notas do PAS 1 e 2. Nosso modelo prevê seu Argumento Final e sua probabilidade percentual de aprovação.
              </p>
            </div>

            <div className="bg-[#001D3D] border border-white/10 rounded-2xl p-8 hover:border-[#00843D]/50 transition-all">
              <div className="w-12 h-12 bg-[#00843D]/20 text-[#00843D] rounded-xl flex items-center justify-center font-mono text-xl font-bold mb-6">
                02
              </div>
              <h3 className="font-heading text-xl font-bold mb-3 text-white">Calculadora "Quanto Falta"</h3>
              <p className="text-sm text-white/70 leading-relaxed">
                Cálculo reverso: saiba exatamente qual Escore Bruto você precisa tirar na prova do PAS 3 para alcançar o curso dos seus sonhos.
              </p>
            </div>

            <div className="bg-[#001D3D] border border-white/10 rounded-2xl p-8 hover:border-white/30 transition-all">
              <div className="w-12 h-12 bg-white/10 text-white rounded-xl flex items-center justify-center font-mono text-xl font-bold mb-6">
                03
              </div>
              <h3 className="font-heading text-xl font-bold mb-3 text-white">Análise Histórica Cebraspe</h3>
              <p className="text-sm text-white/70 leading-relaxed">
                Evolução histórica das notas de corte por curso e estatísticas médias das edições anteriores do PAS/UnB.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="border-t border-white/10 py-10 text-xs text-white/50 bg-[#001730]">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <p>© {new Date().getFullYear()} Vetor PAS — Projeto independente sem vínculo oficial com a UnB ou o Cebraspe.</p>
          <p className="font-mono text-[#7FD8F7]">Variante A — Hero Split Form</p>
        </div>
      </footer>
    </div>
  );
}
