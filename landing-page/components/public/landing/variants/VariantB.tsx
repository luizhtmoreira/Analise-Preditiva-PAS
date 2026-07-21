import { BrandMark } from "@/components/brand/BrandMark";
import { WaitlistForm } from "../WaitlistForm";

export function VariantB() {
  return (
    <div className="landing-root bg-[#001D3D] text-white min-h-screen selection:bg-[#00843D] selection:text-white">
      {/* ============ HEADER ============ */}
      <header className="border-b border-white/10 bg-[#001730]">
        <nav className="max-w-6xl mx-auto flex items-center justify-between px-6 py-5">
          <BrandMark />
          <span className="font-mono text-xs bg-[#00843D]/20 text-[#33D17A] border border-[#00843D]/40 px-3 py-1 rounded-full uppercase tracking-wider">
            História & Lançamento
          </span>
        </nav>
      </header>

      {/* ============ HERO (NARRATIVE HOOK) ============ */}
      <section className="py-16 sm:py-24 max-w-4xl mx-auto px-6 text-center space-y-6">
        <span className="inline-block font-mono text-xs text-[#00AEEF] uppercase tracking-[0.2em] bg-[#00AEEF]/10 border border-[#00AEEF]/30 px-3 py-1 rounded-md">
          Pré-Lançamento Exclusivo PAS 3
        </span>
        <h1 className="font-heading text-4xl sm:text-6xl font-extrabold tracking-tight leading-tight">
          Chega de tentar adivinhar se sua nota do PAS vai dar para a UnB.
        </h1>
        <p className="text-lg sm:text-xl text-white/75 leading-relaxed max-w-2xl mx-auto">
          Um modelo estatístico criado por quem passou exatamente pela mesma dúvida que você está passando agora.
        </p>
      </section>

      {/* ============ HISTÓRIA EM DESTAQUE (STORY FIRST) ============ */}
      <section className="pb-16 max-w-3xl mx-auto px-6">
        <div className="bg-[#002147] border border-[#00AEEF]/30 rounded-3xl p-8 sm:p-12 shadow-2xl relative">
          <div className="absolute top-0 right-10 -translate-y-1/2 bg-[#00AEEF] text-[#002147] font-mono text-xs font-bold uppercase tracking-wider px-4 py-1.5 rounded-full shadow-lg">
            Carta do Criador
          </div>
          <h2 className="font-heading text-2xl sm:text-3xl font-bold mb-6 text-white leading-snug">
            "No meu PAS 3, eu perdi semanas sem saber quanto precisava tirar na prova."
          </h2>
          <div className="space-y-4 text-white/80 leading-relaxed text-sm sm:text-base">
            <p>
              Em vez de focar 100% nos estudos para a terceira etapa, eu ficava procurando notas de corte antigas do Cebraspe e tentando fazer contas manuais que não me davam segurança alguma.
            </p>
            <p>
              Anos depois, combinando ciência de dados e o histórico oficial da UnB, desenvolvi o <strong>Vetor PAS</strong>. Ele analisa a volatilidade das suas notas e calcula a probabilidade real de aprovação no curso que você escolher.
            </p>
            <p>
              Estamos finalizando os ajustes do sistema para o PAS 3 deste ano. Cadastre-se abaixo para garantir seu acesso prioritário assim que abrirmos as vagas!
            </p>
          </div>
        </div>
      </section>

      {/* ============ SEÇÃO DA LISTA DE ESPERA (FORMULÁRIO IMERSIVO) ============ */}
      <section id="waitlist-section" className="py-16 bg-[#001730] border-t border-white/10">
        <div className="max-w-xl mx-auto px-6 space-y-6">
          <div className="text-center space-y-2">
            <h2 className="font-heading text-2xl sm:text-3xl font-bold text-white">
              Garanta seu Acesso Antecipado
            </h2>
            <p className="text-sm text-white/60">
              Receba o aviso em primeira mão no seu e-mail assim que o Preditor PAS 3 for liberado.
            </p>
          </div>
          <WaitlistForm variantStyle="card" buttonText="Entrar na Lista VIP" />
        </div>
      </section>

      {/* ============ B2B TEASER ============ */}
      <section className="py-16 bg-[#002147] border-t border-white/10">
        <div className="max-w-4xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-8 bg-[#00843D]/20 border border-[#00843D]/40 rounded-2xl p-8">
          <div className="space-y-2">
            <span className="font-mono text-xs text-[#33D17A] uppercase tracking-wider">Para Coordenadores & Escolas</span>
            <h3 className="font-heading text-xl font-bold text-white">Sua escola quer o relatório completo da turma?</h3>
            <p className="text-xs text-white/70">Acompanhamento whitelabel de risco por aluno e comparativo com a população do PAS.</p>
          </div>
          <a
            href="mailto:contato@vetorpas.com.br"
            className="px-6 py-3 rounded-xl bg-[#00843D] hover:bg-[#009c48] font-semibold text-white text-sm whitespace-nowrap transition-colors"
          >
            Falar com a equipe
          </a>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="border-t border-white/10 py-8 text-xs text-white/50 bg-[#001226] text-center">
        <p>© {new Date().getFullYear()} Vetor PAS — Projeto independente sem vínculo oficial com a UnB ou o Cebraspe.</p>
        <p className="font-mono text-[#00AEEF] mt-1">Variante B — Narrative Story First</p>
      </footer>
    </div>
  );
}
