import Link from "next/link";
import { CurvaGaussiana } from "@/components/brand/CurvaGaussiana";
import { PublicHeader } from "@/components/public/PublicHeader";

const FEATURES_PUBLICAS = [
  {
    titulo: "Preditor PAS 3",
    href: "/predict",
    descricao:
      "Insira seus escores das duas primeiras etapas e receba a previsão do seu Argumento Final — com a probabilidade de aprovação no curso que você quer.",
    detalhe: "ensemble de 4 modelos · ponderado pela sua volatilidade",
  },
  {
    titulo: "Calculadora de Estratégia",
    href: "/calculadora",
    descricao:
      "Defina seu curso alvo e descubra qual escore você precisa tirar na Parte 2 do PAS 3 para alcançar a nota de corte.",
    detalhe: "reality check histórico · customização de expectativas",
  },
  {
    titulo: "Análise Temporal",
    href: "/temporal",
    descricao:
      "Veja como as notas de corte e as médias das provas evoluíram etapa a etapa, e entenda onde você está em relação ao histórico.",
    detalhe: "séries históricas oficiais · projeção por regressão",
  },
];

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

const PASSOS = [
  {
    numero: "01",
    titulo: "Informe seus escores",
    descricao:
      "Escore Bruto e Redação do PAS 1 e do PAS 2 — os mesmos números que o Cebraspe publicou no seu boletim.",
  },
  {
    numero: "02",
    titulo: "O ensemble decide",
    descricao:
      "Quatro modelos preveem sua terceira etapa. Alunos consistentes pesam mais na regressão linear; trajetórias voláteis, nos modelos de árvore.",
  },
  {
    numero: "03",
    titulo: "Probabilidade, não chute",
    descricao:
      "Convertemos a previsão em probabilidade de aprovação para o seu curso, usando a distribuição do erro do modelo sobre a nota de corte.",
  },
];

/**
 * Acento por posição no grid — a rotação cyan → verde → navy da identidade.
 * As classes ficam escritas por extenso porque o Tailwind varre o código-fonte:
 * `group-hover:${variável}` nunca chegaria a existir no CSS gerado.
 */
const ACENTOS = [
  {
    borderTop: "border-t-[#00AEEF]",
    text: "text-[#00AEEF]",
    chip: "bg-[#00AEEF]/10",
    hoverText: "group-hover:text-[#00AEEF]",
  },
  {
    borderTop: "border-t-[#00843D]",
    text: "text-[#00843D]",
    chip: "bg-[#00843D]/10",
    hoverText: "group-hover:text-[#00843D]",
  },
  {
    borderTop: "border-t-[#002147]",
    text: "text-[#002147]",
    chip: "bg-[#002147]/5",
    hoverText: "group-hover:text-[#002147]",
  },
];

export function LandingPage() {
  return (
    <div className="landing-root bg-[#F8F9FA] text-[#1D1D1F] min-h-screen selection:bg-[#00843D] selection:text-white font-sans antialiased overflow-x-clip">
      <PublicHeader />

      {/* ══════════════════════════════════════════════════════════════════
           Wrapper do topo: Hero + curva.
           `.vp-wash` põe um único gradiente radial atrás dos dois, sumindo
           conforme desce — cheio no hero, invisível antes da seção 2.
          ══════════════════════════════════════════════════════════════════ */}
      <div className="vp-wash relative bg-white overflow-hidden">
        {/* ============ HERO ============ */}
        <header className="relative z-10 text-[#002147] pt-16 pb-0 sm:pt-24">
          <div className="relative z-10 max-w-6xl mx-auto px-6">
            <div className="max-w-3xl">
              <span className="landing-reveal vp-eyebrow">
                Análise preditiva · PAS/UnB
              </span>
              <h1
                className="landing-reveal font-heading text-4xl sm:text-6xl font-extrabold tracking-tight leading-[1.08] text-[#002147] mt-6 mb-6"
                style={{ animationDelay: "90ms" }}
              >
                Sua chance de passar,
                <br />
                <span className="text-[#00843D] relative inline-block">
                  calculada
                  <span className="absolute bottom-1 left-0 w-full h-[4px] bg-[#00AEEF]/40 rounded-full" />
                </span>{" "}
                — não chutada.
              </h1>
              <p
                className="landing-reveal text-lg sm:text-xl text-[#4A5568] leading-relaxed max-w-xl mb-9"
                style={{ animationDelay: "180ms" }}
              >
                O Vetor PAS usa machine learning treinado em dados históricos do
                PAS para prever seu Argumento Final e sua probabilidade real de
                aprovação na UnB — antes da terceira etapa.
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

              <dl
                className="landing-reveal flex flex-wrap gap-x-10 gap-y-4 mt-12 mb-4"
                style={{ animationDelay: "360ms" }}
              >
                {[
                  ["±13,49", "erro médio (RMSE) do modelo"],
                  ["4", "modelos no ensemble dinâmico"],
                  ["3", "etapas do PAS no cálculo oficial"],
                ].map(([valor, rotulo]) => (
                  <div key={rotulo}>
                    <dt className="sr-only">{rotulo}</dt>
                    <dd className="font-mono text-2xl font-black text-[#002147] tabular-nums">
                      {valor}
                    </dd>
                    <dd className="text-[0.8rem] text-[#718096] mt-0.5">{rotulo}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </header>

        {/* a curva é o produto: distribuição do argumento previsto vs. nota de corte */}
        <div className="relative z-0 -mt-6 sm:-mt-16">
          <CurvaGaussiana />
        </div>

        {/* Borda de separação para a seção seguinte */}
        <div className="border-t border-[#E2E8F0]" />
      </div>

      {/* ============ COMO FUNCIONA ============ */}
      <section className="bg-[#F8F9FA] py-20 sm:py-24 border-b border-[#E2E8F0]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-14 space-y-4">
            <span className="vp-eyebrow">Como funciona</span>
            <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-[#002147]">
              Três passos entre o seu boletim e uma resposta honesta.
            </h2>
          </div>
          <ol className="grid md:grid-cols-3 gap-6">
            {PASSOS.map((passo, i) => {
              const acento = ACENTOS[i % ACENTOS.length];
              return (
                <li
                  key={passo.numero}
                  className={`group vp-card vp-card-lift border-t-4 ${acento.borderTop} p-8`}
                >
                  <span
                    className={`inline-block font-mono text-xs font-bold ${acento.text} ${acento.chip} px-3 py-1 rounded-md mb-6 transition-transform group-hover:scale-110`}
                  >
                    {passo.numero}
                  </span>
                  <h3
                    className={`font-heading text-xl font-bold mb-3 text-[#002147] transition-colors ${acento.hoverText}`}
                  >
                    {passo.titulo}
                  </h3>
                  <p className="text-sm text-[#4A5568] leading-relaxed">{passo.descricao}</p>
                </li>
              );
            })}
          </ol>
        </div>
      </section>

      {/* ============ FERRAMENTAS PÚBLICAS ============ */}
      <section className="bg-white py-20 sm:py-28 border-b border-[#E2E8F0]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mb-14 space-y-4">
            <span className="vp-eyebrow">Para estudantes · grátis, sem conta</span>
            <h2 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-[#002147]">
              Duas ferramentas abertas para quem está na disputa.
            </h2>
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            {FEATURES_PUBLICAS.map((feature, i) => {
              const acento = ACENTOS[i % ACENTOS.length];
              return (
                <Link
                  key={feature.href}
                  href={feature.href}
                  className={`group relative block vp-card vp-card-lift border-t-4 ${acento.borderTop} bg-[#F8F9FA] p-8 sm:p-10 overflow-hidden`}
                >
                  <h3 className="font-heading text-2xl font-bold tracking-tight text-[#002147] mb-3 flex items-center gap-3">
                    {feature.titulo}
                    <span className={`${acento.text} transition-transform group-hover:translate-x-1.5`}>
                      →
                    </span>
                  </h3>
                  <p className="text-[0.97rem] leading-relaxed text-[#4A5568] mb-5">
                    {feature.descricao}
                  </p>
                  <p className="font-mono text-[0.72rem] tracking-[0.14em] uppercase text-[#718096] font-bold">
                    {feature.detalhe}
                  </p>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* ============ B2B / ESCOLAS ============ */}
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

      {/* ============ CTA FINAL ============ */}
      <section className="py-20 bg-white border-b border-[#E2E8F0]">
        <div className="max-w-5xl mx-auto px-6">
          <div className="vp-banner py-16 px-8 sm:px-16 text-center text-white">
            <div className="relative z-10 space-y-6">
              <span className="vp-eyebrow vp-eyebrow-on-dark">Acesso gratuito · sem conta</span>
              <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">
                A terceira etapa ainda não aconteceu.
                <br />
                <span className="text-[#00AEEF]">O seu plano pode.</span>
              </h2>
              <div className="pt-4">
                <Link href="/predict" className="vp-btn vp-btn-cyan px-8 py-4 text-base sm:text-lg">
                  Calcular minha previsão →
                </Link>
              </div>
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
