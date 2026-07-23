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

export function LandingPage() {
  return (
    <div className="landing-root">
      <PublicHeader />
      {/* ============ HERO ============ */}
      <header
        className="relative overflow-hidden text-white"
        style={{
          background:
            "linear-gradient(168deg, #002147 0%, #003366 52%, #003A70 100%)",
        }}
      >
        {/* grade de fundo, sutil */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px)",
            backgroundSize: "56px 56px",
            maskImage: "radial-gradient(ellipse 90% 70% at 50% 30%, black, transparent)",
          }}
        />

        <div className="relative z-10 max-w-6xl mx-auto px-6 pt-16 pb-0 sm:pt-24">
          <div className="max-w-3xl">
            <p className="landing-reveal font-mono text-[0.78rem] tracking-[0.22em] uppercase text-[#7FD8F7] mb-5">
              Análise preditiva · PAS/UnB
            </p>
            <h1
              className="landing-reveal font-heading text-[2.6rem] leading-[1.04] sm:text-6xl font-extrabold tracking-[-0.03em] mb-6"
              style={{ animationDelay: "90ms" }}
            >
              Sua chance de passar,
              <br />
              <span className="text-[#00AEEF]">calculada</span> — não chutada.
            </h1>
            <p
              className="landing-reveal text-lg sm:text-xl text-white/70 leading-relaxed max-w-xl mb-9"
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
              <Link
                href="/predict"
                className="group inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-[#002147] bg-[#00AEEF] hover:bg-[#33C1F3] transition-all hover:-translate-y-0.5 shadow-[0_8px_30px_rgba(0,174,239,0.35)]"
              >
                Calcular minha previsão
                <span className="transition-transform group-hover:translate-x-1">→</span>
              </Link>
              <Link
                href="/temporal"
                className="inline-flex items-center justify-center px-7 py-3.5 rounded-xl font-semibold text-white border border-white/25 hover:bg-white/10 transition-colors"
              >
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
                  <dd className="font-mono text-2xl font-bold text-white">{valor}</dd>
                  <dd className="text-[0.8rem] text-white/55 mt-0.5">{rotulo}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        {/* a curva é o produto: distribuição do argumento previsto vs. nota de corte */}
        <div className="relative z-0 -mt-6 sm:-mt-16">
          <CurvaGaussiana />
        </div>
      </header>

      {/* ============ COMO FUNCIONA ============ */}
      <section className="bg-[#F5F5F7] py-20 sm:py-28">
        <div className="max-w-6xl mx-auto px-6">
          <p className="font-mono text-[0.78rem] tracking-[0.22em] uppercase text-[#00843D] mb-4">
            Como funciona
          </p>
          <h2 className="font-heading text-3xl sm:text-4xl font-bold tracking-[-0.025em] text-[#1D1D1F] max-w-2xl mb-14">
            Três passos entre o seu boletim e uma resposta honesta.
          </h2>
          <ol className="grid sm:grid-cols-3 gap-px bg-[#E6E6E8] border border-[#E6E6E8] rounded-2xl overflow-hidden">
            {PASSOS.map((passo) => (
              <li
                key={passo.numero}
                className="bg-white p-8 hover:bg-[#FAFAFC] transition-colors"
              >
                <span
                  className="block font-heading text-[2.75rem] font-extrabold leading-none text-transparent mb-6"
                  style={{ WebkitTextStroke: "1.5px #00AEEF" }}
                >
                  {passo.numero}
                </span>
                <h3 className="text-lg font-semibold text-[#1D1D1F] mb-2.5 tracking-[-0.02em]">
                  {passo.titulo}
                </h3>
                <p className="text-[0.95rem] leading-relaxed text-[#6E6E73]">
                  {passo.descricao}
                </p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ============ FERRAMENTAS PÚBLICAS ============ */}
      <section className="bg-white py-20 sm:py-28">
        <div className="max-w-6xl mx-auto px-6">
          <p className="font-mono text-[0.78rem] tracking-[0.22em] uppercase text-[#00843D] mb-4">
            Para estudantes · grátis, sem conta
          </p>
          <h2 className="font-heading text-3xl sm:text-4xl font-bold tracking-[-0.025em] text-[#1D1D1F] max-w-2xl mb-14">
            Duas ferramentas abertas para quem está na disputa.
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {FEATURES_PUBLICAS.map((feature) => (
              <Link
                key={feature.href}
                href={feature.href}
                className="group relative block rounded-2xl border border-[#E6E6E8] p-8 sm:p-10 overflow-hidden transition-all hover:border-[#00AEEF] hover:shadow-[0_20px_50px_rgba(0,51,102,0.10)] hover:-translate-y-1"
              >
                <div
                  className="absolute top-0 left-0 right-0 h-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: "linear-gradient(90deg, #00AEEF, #00843D)" }}
                />
                <h3 className="font-heading text-2xl font-bold tracking-[-0.02em] text-[#003366] mb-3 flex items-center gap-3">
                  {feature.titulo}
                  <span className="text-[#00AEEF] transition-transform group-hover:translate-x-1.5">
                    →
                  </span>
                </h3>
                <p className="text-[0.97rem] leading-relaxed text-[#3A3A3C] mb-5">
                  {feature.descricao}
                </p>
                <p className="font-mono text-[0.72rem] tracking-[0.14em] uppercase text-[#6E6E73]">
                  {feature.detalhe}
                </p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ============ B2B / ESCOLAS ============ */}
      <section className="relative overflow-hidden py-20 sm:py-28 text-white bg-[#00843D]">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 60% 80% at 85% 20%, rgba(0,33,71,0.35), transparent)",
          }}
        />
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="grid lg:grid-cols-[1fr_1.2fr] gap-12 lg:gap-20 items-start">
            <div>
              <p className="font-mono text-[0.78rem] tracking-[0.22em] uppercase text-white/70 mb-4">
                Para escolas parceiras
              </p>
              <h2 className="font-heading text-3xl sm:text-4xl font-bold tracking-[-0.025em] mb-5">
                A coordenação enxerga a turma inteira de uma vez.
              </h2>
              <p className="text-white/80 leading-relaxed mb-8">
                O painel B2B transforma os escores dos seus alunos em um mapa de
                risco acionável — quem está seguro, quem precisa de reforço e
                onde a escola se posiciona contra toda a população do PAS.
              </p>
              <Link
                href="/auth/login"
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold bg-white text-[#00843D] hover:bg-[#EAF6EF] transition-all hover:-translate-y-0.5"
              >
                Acessar painel da escola →
              </Link>
            </div>
            <ul className="grid sm:grid-cols-2 gap-4">
              {FEATURES_B2B.map((feature) => (
                <li
                  key={feature.titulo}
                  className="rounded-xl bg-white/10 border border-white/15 backdrop-blur-sm p-6 hover:bg-white/15 transition-colors"
                >
                  <h3 className="font-semibold mb-1.5 tracking-[-0.01em]">
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
      </section>

      {/* ============ CTA FINAL + FOOTER ============ */}
      <footer className="bg-[#002147] text-white">
        <div className="max-w-6xl mx-auto px-6 py-20 text-center">
          <h2 className="font-heading text-3xl sm:text-5xl font-extrabold tracking-[-0.03em] mb-6">
            A terceira etapa ainda não aconteceu.
            <br />
            <span className="text-[#00AEEF]">O seu plano pode.</span>
          </h2>
          <Link
            href="/predict"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-[#002147] bg-[#00AEEF] hover:bg-[#33C1F3] transition-all hover:-translate-y-0.5 shadow-[0_8px_30px_rgba(0,174,239,0.35)]"
          >
            Calcular minha previsão →
          </Link>
        </div>
        <div className="border-t border-white/10">
          <div className="max-w-6xl mx-auto px-6 py-7 flex flex-col sm:flex-row items-center justify-between gap-4 text-[0.82rem] text-white/45">
            <span>
              © {new Date().getFullYear()} Vetor PAS — projeto independente, sem
              vínculo com a UnB ou o Cebraspe.
            </span>
            <div className="flex gap-6">
              <Link href="/predict" className="hover:text-white transition-colors">
                Preditor
              </Link>
              <Link href="/temporal" className="hover:text-white transition-colors">
                Análise Temporal
              </Link>
              <Link href="/auth/login" className="hover:text-white transition-colors">
                Coordenadores
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
