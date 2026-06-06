// TODO: implementar landing page completa (dual-audience: alunos + coordenadores)
// Ver docs/adr/0003-split-features-publico-b2b.md e docs/identidade-visual.md
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4"
      style={{ background: "#003366" }}>
      <div className="text-center max-w-xl">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6"
          style={{ background: "#00AEEF" }}>
          <span className="text-white font-black text-2xl">VP</span>
        </div>
        <h1 className="text-4xl font-black text-white mb-3 tracking-tight">Vetor PAS</h1>
        <p className="text-lg mb-8" style={{ color: "rgba(255,255,255,0.65)" }}>
          Análise preditiva para o PAS/UnB — conheça suas chances antes da prova.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link href="/predict"
            className="px-6 py-3 rounded-xl font-semibold text-sm transition-all"
            style={{ background: "#00AEEF", color: "#fff" }}>
            Calcular minha previsão →
          </Link>
          <Link href="/auth/login"
            className="px-6 py-3 rounded-xl font-semibold text-sm transition-all"
            style={{ background: "rgba(255,255,255,0.1)", color: "#fff", border: "1px solid rgba(255,255,255,0.2)" }}>
            Acesso coordenadores
          </Link>
        </div>
      </div>
    </main>
  );
}
