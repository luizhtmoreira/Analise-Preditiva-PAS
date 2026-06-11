import Link from "next/link";
import { Suspense } from "react";
import { BrandMark } from "@/components/brand/BrandMark";
import { AlunoSignupForm } from "@/components/auth/AlunoSignupForm";

export default function CadastroAlunoPage() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(168deg, #002147 0%, #003366 60%, #003A70 100%)",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", padding: "32px 20px",
      }}
    >
      {/* Grid decorativo */}
      <div
        style={{
          position: "fixed", inset: 0, pointerEvents: "none",
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
          backgroundSize: "56px 56px",
        }}
      />

      <div style={{ position: "relative", zIndex: 1, width: "100%", maxWidth: 420 }}>
        {/* Header */}
        <div style={{ marginBottom: 32, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <BrandMark sublabel="Cadastro do aluno" />
          <Link
            href="/predict"
            style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", textDecoration: "none" }}
          >
            ← Voltar
          </Link>
        </div>

        {/* Card */}
        <div
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 20, padding: "32px 28px",
            boxShadow: "0 24px 80px rgba(0,10,25,0.4)",
          }}
        >
          {/* Linha cyan no topo */}
          <div style={{
            height: 2, marginBottom: 28, borderRadius: 99,
            background: "linear-gradient(to right, transparent, #00AEEF 30%, transparent)",
          }} />

          <p
            style={{
              fontFamily: "var(--font-geist-mono), monospace",
              fontSize: 10, letterSpacing: "0.2em", textTransform: "uppercase",
              color: "#7FD8F7", marginBottom: 8,
            }}
          >
            Painel Multi-Curso · Gratuito
          </p>
          <h1
            style={{
              fontFamily: "var(--font-display), sans-serif",
              fontSize: 26, fontWeight: 800, letterSpacing: "-0.025em",
              lineHeight: 1.15, color: "#fff", marginBottom: 6,
            }}
          >
            Compare suas chances em{" "}
            <span style={{ color: "#00AEEF" }}>vários cursos</span>
          </h1>
          <p style={{ fontSize: 13, color: "rgba(255,255,255,0.5)", lineHeight: 1.6, marginBottom: 28 }}>
            Salve seus dados uma vez e veja probabilidade + quanto falta no PAS 3 para cada curso que te interessa.
          </p>

          <Suspense>
            <AlunoSignupForm />
          </Suspense>

          <p style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", textAlign: "center", marginTop: 20 }}>
            Já tem conta?{" "}
            <Link
              href="/auth/entrar"
              style={{ color: "#7FD8F7", textDecoration: "none", fontWeight: 500 }}
            >
              Entrar →
            </Link>
          </p>
        </div>

        <p style={{ fontSize: 11, color: "rgba(255,255,255,0.2)", textAlign: "center", marginTop: 20 }}>
          Exclusivo para alunos do PAS/UnB · sem cartão de crédito
        </p>
      </div>
    </div>
  );
}
