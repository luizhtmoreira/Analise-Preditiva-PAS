import Link from "next/link";
import { Suspense } from "react";
import { BrandMark } from "@/components/brand/BrandMark";
import { AlunoLoginForm } from "@/components/auth/AlunoLoginForm";

export default function EntrarAlunoPage() {
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
          <BrandMark sublabel="Área do aluno" />
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
            Painel Multi-Curso
          </p>
          <h1
            style={{
              fontFamily: "var(--font-display), sans-serif",
              fontSize: 26, fontWeight: 800, letterSpacing: "-0.025em",
              lineHeight: 1.15, color: "#fff", marginBottom: 6,
            }}
          >
            Bem-vindo de volta
          </h1>
          <p style={{ fontSize: 13, color: "rgba(255,255,255,0.5)", lineHeight: 1.6, marginBottom: 28 }}>
            Entre para acessar seu painel e comparar suas chances nos cursos da UnB.
          </p>

          <Suspense>
            <AlunoLoginForm />
          </Suspense>

          <p style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", textAlign: "center", marginTop: 20 }}>
            Ainda não tem conta?{" "}
            <Link
              href="/auth/cadastro"
              style={{ color: "#7FD8F7", textDecoration: "none", fontWeight: 500 }}
            >
              Criar conta gratuita →
            </Link>
          </p>
        </div>

        <p style={{ fontSize: 11, color: "rgba(255,255,255,0.2)", textAlign: "center", marginTop: 20 }}>
          Coordenador pedagógico?{" "}
          <Link
            href="/auth/login"
            style={{ color: "rgba(255,255,255,0.35)", textDecoration: "none" }}
          >
            Acesse o painel da escola →
          </Link>
        </p>
      </div>
    </div>
  );
}
