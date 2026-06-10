import Link from "next/link";
import { BrandMark } from "@/components/brand/BrandMark";
import { CurvaGaussiana } from "@/components/brand/CurvaGaussiana";
import { LoginForm } from "@/components/auth/LoginForm";

export default function LoginPage() {
  return (
    <div className="min-h-screen grid lg:grid-cols-[1.1fr_1fr]">
      {/* Painel de marca */}
      <div
        className="relative hidden lg:flex flex-col justify-between overflow-hidden text-white p-10"
        style={{
          background: "linear-gradient(168deg, #002147 0%, #003366 52%, #003A70 100%)",
        }}
      >
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px)",
            backgroundSize: "56px 56px",
            maskImage: "radial-gradient(ellipse 90% 70% at 50% 30%, black, transparent)",
          }}
        />
        <div className="relative z-10">
          <BrandMark sublabel="Painel da escola" />
        </div>
        <div className="relative z-10 max-w-md">
          <p className="font-mono text-[0.78rem] tracking-[0.22em] uppercase text-[#7FD8F7] mb-4">
            Para coordenadores
          </p>
          <h1 className="font-heading text-4xl font-extrabold tracking-[-0.03em] leading-[1.08] mb-4">
            A turma inteira,
            <br />
            <span className="text-[#00AEEF]">em um mapa de risco.</span>
          </h1>
          <p className="text-white/65 leading-relaxed">
            Gestão de Ativos, Escola vs. População, comparação entre grupos e
            relatórios whitelabel — tudo a partir dos escores que você já tem.
          </p>
        </div>
        <div className="relative -mx-10 -mb-10">
          <CurvaGaussiana showLabels={false} />
        </div>
      </div>

      {/* Formulário */}
      <div className="flex items-center justify-center bg-[#F5F5F7] px-6 py-12">
        <div className="w-full max-w-sm">
          <div className="lg:hidden mb-8 flex justify-center">
            <BrandMark light={false} sublabel="Painel da escola" />
          </div>
          <p className="font-mono text-[0.72rem] tracking-[0.22em] uppercase text-[#00843D] mb-3">
            Acesso restrito
          </p>
          <h2 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F] mb-1">
            Entrar no painel
          </h2>
          <p className="text-sm text-[#6E6E73] mb-8">
            Use o email institucional cadastrado pela sua escola.
          </p>
          <LoginForm />
          <p className="text-xs text-[#6E6E73] mt-8 text-center">
            Sua escola ainda não é parceira?{" "}
            <Link href="/" className="text-[#003366] font-medium hover:underline">
              Conheça o Vetor PAS →
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
