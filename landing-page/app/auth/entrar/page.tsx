import Link from "next/link";
import { Suspense } from "react";
import { BrandMark } from "@/components/brand/BrandMark";
import { AlunoLoginForm } from "@/components/auth/AlunoLoginForm";

export default function EntrarAlunoPage() {
  return (
    <div className="vp-wash relative min-h-screen bg-[#F8F9FA] flex flex-col items-center justify-center px-5 py-10 overflow-hidden">
      <div className="relative z-10 w-full max-w-[420px]">
        <div className="mb-8 flex items-center justify-between">
          <BrandMark light={false} sublabel="Área do aluno" />
          <Link href="/predict" className="text-xs text-[#718096] hover:text-[#4A5568] transition-colors">
            ← Voltar
          </Link>
        </div>

        <div className="vp-card p-8">
          <span className="vp-eyebrow vp-eyebrow-cyan mb-2">Área do Aluno</span>
          <h1 className="font-heading text-[1.6rem] font-extrabold tracking-tight leading-tight text-[#002147] mt-3 mb-1.5">
            Bem-vindo de volta
          </h1>
          <p className="text-[0.8rem] text-[#4A5568] leading-relaxed mb-7">
            Entre para acessar seu painel e comparar suas chances nos cursos da UnB.
          </p>

          <Suspense>
            <AlunoLoginForm />
          </Suspense>

          <p className="text-xs text-[#718096] text-center mt-5">
            Ainda não tem conta?{" "}
            <Link href="/auth/cadastro" className="text-[#00843D] font-semibold hover:underline">
              Criar conta gratuita →
            </Link>
          </p>
        </div>

        <p className="text-[0.68rem] text-[#718096] text-center mt-5">
          Coordenador pedagógico?{" "}
          <Link href="/auth/login" className="text-[#4A5568] hover:text-[#002147] transition-colors">
            Acesse o painel da escola →
          </Link>
        </p>
      </div>
    </div>
  );
}
