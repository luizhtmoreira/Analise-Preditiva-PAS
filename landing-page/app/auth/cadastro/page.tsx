import Link from "next/link";
import { Suspense } from "react";
import { BrandMark } from "@/components/brand/BrandMark";
import { AlunoSignupForm } from "@/components/auth/AlunoSignupForm";

export default function CadastroAlunoPage() {
  return (
    <div className="vp-wash relative min-h-screen bg-[#F8F9FA] flex flex-col items-center justify-center px-5 py-10 overflow-hidden">
      <div className="relative z-10 w-full max-w-[420px]">
        <div className="mb-8 flex items-center justify-between">
          <BrandMark light={false} sublabel="Cadastro do aluno" />
          <Link href="/predict" className="text-xs text-[#718096] hover:text-[#4A5568] transition-colors">
            ← Voltar
          </Link>
        </div>

        <div className="vp-card p-8">
          <span className="vp-eyebrow vp-eyebrow-cyan mb-2">Área do Aluno · Cadastro</span>
          <h1 className="font-heading text-[1.6rem] font-extrabold tracking-tight leading-tight text-[#002147] mt-3 mb-1.5">
            Compare suas chances em{" "}
            <span className="text-[#00843D]">vários cursos</span>
          </h1>
          <p className="text-[0.8rem] text-[#4A5568] leading-relaxed mb-7">
            Salve seus dados uma vez e veja probabilidade + quanto falta no PAS 3 para cada curso que te interessa.
          </p>

          <Suspense>
            <AlunoSignupForm />
          </Suspense>

          <p className="text-xs text-[#718096] text-center mt-5">
            Já tem conta?{" "}
            <Link href="/auth/entrar" className="text-[#00843D] font-semibold hover:underline">
              Entrar →
            </Link>
          </p>
        </div>

        <p className="text-[0.68rem] text-[#718096] text-center mt-5">
          Exclusivo para alunos do PAS/UnB · sem cartão de crédito
        </p>
      </div>
    </div>
  );
}
