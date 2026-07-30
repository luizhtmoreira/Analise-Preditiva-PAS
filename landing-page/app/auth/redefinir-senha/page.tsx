import Link from "next/link";
import { BrandMark } from "@/components/brand/BrandMark";
import { RedefinirSenhaForm } from "@/components/auth/RedefinirSenhaForm";

export const metadata = {
  title: "Redefinir Senha | Vetor PAS",
  description: "Crie uma nova senha para acessar sua conta no Vetor PAS.",
};

export default function RedefinirSenhaPage() {
  return (
    <div className="vp-wash relative min-h-screen bg-[#F8F9FA] flex flex-col items-center justify-center px-5 py-10 overflow-hidden">
      <div className="relative z-10 w-full max-w-[420px]">
        <div className="mb-8 flex items-center justify-between">
          <BrandMark light={false} sublabel="Nova senha" />
          <Link href="/auth/entrar" className="text-xs text-[#718096] hover:text-[#4A5568] transition-colors">
            ← Voltar
          </Link>
        </div>

        <div className="vp-card p-8">
          <span className="vp-eyebrow vp-eyebrow-cyan mb-2">Segurança da Conta</span>
          <h1 className="font-heading text-[1.5rem] font-extrabold tracking-tight leading-tight text-[#002147] mt-3 mb-1.5">
            Crie sua nova senha
          </h1>
          <p className="text-[0.8rem] text-[#4A5568] leading-relaxed mb-7">
            Escolha uma senha forte de no mínimo 8 caracteres para proteger seu acesso.
          </p>

          <RedefinirSenhaForm />
        </div>
      </div>
    </div>
  );
}
