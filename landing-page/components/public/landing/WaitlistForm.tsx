"use client";

import { useState } from "react";
import { submitWaitlist } from "@/lib/waitlist";
import { CourseSelect } from "@/components/ui/CourseSelect";



interface WaitlistFormProps {
  variantStyle?: "card" | "inline" | "minimal";
  buttonText?: string;
}

export function WaitlistForm({
  variantStyle = "card",
  buttonText = "Garantir Meu Acesso Antecipado",
}: WaitlistFormProps) {
  const [nome, setNome] = useState("");
  const [email, setEmail] = useState("");
  const [escola, setEscola] = useState("");
  const [curso, setCurso] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setLoading(true);

    try {
      await submitWaitlist({
        nome,
        email,
        escola,
        curso_pretendido: curso,
      });
      setSuccess(true);
    } catch (err: any) {
      setErrorMsg(err.message || "Ocorreu um erro ao realizar a inscrição.");
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="bg-[#00843D]/15 border border-[#00843D]/40 rounded-2xl p-6 text-center text-white backdrop-blur-md animate-fade-in">
        <div className="w-12 h-12 bg-[#00843D] rounded-full flex items-center justify-center mx-auto mb-3 text-2xl">
          ✓
        </div>
        <h3 className="font-heading text-xl font-bold text-[#00AEEF] mb-2">
          Inscrição Confirmada!
        </h3>
        <p className="text-sm text-white/80 leading-relaxed mb-4">
          Obrigado, <strong className="text-white">{nome}</strong>! Você está na lista de espera preferencial para o PAS 3. Avisaremos você assim que abrirmos o acesso.
        </p>
        <span className="inline-block font-mono text-xs text-[#7FD8F7] bg-white/10 px-3 py-1 rounded-full">
          Escola: {escola} · Curso: {curso || "A definir"}
        </span>
      </div>
    );
  }

  return (
    <form
      onSubmit={handleSubmit}
      className={`space-y-4 text-left ${
        variantStyle === "card"
          ? "bg-[#001D3D]/90 border border-white/15 rounded-2xl p-6 sm:p-8 backdrop-blur-md shadow-2xl"
          : "bg-transparent"
      }`}
    >
      <div className="space-y-1">
        <label className="block text-xs font-medium text-white/80 uppercase tracking-wider">
          Seu Nome Completo *
        </label>
        <input
          type="text"
          required
          value={nome}
          onChange={(e) => setNome(e.target.value)}
          placeholder="Ex: Ana Silva"
          className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:border-[#00AEEF] focus:ring-1 focus:ring-[#00AEEF] transition-all text-sm"
        />
      </div>

      <div className="space-y-1">
        <label className="block text-xs font-medium text-white/80 uppercase tracking-wider">
          Seu E-mail Principal *
        </label>
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="seu.email@exemplo.com"
          className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:border-[#00AEEF] focus:ring-1 focus:ring-[#00AEEF] transition-all text-sm"
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-1">
          <label className="block text-xs font-medium text-white/80 uppercase tracking-wider">
            Sua Escola *
          </label>
          <input
            type="text"
            required
            value={escola}
            onChange={(e) => setEscola(e.target.value)}
            placeholder="Ex: Marista, Sigma..."
            className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:border-[#00AEEF] focus:ring-1 focus:ring-[#00AEEF] transition-all text-sm"
          />
        </div>

        <div className="space-y-1">
          <label className="block text-xs font-medium text-white/80 uppercase tracking-wider">
            Curso Pretendido na UnB
          </label>
          <CourseSelect value={curso} onChange={setCurso} />
        </div>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-red-500/20 border border-red-500/40 text-red-200 text-xs">
          ⚠️ {errorMsg}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full py-4 px-6 rounded-xl font-semibold text-[#002147] bg-[#00AEEF] hover:bg-[#33C1F3] active:scale-[0.98] transition-all duration-200 shadow-[0_8px_25px_rgba(0,174,239,0.3)] disabled:opacity-50 text-sm sm:text-base flex items-center justify-center gap-2"
      >
        {loading ? (
          <span className="inline-block animate-spin font-bold">↻</span>
        ) : (
          <>
            <span>{buttonText}</span>
            <span>→</span>
          </>
        )}
      </button>

      <p className="text-[0.75rem] text-center text-white/50 pt-1">
        Sem spam. Usaremos seus dados exclusivamente para notificar seu acesso ao Vetor PAS.
      </p>
    </form>
  );
}
