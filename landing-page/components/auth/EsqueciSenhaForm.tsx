"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { createClient } from "@/lib/supabase/client";

export function EsqueciSenhaForm() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!email.trim()) return;

    setLoading(true);
    setError("");

    const supabase = createClient();
    const redirectTo = `${window.location.origin}/auth/callback?next=/auth/redefinir-senha`;

    const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo,
    });

    if (resetError) {
      setError("Não foi possível enviar as instruções. Verifique se o e-mail está correto.");
      setLoading(false);
      return;
    }

    setDone(true);
    setLoading(false);
  }

  if (done) {
    return (
      <div className="p-6 rounded-2xl bg-[#00843D]/8 border border-[#00843D]/25 text-center">
        <p className="text-2xl mb-2">📧</p>
        <p className="text-[0.95rem] font-bold text-[#002147] mb-2">E-mail de redefinição enviado!</p>
        <p className="text-[0.8rem] text-[#4A5568] leading-relaxed mb-4">
          Enviamos as instruções para <strong className="text-[#002147]">{email}</strong>. Abra o e-mail e clique no link para redefinir sua senha.
        </p>
        <Link href="/auth/entrar" className="text-xs text-[#00843D] font-semibold hover:underline">
          ← Voltar para a tela de login
        </Link>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div>
        <label className="vp-label block mb-1.5">E-mail da sua conta</label>
        <input
          type="email"
          value={email}
          required
          autoComplete="email"
          placeholder="seu@email.com"
          onChange={(e) => setEmail(e.target.value)}
          className="vp-input"
        />
      </div>

      {error && (
        <div className="text-sm rounded-lg px-3 py-2 bg-[#FFCDD2] text-[#B71C1C]">{error}</div>
      )}

      <button type="submit" disabled={loading} className="vp-btn vp-btn-cyan py-3 mt-1">
        {loading ? "Enviando e-mail…" : "Enviar link de redefinição →"}
      </button>
    </form>
  );
}
