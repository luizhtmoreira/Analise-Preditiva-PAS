"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export function RedefinirSenhaForm() {
  const router = useRouter();

  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (password.length < 8) {
      setError("A senha deve ter no mínimo 8 caracteres.");
      return;
    }
    if (password !== confirmPassword) {
      setError("As senhas não coincidem.");
      return;
    }

    setLoading(true);
    setError("");

    const supabase = createClient();
    const { data, error: updateError } = await supabase.auth.updateUser({
      password,
    });

    if (updateError) {
      setError(
        updateError.code === "same_password"
          ? "A nova senha deve ser diferente da senha atual."
          : "Não foi possível atualizar a senha. O link pode ter expirado."
      );
      setLoading(false);
      return;
    }

    setDone(true);
    setLoading(false);

    // Identify user role and redirect
    const metadata = data?.user?.user_metadata || {};
    const isCoordinator = Boolean(metadata.tenant || metadata.role === "coordenador");
    const destination = isCoordinator ? "/gestao" : "/predict";

    setTimeout(() => {
      router.push(destination);
      router.refresh();
    }, 1200);
  }

  if (done) {
    return (
      <div className="p-6 rounded-2xl bg-[#00843D]/8 border border-[#00843D]/25 text-center">
        <p className="text-2xl mb-2">✓</p>
        <p className="text-base font-bold text-[#002147] mb-2">Senha alterada com sucesso!</p>
        <p className="text-[0.8rem] text-[#4A5568] leading-relaxed">
          Redirecionando você para o sistema em instantes…
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div>
        <label className="vp-label block mb-1.5">Nova Senha</label>
        <input
          type="password"
          value={password}
          required
          minLength={8}
          autoComplete="new-password"
          placeholder="mínimo 8 caracteres"
          onChange={(e) => setPassword(e.target.value)}
          className="vp-input"
        />
      </div>

      <div>
        <label className="vp-label block mb-1.5">Confirmar Nova Senha</label>
        <input
          type="password"
          value={confirmPassword}
          required
          minLength={8}
          autoComplete="new-password"
          placeholder="repita a nova senha"
          onChange={(e) => setConfirmPassword(e.target.value)}
          className="vp-input"
        />
      </div>

      {error && (
        <div className="text-sm rounded-lg px-3 py-2 bg-[#FFCDD2] text-[#B71C1C]">{error}</div>
      )}

      <button type="submit" disabled={loading} className="vp-btn vp-btn-cyan py-3 mt-1">
        {loading ? "Salvando nova senha…" : "Redefinir Senha e Entrar →"}
      </button>
    </form>
  );
}
