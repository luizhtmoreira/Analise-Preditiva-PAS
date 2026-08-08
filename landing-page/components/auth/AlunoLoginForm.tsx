"use client";
import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export function AlunoLoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const nextPath = searchParams.get("next") ?? "/predict";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(
    searchParams.get("error") === "auth_callback_failed"
      ? "O link expirou ou é inválido — peça um novo."
      : ""
  );

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const supabase = createClient();
    const { error: signInError } = await supabase.auth.signInWithPassword({ email, password });

    if (signInError) {
      setError("Email ou senha incorretos.");
      setLoading(false);
      return;
    }

    router.push(nextPath);
    router.refresh();
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div>
        <label className="vp-label block mb-1.5">Email</label>
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

      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="vp-label">Senha</label>
          <Link href="/auth/esqueci-senha" className="text-xs text-[#00843D] font-semibold hover:underline">
            Esqueci minha senha
          </Link>
        </div>
        <input
          type="password"
          value={password}
          required
          autoComplete="current-password"
          placeholder="••••••••"
          onChange={(e) => setPassword(e.target.value)}
          className="vp-input"
        />
      </div>

      {error && (
        <div className="text-sm rounded-lg px-3 py-2 bg-[#FFCDD2] text-[#B71C1C]">{error}</div>
      )}

      <button type="submit" disabled={loading} className="vp-btn vp-btn-cyan py-3 mt-1">
        {loading ? "Entrando…" : "Entrar →"}
      </button>
    </form>
  );
}
