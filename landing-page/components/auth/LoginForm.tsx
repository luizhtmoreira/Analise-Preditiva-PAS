"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function LoginForm() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const supabase = createClient();
    const { error } = await supabase.auth.signInWithPassword({ email, password });

    if (error) {
      setError("Email ou senha incorretos.");
      setLoading(false);
      return;
    }

    router.push("/gestao");
    router.refresh();
  }

  return (
    <div className="rounded-2xl border border-[#E6E6E8] bg-white p-6 shadow-[0_8px_30px_rgba(0,51,102,0.06)]">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block mb-1.5 font-mono text-[0.68rem] tracking-[0.14em] uppercase text-[#6E6E73]">
            Email institucional
          </label>
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="coord@escola.edu.br"
            required
            autoComplete="email"
          />
        </div>
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="font-mono text-[0.68rem] tracking-[0.14em] uppercase text-[#6E6E73]">
              Senha
            </label>
            <Link
              href="/auth/esqueci-senha"
              className="text-xs text-[#00AEEF] hover:underline font-medium"
            >
              Esqueci minha senha
            </Link>
          </div>
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
            autoComplete="current-password"
          />
        </div>
        {error && (
          <p className="text-sm rounded-lg px-3 py-2 bg-[#FFCDD2] text-[#B71C1C]">{error}</p>
        )}
        <Button
          type="submit"
          disabled={loading}
          className="w-full text-white bg-[#003366] hover:bg-[#004080] transition-colors"
        >
          {loading ? "Entrando…" : "Entrar →"}
        </Button>
      </form>
    </div>
  );
}
