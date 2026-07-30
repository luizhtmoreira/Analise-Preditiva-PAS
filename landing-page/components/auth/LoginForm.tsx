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
    <div className="vp-card p-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="vp-label block mb-1.5">Email institucional</label>
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="coord@escola.edu.br"
            required
            autoComplete="email"
            className="vp-input h-auto"
          />
        </div>
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="vp-label">Senha</label>
            <Link
              href="/auth/esqueci-senha"
              className="text-xs text-[#00843D] hover:underline font-semibold"
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
            className="vp-input h-auto"
          />
        </div>
        {error && (
          <p className="text-sm rounded-lg px-3 py-2 bg-[#FFCDD2] text-[#B71C1C]">{error}</p>
        )}
        <Button type="submit" disabled={loading} className="vp-btn vp-btn-navy w-full">
          {loading ? "Entrando…" : "Entrar →"}
        </Button>
      </form>
    </div>
  );
}
