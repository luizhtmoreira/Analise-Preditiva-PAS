"use client";
import { useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export function AlunoLoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const nextPath = searchParams.get("next") ?? "/predict";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

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
    <>
      <style>{`
        .aluno-login-input:focus { border-color: rgba(0,174,239,0.6) !important; }
        .aluno-login-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 10px 32px rgba(0,174,239,0.45) !important; background: #33C1F3 !important; }
        .aluno-login-btn { transition: transform .2s, box-shadow .2s, background .2s; }
      `}</style>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div>
          <label style={{ display: "block", marginBottom: 6, fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.45)", fontFamily: "var(--font-geist-mono), monospace" }}>
            Email
          </label>
          <input
            type="email" value={email} required autoComplete="email"
            placeholder="seu@email.com"
            onChange={(e) => setEmail(e.target.value)}
            className="aluno-login-input"
            style={{
              width: "100%", padding: "11px 14px", borderRadius: 10,
              background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.15)",
              color: "#fff", fontSize: 14, outline: "none",
              fontFamily: "var(--font-body), sans-serif", transition: "border-color .2s",
            }}
          />
        </div>

        <div>
          <label style={{ display: "block", marginBottom: 6, fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.45)", fontFamily: "var(--font-geist-mono), monospace" }}>
            Senha
          </label>
          <input
            type="password" value={password} required autoComplete="current-password"
            placeholder="••••••••"
            onChange={(e) => setPassword(e.target.value)}
            className="aluno-login-input"
            style={{
              width: "100%", padding: "11px 14px", borderRadius: 10,
              background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.15)",
              color: "#fff", fontSize: 14, outline: "none",
              fontFamily: "var(--font-body), sans-serif", transition: "border-color .2s",
            }}
          />
        </div>

        {error && (
          <div style={{
            padding: "10px 14px", borderRadius: 10,
            background: "rgba(255,107,107,0.1)", border: "1px solid rgba(255,107,107,0.3)",
            color: "#FF6B6B", fontSize: 13,
          }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="aluno-login-btn"
          style={{
            width: "100%", padding: "14px", borderRadius: 12, border: "none",
            background: "#00AEEF", color: "#002147", fontSize: 14, fontWeight: 700,
            cursor: loading ? "not-allowed" : "pointer",
            fontFamily: "var(--font-body), sans-serif",
            opacity: loading ? 0.6 : 1,
            boxShadow: "0 8px 28px rgba(0,174,239,0.35)",
            marginTop: 4,
          }}
        >
          {loading ? "Entrando…" : "Entrar →"}
        </button>
      </form>
    </>
  );
}
