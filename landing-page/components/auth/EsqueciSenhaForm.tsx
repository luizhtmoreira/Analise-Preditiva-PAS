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
      <div style={{
        padding: "24px 20px", borderRadius: 16,
        background: "rgba(0,194,106,0.08)", border: "1px solid rgba(0,194,106,0.3)",
        textAlign: "center",
      }}>
        <p style={{ fontSize: 24, marginBottom: 8 }}>📧</p>
        <p style={{ fontSize: 15, fontWeight: 700, color: "#fff", marginBottom: 8 }}>
          E-mail de redefinição enviado!
        </p>
        <p style={{ fontSize: 13, color: "rgba(255,255,255,0.65)", lineHeight: 1.6, marginBottom: 16 }}>
          Enviamos as instruções para <strong style={{ color: "#fff" }}>{email}</strong>. Abra o e-mail e clique no link para redefinir sua senha.
        </p>
        <Link
          href="/auth/entrar"
          style={{ fontSize: 12, color: "#7FD8F7", textDecoration: "none", fontWeight: 600 }}
        >
          ← Voltar para a tela de login
        </Link>
      </div>
    );
  }

  return (
    <>
      <style>{`
        .esqueci-input:focus { border-color: rgba(0,174,239,0.6) !important; }
        .esqueci-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 10px 32px rgba(0,174,239,0.45) !important; background: #33C1F3 !important; }
        .esqueci-btn { transition: transform .2s, box-shadow .2s, background .2s; }
      `}</style>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div>
          <label style={{ display: "block", marginBottom: 6, fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.45)", fontFamily: "var(--font-geist-mono), monospace" }}>
            E-mail da sua conta
          </label>
          <input
            type="email"
            value={email}
            required
            autoComplete="email"
            placeholder="seu@email.com"
            onChange={(e) => setEmail(e.target.value)}
            className="esqueci-input"
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
          className="esqueci-btn"
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
          {loading ? "Enviando e-mail…" : "Enviar link de redefinição →"}
        </button>
      </form>
    </>
  );
}
