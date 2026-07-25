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
      setError("Não foi possível atualizar a senha. O link pode ter expirado.");
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
      <div style={{
        padding: "24px 20px", borderRadius: 16,
        background: "rgba(0,194,106,0.08)", border: "1px solid rgba(0,194,106,0.3)",
        textAlign: "center",
      }}>
        <p style={{ fontSize: 24, marginBottom: 8 }}>✓</p>
        <p style={{ fontSize: 16, fontWeight: 700, color: "#fff", marginBottom: 8 }}>
          Senha alterada com sucesso!
        </p>
        <p style={{ fontSize: 13, color: "rgba(255,255,255,0.65)", lineHeight: 1.6 }}>
          Redirecionando você para o sistema em instantes…
        </p>
      </div>
    );
  }

  return (
    <>
      <style>{`
        .redefinir-input:focus { border-color: rgba(0,174,239,0.6) !important; }
        .redefinir-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 10px 32px rgba(0,174,239,0.45) !important; background: #33C1F3 !important; }
        .redefinir-btn { transition: transform .2s, box-shadow .2s, background .2s; }
      `}</style>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div>
          <label style={{ display: "block", marginBottom: 6, fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.45)", fontFamily: "var(--font-geist-mono), monospace" }}>
            Nova Senha
          </label>
          <input
            type="password"
            value={password}
            required
            minLength={8}
            autoComplete="new-password"
            placeholder="mínimo 8 caracteres"
            onChange={(e) => setPassword(e.target.value)}
            className="redefinir-input"
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
            Confirmar Nova Senha
          </label>
          <input
            type="password"
            value={confirmPassword}
            required
            minLength={8}
            autoComplete="new-password"
            placeholder="repita a nova senha"
            onChange={(e) => setConfirmPassword(e.target.value)}
            className="redefinir-input"
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
          className="redefinir-btn"
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
          {loading ? "Salvando nova senha…" : "Redefinir Senha e Entrar →"}
        </button>
      </form>
    </>
  );
}
