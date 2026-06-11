"use client";
import { useState, useRef, useMemo, useEffect, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

/* ─── lista de escolas ──────────────────────────────────────────── */
// Substitua por consulta ao banco quando a tabela de escolas existir.
const ESCOLAS = [
  "Colégio Marista de Brasília",
  "Colégio Ideal",
  "Colégio Dom Bosco",
  "Colégio Salesiano de Brasília",
  "Colégio Militar de Brasília",
  "Centro Educacional La Salle",
  "Colégio JK",
  "Colégio Galois",
  "Colégio Poliedro Brasília",
  "Instituto Federal de Brasília – Campus Brasília",
  "Instituto Federal de Brasília – Campus Taguatinga",
  "CEM 01 de Brasília",
  "CEM 03 de Brasília",
  "CEM 404 de Santa Maria",
  "CEM 01 do Guará",
  "CEM 01 de Sobradinho",
  "CEM Setor Leste",
  "CEM Setor Oeste",
  "Centro Educacional Sigma",
  "Colégio Objetivo Brasília",
  "Escola Americana de Brasília",
  "Colégio Notre Dame",
  "Colégio Adventista de Brasília",
  "Centro de Ensino Médio Elefante Branco",
  "Colégio Santa Doroteia",
  "Colégio Projeção",
  "Escola de Ensino Médio em Período Integral – CEMI",
  "Colégio Ábaco",
  "Lycée Français François-Mitterrand",
  "Outra escola",
].sort((a, b) => a.localeCompare(b, "pt-BR"));

/* ─── autocomplete de escola ─────────────────────────────────────── */

function EscolaCombobox({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    if (!value.trim()) return ESCOLAS.slice(0, 8);
    const normalize = (s: string) =>
      s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
    const q = normalize(value);
    return ESCOLAS.filter((e) => normalize(e).includes(q)).slice(0, 8);
  }, [value]);

  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <input
        type="text"
        value={value}
        placeholder="Buscar sua escola…"
        required
        autoComplete="off"
        onFocus={() => setOpen(true)}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        style={{
          width: "100%", padding: "11px 14px", borderRadius: 10,
          background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.15)",
          color: "#fff", fontSize: 14, outline: "none",
          fontFamily: "var(--font-body), sans-serif",
          transition: "border-color .2s",
        }}
        className="escola-input"
      />
      {open && filtered.length > 0 && (
        <div style={{
          position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, zIndex: 50,
          background: "#00305F", border: "1px solid rgba(0,174,239,0.3)",
          borderRadius: 12, boxShadow: "0 16px 48px rgba(0,10,25,0.6)",
          maxHeight: 220, overflowY: "auto",
        }}>
          {filtered.map((escola) => (
            <div
              key={escola}
              onMouseDown={() => { onChange(escola); setOpen(false); }}
              style={{
                padding: "9px 14px", fontSize: 13, color: "rgba(255,255,255,0.75)",
                cursor: "pointer", borderBottom: "1px solid rgba(255,255,255,0.06)",
                transition: "all .15s",
              }}
              className="escola-item"
            >
              {escola}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── form principal ─────────────────────────────────────────────── */

export function AlunoSignupForm() {
  const searchParams = useSearchParams();
  const nextPath = searchParams.get("next") ?? "/predict";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [escola, setEscola] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!escola.trim()) { setError("Informe sua escola para continuar."); return; }

    setLoading(true);
    setError("");

    const supabase = createClient();
    const { error: signUpError } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { escola: escola.trim(), role: "aluno" },
        emailRedirectTo: `${window.location.origin}/auth/callback?next=${nextPath}`,
      },
    });

    if (signUpError) {
      setError(
        signUpError.message.includes("already registered")
          ? "Este email já possui uma conta. Faça login."
          : "Não foi possível criar a conta. Tente novamente."
      );
      setLoading(false);
      return;
    }

    setDone(true);
  }

  if (done) {
    return (
      <div style={{
        padding: "28px 24px", borderRadius: 16,
        background: "rgba(0,194,106,0.08)", border: "1px solid rgba(0,194,106,0.3)",
        textAlign: "center",
      }}>
        <p style={{ fontSize: 28, marginBottom: 12 }}>✓</p>
        <p style={{ fontSize: 16, fontWeight: 700, color: "#fff", marginBottom: 8 }}>
          Conta criada!
        </p>
        <p style={{ fontSize: 13, color: "rgba(255,255,255,0.6)", lineHeight: 1.6 }}>
          Enviamos um link de confirmação para <strong style={{ color: "#fff" }}>{email}</strong>.
          Abra o email e clique no link para ativar sua conta.
        </p>
      </div>
    );
  }

  return (
    <>
      <style>{`
        .escola-input:focus { border-color: rgba(0,174,239,0.6) !important; }
        .escola-item:hover { background: rgba(0,174,239,0.15); color: #fff; }
        .aluno-input:focus { border-color: rgba(0,174,239,0.6) !important; }
        .signup-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 10px 32px rgba(0,174,239,0.45) !important; background: #33C1F3 !important; }
        .signup-btn { transition: transform .2s, box-shadow .2s, background .2s; }
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
            className="aluno-input"
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
            type="password" value={password} required autoComplete="new-password"
            placeholder="mínimo 8 caracteres"
            minLength={8}
            onChange={(e) => setPassword(e.target.value)}
            className="aluno-input"
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
            Escola <span style={{ textTransform: "none", letterSpacing: 0, fontWeight: 400, fontSize: 11, color: "rgba(255,255,255,0.3)" }}>(obrigatório)</span>
          </label>
          <EscolaCombobox value={escola} onChange={setEscola} />
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
          className="signup-btn"
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
          {loading ? "Criando conta…" : "Criar conta gratuita →"}
        </button>
      </form>
    </>
  );
}
