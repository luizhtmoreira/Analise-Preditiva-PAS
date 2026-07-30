"use client";
import { useState, useRef, useMemo, useEffect, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

/* ─── lista de escolas ──────────────────────────────────────────── */
// Substitua por consulta ao banco quando a tabela de escolas existir.
export const ESCOLAS = [
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

export function EscolaCombobox({ value, onChange }: { value: string; onChange: (v: string) => void }) {
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
    <div ref={ref} className="relative">
      <input
        type="text"
        value={value}
        placeholder="Buscar sua escola…"
        required
        autoComplete="off"
        onFocus={() => setOpen(true)}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        className="vp-input"
      />
      {open && filtered.length > 0 && (
        <div className="vp-dropdown absolute top-[calc(100%+6px)] left-0 right-0 z-50 max-h-[220px] overflow-y-auto">
          {filtered.map((escola) => (
            <div
              key={escola}
              onMouseDown={() => { onChange(escola); setOpen(false); }}
              className="vp-dropdown-item"
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
      <div className="p-6 rounded-2xl bg-[#00843D]/8 border border-[#00843D]/25 text-center">
        <p className="text-2xl mb-2">✓</p>
        <p className="text-base font-bold text-[#002147] mb-2">Conta criada!</p>
        <p className="text-[0.8rem] text-[#4A5568] leading-relaxed">
          Enviamos um link de confirmação para <strong className="text-[#002147]">{email}</strong>.
          Abra o email e clique no link para ativar sua conta.
        </p>
      </div>
    );
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
        <label className="vp-label block mb-1.5">Senha</label>
        <input
          type="password"
          value={password}
          required
          autoComplete="new-password"
          placeholder="mínimo 8 caracteres"
          minLength={8}
          onChange={(e) => setPassword(e.target.value)}
          className="vp-input"
        />
      </div>

      <div>
        <label className="vp-label block mb-1.5">
          Escola <span className="normal-case tracking-normal font-normal text-[#718096]">(obrigatório)</span>
        </label>
        <EscolaCombobox value={escola} onChange={setEscola} />
      </div>

      {error && (
        <div className="text-sm rounded-lg px-3 py-2 bg-[#FFCDD2] text-[#B71C1C]">{error}</div>
      )}

      <button type="submit" disabled={loading} className="vp-btn vp-btn-cyan py-3 mt-1">
        {loading ? "Criando conta…" : "Criar conta gratuita →"}
      </button>
    </form>
  );
}
