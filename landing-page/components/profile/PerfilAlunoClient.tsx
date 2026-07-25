"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { User, Building2, Target, LogOut, ShieldCheck, CheckCircle2, ArrowRight, Save, KeyRound } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { PublicHeader } from "@/components/public/PublicHeader";
import { EscolaCombobox } from "@/components/auth/AlunoSignupForm";

export function PerfilAlunoClient() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [escola, setEscola] = useState("");
  const [savingEscola, setSavingEscola] = useState(false);
  const [escolaSavedMsg, setEscolaSavedMsg] = useState("");
  const [loggingOut, setLoggingOut] = useState(false);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (!user) {
        router.push("/auth/entrar?next=/perfil");
      } else {
        setUser(user);
        setEscola(user.user_metadata?.escola || "");
      }
      setLoading(false);
    });
  }, [router]);

  async function handleSaveEscola(e: React.FormEvent) {
    e.preventDefault();
    if (!escola.trim()) return;

    setSavingEscola(true);
    setEscolaSavedMsg("");

    const supabase = createClient();
    const { error } = await supabase.auth.updateUser({
      data: { escola: escola.trim() },
    });

    if (!error) {
      setEscolaSavedMsg("Escola atualizada com sucesso!");
      setTimeout(() => setEscolaSavedMsg(""), 3500);
    }
    setSavingEscola(false);
  }

  async function handleLogout() {
    setLoggingOut(true);
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/predict");
    router.refresh();
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-[#001a35] flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-[#00AEEF] border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-white/50 font-medium">Carregando dados do perfil…</p>
        </div>
      </div>
    );
  }

  if (!user) return null;

  const email = user.email || "";
  const initial = email.charAt(0).toUpperCase();

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#001a35] via-[#002147] to-[#003366] text-white flex flex-col">
      <PublicHeader />

      <main className="flex-1 max-w-3xl w-full mx-auto px-4 py-8 sm:py-12 space-y-6">
        {/* Page Title */}
        <div className="flex items-center justify-between border-b border-white/10 pb-5">
          <div>
            <span className="text-[11px] font-mono tracking-widest uppercase text-[#7FD8F7] font-semibold">
              Área do Aluno
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight mt-1 text-white">
              Meu Perfil
            </h1>
          </div>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#00AEEF]/15 border border-[#00AEEF]/30 text-[#7FD8F7]">
            <ShieldCheck size={14} />
            Aluno Cadastrado
          </span>
        </div>

        {/* User Card */}
        <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md shadow-xl flex flex-col sm:flex-row items-start sm:items-center gap-5">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-[#00AEEF] to-[#0055A5] flex items-center justify-center text-white font-black text-2xl shadow-lg shadow-[#00AEEF]/20 flex-shrink-0">
            {initial}
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-bold text-white truncate">{email}</h2>
            <p className="text-xs text-white/60 mt-0.5 flex items-center gap-1.5">
              <Building2 size={13} className="text-[#00AEEF]" />
              <span className="truncate">{escola || "Escola não informada"}</span>
            </p>
          </div>
        </div>

        {/* School Update Card */}
        <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md shadow-xl space-y-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Building2 size={18} className="text-[#00AEEF]" />
              Escola Vinculada
            </h3>
            <p className="text-xs text-white/60 mt-1 leading-relaxed">
              Mantenha sua escola atualizada para acompanhar estatísticas e simulações alinhadas ao seu colégio.
            </p>
          </div>

          <form onSubmit={handleSaveEscola} className="space-y-4">
            <div>
              <label className="block text-[11px] font-semibold text-white/50 uppercase tracking-wider mb-2 font-mono">
                Selecione ou busque sua escola
              </label>
              <EscolaCombobox value={escola} onChange={setEscola} />
            </div>

            {escolaSavedMsg && (
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs flex items-center gap-2">
                <CheckCircle2 size={16} />
                <span>{escolaSavedMsg}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={savingEscola || !escola.trim()}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#00AEEF] text-[#002147] text-xs font-bold hover:bg-[#33C1F3] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-[#00AEEF]/20"
            >
              <Save size={15} />
              {savingEscola ? "Salvando…" : "Salvar Alterações"}
            </button>
          </form>
        </div>

        {/* Predictor Shortcut Card */}
        <div className="p-6 rounded-2xl bg-gradient-to-r from-[#00AEEF]/10 via-transparent to-transparent border border-[#00AEEF]/30 backdrop-blur-md shadow-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="space-y-1 max-w-md">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Target size={18} className="text-[#00AEEF]" />
              Preditor PAS 3
            </h3>
            <p className="text-xs text-white/70 leading-relaxed">
              Simule suas notas no PAS 3 e acompanhe sua probabilidade matemática de aprovação na UnB.
            </p>
          </div>

          <Link
            href="/predict"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-[#00AEEF] text-[#002147] text-xs font-bold hover:bg-[#33C1F3] transition-all shadow-lg shadow-[#00AEEF]/25 flex-shrink-0"
          >
            <span>Ir para o Preditor</span>
            <ArrowRight size={15} />
          </Link>
        </div>

        {/* Security & Account Actions Card */}
        <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md shadow-xl space-y-4">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <KeyRound size={18} className="text-[#00AEEF]" />
            Sessão & Conta
          </h3>
          <p className="text-xs text-white/60 leading-relaxed">
            Deseja sair da sua conta neste dispositivo? Suas simulações continuarão salvas em seu e-mail.
          </p>

          <div className="pt-2">
            <button
              onClick={handleLogout}
              disabled={loggingOut}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-red-500/15 border border-red-500/30 text-red-400 text-xs font-bold hover:bg-red-500/25 transition-all disabled:opacity-50"
            >
              <LogOut size={15} />
              {loggingOut ? "Saindo da conta…" : "Sair da Conta (Logout)"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
