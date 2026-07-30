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
      <div className="min-h-screen bg-[#F8F9FA] flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-[#00843D] border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-[#4A5568] font-medium">Carregando dados do perfil…</p>
        </div>
      </div>
    );
  }

  if (!user) return null;

  const email = user.email || "";
  const initial = email.charAt(0).toUpperCase();

  return (
    <div className="min-h-screen bg-[#F8F9FA] flex flex-col">
      <PublicHeader />

      <main className="flex-1 max-w-3xl w-full mx-auto px-4 py-8 sm:py-12 space-y-6">
        {/* Page Title */}
        <div className="flex items-center justify-between border-b border-[#E2E8F0] pb-5">
          <div>
            <span className="vp-eyebrow vp-eyebrow-cyan">Área do Aluno</span>
            <h1 className="font-heading text-2xl sm:text-3xl font-extrabold tracking-tight mt-2 text-[#002147]">
              Meu Perfil
            </h1>
          </div>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#00843D]/8 border border-[#00843D]/25 text-[#00843D]">
            <ShieldCheck size={14} />
            Aluno Cadastrado
          </span>
        </div>

        {/* User Card */}
        <div className="vp-card p-6 flex flex-col sm:flex-row items-start sm:items-center gap-5">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-[#00AEEF] to-[#002147] flex items-center justify-center text-white font-black text-2xl shadow-lg shadow-[#00AEEF]/20 flex-shrink-0">
            {initial}
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-bold text-[#002147] truncate">{email}</h2>
            <p className="text-xs text-[#4A5568] mt-0.5 flex items-center gap-1.5">
              <Building2 size={13} className="text-[#00AEEF]" />
              <span className="truncate">{escola || "Escola não informada"}</span>
            </p>
          </div>
        </div>

        {/* School Update Card */}
        <div className="vp-card p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-[#002147] flex items-center gap-2">
              <Building2 size={18} className="text-[#00AEEF]" />
              Escola Vinculada
            </h3>
            <p className="text-xs text-[#4A5568] mt-1 leading-relaxed">
              Mantenha sua escola atualizada para acompanhar estatísticas e simulações alinhadas ao seu colégio.
            </p>
          </div>

          <form onSubmit={handleSaveEscola} className="space-y-4">
            <div>
              <label className="vp-label block mb-2">Selecione ou busque sua escola</label>
              <EscolaCombobox value={escola} onChange={setEscola} />
            </div>

            {escolaSavedMsg && (
              <div className="p-3 rounded-xl bg-[#00843D]/8 border border-[#00843D]/25 text-[#00843D] text-xs flex items-center gap-2">
                <CheckCircle2 size={16} />
                <span>{escolaSavedMsg}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={savingEscola || !escola.trim()}
              className="vp-btn vp-btn-cyan px-5 py-2.5 text-xs"
            >
              <Save size={15} />
              {savingEscola ? "Salvando…" : "Salvar Alterações"}
            </button>
          </form>
        </div>

        {/* Predictor Shortcut Card */}
        <div className="vp-card border-t-4 border-t-[#00AEEF] p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="space-y-1 max-w-md">
            <h3 className="text-base font-bold text-[#002147] flex items-center gap-2">
              <Target size={18} className="text-[#00AEEF]" />
              Preditor PAS 3
            </h3>
            <p className="text-xs text-[#4A5568] leading-relaxed">
              Simule suas notas no PAS 3 e acompanhe sua probabilidade matemática de aprovação na UnB.
            </p>
          </div>

          <Link href="/predict" className="vp-btn vp-btn-cyan px-5 py-3 text-xs flex-shrink-0">
            <span>Ir para o Preditor</span>
            <ArrowRight size={15} />
          </Link>
        </div>

        {/* Security & Account Actions Card */}
        <div className="vp-card p-6 space-y-4">
          <h3 className="text-base font-bold text-[#002147] flex items-center gap-2">
            <KeyRound size={18} className="text-[#00AEEF]" />
            Sessão & Conta
          </h3>
          <p className="text-xs text-[#4A5568] leading-relaxed">
            Deseja sair da sua conta neste dispositivo? Suas simulações continuarão salvas em seu e-mail.
          </p>

          <div className="pt-2">
            <button
              onClick={handleLogout}
              disabled={loggingOut}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#FFCDD2]/60 border border-[#B71C1C]/20 text-[#B71C1C] text-xs font-bold hover:bg-[#FFCDD2] transition-all disabled:opacity-50"
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
