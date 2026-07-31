"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { User, LogOut, Target, ChevronDown, LayoutGrid, Building2, Sparkles } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { BrandMark } from "@/components/brand/BrandMark";
import type { User as SupabaseUser } from "@supabase/supabase-js";

export function PublicHeader() {
  const pathname = usePathname();
  const router = useRouter();

  const [user, setUser] = useState<SupabaseUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const supabase = createClient();

    // Initial user fetch
    supabase.auth.getUser().then(({ data: { user } }) => {
      setUser(user ?? null);
      setLoading(false);
    });

    // Real-time auth state listener
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Fecha o menu ao navegar: mesmo padrão do drawer da Sidebar — ajuste durante o render, e não
  // um efeito, que deixaria o menu aberto por um quadro na rota nova.
  const [rotaAnterior, setRotaAnterior] = useState(pathname);
  if (rotaAnterior !== pathname) {
    setRotaAnterior(pathname);
    setMenuOpen(false);
  }

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    setUser(null);
    setMenuOpen(false);
    if (pathname === "/perfil") {
      router.push("/predict");
    }
    router.refresh();
  }

  const links = [
    { href: "/predict", label: "Preditor" },
    { href: "/calculadora", label: "Calculadora" },
    { href: "/temporal", label: "Análise Temporal" },
  ];

  const metadata = user?.user_metadata || {};
  const isCoordinator = Boolean(metadata.tenant || metadata.role === "coordenador");
  const userEscola = metadata.escola || metadata.tenant || "";
  const userEmail = user?.email || "";

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[#E2E8F0] bg-white/95 backdrop-blur-md transition-all duration-300">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <BrandMark light={false} />

        <nav className="flex items-center gap-1.5 sm:gap-3">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`text-[0.78rem] sm:text-sm px-2.5 sm:px-3.5 py-2 rounded-lg transition-colors ${
                  active
                    ? "text-[#00843D] bg-[#00843D]/8 font-bold"
                    : "text-[#002147] hover:text-[#00843D] font-semibold"
                }`}
              >
                {link.label}
              </Link>
            );
          })}

          {!loading && user ? (
            <div className="relative ml-1" ref={menuRef}>
              <button
                onClick={() => setMenuOpen(!menuOpen)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-black/10 bg-[#F8F9FA] hover:bg-white hover:border-[#00843D]/30 transition-all text-left shadow-sm active:scale-95"
                aria-expanded={menuOpen}
              >
                <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-[#00AEEF] to-[#002147] flex items-center justify-center text-white font-bold text-xs shadow-sm">
                  {userEmail ? userEmail.charAt(0).toUpperCase() : "U"}
                </div>
                <div className="hidden sm:flex flex-col max-w-[130px]">
                  <span className="text-xs font-bold text-[#002147] truncate leading-tight">
                    {userEmail.split("@")[0]}
                  </span>
                  <span className="text-[10px] text-[#00843D] font-semibold leading-tight truncate">
                    {isCoordinator ? "Coordenação" : (userEscola || "Aluno Cadastrado")}
                  </span>
                </div>
                <ChevronDown
                  size={14}
                  className={`text-[#4A5568] transition-transform duration-200 ${
                    menuOpen ? "rotate-180 text-[#002147]" : ""
                  }`}
                />
              </button>

              {menuOpen && (
                <div className="absolute right-0 top-full mt-2 w-64 rounded-2xl bg-white border border-black/5 shadow-[0_15px_40px_rgba(0,33,71,0.12)] p-2 z-50 animate-in fade-in slide-in-from-top-2 duration-150">
                  {/* Account Summary Header */}
                  <div className="p-3 mb-1 rounded-xl bg-[#F8F9FA] border border-black/5">
                    <p className="text-[11px] font-mono uppercase tracking-wider text-[#00843D] font-bold flex items-center gap-1.5">
                      {isCoordinator ? "Coordenador Pedagógico" : "Aluno Cadastrado"}
                    </p>
                    <p className="text-xs font-bold text-[#002147] truncate mt-1">{userEmail}</p>
                    {userEscola && (
                      <p className="text-[11px] text-[#4A5568] truncate flex items-center gap-1 mt-0.5">
                        <Building2 size={11} className="text-[#718096] flex-shrink-0" />
                        <span className="truncate">{userEscola}</span>
                      </p>
                    )}
                  </div>

                  {/* Options */}
                  <div className="space-y-0.5">
                    {!isCoordinator ? (
                      <Link
                        href="/perfil"
                        className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-[#4A5568] hover:text-[#002147] hover:bg-[#F8F9FA] transition-colors font-semibold"
                      >
                        <User size={14} className="text-[#00AEEF]" />
                        <span>Meu Perfil</span>
                      </Link>
                    ) : (
                      <Link
                        href="/gestao"
                        className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-[#4A5568] hover:text-[#002147] hover:bg-[#F8F9FA] transition-colors font-semibold"
                      >
                        <LayoutGrid size={14} className="text-[#00AEEF]" />
                        <span>Painel da Escola</span>
                      </Link>
                    )}
                  </div>

                  <div className="my-1 border-t border-black/5" />

                  {/* Logout Action */}
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-[#B71C1C] hover:bg-[#FFCDD2]/50 transition-colors font-semibold text-left"
                  >
                    <LogOut size={14} />
                    <span>Sair da Conta</span>
                  </button>
                </div>
              )}
            </div>
          ) : (
            <Link
              href="/auth/entrar"
              className="text-[0.78rem] sm:text-sm px-4 py-2 rounded-xl font-semibold bg-[#002147] text-white hover:bg-[#003366] transition-all whitespace-nowrap shadow-sm active:scale-95 ml-1"
            >
              Entrar
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
}

