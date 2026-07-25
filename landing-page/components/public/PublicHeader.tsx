"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { User, LogOut, Target, ChevronDown, LayoutGrid, Building2, Sparkles } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { BrandMark } from "@/components/brand/BrandMark";

export function PublicHeader() {
  const pathname = usePathname();
  const router = useRouter();

  const [user, setUser] = useState<any>(null);
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

  // Close menu when route changes
  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

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
    <header className="sticky top-0 z-50 w-full border-b border-white/10 bg-[#001a35]/85 backdrop-blur-md">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        <BrandMark />

        <nav className="flex items-center gap-1.5 sm:gap-3">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`text-[0.78rem] sm:text-sm px-2.5 sm:px-3.5 py-2 rounded-lg transition-all ${
                  active
                    ? "text-[#00AEEF] bg-[#00AEEF]/10 font-semibold"
                    : "text-white/70 hover:text-white hover:bg-white/5 font-medium"
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
                className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/15 bg-white/5 hover:bg-white/10 hover:border-white/25 transition-all text-left"
                aria-expanded={menuOpen}
              >
                <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-[#00AEEF] to-[#0055A5] flex items-center justify-center text-white font-bold text-xs shadow-sm">
                  {userEmail ? userEmail.charAt(0).toUpperCase() : "U"}
                </div>
                <div className="hidden sm:flex flex-col max-w-[130px]">
                  <span className="text-xs font-semibold text-white truncate leading-tight">
                    {userEmail.split("@")[0]}
                  </span>
                  <span className="text-[10px] text-[#7FD8F7] font-medium leading-tight truncate">
                    {isCoordinator ? "Coordenação" : (userEscola || "Aluno Cadastrado")}
                  </span>
                </div>
                <ChevronDown
                  size={14}
                  className={`text-white/60 transition-transform duration-200 ${
                    menuOpen ? "rotate-180 text-white" : ""
                  }`}
                />
              </button>

              {menuOpen && (
                <div className="absolute right-0 top-full mt-2 w-64 rounded-2xl bg-[#002147] border border-white/15 shadow-2xl p-2 z-50 text-white animate-in fade-in slide-in-from-top-2 duration-150">
                  {/* Account Summary Header */}
                  <div className="p-3 mb-1 rounded-xl bg-white/5 border border-white/10">
                    <p className="text-[11px] font-mono uppercase tracking-wider text-[#7FD8F7] font-medium flex items-center gap-1.5">
                      <Sparkles size={12} />
                      {isCoordinator ? "Coordenador Pedagógico" : "Aluno Cadastrado"}
                    </p>
                    <p className="text-xs font-bold text-white truncate mt-1">{userEmail}</p>
                    {userEscola && (
                      <p className="text-[11px] text-white/60 truncate flex items-center gap-1 mt-0.5">
                        <Building2 size={11} className="text-white/40 flex-shrink-0" />
                        <span className="truncate">{userEscola}</span>
                      </p>
                    )}
                  </div>

                  {/* Options */}
                  <div className="space-y-0.5">
                    {!isCoordinator ? (
                      <>
                        <Link
                          href="/perfil"
                          className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-white/80 hover:text-white hover:bg-white/10 transition-colors font-medium"
                        >
                          <User size={14} className="text-[#00AEEF]" />
                          <span>Meu Perfil</span>
                        </Link>
                        <Link
                          href="/predict"
                          className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-white/80 hover:text-white hover:bg-white/10 transition-colors font-medium"
                        >
                          <Target size={14} className="text-[#00AEEF]" />
                          <span>Painel Multi-Curso</span>
                        </Link>
                      </>
                    ) : (
                      <Link
                        href="/gestao"
                        className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-white/80 hover:text-white hover:bg-white/10 transition-colors font-medium"
                      >
                        <LayoutGrid size={14} className="text-[#00AEEF]" />
                        <span>Painel da Escola</span>
                      </Link>
                    )}
                  </div>

                  <div className="my-1 border-t border-white/10" />

                  {/* Logout Action */}
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors font-medium text-left"
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
              className="text-[0.78rem] sm:text-sm px-4 py-2 rounded-lg font-semibold bg-[#00AEEF] text-[#002147] hover:bg-[#33C1F3] transition-all whitespace-nowrap shadow-md shadow-[#00AEEF]/20 ml-1"
            >
              Entrar
            </Link>
          )}
        </nav>
      </div>
    </header>
  );
}

