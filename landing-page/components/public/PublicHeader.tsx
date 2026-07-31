"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { User, LogOut, ChevronDown, LayoutGrid, Building2, Menu, X } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { BrandMark } from "@/components/brand/BrandMark";
import type { User as SupabaseUser } from "@supabase/supabase-js";

export function PublicHeader() {
  const pathname = usePathname();
  const router = useRouter();

  const [user, setUser] = useState<SupabaseUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const mobileRef = useRef<HTMLDivElement>(null);

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
      if (mobileRef.current && !mobileRef.current.contains(event.target as Node)) {
        setMobileMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Fecha os menus ao navegar
  const [rotaAnterior, setRotaAnterior] = useState(pathname);
  if (rotaAnterior !== pathname) {
    setRotaAnterior(pathname);
    setMenuOpen(false);
    setMobileMenuOpen(false);
  }

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    setUser(null);
    setMenuOpen(false);
    setMobileMenuOpen(false);
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
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        <BrandMark light={false} />

        {/* Desktop Navigation & User Menu */}
        <nav className="hidden md:flex items-center gap-3">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm px-3.5 py-2 rounded-lg transition-colors ${
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
                <div className="flex flex-col max-w-[130px]">
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
              className="text-sm px-4 py-2 rounded-xl font-semibold bg-[#002147] text-white hover:bg-[#003366] transition-all whitespace-nowrap shadow-sm active:scale-95 ml-1"
            >
              Entrar
            </Link>
          )}
        </nav>

        {/* Mobile Hamburger Controls */}
        <div className="md:hidden flex items-center gap-2" ref={mobileRef}>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-xl border border-black/10 bg-[#F8F9FA] hover:bg-white text-[#002147] transition-all active:scale-95"
            aria-label="Abrir menu"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>

          {/* Mobile Sandwich Menu Dropdown */}
          {mobileMenuOpen && (
            <div className="absolute top-full left-0 right-0 w-full bg-white border-b border-[#E2E8F0] shadow-xl p-4 z-50 flex flex-col gap-3 animate-in fade-in slide-in-from-top-2 duration-150">
              <nav className="flex flex-col gap-1">
                {links.map((link) => {
                  const active = pathname === link.href;
                  return (
                    <Link
                      key={link.href}
                      href={link.href}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`text-sm px-4 py-2.5 rounded-xl transition-colors font-semibold ${
                        active
                          ? "text-[#00843D] bg-[#00843D]/10 font-bold"
                          : "text-[#002147] hover:bg-[#F8F9FA]"
                      }`}
                    >
                      {link.label}
                    </Link>
                  );
                })}
              </nav>

              <div className="border-t border-[#E2E8F0] pt-3">
                {!loading && user ? (
                  <div className="flex flex-col gap-2">
                    <div className="p-3 rounded-xl bg-[#F8F9FA] border border-black/5">
                      <p className="text-[11px] font-mono uppercase tracking-wider text-[#00843D] font-bold">
                        {isCoordinator ? "Coordenador Pedagógico" : "Aluno Cadastrado"}
                      </p>
                      <p className="text-xs font-bold text-[#002147] truncate mt-0.5">{userEmail}</p>
                      {userEscola && (
                        <p className="text-[11px] text-[#4A5568] truncate flex items-center gap-1 mt-0.5">
                          <Building2 size={11} className="text-[#718096] flex-shrink-0" />
                          <span className="truncate">{userEscola}</span>
                        </p>
                      )}
                    </div>

                    {!isCoordinator ? (
                      <Link
                        href="/perfil"
                        onClick={() => setMobileMenuOpen(false)}
                        className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-bold text-[#002147] hover:bg-[#F8F9FA]"
                      >
                        <User size={16} className="text-[#00AEEF]" />
                        <span>Meu Perfil</span>
                      </Link>
                    ) : (
                      <Link
                        href="/gestao"
                        onClick={() => setMobileMenuOpen(false)}
                        className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-bold text-[#002147] hover:bg-[#F8F9FA]"
                      >
                        <LayoutGrid size={16} className="text-[#00AEEF]" />
                        <span>Painel da Escola</span>
                      </Link>
                    )}

                    <button
                      onClick={handleLogout}
                      className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-bold text-[#B71C1C] hover:bg-[#FFCDD2]/50 text-left"
                    >
                      <LogOut size={16} />
                      <span>Sair da Conta</span>
                    </button>
                  </div>
                ) : (
                  <Link
                    href="/auth/entrar"
                    onClick={() => setMobileMenuOpen(false)}
                    className="flex items-center justify-center text-sm px-4 py-2.5 rounded-xl font-bold bg-[#002147] text-white hover:bg-[#003366] transition-all w-full"
                  >
                    Entrar
                  </Link>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
