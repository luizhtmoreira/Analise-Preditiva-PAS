"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  LayoutGrid,
  School,
  Scale,
  FileText,
  Target,
  TrendingUp,
  Calculator,
  LogOut,
  Menu,
  X,
  type LucideIcon,
} from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { BrandMark } from "@/components/brand/BrandMark";

const NAV: { href: string; label: string; icon: LucideIcon }[] = [
  { href: "/gestao",     label: "Gestão de Ativos",     icon: LayoutGrid },
  { href: "/escola",     label: "Escola vs. População", icon: School },
  { href: "/comparacao", label: "Comparação de Grupos", icon: Scale },
  { href: "/relatorios", label: "Relatórios PDF",       icon: FileText },
];

const PUBLIC_NAV: { href: string; label: string; icon: LucideIcon }[] = [
  { href: "/predict",     label: "Preditor PAS 3",            icon: Target },
  { href: "/calculadora", label: "Calculadora de Estratégia", icon: Calculator },
  { href: "/temporal",    label: "Análise Temporal",          icon: TrendingUp },
];


const TENANT_LABELS: Record<string, string> = {
  marista: "Colégio Marista",
  ideal:   "Colégio Ideal",
  default: "Vetor PAS",
};

function NavLink({ href, label, icon: Icon, active }: {
  href: string; label: string; icon: LucideIcon; active: boolean;
}) {
  return (
    <Link
      href={href}
      className={`relative flex items-center gap-3 px-3 py-2 rounded-lg text-[0.84rem] transition-colors ${
        active
          ? "bg-white/12 text-white font-semibold"
          : "text-white/55 hover:text-white hover:bg-white/8"
      }`}
    >
      {active && (
        <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-[var(--vp-cyan)]" />
      )}
      <Icon size={16} strokeWidth={2} className={active ? "text-[var(--vp-cyan)]" : ""} />
      <span>{label}</span>
    </Link>
  );
}

function NavSections({ pathname }: { pathname: string }) {
  return (
    <>
      <p className="font-mono text-[0.62rem] tracking-[0.2em] uppercase px-3 mb-1.5 text-white/35">
        Coordenação
      </p>
      {NAV.map((item) => (
        <NavLink key={item.href} {...item} active={pathname === item.href} />
      ))}

      <p className="font-mono text-[0.62rem] tracking-[0.2em] uppercase px-3 mt-5 mb-1.5 text-white/35">
        Público
      </p>
      {PUBLIC_NAV.map((item) => (
        <NavLink key={item.href} {...item} active={pathname === item.href} />
      ))}
    </>
  );
}

function UserFooter({ userEmail, onLogout }: { userEmail: string; onLogout: () => void }) {
  return (
    <div className="mt-auto pt-4 border-t border-white/12">
      <p className="text-xs px-3 mb-1.5 truncate text-white/40">{userEmail}</p>
      <button
        onClick={onLogout}
        className="w-full flex items-center gap-2 text-left px-3 py-2 rounded-lg text-xs text-white/40 hover:text-white hover:bg-white/8 transition-colors"
      >
        <LogOut size={13} />
        Sair
      </button>
    </div>
  );
}

export function DashboardSidebar({ tenant, userEmail }: { tenant: string; userEmail: string }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);

  // Fecha o drawer ao navegar: ajuste de estado durante o render quando a rota muda, e não um
  // efeito — o efeito renderizava o drawer aberto na rota nova por um quadro antes de fechar.
  const [rotaAnterior, setRotaAnterior] = useState(pathname);
  if (rotaAnterior !== pathname) {
    setRotaAnterior(pathname);
    setOpen(false);
  }

  // Trava o scroll do body enquanto o drawer está aberto
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/auth/login");
    router.refresh();
  }

  const sublabel = TENANT_LABELS[tenant] ?? tenant;

  return (
    <>
      {/* ── Top bar (mobile) ── */}
      <div
        className="md:hidden flex-shrink-0 sticky top-0 z-50 flex items-center justify-between px-4 h-14 border-b border-white/10 print:hidden"
        style={{ background: "var(--vp-ink)" }}
      >
        <BrandMark sublabel={sublabel} />
        <button
          onClick={() => setOpen(!open)}
          aria-label={open ? "Fechar menu" : "Abrir menu"}
          aria-expanded={open}
          className="p-2 -mr-2 text-white/70 hover:text-white transition-colors"
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* ── Drawer (mobile) ── */}
      {open && (
        <div className="md:hidden fixed inset-0 top-14 z-40 print:hidden">
          <div
            className="absolute inset-0 bg-black/45 backdrop-blur-[2px]"
            onClick={() => setOpen(false)}
          />
          <nav
            className="absolute left-0 top-0 bottom-0 w-72 max-w-[85vw] flex flex-col gap-0.5 py-5 px-4 overflow-y-auto shadow-2xl"
            style={{ background: "linear-gradient(180deg, var(--vp-ink) 0%, #003366 100%)" }}
          >
            <NavSections pathname={pathname} />
            <UserFooter userEmail={userEmail} onLogout={handleLogout} />
          </nav>
        </div>
      )}

      {/* ── Sidebar (desktop) ── */}
      <aside
        className="hidden md:flex w-60 flex-shrink-0 flex-col py-6 px-4 print:hidden"
        style={{ background: "linear-gradient(180deg, #002147 0%, #003366 100%)" }}
      >
        <div className="px-2 mb-9">
          <BrandMark sublabel={sublabel} />
        </div>
        <nav className="flex flex-col gap-0.5 flex-1">
          <NavSections pathname={pathname} />
        </nav>
        <UserFooter userEmail={userEmail} onLogout={handleLogout} />
      </aside>
    </>
  );
}
