"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { useRouter } from "next/navigation";

const NAV = [
  { href: "/gestao",      label: "Gestão de Ativos",    icon: "📊" },
  { href: "/escola",      label: "Escola vs. População", icon: "🏫" },
  { href: "/comparacao",  label: "Comparação de Grupos", icon: "⚖️"  },
  { href: "/relatorios",  label: "Relatórios PDF",       icon: "📄" },
];

const PUBLIC_NAV = [
  { href: "/predict",  label: "Preditor PAS 3",   icon: "🎯" },
  { href: "/temporal", label: "Análise Temporal",  icon: "📈" },
];

const TENANT_LABELS: Record<string, string> = {
  marista:  "Colégio Marista",
  ideal:    "Colégio Ideal",
  default:  "Vetor PAS",
};

export function DashboardSidebar({ tenant, userEmail }: { tenant: string; userEmail: string }) {
  const pathname = usePathname();
  const router = useRouter();

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/auth/login");
    router.refresh();
  }

  return (
    <aside className="w-60 flex-shrink-0 flex flex-col py-6 px-4" style={{ background: "#003366" }}>
      <div className="flex items-center gap-2.5 px-2 mb-8">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold flex-shrink-0"
          style={{ background: "#00AEEF", color: "#fff" }}>VP</div>
        <div className="min-w-0">
          <p className="text-sm font-bold text-white leading-tight truncate">Vetor PAS</p>
          <p className="text-xs truncate" style={{ color: "#00AEEF" }}>
            {TENANT_LABELS[tenant] ?? tenant}
          </p>
        </div>
      </div>

      <nav className="flex flex-col gap-0.5 flex-1">
        <p className="text-[10px] font-semibold uppercase px-3 mb-1" style={{ color: "rgba(255,255,255,0.35)" }}>
          Coordenação
        </p>
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link key={item.href} href={item.href}
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all"
              style={{
                background: active ? "rgba(255,255,255,0.12)" : "transparent",
                color: active ? "#fff" : "rgba(255,255,255,0.55)",
                fontWeight: active ? 600 : 400,
              }}>
              <span className="text-base">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}

        <p className="text-[10px] font-semibold uppercase px-3 mt-4 mb-1" style={{ color: "rgba(255,255,255,0.35)" }}>
          Público
        </p>
        {PUBLIC_NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link key={item.href} href={item.href}
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all"
              style={{
                background: active ? "rgba(255,255,255,0.12)" : "transparent",
                color: active ? "#fff" : "rgba(255,255,255,0.45)",
                fontWeight: active ? 600 : 400,
              }}>
              <span className="text-base">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto pt-4 border-t" style={{ borderColor: "rgba(255,255,255,0.12)" }}>
        <p className="text-xs px-2 mb-2 truncate" style={{ color: "rgba(255,255,255,0.4)" }}>
          {userEmail}
        </p>
        <button onClick={handleLogout}
          className="w-full text-left px-3 py-2 rounded-lg text-xs transition-all"
          style={{ color: "rgba(255,255,255,0.4)" }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "#fff")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.4)")}>
          Sair →
        </button>
      </div>
    </aside>
  );
}
