"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrandMark } from "@/components/brand/BrandMark";

export function PublicHeader() {
  const pathname = usePathname();

  const links = [
    { href: "/predict", label: "Preditor" },
    { href: "/calculadora", label: "Calculadora" },
    { href: "/temporal", label: "Análise Temporal" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/10 bg-[#001a35]/75 backdrop-blur-md">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        <BrandMark />

        <nav className="flex items-center gap-1 sm:gap-3">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`text-[0.78rem] sm:text-sm px-2.5 sm:px-3.5 py-2 rounded-lg transition-all ${
                  active
                    ? "text-[#00AEEF] bg-[#00AEEF]/8 font-semibold"
                    : "text-white/70 hover:text-white hover:bg-white/5 font-medium"
                }`}
              >
                {link.label}
              </Link>
            );
          })}

          <Link
            href="/auth/login"
            className="text-[0.78rem] sm:text-sm px-3 sm:px-4 py-2 rounded-lg font-semibold border border-white/25 text-white hover:bg-white hover:text-[#002147] transition-all ml-1 whitespace-nowrap"
          >
            Coordenação
          </Link>
        </nav>
      </div>
    </header>
  );
}
