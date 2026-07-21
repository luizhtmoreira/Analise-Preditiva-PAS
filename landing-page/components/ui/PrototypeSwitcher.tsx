"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";

interface VariantInfo {
  id: string;
  name: string;
}

interface PrototypeSwitcherProps {
  variants: VariantInfo[];
  current: string;
}

export function PrototypeSwitcher({ variants, current }: PrototypeSwitcherProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const currentIndex = variants.findIndex((v) => v.id === current);
  const activeIndex = currentIndex >= 0 ? currentIndex : 0;

  const setVariant = (id: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("variant", id);
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  const nextVariant = () => {
    const nextIdx = (activeIndex + 1) % variants.length;
    setVariant(variants[nextIdx].id);
  };

  const prevVariant = () => {
    const prevIdx = (activeIndex - 1 + variants.length) % variants.length;
    setVariant(variants[prevIdx].id);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isInput =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.tagName === "SELECT" ||
        (activeElement as HTMLElement)?.isContentEditable;

      if (isInput) return;

      if (e.key === "ArrowLeft") {
        prevVariant();
      } else if (e.key === "ArrowRight") {
        nextVariant();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeIndex]);

  const activeVariantObj = variants[activeIndex];

  return (
    <div className="fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-[#002147]/95 border border-[#00AEEF]/40 backdrop-blur-md px-4 py-2.5 rounded-full shadow-[0_10px_30px_rgba(0,0,0,0.5)] text-white text-xs sm:text-sm select-none">
      <button
        onClick={prevVariant}
        className="w-7 h-7 flex items-center justify-center rounded-full bg-white/10 hover:bg-white/20 active:scale-95 transition-all text-white font-bold"
        title="Variante Anterior (←)"
      >
        ←
      </button>

      <div className="flex items-center gap-2 font-mono">
        <span className="bg-[#00AEEF] text-[#002147] font-extrabold px-2 py-0.5 rounded text-xs">
          VARIANTE {activeVariantObj.id}
        </span>
        <span className="hidden sm:inline text-white/80 font-sans font-medium">
          {activeVariantObj.name}
        </span>
      </div>

      <button
        onClick={nextVariant}
        className="w-7 h-7 flex items-center justify-center rounded-full bg-white/10 hover:bg-white/20 active:scale-95 transition-all text-white font-bold"
        title="Próxima Variante (→)"
      >
        →
      </button>
    </div>
  );
}
