"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { LandingPage } from "./LandingPage";
import { VariantA } from "./variants/VariantA";
import { VariantB } from "./variants/VariantB";
import { VariantC } from "./variants/VariantC";
import { PrototypeSwitcher } from "@/components/ui/PrototypeSwitcher";

const VARIANTS = [
  { id: "production", name: "Produção (Atual)" },
  { id: "A", name: "Tech Minimalist (Estilo Linear)" },
  { id: "B", name: "UnB Premium Glow (Glassmorphism)" },
  { id: "C", name: "Neo-Brutalism Clean (Estilo Figma)" },
];

function SwitcherContainer() {
  const searchParams = useSearchParams();
  const variantId = searchParams.get("variant") || "production";

  const renderActiveVariant = () => {
    switch (variantId) {
      case "A":
        return <VariantA />;
      case "B":
        return <VariantB />;
      case "C":
        return <VariantC />;
      default:
        return <LandingPage />;
    }
  };

  return (
    <>
      {renderActiveVariant()}
      <PrototypeSwitcher variants={VARIANTS} current={variantId} />
    </>
  );
}

export function LandingPageWrapper() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#002147] text-white flex items-center justify-center font-mono">Carregando protótipo...</div>}>
      <SwitcherContainer />
    </Suspense>
  );
}
