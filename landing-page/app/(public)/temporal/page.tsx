import Link from "next/link";
import { BrandMark } from "@/components/brand/BrandMark";
import { TemporalPage } from "@/components/public/temporal/TemporalPage";
import { fetchTemporal } from "@/lib/api";

// ISR: dados oficiais mudam raramente; revalida a cada hora (inclusive após erro da API)
export const revalidate = 3600;

export default async function TemporalServerPage() {
  let data;
  try {
    data = await fetchTemporal();
  } catch {
    return (
      <div
        className="min-h-screen text-white flex flex-col"
        style={{ background: "linear-gradient(168deg, #002147 0%, #003366 60%, #003A70 100%)" }}
      >
        <div className="border-b border-white/10">
          <div className="max-w-3xl mx-auto px-5 py-3.5 flex items-center justify-between">
            <BrandMark sublabel="Análise Temporal" />
            <Link href="/" className="text-xs text-white/55 hover:text-white transition-colors">← Início</Link>
          </div>
        </div>
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="rounded-2xl border border-[#FF6B6B]/30 bg-[#FF6B6B]/10 px-6 py-5 text-sm text-[#FF9B9B] max-w-md text-center">
            Serviço de dados indisponível. Verifique se a API está rodando.
          </div>
        </div>
      </div>
    );
  }

  return <TemporalPage data={data} />;
}
