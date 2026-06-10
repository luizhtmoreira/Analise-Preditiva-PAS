import { createClient } from "@/lib/supabase/server";
import { fetchEscola } from "@/lib/api";
import { EscolaPage } from "@/components/dashboard/escola/EscolaPage";

function ErrorShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="p-6">
      <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F] mb-3">
        Escola vs. População
      </h1>
      {children}
    </div>
  );
}

export default async function EscolaServerPage() {
  const supabase = await createClient();

  const { data: rows, error } = await supabase
    .from("tabela_mestra")
    .select("inscricao,trienio");

  const inscricoes = (rows ?? [])
    .map((r) => (r.inscricao != null ? String(r.inscricao).trim() : ""))
    .filter(Boolean);

  if (error || inscricoes.length === 0) {
    return (
      <ErrorShell>
        <div className="rounded-xl p-4 text-sm bg-[#FFF9C4] text-[#F57F17]">
          Nenhuma inscrição encontrada na base da escola. O match com a população do PAS é
          feito pelo número de inscrição — verifique se a planilha enviada contém essa coluna.
        </div>
      </ErrorShell>
    );
  }

  // Triênio concluído de referência: derivado do triênio em andamento da base
  const trienioBase = rows?.find((r) => r.trienio)?.trienio ?? "";

  let data;
  try {
    data = await fetchEscola(inscricoes, trienioBase);
  } catch {
    return (
      <ErrorShell>
        <div className="rounded-xl p-4 text-sm bg-[#FFCDD2] text-[#B71C1C]">
          Serviço de análise indisponível. Verifique se a API está rodando em{" "}
          <code className="font-mono">{process.env.API_URL ?? "http://localhost:8000"}</code>.
        </div>
      </ErrorShell>
    );
  }

  return <EscolaPage initial={data} inscricoes={inscricoes} />;
}
