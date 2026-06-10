import { createClient } from "@/lib/supabase/server";
import { fetchGestao } from "@/lib/api";
import { RelatoriosPage } from "@/components/dashboard/relatorios/RelatoriosPage";

const TENANT_LABELS: Record<string, string> = {
  marista: "Colégio Marista",
  ideal:   "Colégio Ideal",
  default: "Vetor PAS",
};

function ErrorShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="p-6">
      <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F] mb-3">
        Relatórios PDF
      </h1>
      {children}
    </div>
  );
}

export default async function RelatoriosServerPage() {
  const supabase = await createClient();

  const { data: { user } } = await supabase.auth.getUser();
  const { data: profile } = user
    ? await supabase.from("profiles").select("tenant").eq("id", user.id).single()
    : { data: null };
  const tenantLabel = TENANT_LABELS[profile?.tenant ?? "default"] ?? profile?.tenant ?? "Vetor PAS";

  const { data: students, error } = await supabase
    .from("tabela_mestra")
    .select("nome,turma,unidade,curso_alvo,cota,trienio,p1_pas1,p2_pas1,red_pas1,p1_pas2,p2_pas2,red_pas2");

  if (error || !students || students.length === 0) {
    return (
      <ErrorShell>
        <div className="rounded-xl p-4 text-sm bg-[#FFF9C4] text-[#F57F17]">
          Nenhum dado encontrado. Faça o upload da planilha da turma para começar.
        </div>
      </ErrorShell>
    );
  }

  const payload = students.map((s) => ({
    nome:        s.nome ?? "",
    turma:       s.turma ?? "",
    unidade:     s.unidade ?? "",
    curso_alvo:  s.curso_alvo ?? "",
    cota:        s.cota ?? "Sistema Universal",
    ano_trienio: s.trienio ?? "2024-2026",
    p1_pas1:     Number(s.p1_pas1 ?? 0),
    p2_pas1:     Number(s.p2_pas1 ?? 0),
    red_pas1:    Number(s.red_pas1 ?? 6),
    p1_pas2:     Number(s.p1_pas2 ?? 0),
    p2_pas2:     Number(s.p2_pas2 ?? 0),
    red_pas2:    Number(s.red_pas2 ?? 6),
  }));

  let data;
  try {
    data = await fetchGestao(payload);
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

  return <RelatoriosPage data={data} tenantLabel={tenantLabel} />;
}
