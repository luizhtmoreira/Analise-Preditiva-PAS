import { createClient } from "@/lib/supabase/server";
import { ComparacaoPage } from "@/components/dashboard/comparacao/ComparacaoPage";
import type { StudentRow } from "@/lib/types";

export default async function ComparacaoServerPage() {
  const supabase = await createClient();

  const { data: rows, error } = await supabase
    .from("tabela_mestra")
    .select("nome,turma,unidade,p1_pas1,p2_pas1,red_pas1,p1_pas2,p2_pas2,red_pas2");

  if (error || !rows || rows.length === 0) {
    return (
      <div className="p-6">
        <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F] mb-3">
          Comparação Entre Grupos
        </h1>
        <div className="rounded-xl p-4 text-sm bg-[#FFF9C4] text-[#F57F17]">
          Nenhum dado encontrado. Faça o upload da planilha da turma para começar.
        </div>
      </div>
    );
  }

  const students: StudentRow[] = rows.map((r) => ({
    nome: r.nome ?? "",
    turma: r.turma ?? "",
    unidade: r.unidade ?? "",
    curso_alvo: "",
    cota: "",
    trienio: "",
    p1_pas1: Number(r.p1_pas1 ?? 0),
    p2_pas1: Number(r.p2_pas1 ?? 0),
    red_pas1: Number(r.red_pas1 ?? 0),
    p1_pas2: Number(r.p1_pas2 ?? 0),
    p2_pas2: Number(r.p2_pas2 ?? 0),
    red_pas2: Number(r.red_pas2 ?? 0),
  }));

  return <ComparacaoPage students={students} />;
}
