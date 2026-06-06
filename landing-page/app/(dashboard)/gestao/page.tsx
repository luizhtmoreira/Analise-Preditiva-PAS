import { createClient } from "@/lib/supabase/server";
import { fetchGestao } from "@/lib/api";
import { GestaoPage } from "@/components/dashboard/gestao/GestaoPage";

export default async function GestaoServerPage() {
  const supabase = await createClient();

  const { data: students, error } = await supabase
    .from("tabela_mestra")
    .select("nome,turma,unidade,curso_alvo,cota,trienio,p1_pas1,p2_pas1,red_pas1,p1_pas2,p2_pas2,red_pas2");

  if (error || !students || students.length === 0) {
    return (
      <div className="p-6">
        <h1 className="text-xl font-bold mb-2" style={{ color: "#1D1D1F" }}>Gestão de Ativos</h1>
        <div className="rounded-xl p-4 text-sm" style={{ background: "#FFF9C4", color: "#F57F17" }}>
          Nenhum dado encontrado. Faça o upload da planilha da turma para começar.
        </div>
      </div>
    );
  }

  // Mapeia colunas snake_case do Supabase para o schema da API
  const payload = students.map((s) => ({
    nome:       s.nome ?? "",
    turma:      s.turma ?? "",
    unidade:    s.unidade ?? "",
    curso_alvo: s.curso_alvo ?? "",
    cota:       s.cota ?? "Sistema Universal",
    ano_trienio: s.trienio ?? "2024-2026",
    p1_pas1:    Number(s.p1_pas1 ?? 0),
    p2_pas1:    Number(s.p2_pas1 ?? 0),
    red_pas1:   Number(s.red_pas1 ?? 6),
    p1_pas2:    Number(s.p1_pas2 ?? 0),
    p2_pas2:    Number(s.p2_pas2 ?? 0),
    red_pas2:   Number(s.red_pas2 ?? 6),
  }));

  let data;
  try {
    data = await fetchGestao(payload);
  } catch {
    return (
      <div className="p-6">
        <h1 className="text-xl font-bold mb-2" style={{ color: "#1D1D1F" }}>Gestão de Ativos</h1>
        <div className="rounded-xl p-4 text-sm" style={{ background: "#FFCDD2", color: "#B71C1C" }}>
          Serviço de análise indisponível. Verifique se a API está rodando em{" "}
          <code className="font-mono">{process.env.API_URL ?? "http://localhost:8000"}</code>.
        </div>
      </div>
    );
  }

  return <GestaoPage data={data} />;
}
