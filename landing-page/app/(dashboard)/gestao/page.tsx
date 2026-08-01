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
        <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F] mb-3">Gestão de Ativos</h1>
        <div className="rounded-xl p-4 text-sm bg-[#FFF9C4] text-[#F57F17]">
          Nenhum dado encontrado. Faça o upload da planilha da turma para começar.
        </div>
      </div>
    );
  }

  // Mapeia colunas snake_case do Supabase para o schema da API.
  //
  // Nenhuma nota faltante é preenchida com valor inventado. Antes, `?? 0` nas objetivas e
  // `?? 6` nas Redações substituíam "não informado" por um Aluno plausível que não existe:
  // a Redação 6,0 sozinha inflava o Argumento da Etapa em ~3,3 pontos na direção do
  // otimismo, e nada na tela registrava que tinha sido inventada.
  //
  // "Não informado" é um estado real e distinto de "não fez a Etapa" (que o Edital publica
  // como 0,000) e de "tirou zero". Ver ADR-0008. Aqui ele é separado e mostrado, não suprido.
  const linhas = students.map((s) => {
    const notas = {
      p1_pas1: s.p1_pas1, p2_pas1: s.p2_pas1, red_pas1: s.red_pas1,
      p1_pas2: s.p1_pas2, p2_pas2: s.p2_pas2, red_pas2: s.red_pas2,
    };
    const faltantes = Object.entries(notas)
      .filter(([, v]) => v === null || v === undefined || Number.isNaN(Number(v)))
      .map(([campo]) => campo);
    return { s, notas, faltantes };
  });

  const completos   = linhas.filter((l) => l.faltantes.length === 0);
  const incompletos = linhas.filter((l) => l.faltantes.length > 0);

  const avisoIncompletos =
    incompletos.length === 0 ? null : (
      <div className="px-6 pt-6">
        <div className="rounded-xl p-4 text-sm bg-[#FFF9C4] text-[#F57F17]">
          <strong>{incompletos.length}</strong>{" "}
          {incompletos.length === 1 ? "aluno ficou fora da análise" : "alunos ficaram fora da análise"}{" "}
          por ter nota não informada. Complete o cadastro para incluí-{incompletos.length === 1 ? "lo" : "los"}:
          <ul className="mt-2 space-y-0.5 font-mono text-xs">
            {incompletos.map((l, i) => (
              <li key={i}>
                {l.s.nome ?? "(sem nome)"} — falta {l.faltantes.join(", ")}
              </li>
            ))}
          </ul>
        </div>
      </div>
    );

  if (completos.length === 0) {
    return (
      <div className="p-6">
        <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F] mb-3">Gestão de Ativos</h1>
        <div className="rounded-xl p-4 text-sm bg-[#FFF9C4] text-[#F57F17]">
          Nenhum aluno com as seis notas informadas. Complete o cadastro para começar.
        </div>
        {avisoIncompletos}
      </div>
    );
  }

  const payload = completos.map(({ s, notas }) => ({
    nome:       s.nome ?? "",
    turma:      s.turma ?? "",
    unidade:    s.unidade ?? "",
    curso_alvo: s.curso_alvo ?? "",
    cota:       s.cota ?? "Sistema Universal",
    ano_trienio: s.trienio ?? "2024-2026",
    p1_pas1:    Number(notas.p1_pas1),
    p2_pas1:    Number(notas.p2_pas1),
    red_pas1:   Number(notas.red_pas1),
    p1_pas2:    Number(notas.p1_pas2),
    p2_pas2:    Number(notas.p2_pas2),
    red_pas2:   Number(notas.red_pas2),
  }));

  let data;
  try {
    data = await fetchGestao(payload);
  } catch (e) {
    // O motivo tem que aparecer: uma rejeição de validação e uma API fora do ar exigem
    // conserto em lugares diferentes, e antes as duas liam igual na tela.
    const motivo = e instanceof Error ? e.message : String(e);
    return (
      <div className="p-6">
        <h1 className="font-heading text-2xl font-bold tracking-[-0.025em] text-[#1D1D1F] mb-3">Gestão de Ativos</h1>
        <div className="rounded-xl p-4 text-sm bg-[#FFCDD2] text-[#B71C1C]">
          Análise indisponível. API em{" "}
          <code className="font-mono">{process.env.API_URL ?? "http://localhost:8000"}</code>.
          <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-xs">{motivo}</pre>
        </div>
      </div>
    );
  }

  return (
    <>
      {avisoIncompletos}
      <GestaoPage data={data} />
    </>
  );
}
