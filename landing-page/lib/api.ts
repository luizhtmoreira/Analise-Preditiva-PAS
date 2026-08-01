import type { ChamadaCorte } from "./types";
// Server-side: process.env.API_URL. Client-side: NEXT_PUBLIC_API_URL.
const API_URL =
  (typeof window === "undefined"
    ? process.env.API_URL
    : process.env.NEXT_PUBLIC_API_URL) ?? "https://api.vetorpas.com.br";

export async function fetchGestao(students: unknown[], trienio = "2024-2026", cenario = "padrao") {
  const res = await fetch(`${API_URL}/api/gestao/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ students, trienio, cenario }),
    cache: "no-store",
  });

  if (!res.ok) {
    // O corpo carrega o motivo — um 422 do Pydantic nomeia o campo e a linha. Descartá-lo
    // fazia qualquer rejeição de validação virar "API indisponível" na tela.
    const detalhe = await res.text().catch(() => "");
    throw new Error(`API error: ${res.status}${detalhe ? ` — ${detalhe}` : ""}`);
  }

  return res.json();
}

export async function fetchCourses(cota: string, trienio: string): Promise<string[]> {
  const params = new URLSearchParams({ cota, trienio });
  const res = await fetch(`${API_URL}/api/courses?${params}`);
  if (!res.ok) return [];
  return res.json();
}

export async function fetchTemporal() {
  const res = await fetch(`${API_URL}/api/temporal`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchCorteEvolucao(curso: string) {
  const params = new URLSearchParams({ curso });
  const res = await fetch(`${API_URL}/api/temporal/corte?${params}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchCourseChamadas(curso: string, cota: string, trienio: string, semestre: string): Promise<ChamadaCorte[]> {
  const params = new URLSearchParams({ curso, cota, trienio, semestre });
  const res = await fetch(`${API_URL}/api/courses/chamadas?${params}`);
  if (!res.ok) return [];
  return res.json();
}

export async function fetchEscola(inscricoes: string[], trienio = "") {
  const res = await fetch(`${API_URL}/api/escola/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ inscricoes, trienio }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function compareGroups(body: {
  group_a: number[]; group_b: number[];
  name_a: string; name_b: string; metric: string;
}) {
  const res = await fetch(`${API_URL}/api/comparacao`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchPredict(body: {
  p1_pas1: number; p2_pas1: number; red_pas1: number;
  p1_pas2: number; p2_pas2: number; red_pas2: number;
  // Uma por Etapa, ambas obrigatórias: o backend as exige (422 se faltar qualquer uma), porque
  // o Cebraspe registra a língua por Etapa — 13,9% dos Alunos trocam entre a Etapa 1 e a 2 — e
  // um default silencioso calcularia a Parte 1 de quem trocou com a estatística errada.
  lingua_e1: "inglesa" | "francesa" | "espanhola";
  lingua_e2: "inglesa" | "francesa" | "espanhola";
  cota?: string; trienio?: string; curso_alvo?: string;
  is_logged_in?: boolean;
  semestre?: string;
}) {
  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchCoursesCutoff(curso: string, cota: string, trienio: string, semestre: string): Promise<{ cutoff: number }> {
  const params = new URLSearchParams({ curso, cota, trienio, semestre });
  const res = await fetch(`${API_URL}/api/courses/cutoff?${params}`);
  if (!res.ok) return { cutoff: 0 };
  return res.json();
}

export async function fetchStrategy(body: {
  p1_pas1: number; p2_pas1: number; red_pas1: number;
  p1_pas2: number; p2_pas2: number; red_pas2: number;
  nota_alvo: number;
  ciclo_aluno: string;
  lingua_e1: "inglesa" | "francesa" | "espanhola";
  lingua_e2: "inglesa" | "francesa" | "espanhola";
  p1_override?: number | null;
  red_override?: number | null;
  base_projecao?: string;
  // Ticket 12 (Ano-Âncora): sem elas os cinco cenários caem de volta a comparar contra o único
  // `nota_alvo` já resolvido — cada Ano-Âncora precisa da Nota de Corte do curso no *seu* próprio
  // triênio (2025 → 2023-2025), e só o backend sabe montar essa combinação.
  curso_alvo?: string;
  cota?: string;
  semestre?: string;
}) {
  const res = await fetch(`${API_URL}/api/predict/strategy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

