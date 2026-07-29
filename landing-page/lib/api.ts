// Server-side: process.env.API_URL. Client-side: NEXT_PUBLIC_API_URL.
const API_URL =
  (typeof window === "undefined"
    ? process.env.API_URL
    : process.env.NEXT_PUBLIC_API_URL) ?? "http://localhost:8000";

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
  lingua?: string; cota?: string; trienio?: string; curso_alvo?: string;
}) {
  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
