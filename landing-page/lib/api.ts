const API_URL = process.env.API_URL ?? "http://localhost:8000";

export async function fetchGestao(students: unknown[], trienio = "2024-2026", cenario = "padrao") {
  const res = await fetch(`${API_URL}/api/gestao/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ students, trienio, cenario }),
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return res.json();
}
