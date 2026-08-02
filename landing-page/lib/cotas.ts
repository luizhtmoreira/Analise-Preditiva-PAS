// Os 10 Sistemas de Concorrência do Edital do PAS, na mesma ordem em que ele publica as
// classificações (ver `MAPA_SISTEMAS` em `src/pas_extraction/constants.py`, a fonte que gera
// esses rótulos nos CSVs de nota de corte). O Edital não usa nomenclatura "L1/L2/L9/L10" —
// isso era um empréstimo solto do SiSU que nem batia com a própria ordem do Edital (ticket 16).
export const COTAS = [
  "Sistema Universal",
  "Cota para Negros",
  "EP / Baixa Renda / PPI",
  "EP / Baixa Renda / PPI / PcD",
  "EP / Baixa Renda / Não-PPI",
  "EP / Baixa Renda / Não-PPI / PcD",
  "EP / Alta Renda / PPI",
  "EP / Alta Renda / PPI / PcD",
  "EP / Alta Renda / Não-PPI",
  "EP / Alta Renda / Não-PPI / PcD",
] as const;

export type Cota = (typeof COTAS)[number];

export function ehCotaConhecida(valor: unknown): valor is Cota {
  return typeof valor === "string" && (COTAS as readonly string[]).includes(valor);
}
