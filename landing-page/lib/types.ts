export type RiscoStatus = "green" | "yellow" | "red";

export interface CourseResult {
  curso: string;
  turno: string;
  campus: string;
  nota_corte: number;
  prob: number;
  semestre: string;
}

export interface PredictResponse {
  eb_pas3_previsto: number;
  arg_previsto: number;
  arg_min: number;
  arg_max: number;
  curso_alvo_result: CourseResult | null;
  top_cursos: CourseResult[];
  trienio_ref: string;
  modelo_disponivel: boolean;
}

export interface StudentResult {
  nome: string;
  turma: string;
  unidade: string;
  curso_alvo: string;
  sistema_concorrencia: string;
  arg_previsto: number;
  gap: number;
  chance_display: string;
  historico_pct: number;
  sugestao: string;
  status: RiscoStatus;
  status_label: string;
  prob_1_sem: number;
  prob_2_sem: number;
}

export interface GestaoKpis {
  total: number;
  n_red: number;
  n_yellow: number;
  n_green: number;
}

export interface GestaoResponse {
  results: StudentResult[];
  kpis: GestaoKpis;
  trienio_ref: string;
  modelo_disponivel: boolean;
}

/* ── Análise Temporal (pública) ───────────────────────────────── */

export interface EtapaStat {
  ano: number;
  etapa: number;
  p1_medio: number;
  p2_medio: number;
  eb_medio: number;
  red_media: number;
}

export interface TemporalResponse {
  etapas: EtapaStat[];
  cursos: string[];
}

export interface CorteEvolucao {
  trienio: string;
  corte_1sem: number | null;
  corte_2sem: number | null;
}

export interface ChamadaCorte {
  chamada: string;
  campus: string;
  turno: string;
  nota_corte: number;
}

/* ── Escola vs. População ─────────────────────────────────────── */

export interface EtapaComparativo {
  etapa: string;
  escola: number;
  geral: number;
}

export interface HistBin {
  x0: number;
  x1: number;
  count: number;
}

export interface EscolaResponse {
  n_enviados: number;
  n_encontrados: number;
  taxa_match: number;
  trienio: string;
  trienios_disponiveis: string[];
  media_escola: number;
  media_geral: number;
  std_escola: number;
  std_geral: number;
  diff: number;
  percentil: number;
  p_value: number;
  significativo: boolean;
  etapas: EtapaComparativo[];
  hist_arg: HistBin[];
}

/* ── Comparação entre grupos (A/B) ────────────────────────────── */

export interface BoxStats {
  min: number;
  q1: number;
  med: number;
  q3: number;
  max: number;
  mean: number;
  n: number;
}

export interface ComparacaoResponse {
  mean_a: number;
  mean_b: number;
  diff: number;
  p_value: number;
  t_statistic: number;
  effect_size: number;
  significativo: boolean;
  box_a: BoxStats;
  box_b: BoxStats;
}

/* ── Linha crua da tabela_mestra (Supabase) ───────────────────── */

export interface StudentRow {
  nome: string;
  turma: string;
  unidade: string;
  curso_alvo: string;
  cota: string;
  trienio: string;
  p1_pas1: number;
  p2_pas1: number;
  red_pas1: number;
  p1_pas2: number;
  p2_pas2: number;
  red_pas2: number;
}
