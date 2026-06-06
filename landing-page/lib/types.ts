export type RiscoStatus = "green" | "yellow" | "red";

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
