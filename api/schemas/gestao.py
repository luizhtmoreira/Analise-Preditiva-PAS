from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    nome: str
    turma: str = ""
    unidade: str = ""
    curso_alvo: str = ""
    cota: str = "Sistema Universal"
    ano_trienio: str = "2024-2026"
    p1_pas1: float
    p2_pas1: float
    red_pas1: float = 6.0
    p1_pas2: float
    p2_pas2: float
    red_pas2: float = 6.0


class StudentResult(BaseModel):
    nome: str
    turma: str
    unidade: str
    curso_alvo: str
    sistema_concorrencia: str
    arg_previsto: float
    gap: float
    chance_display: str
    historico_pct: float
    sugestao: str
    status: str          # "green" | "yellow" | "red"
    status_label: str    # "Baixo Risco" | "Oportunidade (2º Sem)" | "Alto Risco"
    prob_1_sem: float
    prob_2_sem: float


class GestaoKpis(BaseModel):
    total: int
    n_red: int
    n_yellow: int
    n_green: int


class GestaoRequest(BaseModel):
    students: list[StudentInput]
    trienio: str = "2024-2026"
    cenario: str = "padrao"   # "padrao" | "tendencia"


class GestaoResponse(BaseModel):
    results: list[StudentResult]
    kpis: GestaoKpis
    trienio_ref: str
    modelo_disponivel: bool
