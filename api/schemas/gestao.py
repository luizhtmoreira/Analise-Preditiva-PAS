from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    nome: str
    turma: str = ""
    unidade: str = ""
    curso_alvo: str = ""
    cota: str = "Sistema Universal"
    ano_trienio: str = "2024-2026"
    # As seis notas são obrigatórias e não têm default. `red_pas1`/`red_pas2` já tiveram
    # default 6.0 — uma Redação mediana inventada onde a nota não foi informada, que inflava
    # o Argumento da Etapa em ~3,3 pontos na direção do otimismo, sem deixar rastro.
    # Nota faltante agora é 422 nomeando o campo, nunca um número plausível. Ver ADR-0008.
    p1_pas1: float
    p2_pas1: float
    red_pas1: float
    p1_pas2: float
    p2_pas2: float
    red_pas2: float


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
