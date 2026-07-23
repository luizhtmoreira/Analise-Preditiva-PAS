from typing import Optional
from pydantic import BaseModel


class PredictInput(BaseModel):
    p1_pas1: float
    p2_pas1: float
    red_pas1: float
    p1_pas2: float
    p2_pas2: float
    red_pas2: float
    cota: str = "Sistema Universal"
    trienio: str = "2024-2026"
    curso_alvo: Optional[str] = None
    is_logged_in: bool = False
    semestre: Optional[str] = "1°"


class CourseResult(BaseModel):
    curso: str
    turno: str
    campus: str
    nota_corte: float
    prob: float
    semestre: str


class PredictResponse(BaseModel):
    eb_pas3_previsto: float
    arg_previsto: float
    arg_min: float
    arg_max: float
    curso_alvo_result: Optional[CourseResult]
    top_cursos: list[CourseResult]
    trienio_ref: str
    modelo_disponivel: bool

class ChamadaCorte(BaseModel):
    chamada: str
    campus: str
    turno: str
    nota_corte: float


class StrategyInput(BaseModel):
    p1_pas1: float
    p2_pas1: float
    red_pas1: float
    p1_pas2: float
    p2_pas2: float
    red_pas2: float
    nota_alvo: float
    ciclo_aluno: str
    p1_override: Optional[float] = None
    red_override: Optional[float] = None
    base_projecao: str = "Utilizar Projeção Tendência"


class StrategyResponse(BaseModel):
    p1_estimado: float
    p2_necessario: float
    red_estimada: float
    total_pas3: float
    arg_pas3_necessario: float
    status: str
    mensagem: str
    prob_hist: float
    amostra: int
    p1_ia: float
    red_ia: float

