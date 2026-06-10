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
