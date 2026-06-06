from fastapi import APIRouter
from api.schemas.gestao import GestaoRequest, GestaoResponse
from api.services.gestao_service import analyze_students

router = APIRouter(prefix="/gestao", tags=["gestao"])


@router.post("/analyze", response_model=GestaoResponse)
def analyze(req: GestaoRequest) -> GestaoResponse:
    return analyze_students(req.students, req.trienio, req.cenario)
