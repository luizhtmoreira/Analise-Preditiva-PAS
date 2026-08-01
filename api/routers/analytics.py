from fastapi import APIRouter, HTTPException

from api.schemas.analytics import (
    TemporalResponse,
    CorteEvolucao,
    EscolaRequest,
    EscolaResponse,
    ComparacaoRequest,
    ComparacaoResponse,
)
from api.services import analytics_service

router = APIRouter(tags=["analytics"])


@router.get("/temporal", response_model=TemporalResponse)
def temporal() -> TemporalResponse:
    return analytics_service.get_temporal()


@router.get("/temporal/corte", response_model=list[CorteEvolucao])
def corte_evolucao(curso: str, cota: str = "Sistema Universal") -> list[CorteEvolucao]:
    return analytics_service.get_corte_evolucao(curso, cota)


@router.post("/escola/analyze", response_model=EscolaResponse)
def escola(req: EscolaRequest) -> EscolaResponse:
    try:
        return analytics_service.analyze_escola(req.inscricoes, req.trienio)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/comparacao", response_model=ComparacaoResponse)
def comparacao(req: ComparacaoRequest) -> ComparacaoResponse:
    try:
        return analytics_service.compare(req.group_a, req.group_b, req.name_a, req.name_b, req.metric)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
