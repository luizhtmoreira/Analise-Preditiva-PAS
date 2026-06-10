from fastapi import APIRouter, Query
from api.schemas.predict import PredictInput, PredictResponse
from api.services.predict_service import predict_student, get_courses

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictInput) -> PredictResponse:
    return predict_student(body)


@router.get("/courses", response_model=list[str])
def courses(
    cota: str = Query(default="Sistema Universal"),
    trienio: str = Query(default="2024-2026"),
) -> list[str]:
    return get_courses(cota, trienio)
