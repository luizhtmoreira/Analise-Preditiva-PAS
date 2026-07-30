import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analytics, gestao, predict
from api.services.gestao_service import load_resources

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:3001"
DEFAULT_CORS_ORIGIN_REGEX = r"https://.*\.vercel\.app"


def cors_origins_do_ambiente() -> list[str]:
    """Lê `CORS_ALLOW_ORIGINS` (CSV) do ambiente; cai para os defaults de DEV se ausente."""
    bruto = os.environ.get("CORS_ALLOW_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origem.strip() for origem in bruto.split(",") if origem.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="Vetor PAS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_do_ambiente(),
    allow_origin_regex=os.environ.get("CORS_ALLOW_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gestao.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
