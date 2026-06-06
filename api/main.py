from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import gestao
from api.services.gestao_service import load_resources


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="Vetor PAS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gestao.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
