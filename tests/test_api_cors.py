"""CORS deixa de ser hardcoded: as origens permitidas vêm de `CORS_ALLOW_ORIGINS` (ticket 03).

O bug latente que motivou o ticket: `"https://*.vercel.app"` era passado como item literal de
`allow_origins`, mas o `CORSMiddleware` faz comparação exata, não glob — o wildcard nunca
casava com nada. Preview deployments da Vercel viravam CORS silencioso.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore


@pytest.fixture
def api_main(monkeypatch):
    """Recarrega `api.main` a cada teste para que a leitura do ambiente seja refeita."""
    monkeypatch.setattr("api.services.gestao_service.load_resources", lambda: None)
    import api.main as main

    return importlib.reload(main)


def test_usa_defaults_de_dev_quando_variavel_ausente(monkeypatch, api_main):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    origens = api_main.cors_origins_do_ambiente()
    assert origens == ["http://localhost:3000", "http://localhost:3001"]


def test_le_lista_csv_da_variavel_de_ambiente(monkeypatch, api_main):
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS", "https://app.vetorpas.com.br, https://staging.vetorpas.com.br"
    )
    origens = api_main.cors_origins_do_ambiente()
    assert origens == ["https://app.vetorpas.com.br", "https://staging.vetorpas.com.br"]


def test_regex_de_preview_da_vercel_casa_com_subdominio(monkeypatch, api_main):
    monkeypatch.delenv("CORS_ALLOW_ORIGIN_REGEX", raising=False)
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    importlib.reload(api_main)

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as client:
        resposta = client.get(
            "/health",
            headers={"Origin": "https://vetor-pas-git-preview-abc123.vercel.app"},
        )

    assert resposta.headers["access-control-allow-origin"] == (
        "https://vetor-pas-git-preview-abc123.vercel.app"
    )


def test_origem_de_producao_e_aceita(monkeypatch, api_main):
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS", "https://vetorpas.com.br,https://www.vetorpas.com.br"
    )
    monkeypatch.delenv("CORS_ALLOW_ORIGIN_REGEX", raising=False)
    importlib.reload(api_main)

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as client:
        resposta_apex = client.get(
            "/health", headers={"Origin": "https://vetorpas.com.br"}
        )
        resposta_www = client.get(
            "/health", headers={"Origin": "https://www.vetorpas.com.br"}
        )

    assert resposta_apex.headers["access-control-allow-origin"] == "https://vetorpas.com.br"
    assert resposta_www.headers["access-control-allow-origin"] == "https://www.vetorpas.com.br"


def test_origem_desconhecida_e_recusada(monkeypatch, api_main):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://vetorpas.com.br")
    monkeypatch.delenv("CORS_ALLOW_ORIGIN_REGEX", raising=False)
    importlib.reload(api_main)

    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as client:
        resposta = client.get(
            "/health", headers={"Origin": "https://site-malicioso.com"}
        )

    assert "access-control-allow-origin" not in resposta.headers
