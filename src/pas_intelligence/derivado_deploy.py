"""Fonte única das colunas do Derivado de Deploy — os CSVs reduzidos, sem PII, que a API
hospedada lê (ADR-0014, ticket 08a).

Lida tanto por quem publica (`deploy/publicar_pacote.py`) quanto pelos serviços que consomem
os CSVs em runtime (`api/services/gestao_service.py`, `api/services/analytics_service.py`).
Antes deste módulo a lista existia duplicada nesses dois serviços; um terceiro lugar (o
publicador) faria uma coluna nova lida em produção quebrar com `KeyError` sem que nada ligasse o
erro ao script de publicação — o mesmo defeito que o `ponteiro.json` já corrigiu uma vez para os
nomes de arquivo de cada artefato.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore

COLUNAS_RESULTADO_FINAL = [
    "inscricao", "trienio", "argumento_final",
    "eb_p1_e1", "eb_p2_e1", "eb_p1_e2", "eb_p2_e2", "eb_p1_e3", "eb_p2_e3",
    "checksum_fecha",
]

COLUNAS_NOTAS_CORTE = [
    "trienio", "semestre", "campus", "curso", "turno",
    "sistema_nome", "chamada", "nota_corte", "checksum_fecha",
]

# Nome do arquivo → colunas que sobrevivem no Derivado. `resultado_final.csv` e
# `notas_corte.csv` são os dois únicos CSVs que a API hospedada lê de `data/` — nenhum dos dois
# tem `nome` nesta lista.
COLUNAS_DERIVADO: dict[str, list[str]] = {
    "resultado_final.csv": COLUNAS_RESULTADO_FINAL,
    "notas_corte.csv": COLUNAS_NOTAS_CORTE,
}


def build_derivado(origem: Path, destino: Path) -> dict[str, tuple[int, int]]:
    """Lê cada CSV de `COLUNAS_DERIVADO` presente em `origem`, grava em `destino` só com as
    colunas listadas, e devolve `{arquivo: (linhas, colunas)}` do que foi escrito.

    Nenhuma linha é descartada aqui: o filtro `checksum_fecha == True` é responsabilidade de
    quem lê (`gestao_service`, `analytics_service`), não do Derivado — cortar coluna e cortar
    linha são decisões independentes.
    """
    destino.mkdir(parents=True, exist_ok=True)
    escritos: dict[str, tuple[int, int]] = {}
    for nome_arquivo, colunas in COLUNAS_DERIVADO.items():
        caminho_origem = origem / nome_arquivo
        if not caminho_origem.exists():
            continue
        df = pd.read_csv(caminho_origem, usecols=lambda c: c in colunas)
        df.to_csv(destino / nome_arquivo, index=False)
        escritos[nome_arquivo] = df.shape
    return escritos
