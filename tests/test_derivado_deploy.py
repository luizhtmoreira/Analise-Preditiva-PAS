"""O Derivado de Deploy é o que sai do disco para dentro da imagem hospedada (ADR-0014, ticket
08a): reduzido às colunas que a API de fato lê, sem `nome`. Estes testes provam a fonte única —
`build_derivado` produz exatamente o que `COLUNAS_DERIVADO` promete, sobre o arquivo escrito, não
sobre a intenção do script.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_intelligence.derivado_deploy import (
    COLUNAS_DERIVADO,
    COLUNAS_NOTAS_CORTE,
    COLUNAS_RESULTADO_FINAL,
    build_derivado,
)

NOTAS_CORTE_CSV = """trienio,semestre,campus,curso,turno,sistema,sistema_nome,chamada,nota_corte,inscricao,nome,checksum_fecha
2020/2022,2,DARCY RIBEIRO,MEDICINA (BACHARELADO),DIURNO,1,Universal,2,199162.872,20168784,Daniel Mota Cardoso,False
"""

RESULTADO_FINAL_CSV = """inscricao,nome,trienio,eb_p1_e1,eb_p2_e1,eb_p1_e2,eb_p2_e2,eb_p1_e3,eb_p2_e3,argumento_final,checksum_fecha
20168784,Daniel Mota Cardoso,2020/2022,3.0,25.0,3.5,28.0,4.0,30.0,39617.0,False
16124894,Pedro de Vilhena Moraes Silva,2016/2018,3.0,25.0,3.5,28.0,4.0,30.0,55.5,True
"""


@pytest.fixture
def origem(tmp_path: Path) -> Path:
    d = tmp_path / "origem"
    d.mkdir()
    (d / "notas_corte.csv").write_text(NOTAS_CORTE_CSV, encoding="utf-8")
    (d / "resultado_final.csv").write_text(RESULTADO_FINAL_CSV, encoding="utf-8")
    return d


def test_nome_nao_sobrevive_em_nenhum_dos_dois_csvs(tmp_path: Path, origem: Path):
    destino = tmp_path / "derivado"
    build_derivado(origem, destino)

    for arquivo in COLUNAS_DERIVADO:
        df = pd.read_csv(destino / arquivo)
        assert "nome" not in df.columns


def test_colunas_do_derivado_batem_exatamente_com_a_fonte_unica(tmp_path: Path, origem: Path):
    destino = tmp_path / "derivado"
    build_derivado(origem, destino)

    df_corte = pd.read_csv(destino / "notas_corte.csv")
    assert set(df_corte.columns) == set(COLUNAS_NOTAS_CORTE)

    df_resultado = pd.read_csv(destino / "resultado_final.csv")
    assert set(df_resultado.columns) == set(COLUNAS_RESULTADO_FINAL)


def test_nenhuma_linha_e_descartada_no_derivado(tmp_path: Path, origem: Path):
    """Cortar coluna e cortar linha são decisões independentes — o filtro `checksum_fecha` é
    de quem lê, não do Derivado."""
    destino = tmp_path / "derivado"
    escritos = build_derivado(origem, destino)

    assert escritos["resultado_final.csv"] == (2, len(COLUNAS_RESULTADO_FINAL))
    assert escritos["notas_corte.csv"] == (1, len(COLUNAS_NOTAS_CORTE))


def test_arquivo_ausente_na_origem_e_ignorado(tmp_path: Path):
    origem = tmp_path / "origem_parcial"
    origem.mkdir()
    (origem / "notas_corte.csv").write_text(NOTAS_CORTE_CSV, encoding="utf-8")

    destino = tmp_path / "derivado"
    escritos = build_derivado(origem, destino)

    assert "notas_corte.csv" in escritos
    assert "resultado_final.csv" not in escritos
    assert not (destino / "resultado_final.csv").exists()
