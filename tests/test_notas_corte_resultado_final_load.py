"""Tradução do schema novo da frente de extração no único ponto de carga (ticket 09).

`notas_corte.csv` e `resultado_final.csv` chegam com nomes de coluna minúsculos, PII
(`inscricao`/`nome`) e uma coluna `checksum_fecha` que aponta a mesma família de corrupção de
escala do ticket 14 (ex. MEDICINA/Darcy Ribeiro/2020-2022 saindo com corte=199.162,872). Estes
testes provam que `load_resources` traduz para o schema que `_build_cutoff_maps` e o Reality
Check já esperavam, sem espalhar a adaptação pelos serviços, e que a PII nunca chega à memória.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # type: ignore
import pytest  # type: ignore

from api.services import gestao_service, analytics_service


NOTAS_CORTE_CSV = """trienio,semestre,campus,curso,turno,sistema,sistema_nome,chamada,nota_corte,inscricao,nome,checksum_fecha
2020/2022,2,DARCY RIBEIRO,MEDICINA (BACHARELADO),DIURNO,1,Universal,2,199162.872,20168784,Daniel Mota Cardoso,False
2016/2018,1,DARCY RIBEIRO,ADMINISTRAÇÃO (BACHARELADO),DIURNO,1,Universal,3,68.787,16124894,Pedro de Vilhena Moraes Silva,True
2018/2020,2,DARCY RIBEIRO,ENFERMAGEM (BACHARELADO),,1,Universal,1,42.5,20100001,Fulano de Tal,True
2018/2020,desconhecido,DARCY RIBEIRO,ENFERMAGEM (BACHARELADO),,1,Universal,1,10.0,20100002,Beltrano da Silva,True
2022/2024,1,UnB CEILÂNDIA (FCE),FARMÁCIA (BACHARELADO),,1,Universal,1,15.4,20220001,Ciclana Souza,True
"""

RESULTADO_FINAL_CSV = """inscricao,nome,trienio,eb_p1_e1,eb_p2_e1,eb_p1_e2,eb_p2_e2,eb_p1_e3,eb_p2_e3,argumento_final,checksum_fecha
20168784,Daniel Mota Cardoso,2020/2022,3.0,25.0,3.5,28.0,4.0,30.0,39617.0,False
16124894,Pedro de Vilhena Moraes Silva,2016/2018,3.0,25.0,3.5,28.0,4.0,30.0,55.5,True
16126633,Rayza Leitao da Cunha,2016/2018,2.0,20.0,2.5,22.0,0.0,0.0,0.0,True
"""


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "notas_corte.csv").write_text(NOTAS_CORTE_CSV, encoding="utf-8")
    (d / "resultado_final.csv").write_text(RESULTADO_FINAL_CSV, encoding="utf-8")
    return tmp_path


@pytest.fixture
def carregado(monkeypatch, data_dir: Path):
    monkeypatch.setattr(gestao_service, "_ROOT", data_dir)
    monkeypatch.setattr(gestao_service, "_pacote", None)  # não testamos o pacote aqui
    gestao_service.load_resources()
    yield
    gestao_service._df_corte = None
    gestao_service._df_cohort = None


# ─── notas_corte.csv → _df_corte ────────────────────────────────────────────────────────────


def test_checksum_reprovado_e_descartado(carregado):
    """A linha MEDICINA/Darcy/2020-2022 (corte=199.162,872) tem checksum_fecha=False — some."""
    df = gestao_service._df_corte
    assert not (df["Min"] == 199162.872).any()


def test_semestre_desconhecido_e_descartado(carregado):
    """Sem saber 1º ou 2º semestre a linha não cabe em nenhum dos dois mapas de corte."""
    df = gestao_service._df_corte
    assert df["Min"].isin([10.0]).sum() == 0


def test_pii_nunca_entra_em_memoria(carregado):
    df = gestao_service._df_corte
    assert "inscricao" not in df.columns
    assert "nome" not in df.columns


def test_schema_traduzido_para_o_formato_canonico(carregado):
    df = gestao_service._df_corte
    linha = df[df["Min"] == 68.787].iloc[0]
    assert linha["Trienio"] == "2016-2018"
    assert linha["Semestre"] == "1°"
    assert linha["Sistema_Nome"] == "Sistema Universal"
    assert linha["Curso_Limpo"] == "ADMINISTRAÇÃO (BACHARELADO)"


def test_turno_ausente_vira_integral(carregado):
    """Enfermagem é período integral no Edital — sem essa tradução a chave do curso levava um
    "nan" literal para a tela."""
    df = gestao_service._df_corte
    linha = df[df["Min"] == 42.5].iloc[0]
    assert linha["Turno"] == "Integral"


def test_campus_com_sigla_e_normalizado(carregado):
    """"UnB CEILÂNDIA (FCE)" tem parênteses aninhados com os da chave de curso
    (`f"{curso} - {turno} ({campus})"`) — sem normalizar, `_parse_course_key` (que faz
    `rfind("(")`) devolvia campus e turno embaralhados."""
    from api.services.predict_service import _parse_course_key

    df = gestao_service._df_corte
    linha = df[df["Min"] == 15.4].iloc[0]
    assert linha["Campus"] == "CEILÂNDIA"

    chave = f"{linha['Curso_Limpo']} - {linha['Turno']} ({linha['Campus']})"
    parsed = _parse_course_key(chave)
    assert parsed["campus"] == "CEILÂNDIA"
    assert parsed["turno"] == "Integral"
    assert parsed["curso"] == "FARMÁCIA (BACHARELADO)"


# ─── resultado_final.csv → _df_cohort (Reality Check) ──────────────────────────────────────


def test_cohort_filtra_por_checksum_e_traduz_colunas(carregado):
    df = gestao_service._df_cohort
    assert len(df) == 1  # só a linha do Pedro sobrevive: Daniel reprova checksum, Rayza tem Arg=0
    linha = df.iloc[0]
    assert linha["EB_PAS1"] == pytest.approx(28.0)
    assert linha["EB_PAS2"] == pytest.approx(31.5)
    assert linha["ARG_FINAL_REAL"] == pytest.approx(55.5)


# ─── resultado_final.csv → analytics_service._load_population ─────────────────────────────


def test_populacao_filtra_por_checksum_e_traduz_trienio(monkeypatch, data_dir: Path):
    monkeypatch.setattr(analytics_service, "_ROOT", data_dir)
    monkeypatch.setattr(analytics_service, "_df_pop", None)

    df = analytics_service._load_population()

    assert len(df) == 2  # Daniel reprova checksum; Pedro e Rayza sobrevivem (Arg=0 é válido aqui)
    assert set(df["Ano_Trienio"]) == {"2016-2018"}
    assert "Inscricao" in df.columns
    assert "nome" not in df.columns
