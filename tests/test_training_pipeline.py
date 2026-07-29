"""Testes do pipeline de treino (ticket 12). Dados 100% sintéticos, nenhum PDF nem CSV real —
gera o próprio `resultado_final.csv` sintético em `tmp_path` e roda o pipeline sobre ele.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_intelligence.argument_calculator import calculate_argument_part
from pas_intelligence.pas_constants import OFFICIAL_STATS
from pas_intelligence.training_dataset import REQUIRED_SOURCE_COLUMNS, _anos_do_trienio
from pas_intelligence.training_pipeline import (
    NOME_ARQUIVO_MANIFESTO,
    NOME_ARQUIVO_MODELO,
    PortaoFechadoError,
    treinar,
)
from pas_intelligence.validation import TRIENIO_LACRADO

TRIENIOS = (
    "2016/2018",
    "2017/2019",
    "2018/2020",
    "2019/2021",
    "2020/2022",
    "2021/2023",
    "2022/2024",
    "2023/2025",
)

# Gate permissivo: só quer exercitar o caminho inteiro do pipeline sobre dado sintético e
# pequeno, não reproduzir o Portão 1 real — esse é medido sobre o `resultado_final.csv`
# verdadeiro, fora do escopo de um teste unitário.
PORTAO_PERMISSIVO = dict(geral=1e6, majoritaria=1e6, minoritaria=1e6, vies_abs=1e6)

# Impossível de bater por construção — usado para testar o caminho de recusa.
PORTAO_IMPOSSIVEL = dict(geral=1e-9, majoritaria=1e-9, minoritaria=1e-9, vies_abs=1e-9)


def _argumento_final(trienio: str, linguas, notas) -> float:
    """Recompõe o Argumento Final do jeito que o Edital faz (mesma fórmula de
    `build_training_dataset`), só para as fixtures nascerem com um valor consistente."""
    anos = _anos_do_trienio(trienio)
    argumentos_etapa = []
    for etapa, ano, lingua, (p1, p2, red) in zip((1, 2, 3), anos, linguas, notas):
        stats = OFFICIAL_STATS[(ano, etapa)]
        valor_lingua = stats.parte_1[lingua]
        arg = (
            calculate_argument_part(p1, valor_lingua.media, valor_lingua.desvio_padrao, 0.72)
            + calculate_argument_part(p2, stats.m_p2, stats.dp_p2, 8.28)
            + calculate_argument_part(red, stats.m_red, stats.dp_red, 1.00)
        )
        argumentos_etapa.append(round(arg, 3))
    a1, a2, a3 = argumentos_etapa
    return round(1 * a1 + 2 * a2 + 3 * a3, 3)


def _linha(inscricao: str, trienio: str, rng: np.random.Generator, *, etapa_1_ausente: bool) -> dict:
    ano_e1, ano_e2, ano_e3 = _anos_do_trienio(trienio)
    stats_e1 = OFFICIAL_STATS[(ano_e1, 1)]
    stats_e2 = OFFICIAL_STATS[(ano_e2, 2)]
    stats_e3 = OFFICIAL_STATS[(ano_e3, 3)]
    lingua = "inglesa"

    if etapa_1_ausente:
        eb_p1_e1, eb_p2_e1, red_e1 = 0.0, 0.0, 0.0
    else:
        valor_lingua_e1 = stats_e1.parte_1[lingua]
        eb_p1_e1 = round(float(rng.normal(valor_lingua_e1.media, valor_lingua_e1.desvio_padrao)), 3)
        eb_p2_e1 = round(float(rng.normal(stats_e1.m_p2, stats_e1.dp_p2)), 3)
        red_e1 = round(float(rng.normal(stats_e1.m_red, stats_e1.dp_red)), 3)

    valor_lingua_e2 = stats_e2.parte_1[lingua]
    eb_p1_e2 = round(float(rng.normal(valor_lingua_e2.media, valor_lingua_e2.desvio_padrao)), 3)
    eb_p2_e2 = round(float(rng.normal(stats_e2.m_p2, stats_e2.dp_p2)), 3)
    red_e2 = round(float(rng.normal(stats_e2.m_red, stats_e2.dp_red)), 3)

    valor_lingua_e3 = stats_e3.parte_1[lingua]
    eb_p1_e3 = round(float(rng.normal(valor_lingua_e3.media, valor_lingua_e3.desvio_padrao)), 3)
    eb_p2_e3 = round(float(rng.normal(stats_e3.m_p2, stats_e3.dp_p2)), 3)
    red_e3 = round(float(rng.normal(stats_e3.m_red, stats_e3.dp_red)), 3)

    linha = {
        "campus": "DARCY RIBEIRO",
        "curso": "ADMINISTRAÇÃO (BACHARELADO)",
        "turno": "DIURNO",
        "inscricao": inscricao,
        "eb_p1_e1": eb_p1_e1,
        "eb_p2_e1": eb_p2_e1,
        "red_e1": red_e1,
        "eb_p1_e2": eb_p1_e2,
        "eb_p2_e2": eb_p2_e2,
        "red_e2": red_e2,
        "eb_p1_e3": eb_p1_e3,
        "eb_p2_e3": eb_p2_e3,
        "red_e3": red_e3,
        "lingua_e1": lingua,
        "lingua_e2": lingua,
        "lingua_e3": lingua,
        "lingua_ambigua": False,
        "trienio": trienio,
        "checksum_fecha": True,
    }
    linha["argumento_final"] = _argumento_final(
        trienio,
        (lingua, lingua, lingua),
        (
            (eb_p1_e1, eb_p2_e1, red_e1),
            (eb_p1_e2, eb_p2_e2, red_e2),
            (eb_p1_e3, eb_p2_e3, red_e3),
        ),
    )
    return linha


def _csv_sintetico(caminho: Path, *, n_por_trienio: int = 30, semente: int = 7) -> Path:
    rng = np.random.default_rng(semente)
    linhas = []
    contador = 0
    for trienio in TRIENIOS:
        for i in range(n_por_trienio):
            contador += 1
            linhas.append(
                _linha(
                    f"{contador:08d}",
                    trienio,
                    rng,
                    etapa_1_ausente=(i % 7 == 0),
                )
            )
    df = pd.DataFrame(linhas)[REQUIRED_SOURCE_COLUMNS]
    df.to_csv(caminho, index=False)
    return caminho


@pytest.fixture
def csv_sintetico(tmp_path) -> Path:
    return _csv_sintetico(tmp_path / "resultado_final.csv")


def test_pipeline_do_csv_ao_pacote(tmp_path, csv_sintetico):
    resultado = treinar(
        csv_sintetico, tmp_path / "pacote", semente=123, portao=PORTAO_PERMISSIVO
    )

    assert resultado.caminho_modelo.exists()
    assert resultado.caminho_manifesto.exists()
    assert resultado.caminho_modelo.name == NOME_ARQUIVO_MODELO
    assert resultado.caminho_manifesto.name == NOME_ARQUIVO_MANIFESTO
    assert resultado.bateu_portao is True

    manifesto = resultado.manifesto
    assert manifesto["dado"]["linhas"] == 30 * len(TRIENIOS)
    assert TRIENIO_LACRADO in manifesto["dado"]["trienios"]  # descrito, não usado no treino
    assert manifesto["modelos"][0]["features"] == list(
        resultado.resultado_validacao.features
    )
    assert manifesto["modelos"][0]["treinado_excluindo_trienio"] == TRIENIO_LACRADO
    assert manifesto["ambiente"]["lightgbm"] != "desconhecida"


def test_manifesto_ganha_bloco_incerteza_com_a_forma_do_relatorio_11(tmp_path, csv_sintetico):
    resultado = treinar(
        csv_sintetico, tmp_path / "pacote", semente=123, portao=PORTAO_PERMISSIVO
    )

    incerteza = resultado.manifesto["incerteza"]
    validacao = resultado.resultado_validacao

    assert incerteza["forma"] == "normal"
    assert incerteza["escala"] == "a3"
    assert incerteza["sigma_por_classe"] == {
        "com_etapa_1": validacao.agrupado_majoritaria.rmse,
        "sem_etapa_1": validacao.agrupado_minoritaria.rmse,
    }
    assert incerteza["sigma_agrupado"] == validacao.agrupado.rmse
    assert set(incerteza["cobertura_verificada"]) == {"0.50", "0.80", "0.90", "0.95"}
    # Cobertura empírica tem que estar perto do nível prometido, não é hardcoded — dado
    # sintético e pequeno, então a folga é generosa (relatório 11 §3.2 mede ~0,4 p.p. no real).
    for nivel_str, cobertura in incerteza["cobertura_verificada"].items():
        nivel = float(nivel_str)
        assert abs(cobertura - nivel) < 0.15


def test_pipeline_nunca_toca_o_trienio_lacrado(tmp_path, csv_sintetico):
    resultado = treinar(
        csv_sintetico, tmp_path / "pacote", semente=123, portao=PORTAO_PERMISSIVO
    )
    previsoes = resultado.resultado_validacao.previsoes
    assert TRIENIO_LACRADO not in set(previsoes["trienio"])


def test_duas_execucoes_com_mesma_entrada_produzem_artefatos_equivalentes(tmp_path, csv_sintetico):
    r1 = treinar(csv_sintetico, tmp_path / "pacote_1", semente=123, portao=PORTAO_PERMISSIVO)
    r2 = treinar(csv_sintetico, tmp_path / "pacote_2", semente=123, portao=PORTAO_PERMISSIVO)

    assert r1.caminho_modelo.read_bytes() == r2.caminho_modelo.read_bytes()

    m1 = dict(r1.manifesto)
    m2 = dict(r2.manifesto)
    m1.pop("criado_em")
    m2.pop("criado_em")
    assert m1 == m2


def test_portao_fechado_recusa_publicar_sem_forcar(tmp_path, csv_sintetico):
    saida = tmp_path / "pacote"
    with pytest.raises(PortaoFechadoError):
        treinar(csv_sintetico, saida, semente=123, portao=PORTAO_IMPOSSIVEL)
    assert not saida.exists()


def test_forcar_publica_com_motivo_gravado_no_manifesto(tmp_path, csv_sintetico):
    resultado = treinar(
        csv_sintetico,
        tmp_path / "pacote",
        semente=123,
        portao=PORTAO_IMPOSSIVEL,
        forcar=True,
        motivo_forca="teste: publicar mesmo sem bater o portão",
    )
    assert resultado.bateu_portao is False
    assert resultado.manifesto["avaliacao"]["forcado"] is True
    assert resultado.manifesto["avaliacao"]["motivo_forca"] == (
        "teste: publicar mesmo sem bater o portão"
    )
    assert resultado.caminho_modelo.exists()


def test_forcar_sem_motivo_e_erro_de_uso(tmp_path, csv_sintetico):
    with pytest.raises(ValueError, match="motivo_forca"):
        treinar(csv_sintetico, tmp_path / "pacote", forcar=True)
