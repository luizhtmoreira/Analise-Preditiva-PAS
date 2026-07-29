import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_intelligence.argument_calculator import calculate_argument_part
from pas_intelligence.pas_constants import OFFICIAL_STATS
from pas_intelligence.training_dataset import (
    REQUIRED_SOURCE_COLUMNS,
    anos_do_trienio,
    build_training_dataset,
)

# Dados 100% sintéticos — nenhuma inscrição ou nome real.


def _argumento_final_esperado(
    trienio: str,
    linguas: tuple[str, str, str],
    notas: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> float:
    """Recompõe o Argumento Final do jeito que o Edital faz, para dar às fixtures um valor
    consistente sem precisar calcular à mão — mesma fórmula que `build_training_dataset` usa
    para validar, então serve para construir dado de teste válido, não para testar a fórmula."""
    anos = anos_do_trienio(trienio)
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


def _linha_base(inscricao: str, trienio: str = "2016/2018", **overrides) -> dict:
    ano_e1, ano_e2, ano_e3 = anos_do_trienio(trienio)
    stats_e1 = OFFICIAL_STATS[(ano_e1, 1)]
    stats_e2 = OFFICIAL_STATS[(ano_e2, 2)]
    stats_e3 = OFFICIAL_STATS[(ano_e3, 3)]

    linha = {
        "campus": "DARCY RIBEIRO",
        "curso": "ADMINISTRAÇÃO (BACHARELADO)",
        "turno": "DIURNO",
        "inscricao": inscricao,
        "eb_p1_e1": stats_e1.parte_1["inglesa"].media,
        "eb_p2_e1": stats_e1.m_p2,
        "red_e1": stats_e1.m_red,
        "eb_p1_e2": stats_e2.parte_1["inglesa"].media,
        "eb_p2_e2": stats_e2.m_p2,
        "red_e2": stats_e2.m_red,
        "eb_p1_e3": stats_e3.parte_1["inglesa"].media,
        "eb_p2_e3": stats_e3.m_p2,
        "red_e3": stats_e3.m_red,
        "lingua_e1": "inglesa",
        "lingua_e2": "inglesa",
        "lingua_e3": "inglesa",
        "lingua_ambigua": False,
        "trienio": trienio,
        "checksum_fecha": True,
    }
    linha.update(overrides)

    if "argumento_final" not in overrides:
        linha["argumento_final"] = _argumento_final_esperado(
            trienio,
            (linha["lingua_e1"], linha["lingua_e2"], linha["lingua_e3"]),
            (
                (linha["eb_p1_e1"], linha["eb_p2_e1"], linha["red_e1"]),
                (linha["eb_p1_e2"], linha["eb_p2_e2"], linha["red_e2"]),
                (linha["eb_p1_e3"], linha["eb_p2_e3"], linha["red_e3"]),
            ),
        )
    return linha


def _df(linhas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(linhas)
    assert set(REQUIRED_SOURCE_COLUMNS) <= set(df.columns)
    return df[REQUIRED_SOURCE_COLUMNS]


def test_exclui_linha_com_checksum_falhando():
    source = _df(
        [
            _linha_base("10000001"),
            _linha_base("10000002", checksum_fecha=False),
        ]
    )
    resultado = build_training_dataset(source)
    assert len(resultado) == 1
    assert resultado.iloc[0]["id_pseudonimo"] != ""


def test_aluno_na_media_tem_argumentos_proximos_de_zero():
    source = _df([_linha_base("10000001")])
    resultado = build_training_dataset(source)
    linha = resultado.iloc[0]
    assert linha["a1"] == pytest.approx(0.0, abs=1e-6)
    assert linha["a2"] == pytest.approx(0.0, abs=1e-6)
    assert linha["a3"] == pytest.approx(0.0, abs=1e-6)


def test_etapa_1_ausente_detectada_por_tres_zeros():
    com_etapa1 = _linha_base("10000001")
    sem_etapa1 = _linha_base(
        "10000002", eb_p1_e1=0.0, eb_p2_e1=0.0, red_e1=0.0
    )
    resultado = build_training_dataset(_df([com_etapa1, sem_etapa1]))
    linha_ausente = resultado[resultado["eb_p1_e1"] == 0].iloc[0]
    linha_presente = resultado[resultado["eb_p1_e1"] != 0].iloc[0]
    assert bool(linha_ausente["etapa_1_ausente"]) is True
    assert bool(linha_presente["etapa_1_ausente"]) is False


def test_nome_e_inscricao_nao_saem_no_dataset():
    source = _df([_linha_base("10000001")])
    resultado = build_training_dataset(source)
    assert "nome" not in resultado.columns
    assert "inscricao" not in resultado.columns
    assert "id_pseudonimo" in resultado.columns


def test_id_pseudonimo_e_deterministico_e_nao_e_a_inscricao_em_texto_puro():
    source = _df([_linha_base("10000001")])
    resultado_1 = build_training_dataset(source)
    resultado_2 = build_training_dataset(source)
    assert resultado_1.iloc[0]["id_pseudonimo"] == resultado_2.iloc[0]["id_pseudonimo"]
    assert resultado_1.iloc[0]["id_pseudonimo"] != "10000001"


def test_inscricao_repetida_entre_trienios_e_marcada():
    source = _df(
        [
            _linha_base("10000001", trienio="2016/2018"),
            _linha_base("10000001", trienio="2017/2019"),
            _linha_base("10000002", trienio="2016/2018"),
        ]
    )
    resultado = build_training_dataset(source)
    repetidos = resultado[resultado["inscricao_repetida_entre_trienios"]]
    unicos = resultado[~resultado["inscricao_repetida_entre_trienios"]]
    assert len(repetidos) == 2
    assert len(unicos) == 1


def test_argumento_final_inconsistente_com_a_lingua_gravada_falha_o_build():
    # notas na média da língua inglesa, mas gravadas como espanhola: o Argumento Final impresso
    # (calculado supondo inglesa) diverge do recomposto (que usa a língua gravada) além da
    # tolerância — exatamente o cenário de língua errada que o checksum não pega sozinho.
    linha = _linha_base("10000001", lingua_e1="espanhola", argumento_final=0.0)
    with pytest.raises(ValueError, match="fora da tolerância"):
        build_training_dataset(_df([linha]))


def test_stats_faltando_para_ano_lingua_leva_a_erro_claro():
    # 2024/2026 pede (2024,1), que ainda não existe em OFFICIAL_STATS (só sai quando o Edital
    # daquele triênio for publicado) — monta a linha à mão para não depender do lookup que falta.
    linha = {
        "campus": "DARCY RIBEIRO",
        "curso": "ADMINISTRAÇÃO (BACHARELADO)",
        "turno": "DIURNO",
        "inscricao": "10000001",
        "eb_p1_e1": 5.0,
        "eb_p2_e1": 25.0,
        "red_e1": 6.0,
        "eb_p1_e2": 5.0,
        "eb_p2_e2": 25.0,
        "red_e2": 6.0,
        "eb_p1_e3": 5.0,
        "eb_p2_e3": 25.0,
        "red_e3": 6.0,
        "argumento_final": 0.0,
        "lingua_e1": "inglesa",
        "lingua_e2": "inglesa",
        "lingua_e3": "inglesa",
        "lingua_ambigua": False,
        "trienio": "2024/2026",
        "checksum_fecha": True,
    }
    with pytest.raises(ValueError, match="OFFICIAL_STATS não cobre"):
        build_training_dataset(_df([linha]))
