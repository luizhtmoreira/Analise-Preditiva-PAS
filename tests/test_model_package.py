"""Testes de `model_package.py` — o pacote promovido servindo uma previsão (ticket 13).

Dado sintético do começo ao fim: o pacote destes testes é um LightGBM de brinquedo treinado sobre
ruído, escrito em disco no mesmo formato do pipeline. Nenhum artefato de produção e nenhuma linha
de Aluno entram aqui ([[project_parser_privacy]]).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_intelligence.argument_calculator import calculate_argument_etapa
from pas_intelligence.dataset_pas3 import FEATURES_CANONICAS, FEATURES_ETAPA1
from pas_intelligence.model_package import (
    EntradaDePrevisao,
    EstatisticasIndisponiveisError,
    LINGUAS_OFICIAIS,
    NotasDeEtapa,
    PacoteDeModelo,
    PacoteIndisponivelError,
    anos_do_trienio,
)
from pas_intelligence.training_dataset import stats_da_prova
from tests.conftest import SIGMA_COM_ETAPA_1 as SIGMA_COM, SIGMA_SEM_ETAPA_1 as SIGMA_SEM


def entrada(**alteracoes) -> EntradaDePrevisao:
    """Um Aluno sintético do triênio 2023-2025 — o mais recente que `OFFICIAL_STATS` cobre."""
    padrao = dict(
        etapa_1=NotasDeEtapa(p1=4.0, p2=30.0, redacao=7.0),
        etapa_2=NotasDeEtapa(p1=5.0, p2=35.0, redacao=7.5),
        lingua="inglesa",
        trienio="2023-2025",
    )
    return EntradaDePrevisao(**{**padrao, **alteracoes})


# ─── Triênio ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("texto", ["2023-2025", "2023/2025"])
def test_trienio_aceita_os_dois_separadores(texto):
    """A API fala `2024-2026` e o `resultado_final.csv` fala `2024/2026` — o mesmo triênio."""
    assert anos_do_trienio(texto) == (2023, 2024, 2025)


def test_trienio_malformado_e_erro():
    with pytest.raises(ValueError):
        anos_do_trienio("2023")


# ─── Aritmética exata ───────────────────────────────────────────────────────────────────────


def test_a1_e_a2_saem_da_formula_oficial_da_etapa(pacote):
    """`A1` e `A2` são **aritmética exata** (ADR-0009), não previsão: precisam bater com
    `calculate_argument_etapa` no dígito, com a estatística da língua que o Aluno declarou."""
    previsao = pacote.prever(entrada())

    esperado_a1 = calculate_argument_etapa(4.0, 30.0, 7.0, stats_da_prova(2023, 1, "inglesa"))
    esperado_a2 = calculate_argument_etapa(5.0, 35.0, 7.5, stats_da_prova(2024, 2, "inglesa"))
    assert previsao.a1 == pytest.approx(esperado_a1)
    assert previsao.a2 == pytest.approx(esperado_a2)


def test_argumento_final_e_a_soma_ponderada_das_tres_etapas(pacote):
    """`Argumento Final = 1·A1 + 2·A2 + 3·A3` — a única rota, para que não exista um segundo
    número na tela capaz de discordar dela (ticket 04 §3)."""
    previsao = pacote.prever(entrada())
    assert previsao.argumento_final == pytest.approx(
        previsao.a1 + 2 * previsao.a2 + 3 * previsao.a3
    )


def test_largura_em_argumento_final_e_o_triplo_da_largura_em_a3(pacote):
    """`σ(Argumento Final) = 3 × σ(A3)`, exato porque `A1` e `A2` têm variância zero."""
    previsao = pacote.prever(entrada())
    assert previsao.largura_argumento_final == pytest.approx(3 * previsao.largura_a3)


def test_lingua_declarada_muda_a1_e_a2(pacote):
    """O Cebraspe normaliza a Parte 1 **por língua**; agrupar embute viés contra a minoria
    (ticket 04 §5). Se a língua não mudasse nada, o campo do formulário seria decorativo."""
    assert pacote.prever(entrada(lingua="inglesa")).a1 != pytest.approx(
        pacote.prever(entrada(lingua="espanhola")).a1
    )


def test_lingua_fora_das_tres_oficiais_e_recusada(pacote):
    with pytest.raises(ValueError, match="língua"):
        pacote.prever(entrada(lingua="alema"))
    assert set(LINGUAS_OFICIAIS) == {"inglesa", "francesa", "espanhola"}


# ─── Aluno sem Etapa 1 ──────────────────────────────────────────────────────────────────────


def test_etapa_1_zerada_e_detectada_como_classe(pacote):
    """A mesma regra de `training_dataset` (as três notas da Etapa 1 iguais a zero). Sem ela o
    modelo lê *"fez a prova e tirou zero"* e erra a **previsão**, não só a largura (ADR-0012 §8)."""
    previsao = pacote.prever(entrada(etapa_1=NotasDeEtapa(p1=0.0, p2=0.0, redacao=0.0)))
    assert previsao.etapa_1_ausente is True
    assert pacote.prever(entrada()).etapa_1_ausente is False


def test_aluno_sem_etapa_1_recebe_nan_nas_features_derivadas_da_etapa_1(pacote):
    """Decisão do ticket 10: `NaN` nativo, não zero literal — e a mesma troca do treino, ou o
    modelo vê em produção uma codificação que nunca viu no treino."""
    features = pacote.montar_features(entrada(etapa_1=NotasDeEtapa(0.0, 0.0, 0.0)))
    for coluna in FEATURES_ETAPA1:
        assert np.isnan(features.iloc[0][coluna]), coluna
    assert not features.iloc[0][["a2", "EB_PAS2", "Red_PAS2"]].isna().any()


def test_a1_do_aluno_sem_etapa_1_continua_exato_na_recomposicao(pacote):
    """O `NaN` é **codificação de feature**, não o valor do domínio: o `A1` dele é o z de zero
    (ADR-0008) e entra inteiro no Argumento Final. Recompor com `NaN` daria `NaN` para 10% da
    base — foi o defeito que a primeira rodada do lacre encontrou."""
    previsao = pacote.prever(entrada(etapa_1=NotasDeEtapa(0.0, 0.0, 0.0)))
    assert not np.isnan(previsao.a1)
    assert previsao.a1 == pytest.approx(
        calculate_argument_etapa(0.0, 0.0, 0.0, stats_da_prova(2023, 1, "inglesa"))
    )
    assert not np.isnan(previsao.argumento_final)


def test_largura_e_a_da_classe_do_aluno(pacote):
    """Dois números, por classe — não por acurácia média (empate), mas para não mentir com a
    minoria: com largura emprestada a cobertura dela cai de 79,5% para 77,8% (ADR-0012 §5)."""
    com = pacote.prever(entrada())
    sem = pacote.prever(entrada(etapa_1=NotasDeEtapa(0.0, 0.0, 0.0)))
    assert com.largura_a3 == pytest.approx(SIGMA_COM)
    assert sem.largura_a3 == pytest.approx(SIGMA_SEM)


# ─── Falhas ruidosas ────────────────────────────────────────────────────────────────────────


def test_trienio_sem_estatistica_oficial_falha_alto(pacote):
    """A turma viva (2024-2026) precisa de `(2024, Etapa 1)` e `(2025, Etapa 2)`, que ainda não
    foram extraídas dos Editais de Etapa. O caminho degradado é **recusar**, com a chave que
    falta na mensagem — nunca aproximar em silêncio, porque `A1` e `A2` são a parte *exata* da
    conta (ticket 04 §9)."""
    with pytest.raises(EstatisticasIndisponiveisError, match="2024"):
        pacote.prever(entrada(trienio="2024-2026"))


def test_pacote_ausente_falha_alto(tmp_path):
    """Sem pacote não há previsão *nem* largura — o estado "previsão sim, largura não" não é
    representável (ADR-0012 §6). É a resposta ao `target_calculator.py:66`, que engolia falha de
    carregamento com um `print` e seguia respondendo em produção."""
    with pytest.raises(PacoteIndisponivelError):
        PacoteDeModelo.carregar(tmp_path / "nao-existe")


def test_pacote_com_features_fora_de_ordem_e_recusado(tmp_path, pacote):
    """A ordem das features é contrato: passar o vetor trocado não levanta erro, produz número
    errado — foi o que invalidou o ADR-0007 (`baseline_avaliacao.py:55`, `R² = −83`)."""
    manifesto = json.loads((pacote.diretorio / "manifest.json").read_text(encoding="utf-8"))
    manifesto["modelos"][0]["features"] = list(reversed(FEATURES_CANONICAS))
    (pacote.diretorio / "manifest.json").write_text(json.dumps(manifesto), encoding="utf-8")

    with pytest.raises(PacoteIndisponivelError, match="features"):
        PacoteDeModelo.carregar(pacote.diretorio)


def test_features_montadas_na_ordem_do_manifesto(pacote):
    assert list(pacote.montar_features(entrada()).columns) == list(FEATURES_CANONICAS)


# ─── Paridade treino/runtime ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "notas_etapa_1", [(4.0, 30.0, 7.0), (0.0, 0.0, 0.0)], ids=["com Etapa 1", "sem Etapa 1"]
)
def test_o_runtime_monta_as_mesmas_features_que_o_treino(pacote, notas_etapa_1):
    """O teste que justifica este módulo existir.

    O mesmo Aluno entrando pelas duas portas — pelo `resultado_final.csv` no treino e pelas seis
    notas do request no runtime — tem que produzir o **mesmo vetor**. O desencontro entre as duas
    montagens (*train/serve skew*) não levanta exceção nenhuma: ele devolve previsão errada com
    cara de certa, para sempre.
    """
    from pas_intelligence.training_dataset import build_training_dataset
    from pas_intelligence.training_pipeline import montar_features as montar_features_do_treino

    p1_e1, p2_e1, red_e1 = notas_etapa_1
    a1 = calculate_argument_etapa(p1_e1, p2_e1, red_e1, stats_da_prova(2023, 1, "inglesa"))
    a2 = calculate_argument_etapa(5.0, 35.0, 7.5, stats_da_prova(2024, 2, "inglesa"))
    a3 = calculate_argument_etapa(6.0, 40.0, 8.0, stats_da_prova(2025, 3, "inglesa"))

    fonte = pd.DataFrame(
        [
            {
                "campus": "X", "curso": "Y", "turno": "DIURNO", "inscricao": 1,
                "eb_p1_e1": p1_e1, "eb_p2_e1": p2_e1, "red_e1": red_e1,
                "eb_p1_e2": 5.0, "eb_p2_e2": 35.0, "red_e2": 7.5,
                "eb_p1_e3": 6.0, "eb_p2_e3": 40.0, "red_e3": 8.0,
                "argumento_final": round(round(a1, 3) + 2 * round(a2, 3) + 3 * round(a3, 3), 3),
                "lingua_e1": "inglesa", "lingua_e2": "inglesa", "lingua_e3": "inglesa",
                "lingua_ambigua": False, "trienio": "2023/2025", "checksum_fecha": True,
            }
        ]
    )
    do_treino = montar_features_do_treino(build_training_dataset(fonte))[list(FEATURES_CANONICAS)]
    do_runtime = pacote.montar_features(
        entrada(etapa_1=NotasDeEtapa(p1_e1, p2_e1, red_e1))
    )

    pd.testing.assert_frame_equal(
        do_runtime.reset_index(drop=True),
        do_treino.reset_index(drop=True),
        check_dtype=False,
        atol=1e-3,  # o treino arredonda A1/A2 a 3 casas; o runtime não precisa
    )
