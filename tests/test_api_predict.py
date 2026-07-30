"""Testes da fiação `api/services` ↔ pacote de modelo (ticket 13).

O que se testa aqui é a **costura**, não o modelo: que o Argumento Final da resposta é a soma
ponderada exata, que a probabilidade usa a largura do pacote e não uma constante, e que a
ausência de Edital de Etapa vira `modelo_disponivel: False` em vez de zeros mudos.

Pacote sintético (`conftest.diretorio_do_pacote`) — `models/` fica fora do git.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # type: ignore
import pytest  # type: ignore

from api.schemas.gestao import StudentInput
from api.schemas.predict import PredictInput
from api.services import gestao_service, predict_service
from pas_intelligence.statistics import calculate_approval_probability

TRIENIO_COM_EDITAL = "2023-2025"
TRIENIO_DA_TURMA_VIVA = "2024-2026"
"""`(2024, Etapa 1)` e `(2025, Etapa 2)` ainda não foram extraídos do Edital de Etapa."""


@pytest.fixture
def api_com_pacote(monkeypatch, pacote):
    monkeypatch.setattr(gestao_service, "_pacote", pacote)
    return pacote


@pytest.fixture
def api_sem_pacote(monkeypatch):
    monkeypatch.setattr(gestao_service, "_pacote", None)


def entrada_predict(**alteracoes) -> PredictInput:
    padrao = dict(
        p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
        p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5,
        lingua="inglesa", trienio=TRIENIO_COM_EDITAL,
    )
    return PredictInput(**{**padrao, **alteracoes})


def aluno_gestao(**alteracoes) -> StudentInput:
    padrao = dict(
        nome="Aluno Sintético", curso_alvo="Medicina", ano_trienio=TRIENIO_COM_EDITAL,
        p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
        p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5,
    )
    return StudentInput(**{**padrao, **alteracoes})


# ─── Preditor público ───────────────────────────────────────────────────────────────────────


def test_argumento_final_da_resposta_e_a_soma_ponderada_exata(api_com_pacote):
    """Uma previsão só na resposta. A tela antiga tinha duas — Argumento Final e EB PAS 3, de
    modelos independentes — que discordavam sobre passar em 11% dos Alunos (relatório 04 §3.2)."""
    resposta = predict_service.predict_student(entrada_predict())

    assert resposta.modelo_disponivel is True
    assert resposta.arg_previsto == pytest.approx(
        resposta.a1 + 2 * resposta.a2 + 3 * resposta.a3_previsto, abs=0.05
    )


def test_largura_da_resposta_vem_do_manifesto_do_pacote(api_com_pacote):
    """Não de constante de código. Trocar o pacote tem que trocar a largura sozinho — era o que
    o `13,49` de `gestao_service.py:34` não fazia (ADR-0012)."""
    resposta = predict_service.predict_student(entrada_predict())
    esperado = 3 * api_com_pacote.largura_de_incerteza(etapa_1_ausente=False)
    assert resposta.largura_incerteza == pytest.approx(esperado, abs=0.01)


def test_aluno_sem_etapa_1_e_declarado_e_recebe_a_largura_da_classe_dele(api_com_pacote):
    """As três notas da Etapa 1 em zero são uma **declaração** do Aluno (o botão do formulário),
    não "tirou zero" — e a classe muda a previsão, não só a largura (ADR-0012 §8)."""
    resposta = predict_service.predict_student(
        entrada_predict(p1_pas1=0.0, p2_pas1=0.0, red_pas1=0.0)
    )
    assert resposta.etapa_1_ausente is True
    assert resposta.largura_incerteza == pytest.approx(
        3 * api_com_pacote.largura_de_incerteza(etapa_1_ausente=True), abs=0.01
    )


def test_turma_viva_sem_edital_de_etapa_devolve_indisponivel_com_motivo(api_com_pacote):
    """`A1` e `A2` são a parte **exata** da conta. Sem o Edital daquela Etapa a resposta honesta
    é recusar dizendo qual chave falta — nunca aproximar em silêncio (ticket 04 §9)."""
    resposta = predict_service.predict_student(entrada_predict(trienio=TRIENIO_DA_TURMA_VIVA))

    assert resposta.modelo_disponivel is False
    assert resposta.top_cursos == []
    assert "2024" in (resposta.motivo_indisponivel or "")


def test_sem_pacote_nao_ha_previsao_nem_largura(api_sem_pacote):
    """O estado "previsão sim, largura não" não é representável (ADR-0012 §6)."""
    resposta = predict_service.predict_student(entrada_predict())

    assert resposta.modelo_disponivel is False
    assert resposta.arg_previsto == 0.0
    assert resposta.largura_incerteza == 0.0
    assert resposta.motivo_indisponivel


def test_trienio_malformado_nao_derruba_o_endpoint(api_com_pacote):
    """O Preditor é público e `trienio` é texto livre (clientes antigos). Entrada ruim tem que
    virar resposta, não 500."""
    resposta = predict_service.predict_student(entrada_predict(trienio="2024"))

    assert resposta.modelo_disponivel is False
    assert "malformado" in (resposta.motivo_indisponivel or "")


def test_lingua_fora_das_tres_e_recusada_pelo_schema():
    """422 nomeando o campo, antes de chegar ao serviço — como as seis notas."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        entrada_predict(lingua="alema")


def test_a_faixa_sem_rotulo_nao_esta_na_resposta():
    """ADR-0012 §7: `previsto ± 13,49` acertava 63% dos Alunos e respondia uma pergunta que
    ninguém fez. Se voltar, volta rotulada — este teste é o lembrete."""
    from api.schemas.predict import PredictResponse

    assert "arg_min" not in PredictResponse.model_fields
    assert "arg_max" not in PredictResponse.model_fields


def test_probabilidade_do_curso_alvo_usa_a_largura_do_pacote(api_com_pacote, monkeypatch):
    """A probabilidade tem que ser reproduzível a partir dos números que a própria resposta
    devolve — senão a tela e a conta podem divergir sem ninguém notar."""
    monkeypatch.setattr(
        predict_service,
        "_build_cutoff_maps",
        lambda _: ({"Sistema Universal": {"Medicina - Diurno (Darcy Ribeiro)": 100.0}}, {}, "2023-2025"),
    )
    resposta = predict_service.predict_student(entrada_predict(curso_alvo="Medicina"))

    assert resposta.curso_alvo_result is not None
    esperado = calculate_approval_probability(
        resposta.arg_previsto, 100.0, largura_incerteza=resposta.largura_incerteza
    )
    assert resposta.curso_alvo_result.prob == pytest.approx(round(esperado * 100, 1), abs=0.2)


# ─── Gestão de Ativos ───────────────────────────────────────────────────────────────────────


def test_aluno_sem_previsao_nao_vira_alto_risco(api_com_pacote):
    """Zerar o Argumento faria a tela da coordenação afirmar "Alto Risco" sobre um Aluno que
    ninguém mediu. `grey` diz "não sei", que é a verdade."""
    resposta = gestao_service.analyze_students(
        [aluno_gestao(ano_trienio=TRIENIO_DA_TURMA_VIVA)], TRIENIO_DA_TURMA_VIVA, "padrao"
    )

    assert resposta.results[0].status == "grey"
    assert resposta.results[0].status_label == "Sem previsão"
    assert resposta.results[0].chance_display == "—"
    assert resposta.kpis.n_sem_previsao == 1
    assert "2024" in (resposta.motivo_sem_previsao or "")


def test_modelo_disponivel_diz_so_que_o_pacote_carregou(api_com_pacote):
    """Dois problemas diferentes, dois campos diferentes. "O modelo não carregou" é ação de quem
    opera o sistema; "o Edital desta turma ainda não saiu" é espera. Um booleano só, cobrindo os
    dois, mandaria a coordenação procurar o problema errado."""
    resposta = gestao_service.analyze_students(
        [aluno_gestao(ano_trienio=TRIENIO_DA_TURMA_VIVA)], TRIENIO_DA_TURMA_VIVA, "padrao"
    )

    assert resposta.modelo_disponivel is True  # o pacote está lá
    assert resposta.kpis.n_sem_previsao == 1  # o Edital é que não está


def test_sem_pacote_a_gestao_diz_que_o_modelo_nao_carregou(api_sem_pacote):
    resposta = gestao_service.analyze_students([aluno_gestao()], TRIENIO_COM_EDITAL, "padrao")

    assert resposta.modelo_disponivel is False
    assert resposta.kpis.n_sem_previsao == 1
    assert "pacote" in (resposta.motivo_sem_previsao or "").lower()


def test_gestao_preve_quando_o_trienio_tem_edital(api_com_pacote):
    resposta = gestao_service.analyze_students(
        [aluno_gestao()], TRIENIO_COM_EDITAL, "padrao"
    )

    assert resposta.results[0].status != "grey"
    assert resposta.kpis.n_sem_previsao == 0
    assert resposta.modelo_disponivel is True


def test_trienniumstats_e_stats_pas3_trend_nao_existem_mais():
    """Ticket 05: só existe um lugar de onde sai média e desvio oficiais — `OFFICIAL_STATS`."""
    assert not hasattr(gestao_service, "TRIENNIUM_STATS")
    assert not hasattr(gestao_service, "STATS_PAS3_TREND")


def test_reality_check_le_official_stats_pela_porta_unica(api_com_pacote, monkeypatch, caplog):
    """O Reality Check (cohort) tem que continuar funcionando, agora só com os números do
    Edital — nunca com um dicionário próprio que possa divergir do `A1`/`A2` do Preditor."""
    monkeypatch.setattr(
        gestao_service,
        "_df_cohort",
        pd.DataFrame({
            "P1_PAS1": [4.0] * 5, "P2_PAS1": [30.0] * 5,
            "P1_PAS2": [5.0] * 5, "P2_PAS2": [35.0] * 5,
            "P1_PAS3": [5.0] * 5, "P2_PAS3": [40.0] * 5,
            "ARG_FINAL_REAL": [1.0] * 5,
            "EB_PAS1": [34.0] * 5, "EB_PAS2": [40.0] * 5,
        }),
    )
    monkeypatch.setattr(
        gestao_service,
        "_build_cutoff_maps",
        lambda _: (
            {"Sistema Universal": {"Medicina - Diurno (Darcy Ribeiro)": 5.0}}, {}, TRIENIO_COM_EDITAL,
        ),
    )

    with caplog.at_level("ERROR"):
        resposta = gestao_service.analyze_students(
            [aluno_gestao(curso_alvo="Medicina")], TRIENIO_COM_EDITAL, "padrao"
        )

    assert "falhou" not in caplog.text
    assert resposta.results[0].historico_pct > 0.0


def test_lote_misto_conta_os_dois_lados(api_com_pacote):
    """Uma planilha de escola pode trazer turmas de triênios diferentes; a que não dá para prever
    não pode contaminar a que dá."""
    resposta = gestao_service.analyze_students(
        [aluno_gestao(), aluno_gestao(ano_trienio=TRIENIO_DA_TURMA_VIVA)],
        TRIENIO_COM_EDITAL,
        "padrao",
    )

    assert resposta.kpis.n_sem_previsao == 1
    assert resposta.kpis.total == 2
    assert resposta.modelo_disponivel is True
