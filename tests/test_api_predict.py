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
"""Ticket 07: `(2024, Etapa 1)` e `(2025, Etapa 2)` entraram em `OFFICIAL_STATS` como
`DERIVADA` — o Preditor responde, apoiado em estatística estimada (ticket 06), não recusa
mais."""
TRIENIO_SEM_EDITAL = "2025-2027"
"""O triênio seguinte à Turma viva: nem Edital nem derivada cobrem `(2025, Etapa 1)`. É o que
resta para testar a política de recusa — ela não foi afrouxada, só parou de disparar para
2024-2026."""


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
        lingua_e1="inglesa", lingua_e2="inglesa", trienio=TRIENIO_COM_EDITAL,
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


def test_turma_viva_responde_apoiada_em_estatistica_derivada(api_com_pacote):
    """Ticket 07: `(2024, Etapa 1)` e `(2025, Etapa 2)` entraram em `OFFICIAL_STATS` como
    `DERIVADA` (Edital isolado de Etapa corrigido pelo ticket 06) — o Preditor deixa de recusar
    a Turma viva, e a resposta carrega o aviso de que a previsão se apoia em estimativa."""
    resposta = predict_service.predict_student(entrada_predict(trienio=TRIENIO_DA_TURMA_VIVA))

    assert resposta.modelo_disponivel is True
    assert resposta.motivo_indisponivel is None
    assert resposta.usa_estatistica_derivada is True


def test_trienio_com_edital_nao_carrega_aviso_de_estatistica_derivada(api_com_pacote):
    """Um triênio cujas duas Etapas já têm Edital de médias e desvios do Cebraspe não é
    estimativa — a tela não pode mostrar o aviso onde ele não se aplica."""
    resposta = predict_service.predict_student(entrada_predict())

    assert resposta.modelo_disponivel is True
    assert resposta.usa_estatistica_derivada is False


def test_trienio_sem_edital_de_etapa_continua_devolvendo_indisponivel_com_motivo(api_com_pacote):
    """A política de recusa não foi afrouxada pelo ticket 07 — só deixou de disparar para
    2024-2026. `A1` e `A2` são a parte **exata** da conta: sem o Edital (nem derivada) daquela
    Etapa, a resposta honesta é recusar dizendo qual chave falta (ticket 04 §9)."""
    resposta = predict_service.predict_student(entrada_predict(trienio=TRIENIO_SEM_EDITAL))

    assert resposta.modelo_disponivel is False
    assert resposta.top_cursos == []
    assert "2025" in (resposta.motivo_indisponivel or "")


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
        entrada_predict(lingua_e1="alema")
    with pytest.raises(ValidationError):
        entrada_predict(lingua_e2="alema")


def test_lingua_e_por_etapa_faltando_qualquer_uma_e_422():
    """`lingua_e1` e `lingua_e2` são obrigatórias, sem default (ticket 13, defeito 11): o Aluno
    é quem troca de língua entre Etapas, não um dado só do Aluno. Faltar qualquer uma nomeia o
    campo, como as seis notas — nunca um default silencioso que embutiria viés."""
    from pydantic import ValidationError

    from api.schemas.predict import PredictInput

    base = dict(
        p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
        p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5, trienio=TRIENIO_COM_EDITAL,
    )
    with pytest.raises(ValidationError, match="lingua_e1"):
        PredictInput(**base, lingua_e2="inglesa")
    with pytest.raises(ValidationError, match="lingua_e2"):
        PredictInput(**base, lingua_e1="inglesa")


def test_lingua_por_etapa_produz_a1_e_a2_diferentes(api_com_pacote):
    """O Aluno que fez inglês na Etapa 1 e espanhol na Etapa 2 tem `A1` e `A2` normalizados cada
    um pela sua língua — não os dois pela mesma, que era o defeito 11."""
    resposta = predict_service.predict_student(
        entrada_predict(lingua_e1="inglesa", lingua_e2="espanhola")
    )
    resposta_so_ingles = predict_service.predict_student(
        entrada_predict(lingua_e1="inglesa", lingua_e2="inglesa")
    )

    assert resposta.a1 == pytest.approx(resposta_so_ingles.a1)
    assert resposta.a2 != pytest.approx(resposta_so_ingles.a2)


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
    ninguém mediu. `grey` diz "não sei", que é a verdade. `TRIENIO_SEM_EDITAL` (não a Turma viva,
    que o ticket 07 passou a atender) é o triênio que ainda não tem cobertura nenhuma."""
    resposta = gestao_service.analyze_students(
        [aluno_gestao(ano_trienio=TRIENIO_SEM_EDITAL)], TRIENIO_SEM_EDITAL, "padrao"
    )

    assert resposta.results[0].status == "grey"
    assert resposta.results[0].status_label == "Sem previsão"
    assert resposta.results[0].chance_display == "—"
    assert resposta.kpis.n_sem_previsao == 1
    assert "2025" in (resposta.motivo_sem_previsao or "")


def test_modelo_disponivel_diz_so_que_o_pacote_carregou(api_com_pacote):
    """Dois problemas diferentes, dois campos diferentes. "O modelo não carregou" é ação de quem
    opera o sistema; "o Edital desta turma ainda não saiu" é espera. Um booleano só, cobrindo os
    dois, mandaria a coordenação procurar o problema errado."""
    resposta = gestao_service.analyze_students(
        [aluno_gestao(ano_trienio=TRIENIO_SEM_EDITAL)], TRIENIO_SEM_EDITAL, "padrao"
    )

    assert resposta.modelo_disponivel is True  # o pacote está lá
    assert resposta.kpis.n_sem_previsao == 1  # o Edital é que não está


def test_gestao_preve_para_a_turma_viva_apoiada_em_estatistica_derivada(api_com_pacote):
    """Ticket 07: a Gestão de Ativos usa a mesma `stats_da_prova` do Preditor público — as duas
    entradas `DERIVADA` de `OFFICIAL_STATS` a atendem sem qualquer mudança neste módulo."""
    resposta = gestao_service.analyze_students(
        [aluno_gestao(ano_trienio=TRIENIO_DA_TURMA_VIVA)], TRIENIO_DA_TURMA_VIVA, "padrao"
    )

    assert resposta.results[0].status != "grey"
    assert resposta.kpis.n_sem_previsao == 0
    assert resposta.modelo_disponivel is True


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
        [aluno_gestao(), aluno_gestao(ano_trienio=TRIENIO_SEM_EDITAL)],
        TRIENIO_COM_EDITAL,
        "padrao",
    )

    assert resposta.kpis.n_sem_previsao == 1
    assert resposta.kpis.total == 2
    assert resposta.modelo_disponivel is True


# ─── Merge do portal sobre o modelo (ticket 10) ──────────────────────────────────────────────


def _mapa_de_cursos(monkeypatch, m1: dict, m2: dict) -> None:
    """Fixa as notas de corte dos dois semestres, para que o teste fale de seleção e não de CSV."""
    # `predict_service` importou a função por valor, então é nele que o patch tem que entrar.
    monkeypatch.setattr(
        predict_service,
        "_build_cutoff_maps",
        lambda _trienio: ({"Sistema Universal": m1}, {"Sistema Universal": m2}, TRIENIO_COM_EDITAL),
    )


def test_probabilidade_dos_top_cursos_usa_a_largura_do_pacote(api_com_pacote, monkeypatch):
    """O lado do portal chamava `calculate_approval_probability(..., rmse=ARG_FINAL_MAE)` — uma
    constante de um modelo aposentado. Depois do merge é a Largura do manifesto, a mesma que o
    Curso Alvo e a Gestão usam (ADR-0012)."""
    _mapa_de_cursos(monkeypatch, {"Medicina - Diurno (Darcy Ribeiro)": 10.0}, {})

    resposta = predict_service.predict_student(
        entrada_predict(is_logged_in=True, curso_alvo="Medicina - Diurno (Darcy Ribeiro)")
    )

    esperado = calculate_approval_probability(
        resposta.arg_previsto, 10.0, largura_incerteza=resposta.largura_incerteza
    )
    assert resposta.top_cursos[0].prob == pytest.approx(round(esperado * 100, 1), abs=0.1)
    assert resposta.curso_alvo_result.prob == resposta.top_cursos[0].prob


def test_semestre_filtra_so_para_aluno_logado(api_com_pacote, monkeypatch):
    """As duas evoluções juntas: o semestre veio do portal, e é filtro de consulta — quem não
    está logado não tem o seletor, então enxerga os dois semestres."""
    _mapa_de_cursos(
        monkeypatch,
        {"Curso Do 1 - Diurno (Darcy Ribeiro)": 10.0},
        {"Curso Do 2 - Noturno (Darcy Ribeiro)": 10.0},
    )

    so_primeiro = predict_service.predict_student(entrada_predict(is_logged_in=True, semestre="1°"))
    assert {c.semestre for c in so_primeiro.top_cursos} == {"1°"}

    deslogado = predict_service.predict_student(entrada_predict(is_logged_in=False, semestre="1°"))
    assert {c.semestre for c in deslogado.top_cursos} == {"1°", "2°"}


def test_calculadora_le_official_stats_e_nao_um_dicionario_proprio():
    """`get_strategy_prediction` consumia o `TRIENNIUM_STATS` que o ticket 05 apagou. A rota
    agora resolve média e desvio pela mesma porta única do Preditor e do treino."""
    from api.schemas.predict import StrategyInput
    from pas_intelligence.training_dataset import stats_da_prova

    stats_p1, stats_p2, _, e3_e_ancora = predict_service._stats_do_ciclo(TRIENIO_COM_EDITAL, "espanhola", "espanhola")

    assert stats_p1 == stats_da_prova(2023, 1, "espanhola")
    assert stats_p2 == stats_da_prova(2024, 2, "espanhola")
    # 2025 (Etapa 3 de 2023-2025) já está publicada em OFFICIAL_STATS — nenhum Ano-Âncora aqui.
    assert e3_e_ancora is False

    resposta = predict_service.predict_strategy(
        StrategyInput(
            p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
            p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5,
            nota_alvo=120.0, ciclo_aluno=TRIENIO_COM_EDITAL, lingua_e1="espanhola", lingua_e2="espanhola",
        )
    )
    assert resposta.status != "indisponivel"
    assert resposta.p2_necessario > 0
    # Etapa 3 de 2023-2025 já é real (2025 publicado): nenhum Ano-Âncora precisa ser simulado.
    assert resposta.anos_ancora == []


def test_calculadora_preve_para_a_turma_viva_apoiada_em_estatistica_derivada():
    """Ticket 07: a Calculadora de Estratégia lê `OFFICIAL_STATS` pela mesma porta única
    (`stats_da_prova`) que o Preditor — as duas entradas `DERIVADA` a atendem de graça."""
    from api.schemas.predict import StrategyInput

    resposta = predict_service.predict_strategy(
        StrategyInput(
            p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
            p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5,
            nota_alvo=120.0, ciclo_aluno=TRIENIO_DA_TURMA_VIVA, lingua_e1="espanhola", lingua_e2="espanhola",
        )
    )

    assert resposta.status != "indisponivel"
    assert resposta.p2_necessario > 0


def test_calculadora_turma_viva_traz_os_cinco_anos_ancora_mais_recentes():
    """Ticket 12: sem a própria Etapa 3 publicada (turma viva), a Calculadora troca o cenário
    único do ticket 11 por cinco — um por Ano-Âncora, do mais recente ao mais antigo — em vez de
    reviver a regressão do `STATS_PAS3_TREND` que o ticket 05 apagou."""
    from api.schemas.predict import StrategyInput
    from pas_intelligence.pas_constants import anos_ancora

    resposta = predict_service.predict_strategy(
        StrategyInput(
            p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
            p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5,
            nota_alvo=120.0, ciclo_aluno=TRIENIO_DA_TURMA_VIVA, lingua_e1="espanhola", lingua_e2="espanhola",
        )
    )

    anos_esperados = anos_ancora()
    assert len(resposta.anos_ancora) == 5
    assert [a.ano for a in resposta.anos_ancora] == anos_esperados
    assert all(a.status != "" for a in resposta.anos_ancora)
    # O primeiro Ano-Âncora (o mais recente) é o que os campos de topo replicam, para o cliente
    # que ainda não sabe do ticket 12.
    primeiro = resposta.anos_ancora[0]
    assert resposta.p1_estimado == primeiro.p1_estimado
    assert resposta.p2_necessario == primeiro.p2_necessario
    assert resposta.status == primeiro.status
    assert resposta.arg_pas3_necessario == primeiro.arg_pas3_necessario
    assert resposta.arg_pas3_necessario != 0.0
    # Sem `curso_alvo`, os cinco cenários caem de volta no único `nota_alvo` do cliente — mas
    # cada um usa a *sua própria* estatística de Etapa 3, então P2 necessário varia entre eles.
    assert len({a.p2_necessario for a in resposta.anos_ancora}) > 1


def test_calculadora_ano_ancora_varia_a_nota_de_corte_por_trienio(monkeypatch):
    """Cada Ano-Âncora deve usar a Nota de Corte do curso no *seu* triênio (Ano-Âncora 2025 →
    triênio 2023-2025), não o `nota_alvo` de um triênio só — a decisão do relatório 04 §7.2."""
    from api.schemas.predict import StrategyInput
    from pas_intelligence.pas_constants import anos_ancora

    cortes_por_trienio = {
        f"{ano - 2}-{ano}": float(ano) for ano in anos_ancora()
    }

    def _get_course_cutoff_falso(curso, cota, trienio, semestre):
        assert curso == "Medicina - Diurno (Darcy Ribeiro)"
        return cortes_por_trienio.get(trienio, 0.0)

    monkeypatch.setattr(predict_service, "get_course_cutoff", _get_course_cutoff_falso)

    resposta = predict_service.predict_strategy(
        StrategyInput(
            p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
            p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5,
            nota_alvo=999.0, ciclo_aluno=TRIENIO_DA_TURMA_VIVA, lingua_e1="espanhola", lingua_e2="espanhola",
            curso_alvo="Medicina - Diurno (Darcy Ribeiro)",
        )
    )

    for resultado in resposta.anos_ancora:
        assert resultado.nota_corte == float(resultado.ano)
        assert resultado.trienio_corte == f"{resultado.ano - 2}-{resultado.ano}"


def test_calculadora_recusa_o_trienio_sem_edital_em_vez_de_estourar():
    """`TRIENIO_SEM_EDITAL` não tem o Edital da Etapa 1, nem derivada. Sem este caminho a rota
    respondia 500; com um `stats` chutado, responderia um P2 necessário que ninguém mediu."""
    from api.schemas.predict import StrategyInput

    resposta = predict_service.predict_strategy(
        StrategyInput(
            p1_pas1=4.0, p2_pas1=30.0, red_pas1=7.0,
            p1_pas2=5.0, p2_pas2=35.0, red_pas2=7.5,
            nota_alvo=120.0, ciclo_aluno=TRIENIO_SEM_EDITAL, lingua_e1="espanhola", lingua_e2="espanhola",
        )
    )

    assert resposta.status == "indisponivel"
    assert resposta.p2_necessario == 0.0
    assert "Edital" in resposta.mensagem


def test_ensemble_aposentado_nao_voltou_no_merge():
    """ADR-0011: o ensemble batia seu melhor componente por 0,10%, dentro do ruído de dobra a
    dobra, e nunca rodou em produção. A branch do portal ainda o trazia."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pas_intelligence.ensemble")


def test_resolver_sistema_cota_negros_nao_cai_no_fallback_universal():
    """Ticket 16: antes do `COTA_ALIASES` (rótulos L1/L2/L9/L10) ser removido, qualquer cota que
    não fosse "Sistema Universal" resolvia, em silêncio, para o mapa de Universal — o `.get`
    de quem consumia `_resolver_sistema` escondia o defeito. Este teste falharia se isso
    voltasse a acontecer."""
    available_systems = [
        "Sistema Universal",
        "Cota para Negros",
        "EP / Baixa Renda / PPI",
        "EP / Alta Renda / Não-PPI",
    ]

    resolvido = gestao_service._resolver_sistema("Cota para Negros", available_systems)

    assert resolvido == "Cota para Negros"
    assert resolvido != "Sistema Universal"


@pytest.mark.parametrize(
    "cota,esperado",
    [
        ("Sistema Universal", "Sistema Universal"),
        ("Cota para Negros", "Cota para Negros"),
        ("EP / Baixa Renda / PPI", "EP / Baixa Renda / PPI"),
        ("EP / Baixa Renda / PPI / PcD", "EP / Baixa Renda / PPI / PcD"),
        ("EP / Baixa Renda / Não-PPI", "EP / Baixa Renda / Não-PPI"),
        ("EP / Baixa Renda / Não-PPI / PcD", "EP / Baixa Renda / Não-PPI / PcD"),
        ("EP / Alta Renda / PPI", "EP / Alta Renda / PPI"),
        ("EP / Alta Renda / PPI / PcD", "EP / Alta Renda / PPI / PcD"),
        ("EP / Alta Renda / Não-PPI", "EP / Alta Renda / Não-PPI"),
        ("EP / Alta Renda / Não-PPI / PcD", "EP / Alta Renda / Não-PPI / PcD"),
    ],
)
def test_resolver_sistema_resolve_os_dez_sistemas_do_mapa(cota, esperado):
    """O seletor da tela manda o rótulo de `MAPA_SISTEMAS` (ticket 16) — cada um dos 10 deve
    bater exato contra o `Sistema_Nome` do CSV, sem depender de alias nem de fuzzy match."""
    available_systems = [
        "Sistema Universal", "Cota para Negros",
        "EP / Baixa Renda / PPI", "EP / Baixa Renda / PPI / PcD",
        "EP / Baixa Renda / Não-PPI", "EP / Baixa Renda / Não-PPI / PcD",
        "EP / Alta Renda / PPI", "EP / Alta Renda / PPI / PcD",
        "EP / Alta Renda / Não-PPI", "EP / Alta Renda / Não-PPI / PcD",
    ]

    assert gestao_service._resolver_sistema(cota, available_systems) == esperado
