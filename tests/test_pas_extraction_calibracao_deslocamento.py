"""Testes do módulo de calibração do Deslocamento (ticket 06).

O parser de PDF (`etapa.py`) já é testado em `test_pas_extraction_etapa.py`; aqui o alvo é o
**cálculo** — Deslocamento por Etapa, correção, e o portão. Por isso os testes montam
`StatsEmpiricos` e `HistoricalStats` na mão e injetam `stats_oficiais`, em vez de ler PDF ou
importar o `OFFICIAL_STATS` de produção: o mesmo cuidado de `test_pas_extraction_official_stats
.py`, para o teste não quebrar toda vez que um Edital novo entrar em `pas_constants.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_extraction.calibracao_deslocamento import (  # type: ignore
    DeslocamentoEtapa,
    MedicaoTrienio,
    PortaoReprovadoError,
    RelatorioCalibracao,
    ResumoErro,
    StatsEmpiricos,
    calcular_delta_por_etapa,
    gerar_relatorio,
    medir_trienios,
    montar_deslocamento,
    verificar_portao,
)

from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore

# Uma HistoricalStats "oficial" simples, reaproveitada nos testes que não perguntam pelo valor
# exato do Argumento, só pela presença/ausência do delta.
_OFICIAL = HistoricalStats(mean_p1=5.0, std_p1=2.0, mean_p2=25.0, std_p2=10.0, mean_red=6.0, std_red=2.0)


def _linha(ano, etapa, p1, p2, red, lingua="inglesa", trienio=None, checksum=True):
    return {
        "trienio": trienio or f"{ano - etapa + 1}/{ano - etapa + 3}",
        f"_ano_e{etapa}": ano,
        f"eb_p1_e{etapa}": p1,
        f"eb_p2_e{etapa}": p2,
        f"red_e{etapa}": red,
        f"lingua_e{etapa}": lingua,
    }


class TestCalcularDeltaPorEtapa:
    def test_delta_positivo_quando_media_empirica_e_menor(self):
        # Média empírica de P2 menor que a oficial -> nota fixa vale mais Argumento (z maior)
        # -> Argumento empírico > oficial -> delta positivo. É exatamente o viés otimista do
        # ticket: Edital isolado (empírico) mais generoso que o Cebraspe (oficial).
        df = pd.DataFrame([_linha(2020, 1, p1=5.0, p2=25.0, red=6.0)])
        stats_empiricos = {
            (2020, 1): StatsEmpiricos(
                ano=2020, etapa=1, n=100, m_p1=5.0, dp_p1=2.0,
                m_p2=20.0, dp_p2=10.0, m_red=6.0, dp_red=2.0, arquivo_origem="x.pdf",
            )
        }
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        delta = calcular_delta_por_etapa(df, 1, stats_empiricos, stats_oficiais)

        assert delta.iloc[0] > 0

    def test_nan_quando_aluno_nao_prestou_a_etapa(self):
        df = pd.DataFrame([_linha(2020, 1, p1=0.0, p2=0.0, red=0.0)])
        stats_empiricos = {
            (2020, 1): StatsEmpiricos(2020, 1, 100, 5.0, 2.0, 20.0, 10.0, 6.0, 2.0, "x.pdf")
        }
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        delta = calcular_delta_por_etapa(df, 1, stats_empiricos, stats_oficiais)

        assert delta.isna().all()

    def test_nan_quando_nao_ha_edital_isolado_coletado(self):
        df = pd.DataFrame([_linha(2020, 1, p1=5.0, p2=25.0, red=6.0)])
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        delta = calcular_delta_por_etapa(df, 1, stats_empiricos={}, stats_oficiais=stats_oficiais)

        assert delta.isna().all()

    def test_levanta_se_official_stats_nao_cobre_a_lingua(self):
        df = pd.DataFrame([_linha(2020, 1, p1=5.0, p2=25.0, red=6.0, lingua="francesa")])
        stats_empiricos = {
            (2020, 1): StatsEmpiricos(2020, 1, 100, 5.0, 2.0, 20.0, 10.0, 6.0, 2.0, "x.pdf")
        }
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}  # falta "francesa"

        from pas_intelligence.training_dataset import EstatisticaOficialAusenteError  # type: ignore

        with pytest.raises(EstatisticaOficialAusenteError):
            calcular_delta_por_etapa(df, 1, stats_empiricos, stats_oficiais)


class TestMontarDeslocamento:
    def test_media_e_dispersao_entre_anos(self):
        df = pd.DataFrame({"_ano_e1": [2018, 2018, 2019], "dA1": [1.0, 3.0, 5.0]})

        deslocamento = montar_deslocamento(df, 1)

        assert deslocamento.pontos == {2018: pytest.approx(2.0), 2019: pytest.approx(5.0)}
        assert deslocamento.media == pytest.approx(3.5)  # média dos ANOS, não dos Alunos
        assert deslocamento.desvio == pytest.approx(2.1213, abs=1e-3)
        assert deslocamento.n_trienios == 2

    def test_desvio_zero_com_um_so_ano(self):
        df = pd.DataFrame({"_ano_e1": [2018, 2018], "dA1": [1.0, 3.0]})

        deslocamento = montar_deslocamento(df, 1)

        assert deslocamento.desvio == 0.0
        assert deslocamento.n_trienios == 1

    def test_linhas_nan_sao_ignoradas(self):
        df = pd.DataFrame({"_ano_e1": [2018, 2018, 2019], "dA1": [1.0, 3.0, float("nan")]})

        deslocamento = montar_deslocamento(df, 1)

        assert deslocamento.n_trienios == 1
        assert 2019 not in deslocamento.pontos


class TestMedirTrienios:
    def _df(self):
        return pd.DataFrame([
            {"trienio": "2018/2020", "_ano_e1": 2018, "_ano_e2": 2019, "dA1": 2.0, "dA2": 4.0},
            {"trienio": "2018/2020", "_ano_e1": 2018, "_ano_e2": 2019, "dA1": 2.0, "dA2": -4.0},
            {"trienio": "2019/2021", "_ano_e1": 2019, "_ano_e2": 2020, "dA1": 1.0, "dA2": 1.0},
            # Sem a Etapa 2 medida (2022/2024 hoje): não entra na tabela por triênio.
            {"trienio": "2022/2024", "_ano_e1": 2022, "_ano_e2": 2023, "dA1": 3.0, "dA2": float("nan")},
        ])

    def test_triennio_sem_as_duas_etapas_fica_de_fora(self):
        deslocamento_e1 = DeslocamentoEtapa(1, {2018: 0.0, 2019: 0.0, 2022: 0.0})
        deslocamento_e2 = DeslocamentoEtapa(2, {2019: 0.0, 2020: 0.0})
        medicoes = medir_trienios(self._df(), {}, deslocamento_e1, deslocamento_e2)

        trienios = {m.trienio for m in medicoes}
        assert trienios == {"2018/2020", "2019/2021"}

    def test_correcao_desloca_o_erro_pelo_deslocamento_medio(self):
        deslocamento_e1 = DeslocamentoEtapa(1, {2018: 2.0, 2019: 1.0})
        deslocamento_e2 = DeslocamentoEtapa(2, {2019: 0.0, 2020: 1.0})

        medicoes = medir_trienios(self._df(), {}, deslocamento_e1, deslocamento_e2)
        por_trienio = {m.trienio: m for m in medicoes}

        # 2019/2021: dA1=1, dA2=1; deslocamento média_e1=(2+1)/2=1.5, média_e2=(0+1)/2=0.5
        # corrigido: dA1_corr = 1 - 1.5 = -0.5 ; dA2_corr = 1 - 0.5 = 0.5
        # erro_corrigido = 1*(-0.5) + 2*(0.5) = 0.5
        assert por_trienio["2019/2021"].erro_corrigido.maximo == pytest.approx(0.5)
        assert por_trienio["2019/2021"].n_alunos == 1

    def test_resumo_erro_bruto_e_por_valor_absoluto(self):
        medicoes = medir_trienios(
            self._df(), {}, DeslocamentoEtapa(1, {2018: 0.0}), DeslocamentoEtapa(2, {2019: 0.0}),
        )
        por_trienio = {m.trienio: m for m in medicoes}

        # 2018/2020: erro_bruto = 1*2+2*4=10 e 1*2+2*(-4)=-6 -> |.| = [10, 6]
        assert por_trienio["2018/2020"].erro_bruto.medio == pytest.approx(8.0)
        assert por_trienio["2018/2020"].erro_bruto.maximo == pytest.approx(10.0)


class TestPortao:
    def _relatorio(self, residuais):
        trienios = [
            MedicaoTrienio(
                trienio=f"trienio-{i}", n_alunos=10, n_edital_e1=100, n_edital_e2=100,
                erro_bruto=ResumoErro(0, 0, 0),
                erro_corrigido=ResumoErro(medio=r / 2, p95=r * 0.9, maximo=r),
            )
            for i, r in enumerate(residuais)
        ]
        return RelatorioCalibracao(
            deslocamento_e1=DeslocamentoEtapa(1, {}), deslocamento_e2=DeslocamentoEtapa(2, {}),
            trienios=trienios,
        )

    def test_aprova_com_quatro_trienios_e_residual_abaixo_do_limiar(self):
        relatorio = self._relatorio([1.0, 2.0, 3.0, 4.0])

        assert relatorio.portao_aprovado is True
        verificar_portao(relatorio)  # não levanta

    def test_reprova_por_poucos_trienios(self):
        relatorio = self._relatorio([1.0, 2.0, 3.0])

        assert relatorio.portao_aprovado is False
        with pytest.raises(PortaoReprovadoError, match="pelo menos 4"):
            verificar_portao(relatorio)

    def test_reprova_por_residual_acima_do_limiar(self):
        relatorio = self._relatorio([1.0, 2.0, 3.0, 5.5])

        assert relatorio.portao_aprovado is False
        with pytest.raises(PortaoReprovadoError, match="resíduo máximo"):
            verificar_portao(relatorio)

    def test_residual_maximo_e_o_maior_entre_todos_os_trienios(self):
        relatorio = self._relatorio([1.0, 4.99, 2.0, 3.0])

        assert relatorio.residual_maximo == pytest.approx(4.99)


class TestGerarRelatorioIntegracao:
    def test_fluxo_completo_de_ponta_a_ponta(self):
        stats_empiricos = {
            (2018, 1): StatsEmpiricos(2018, 1, 100, 5.0, 2.0, 18.0, 10.0, 6.0, 2.0, "e1.pdf"),
            (2019, 2): StatsEmpiricos(2019, 2, 100, 5.0, 2.0, 22.0, 10.0, 6.0, 2.0, "e2.pdf"),
        }
        stats_oficiais = {
            (2018, 1, "inglesa"): _OFICIAL, (2018, 1, "francesa"): _OFICIAL,
            (2019, 2, "inglesa"): _OFICIAL, (2019, 2, "francesa"): _OFICIAL,
        }
        linhas = [
            {**_linha(2018, 1, 5.0, 25.0, 6.0, trienio="2018/2020"),
             **_linha(2019, 2, 5.0, 25.0, 6.0, trienio="2018/2020")}
            for _ in range(5)
        ]
        df = pd.DataFrame(linhas)

        relatorio = gerar_relatorio(df, stats_empiricos, stats_oficiais)

        assert relatorio.deslocamento_e1.n_trienios == 1
        assert relatorio.deslocamento_e2.n_trienios == 1
        assert len(relatorio.trienios) == 1
        assert relatorio.trienios[0].trienio == "2018/2020"
        assert relatorio.trienios[0].n_alunos == 5
        # Com o mesmo empírico usado para medir e corrigir, o resíduo some (viés constante).
        assert relatorio.trienios[0].erro_corrigido.maximo == pytest.approx(0.0, abs=1e-9)
