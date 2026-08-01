"""Testes do módulo de calibração do ticket 06.

O parser de PDF (`etapa.py`) já é testado em `test_pas_extraction_etapa.py`; aqui o alvo é o
**cálculo** — os parâmetros por (Etapa, ano), a correção agregada, o resíduo e o portão. Por
isso os testes montam `StatsEmpiricos` e `HistoricalStats` na mão e injetam `stats_oficiais`,
em vez de ler PDF ou importar o `OFFICIAL_STATS` de produção: o mesmo cuidado de
`test_pas_extraction_official_stats.py`, para o teste não quebrar toda vez que um Edital novo
entrar em `pas_constants.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_extraction.calibracao_deslocamento import (  # type: ignore
    CalibracaoEtapa,
    Correcao,
    CorrecaoComponente,
    DeslocamentoEtapa,
    MedicaoTrienio,
    PortaoReprovadoError,
    RelatorioCalibracao,
    ResumoErro,
    StatsEmpiricos,
    calcular_delta_por_etapa,
    gerar_relatorio,
    medir_trienios,
    montar_calibracao,
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


def _emp(ano, etapa, *, m_p2=20.0, dp_p2=10.0, m_red=6.0, dp_red=2.0, m_p1=5.0, dp_p1=2.0, n=100):
    return StatsEmpiricos(ano, etapa, n, m_p1, dp_p1, m_p2, dp_p2, m_red, dp_red, "x.pdf")


class TestCorrecaoComponente:
    def test_aplicar_desfaz_o_erro_medido(self):
        # O Edital isolado deu média 20 onde o oficial é 25, e desvio 8 onde o oficial é 10.
        correcao = CorrecaoComponente(delta_media=-5.0, razao_desvio=0.8)

        assert correcao.aplicar(20.0, 8.0) == pytest.approx((25.0, 10.0))

    def test_inerte_nao_mexe_em_nada(self):
        assert CorrecaoComponente.INERTE.aplicar(20.0, 8.0) == pytest.approx((20.0, 8.0))


class TestMontarCalibracao:
    def test_mede_delta_da_media_e_razao_do_desvio(self):
        stats_empiricos = {(2020, 1): _emp(2020, 1, m_p2=20.0, dp_p2=8.0)}
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        calibracao = montar_calibracao(stats_empiricos, 1, stats_oficiais)

        p2 = calibracao.pontos[2020]["p2"]
        assert p2.delta_media == pytest.approx(20.0 - 25.0)
        assert p2.razao_desvio == pytest.approx(8.0 / 10.0)

    def test_ignora_ano_sem_edital_oficial(self):
        stats_empiricos = {(2020, 1): _emp(2020, 1), (2099, 1): _emp(2099, 1)}
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        calibracao = montar_calibracao(stats_empiricos, 1, stats_oficiais)

        assert calibracao.n_anos == 1
        assert 2099 not in calibracao.pontos

    def test_so_recolhe_a_etapa_pedida(self):
        stats_empiricos = {(2020, 1): _emp(2020, 1), (2021, 2): _emp(2021, 2)}
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL, (2021, 2, "inglesa"): _OFICIAL}

        assert montar_calibracao(stats_empiricos, 2, stats_oficiais).pontos.keys() == {2021}


class TestCalibracaoCorrecao:
    def _calibracao(self):
        return CalibracaoEtapa(etapa=1, pontos={
            2018: {"p1": CorrecaoComponente(9.0, 9.0), "p2": CorrecaoComponente(1.0, 0.8),
                   "red": CorrecaoComponente(0.5, 1.0)},
            2019: {"p1": CorrecaoComponente(9.0, 9.0), "p2": CorrecaoComponente(2.0, 0.9),
                   "red": CorrecaoComponente(0.7, 1.1)},
            # Ano fora do padrão, como a Etapa 1 de 2021 (volta da pandemia): a mediana o
            # absorve, a média não.
            2020: {"p1": CorrecaoComponente(9.0, 9.0), "p2": CorrecaoComponente(30.0, 0.85),
                   "red": CorrecaoComponente(0.6, 1.05)},
        })

    def test_agrega_por_mediana_e_nao_por_media(self):
        correcao = self._calibracao().correcao()

        # média seria (1+2+30)/3 = 11; mediana é 2.
        assert correcao.por_componente["p2"].delta_media == pytest.approx(2.0)
        assert correcao.por_componente["p2"].razao_desvio == pytest.approx(0.85)

    def test_parte_1_nunca_e_corrigida(self):
        # Mesmo com parâmetros gritantes medidos para a P1, a correção sai inerte: o Edital
        # isolado não diz a língua de ninguém, e calibrar a P1 misturada piora o resíduo.
        correcao = self._calibracao().correcao()

        assert correcao.por_componente["p1"] == CorrecaoComponente.INERTE

    def test_excluir_ano_tira_o_proprio_do_pool(self):
        correcao = self._calibracao().correcao(excluir_ano=2020)

        # Sobram 1,0 e 2,0 -> mediana 1,5.
        assert correcao.por_componente["p2"].delta_media == pytest.approx(1.5)

    def test_levanta_se_excluir_o_unico_ano(self):
        calibracao = CalibracaoEtapa(etapa=1, pontos={
            2018: {c: CorrecaoComponente.INERTE for c in ("p1", "p2", "red")}
        })

        with pytest.raises(ValueError, match="nenhum ano sobra"):
            calibracao.correcao(excluir_ano=2018)


class TestCorrecaoAplicar:
    def test_produz_a_historical_stats_que_o_ticket_07_escreve(self):
        correcao = Correcao(por_componente={
            "p1": CorrecaoComponente.INERTE,
            "p2": CorrecaoComponente(delta_media=-5.0, razao_desvio=0.8),
            "red": CorrecaoComponente(delta_media=0.5, razao_desvio=1.25),
        })

        stats = correcao.aplicar(_emp(2024, 1, m_p2=20.0, dp_p2=8.0, m_red=6.5, dp_red=2.5))

        assert (stats.mean_p2, stats.std_p2) == pytest.approx((25.0, 10.0))
        assert (stats.mean_red, stats.std_red) == pytest.approx((6.0, 2.0))
        assert (stats.mean_p1, stats.std_p1) == pytest.approx((5.0, 2.0))  # intocada


class TestCalcularDeltaPorEtapa:
    def test_delta_positivo_quando_media_empirica_e_menor(self):
        # Média empírica de P2 menor que a oficial -> nota fixa vale mais Argumento (z maior)
        # -> Argumento empírico > oficial -> delta positivo. É exatamente o viés otimista do
        # ticket: Edital isolado (empírico) mais generoso que o Cebraspe (oficial).
        df = pd.DataFrame([_linha(2020, 1, p1=5.0, p2=25.0, red=6.0)])
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        delta = calcular_delta_por_etapa(df, 1, {(2020, 1): _emp(2020, 1)}, stats_oficiais)

        assert delta.iloc[0] > 0

    def test_correcao_por_ano_zera_o_delta_quando_desfaz_o_erro(self):
        df = pd.DataFrame([_linha(2020, 1, p1=5.0, p2=30.0, red=7.0)])
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}
        stats_empiricos = {(2020, 1): _emp(2020, 1, m_p2=20.0, dp_p2=8.0)}
        # A correção exata: média 20 -> 25, desvio 8 -> 10.
        correcao = Correcao(por_componente={
            "p1": CorrecaoComponente.INERTE,
            "p2": CorrecaoComponente(-5.0, 0.8),
            "red": CorrecaoComponente.INERTE,
        })

        delta = calcular_delta_por_etapa(
            df, 1, stats_empiricos, stats_oficiais, correcao_por_ano={2020: correcao},
        )

        assert delta.iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_nan_quando_aluno_nao_prestou_a_etapa(self):
        df = pd.DataFrame([_linha(2020, 1, p1=0.0, p2=0.0, red=0.0)])
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        delta = calcular_delta_por_etapa(df, 1, {(2020, 1): _emp(2020, 1)}, stats_oficiais)

        assert delta.isna().all()

    def test_nan_quando_nao_ha_edital_isolado_coletado(self):
        df = pd.DataFrame([_linha(2020, 1, p1=5.0, p2=25.0, red=6.0)])
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}

        delta = calcular_delta_por_etapa(df, 1, stats_empiricos={}, stats_oficiais=stats_oficiais)

        assert delta.isna().all()

    def test_levanta_se_official_stats_nao_cobre_a_lingua(self):
        df = pd.DataFrame([_linha(2020, 1, p1=5.0, p2=25.0, red=6.0, lingua="francesa")])
        stats_oficiais = {(2020, 1, "inglesa"): _OFICIAL}  # falta "francesa"

        from pas_intelligence.training_dataset import EstatisticaOficialAusenteError  # type: ignore

        with pytest.raises(EstatisticaOficialAusenteError):
            calcular_delta_por_etapa(df, 1, {(2020, 1): _emp(2020, 1)}, stats_oficiais)


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
            {"trienio": "2018/2020", "_ano_e1": 2018, "_ano_e2": 2019,
             "dA1": 2.0, "dA2": 4.0, "res1": 0.5, "res2": 1.0},
            {"trienio": "2018/2020", "_ano_e1": 2018, "_ano_e2": 2019,
             "dA1": 2.0, "dA2": -4.0, "res1": 0.5, "res2": -1.0},
            {"trienio": "2019/2021", "_ano_e1": 2019, "_ano_e2": 2020,
             "dA1": 1.0, "dA2": 1.0, "res1": -0.5, "res2": 0.5},
            # Sem a Etapa 2 medida (2022/2024 hoje): não entra na tabela por triênio.
            {"trienio": "2022/2024", "_ano_e1": 2022, "_ano_e2": 2023,
             "dA1": 3.0, "dA2": float("nan"), "res1": 1.0, "res2": float("nan")},
        ])

    def test_triennio_sem_as_duas_etapas_fica_de_fora(self):
        medicoes = medir_trienios(self._df(), {})

        assert {m.trienio for m in medicoes} == {"2018/2020", "2019/2021"}

    def test_erro_corrigido_vem_do_residuo_ja_calculado(self):
        por_trienio = {m.trienio: m for m in medir_trienios(self._df(), {})}

        # 2019/2021: res1=-0.5, res2=0.5 -> 1*(-0.5) + 2*(0.5) = 0.5
        assert por_trienio["2019/2021"].erro_corrigido.maximo == pytest.approx(0.5)
        assert por_trienio["2019/2021"].n_alunos == 1

    def test_resumo_erro_bruto_e_por_valor_absoluto(self):
        por_trienio = {m.trienio: m for m in medir_trienios(self._df(), {})}

        # 2018/2020: erro_bruto = 1*2+2*4=10 e 1*2+2*(-4)=-6 -> |.| = [10, 6]
        assert por_trienio["2018/2020"].erro_bruto.medio == pytest.approx(8.0)
        assert por_trienio["2018/2020"].erro_bruto.maximo == pytest.approx(10.0)

    def test_n_do_edital_vem_dos_stats_empiricos(self):
        stats_empiricos = {(2018, 1): _emp(2018, 1, n=22666), (2019, 2): _emp(2019, 2, n=21380)}

        por_trienio = {m.trienio: m for m in medir_trienios(self._df(), stats_empiricos)}

        assert por_trienio["2018/2020"].n_edital_e1 == 22666
        assert por_trienio["2018/2020"].n_edital_e2 == 21380


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
        vazia = CalibracaoEtapa(etapa=1, pontos={})
        return RelatorioCalibracao(
            calibracao_e1=vazia, calibracao_e2=CalibracaoEtapa(etapa=2, pontos={}),
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
    def _cenario(self, m_p2_2018, m_p2_2020):
        """Dois triênios completos, com o mesmo par de Etapas medido em anos diferentes."""
        stats_empiricos = {
            (2018, 1): _emp(2018, 1, m_p2=m_p2_2018),
            (2019, 2): _emp(2019, 2, m_p2=22.0),
            (2020, 1): _emp(2020, 1, m_p2=m_p2_2020),
            (2021, 2): _emp(2021, 2, m_p2=22.0),
        }
        stats_oficiais = {
            (ano, etapa, lingua): _OFICIAL
            for ano, etapa in ((2018, 1), (2019, 2), (2020, 1), (2021, 2))
            for lingua in ("inglesa", "francesa")
        }
        linhas = [
            {**_linha(2018, 1, 5.0, 25.0, 6.0, trienio="2018/2020"),
             **_linha(2019, 2, 5.0, 25.0, 6.0, trienio="2018/2020")}
            for _ in range(5)
        ] + [
            {**_linha(2020, 1, 5.0, 25.0, 6.0, trienio="2020/2022"),
             **_linha(2021, 2, 5.0, 25.0, 6.0, trienio="2020/2022")}
            for _ in range(5)
        ]
        return pd.DataFrame(linhas), stats_empiricos, stats_oficiais

    def test_fluxo_completo_de_ponta_a_ponta(self):
        df, stats_empiricos, stats_oficiais = self._cenario(m_p2_2018=18.0, m_p2_2020=18.0)

        relatorio = gerar_relatorio(df, stats_empiricos, stats_oficiais)

        assert relatorio.calibracao_e1.n_anos == 2
        assert relatorio.calibracao_e2.n_anos == 2
        assert {t.trienio for t in relatorio.trienios} == {"2018/2020", "2020/2022"}
        assert relatorio.trienios[0].n_alunos == 5
        # Dois anos com o MESMO erro: a correção do outro ano descreve este perfeitamente.
        assert relatorio.residual_maximo == pytest.approx(0.0, abs=1e-9)

    def test_o_residuo_e_leave_one_year_out(self):
        # Anos com erros DIFERENTES: corrigir 2018 com o parâmetro de 2020 deixa sobra. Se a
        # correção usasse o próprio ano, o resíduo seria zero e o portão sairia otimista.
        df, stats_empiricos, stats_oficiais = self._cenario(m_p2_2018=18.0, m_p2_2020=22.0)

        relatorio = gerar_relatorio(df, stats_empiricos, stats_oficiais)

        assert relatorio.residual_maximo > 0.1
