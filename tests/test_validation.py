"""Testes da régua (ticket 06). Dados 100% sintéticos — nenhuma inscrição, nome ou nota real.

O teste que carrega o peso é `test_gerador_de_dobras_nunca_produz_o_trienio_lacrado`: com o laço
provado, qualquer número esquisito nos tickets 07 a 12 é do modelo, não do laço.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_intelligence.validation import (
    TRIENIO_LACRADO,
    Erro,
    avaliar,
    erro_de_decisao,
    faixa_de_decisao,
    gerar_dobras,
    holdout_final_use_uma_vez,
)

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

# Taxa de `etapa_1_ausente` por triênio, na forma do achado do relatório 06 §2: quase nula nos
# dois triênios antigos (o filtro do ticket 01 removeu a classe do regime antigo) e ~10% nos
# recentes. É o que faz a trava 1 disparar na dobra 1.
_TAXA_AUSENTE = (0.004, 0.004, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10)


def _dataset_sintetico(n_por_trienio: int = 300, semente: int = 7) -> pd.DataFrame:
    """Um dataset com a forma do `pas3_dataset.parquet`, pequeno e com relação conhecida."""
    rng = np.random.default_rng(semente)
    linhas = []
    for indice, (trienio, taxa) in enumerate(zip(TRIENIOS, _TAXA_AUSENTE)):
        a1 = rng.normal(0, 9, n_por_trienio)
        a2 = rng.normal(0, 9, n_por_trienio)
        a3 = 0.5 * a1 + 0.5 * a2 + rng.normal(0, 4, n_por_trienio)
        ausente = rng.random(n_por_trienio) < taxa
        linhas.append(
            pd.DataFrame(
                {
                    "id_pseudonimo": [
                        f"{indice}{i:06d}" for i in range(n_por_trienio)
                    ],
                    "trienio": trienio,
                    "etapa_1_ausente": ausente,
                    "a1": a1,
                    "a2": a2,
                    "a3": a3,
                }
            )
        )
    return pd.concat(linhas, ignore_index=True)


class _ModeloConstante:
    """Responde sempre o mesmo valor — previsível o bastante para conferir métrica na mão."""

    def __init__(self, valor: float = 0.0, semente: int | None = None):
        self.valor = valor
        self.semente = semente
        self.vezes_treinado = 0

    def fit(self, X, y):
        self.vezes_treinado += 1
        return self

    def predict(self, X):
        return np.full(len(X), self.valor)


def _fabrica_constante(valor: float = 0.0):
    return lambda semente: _ModeloConstante(valor, semente)


# ─── O lacre ────────────────────────────────────────────────────────────────────────────────


def test_gerador_de_dobras_nunca_produz_o_trienio_lacrado():
    for janela in (None, 1, 2, 3, 6, 99):
        dobras = gerar_dobras(TRIENIOS, janela=janela)
        for dobra in dobras:
            assert dobra.trienio_teste != TRIENIO_LACRADO
            assert TRIENIO_LACRADO not in dobra.trienios_treino


def test_holdout_final_e_o_unico_caminho_ate_o_lacre():
    dobra = holdout_final_use_uma_vez(TRIENIOS)
    assert dobra.trienio_teste == TRIENIO_LACRADO
    assert dobra.trienios_treino == TRIENIOS[:-1]


def test_avaliar_nunca_treina_nem_mede_no_lacre():
    dataset = _dataset_sintetico()
    resultado = avaliar(
        dataset,
        _fabrica_constante(),
        nome="constante",
        features=["a1", "a2"],
        semente=1,
    )
    assert TRIENIO_LACRADO not in set(resultado.previsoes["trienio"])
    for dobra in resultado.dobras:
        assert TRIENIO_LACRADO not in dobra.dobra.trienios_treino


def test_trienio_posterior_ao_lacre_falha_em_vez_de_adivinhar():
    with pytest.raises(ValueError, match="posterior"):
        gerar_dobras(TRIENIOS + ("2024/2026",))


# ─── A forma das dobras ─────────────────────────────────────────────────────────────────────


def test_cinco_dobras_treinando_no_passado_e_prevendo_o_futuro():
    dobras = gerar_dobras(TRIENIOS)
    assert len(dobras) == 5
    assert dobras[0].trienios_treino == ("2016/2018", "2017/2019")
    assert dobras[0].trienio_teste == "2018/2020"
    assert dobras[-1].trienios_treino == TRIENIOS[:6]
    assert dobras[-1].trienio_teste == "2022/2024"
    for dobra in dobras:
        assert all(t < dobra.trienio_teste for t in dobra.trienios_treino)


def test_janela_trunca_o_treino_sem_mexer_nos_trienios_de_teste():
    # Comparação pareada dentro da dobra (relatório 06 §5): o que a janela varre é o treino;
    # se ela mexesse no teste, o ticket 08 estaria comparando dobras diferentes entre si.
    expansiva = gerar_dobras(TRIENIOS)
    curta = gerar_dobras(TRIENIOS, janela=2)
    assert [d.trienio_teste for d in curta] == [d.trienio_teste for d in expansiva]
    assert all(len(d.trienios_treino) <= 2 for d in curta)
    assert curta[-1].trienios_treino == ("2020/2022", "2021/2023")


def test_dataset_curto_demais_para_uma_dobra_falha():
    with pytest.raises(ValueError, match="triênios"):
        gerar_dobras(("2016/2018", "2017/2019"))


# ─── A fábrica ──────────────────────────────────────────────────────────────────────────────


def test_cada_dobra_treina_um_modelo_novo():
    dataset = _dataset_sintetico()
    criados = []

    def fabrica(semente):
        modelo = _ModeloConstante(0.0, semente)
        criados.append(modelo)
        return modelo

    avaliar(dataset, fabrica, nome="c", features=["a1", "a2"], semente=42)
    assert len(criados) == 5
    assert all(m.vezes_treinado == 1 for m in criados)
    assert all(m.semente == 42 for m in criados)


def test_fabrica_que_devolve_o_mesmo_modelo_e_recusada():
    # O erro silencioso que a §8 do relatório 06 nomeia: a dobra 2 treinaria em cima do que a
    # dobra 1 já treinou, e o número sairia bom demais sem nada quebrar.
    dataset = _dataset_sintetico()
    pronto = _ModeloConstante()
    with pytest.raises(ValueError, match="fábrica"):
        avaliar(
            dataset,
            lambda semente: pronto,
            nome="c",
            features=["a1", "a2"],
            semente=1,
        )


def test_semente_e_registrada_no_resultado():
    dataset = _dataset_sintetico()
    resultado = avaliar(
        dataset, _fabrica_constante(), nome="c", features=["a1", "a2"], semente=1234
    )
    assert resultado.semente == 1234


# ─── Vazamento pela feature ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("coluna", ["a3", "argumento_final", "eb_pas3", "red_e3"])
def test_feature_que_carrega_a_resposta_e_recusada(coluna):
    dataset = _dataset_sintetico()
    dataset[coluna] = dataset["a3"]
    with pytest.raises(ValueError, match="Etapa 3"):
        avaliar(
            dataset,
            _fabrica_constante(),
            nome="c",
            features=["a1", coluna],
            semente=1,
        )


def test_lingua_da_etapa_3_e_feature_legitima():
    # O Aluno informa a própria língua (ticket 04) — ela é conhecida antes da prova, ao
    # contrário das notas da Etapa 3.
    dataset = _dataset_sintetico()
    dataset["lingua_e3"] = 0.0
    resultado = avaliar(
        dataset,
        _fabrica_constante(),
        nome="c",
        features=["a1", "a2", "lingua_e3"],
        semente=1,
    )
    assert resultado.features == ("a1", "a2", "lingua_e3")


# ─── As métricas ────────────────────────────────────────────────────────────────────────────


def test_metricas_sao_o_que_dizem():
    # resíduos (previsto − real) = +3, −3, +3, −3 → RMSE 3, MAE 3, viés 0
    erro = Erro.medir(np.array([0.0, 0.0, 0.0, 0.0]), np.array([3.0, -3.0, 3.0, -3.0]))
    assert erro.n == 4
    assert erro.rmse == pytest.approx(3.0)
    assert erro.mae == pytest.approx(3.0)
    assert erro.vies == pytest.approx(0.0)


def test_vies_positivo_quer_dizer_modelo_otimista():
    erro = Erro.medir(np.array([10.0, 10.0]), np.array([12.0, 14.0]))
    assert erro.vies == pytest.approx(3.0)


def test_erro_em_argumento_final_e_tres_vezes_o_erro_em_a3():
    erro = Erro.medir(np.array([0.0, 0.0]), np.array([4.0, -2.0]))
    triplo = erro.em_argumento_final
    assert triplo.rmse == pytest.approx(erro.rmse * 3)
    assert triplo.mae == pytest.approx(erro.mae * 3)
    assert triplo.vies == pytest.approx(erro.vies * 3)
    assert triplo.n == erro.n


def test_agrupado_pondera_pelo_tamanho_da_dobra_e_nao_e_a_media_dos_cinco():
    dataset = _dataset_sintetico()
    resultado = avaliar(
        dataset, _fabrica_constante(), nome="c", features=["a1", "a2"], semente=1
    )
    residuos = resultado.previsoes["previsto"] - resultado.previsoes["real"]
    esperado = float(np.sqrt(np.mean(np.square(residuos))))
    assert resultado.agrupado.rmse == pytest.approx(esperado)
    assert resultado.agrupado.n == len(resultado.previsoes)


# ─── As quatro travas da §2.2 ───────────────────────────────────────────────────────────────


def test_trava_1_classe_minoritaria_sem_lastro_no_treino_devolve_vazio():
    dataset = _dataset_sintetico()
    resultado = avaliar(
        dataset, _fabrica_constante(), nome="c", features=["a1", "a2"], semente=1
    )
    dobra_1 = resultado.dobras[0]
    assert dobra_1.minoritaria.n_treino_classe < dobra_1.minoritaria.n_teste_classe
    assert dobra_1.minoritaria.erro is None
    assert dobra_1.minoritaria.motivo_ausencia is not None
    assert dobra_1.majoritaria.erro is not None


def test_trava_2_a_classe_minoritaria_nao_tem_serie_para_ler_tendencia():
    dataset = _dataset_sintetico()
    resultado = avaliar(
        dataset, _fabrica_constante(), nome="c", features=["a1", "a2"], semente=1
    )
    assert not hasattr(resultado, "serie_minoritaria")
    assert len(resultado.serie_majoritaria) == 5
    assert isinstance(resultado.agrupado_minoritaria, Erro)


def test_trava_2_agrupado_da_minoria_ignora_as_dobras_que_nao_qualificam():
    dataset = _dataset_sintetico()
    resultado = avaliar(
        dataset, _fabrica_constante(), nome="c", features=["a1", "a2"], semente=1
    )
    qualificadas = [d for d in resultado.dobras if d.minoritaria.erro is not None]
    n_esperado = sum(d.minoritaria.n_teste_classe for d in qualificadas)
    assert len(qualificadas) < len(resultado.dobras)
    assert resultado.agrupado_minoritaria.n == n_esperado


def test_trava_3_o_erro_nunca_viaja_sem_o_tamanho_da_classe_no_treino():
    dataset = _dataset_sintetico()
    resultado = avaliar(
        dataset, _fabrica_constante(), nome="c", features=["a1", "a2"], semente=1
    )
    for dobra in resultado.dobras:
        for classe in (dobra.majoritaria, dobra.minoritaria):
            assert isinstance(classe.n_treino_classe, int)
            assert 0.0 <= classe.taxa_treino_classe <= 1.0


# ─── Erro de decisão ────────────────────────────────────────────────────────────────────────


def _previsoes_para_decisao() -> pd.DataFrame:
    # Argumento Final = a1 + 2·a2 + 3·a3. Com a1 = a2 = 0, ele é 3·a3 e o corte fica fácil de ler.
    return pd.DataFrame(
        {
            "a1": 0.0,
            "a2": 0.0,
            "real": [10.0, 10.0, -10.0, 0.4],
            "previsto": [11.0, -10.0, -11.0, -0.4],
            "nota_corte": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_erro_de_decisao_conta_quem_troca_de_lado_do_corte():
    # linha 2 (real +30, previsto −30) e linha 4 (real +1,2, previsto −1,2) trocam de lado.
    resultado = erro_de_decisao(_previsoes_para_decisao(), largura_faixa=3.0)
    assert resultado.n == 4
    assert resultado.taxa_erro == pytest.approx(0.5)


def test_erro_de_decisao_mede_a_faixa_so_em_quem_esta_perto_do_corte():
    # faixa de ±3 em Argumento Final: só a linha 4 (AF real = +1,2) está dentro.
    resultado = erro_de_decisao(_previsoes_para_decisao(), largura_faixa=3.0)
    assert resultado.n_na_faixa == 1
    assert resultado.taxa_erro_na_faixa == pytest.approx(1.0)
    assert resultado.erro_na_faixa.n == 1


def test_erro_de_decisao_conta_quem_ficou_sem_corte_em_vez_de_esconder():
    previsoes = _previsoes_para_decisao()
    previsoes.loc[0, "nota_corte"] = np.nan
    resultado = erro_de_decisao(previsoes, largura_faixa=3.0)
    assert resultado.n == 3
    assert resultado.n_sem_corte == 1


def test_faixa_de_decisao_e_um_rmse_de_argumento_final_do_baseline():
    dataset = _dataset_sintetico()
    baseline = avaliar(
        dataset, _fabrica_constante(), nome="c", features=["a1", "a2"], semente=1
    )
    assert faixa_de_decisao(baseline) == pytest.approx(baseline.agrupado.rmse * 3)
