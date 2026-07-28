"""Testes das classes de modelo do ticket 10 (`scripts/familia_de_modelo_ticket10.py`).

Dados 100% sintéticos. O que importa provar aqui não é RMSE — é que o roteamento e a mistura
fazem exatamente o que dizem fazer, porque um erro de indexação booleana (a linha errada indo
pro submodelo errado) não quebra nada e só produz um número plausível e errado — o mesmo tipo de
falha silenciosa que o resto do mapa já documentou em outros lugares.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from familia_de_modelo_ticket10 import (  # noqa: E402
    DoisModelosPorClasse,
    EnsemblePorVolatilidade,
    com_faltante_nativo,
)


class _ModeloConstante:
    """Responde sempre um valor fixo — deixa a mistura/roteamento verificável na mão."""

    def __init__(self, valor: float):
        self.valor = valor

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.valor)


# ─── EnsemblePorVolatilidade ────────────────────────────────────────────────────────────────


def test_ensemble_pondera_por_cv_no_ponto_esperado():
    """CV baixo (aluno estável) pesa pro linear; CV alto (aluno errático) pesa pro LightGBM —
    a mesma direção que `ensemble.py` já documenta."""
    from lightgbm import LGBMRegressor  # type: ignore
    from sklearn.linear_model import Ridge  # type: ignore

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "EB_PAS1": rng.normal(30, 5, n),
            "EB_PAS2": rng.normal(30, 5, n),
            "a1": rng.normal(0, 9, n),
            "a2": rng.normal(0, 9, n),
        }
    )
    y = X["a1"] * 0.5 + X["a2"] * 0.5 + rng.normal(0, 1, n)

    modelo = EnsemblePorVolatilidade(alpha_ridge=1.0, params_lgbm={"n_estimators": 20}, semente=1)
    modelo.fit(X, y)

    # Dois alunos com a1/a2 idênticos, mas volatilidade de EB muito diferente.
    estavel = pd.DataFrame({"EB_PAS1": [30.0], "EB_PAS2": [30.0], "a1": [0.0], "a2": [0.0]})
    erratico = pd.DataFrame({"EB_PAS1": [10.0], "EB_PAS2": [50.0], "a1": [0.0], "a2": [0.0]})

    modelo.linear = Ridge(alpha=1.0).fit(X, y)
    modelo.lgbm = LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
    modelo.linear.predict = lambda Xp: np.full(len(Xp), 0.0)  # type: ignore
    modelo.lgbm.predict = lambda Xp: np.full(len(Xp), 10.0)  # type: ignore

    pred_estavel = modelo.predict(estavel)[0]
    pred_erratico = modelo.predict(erratico)[0]

    # CV(30,30)=0 -> peso baixo pro "arrojado" (10.0); CV(10,50)=100% -> peso alto.
    assert pred_estavel < pred_erratico
    assert 0.0 < pred_estavel < 5.0  # perto do conservador (peso baixo no arrojado)
    assert 5.0 < pred_erratico < 10.0  # puxado pro arrojado


def test_ensemble_nao_quebra_com_media_zero():
    """Aluno com EB_PAS1 == -EB_PAS2 dá média zero — a divisão do CV tem que virar peso 0.5,
    não `inf`/`NaN` vazando pra previsão."""
    modelo = EnsemblePorVolatilidade(alpha_ridge=1.0, params_lgbm={"n_estimators": 10}, semente=1)
    modelo.linear = _ModeloConstante(0.0)
    modelo.lgbm = _ModeloConstante(10.0)

    X = pd.DataFrame({"EB_PAS1": [20.0], "EB_PAS2": [-20.0], "a1": [0.0], "a2": [0.0]})
    pred = modelo.predict(X)
    assert np.isfinite(pred).all()
    assert pred[0] == 5.0  # peso 0.5 nos dois lados


# ─── DoisModelosPorClasse ───────────────────────────────────────────────────────────────────


def test_dois_modelos_roteia_cada_linha_para_o_submodelo_da_propria_classe():
    modelo = DoisModelosPorClasse(lambda semente: _ModeloConstante(0.0))
    # Cada submodelo aprende sua própria constante.
    modelo.maioria = _ModeloConstante(1.0)
    modelo.minoria = _ModeloConstante(9.0)

    X = pd.DataFrame(
        {
            "a1": [0.0, 0.0, 0.0, 0.0],
            "etapa_1_ausente": [False, True, False, True],
        }
    )
    previsto = modelo.predict(X)
    np.testing.assert_array_equal(previsto, [1.0, 9.0, 1.0, 9.0])


def test_dois_modelos_fit_treina_cada_submodelo_só_na_propria_classe():
    class _Registrador:
        def __init__(self):
            self.y_visto = None

        def fit(self, X, y):
            self.y_visto = np.asarray(y, dtype=float)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    registradores = []

    def fabrica(semente):
        m = _Registrador()
        registradores.append(m)
        return m

    modelo = DoisModelosPorClasse(fabrica)
    X = pd.DataFrame(
        {
            "a1": [0.0, 0.0, 0.0, 0.0],
            "etapa_1_ausente": [False, True, False, True],
        }
    )
    y = pd.Series([10.0, 20.0, 30.0, 40.0])
    modelo.fit(X, y)

    maioria, minoria = registradores
    np.testing.assert_array_equal(sorted(maioria.y_visto), [10.0, 30.0])
    np.testing.assert_array_equal(sorted(minoria.y_visto), [20.0, 40.0])
    assert modelo.n_treino_minoria == 2


def test_dois_modelos_nunca_repassa_a_propria_classe_como_feature():
    class _ConfereColunas:
        def __init__(self, semente=None):
            pass

        def fit(self, X, y):
            assert "etapa_1_ausente" not in X.columns
            return self

        def predict(self, X):
            assert "etapa_1_ausente" not in X.columns
            return np.zeros(len(X))

    modelo = DoisModelosPorClasse(_ConfereColunas)
    X = pd.DataFrame({"a1": [0.0, 0.0], "etapa_1_ausente": [False, True]})
    modelo.fit(X, pd.Series([1.0, 2.0]))
    modelo.predict(X)


# ─── com_faltante_nativo ────────────────────────────────────────────────────────────────────


def test_com_faltante_nativo_so_troca_colunas_da_etapa_1_e_so_para_quem_e_ausente():
    df = pd.DataFrame(
        {
            "etapa_1_ausente": [False, True],
            "a1": [5.0, 0.0],
            "a2": [7.0, 7.0],
            "EB_PAS1": [30.0, 0.0],
            "Red_PAS1": [10.0, 0.0],
            "Cresc_EB": [2.0, 15.0],
            "Cresc_Red": [1.0, 4.0],
            "cresc_eb_pct": [0.1, 999.0],
            "cresc_red_pct": [0.1, 999.0],
            "sinal_cresc_eb": [1.0, 1.0],
        }
    )
    resultado = com_faltante_nativo(df)

    # Linha 0 (não ausente): nada muda.
    assert resultado.loc[0, "a1"] == 5.0
    assert resultado.loc[0, "EB_PAS1"] == 30.0

    # Linha 1 (ausente): as colunas da Etapa 1 viram NaN...
    for coluna in ("a1", "EB_PAS1", "Red_PAS1", "Cresc_EB", "Cresc_Red",
                   "cresc_eb_pct", "cresc_red_pct", "sinal_cresc_eb"):
        assert np.isnan(resultado.loc[1, coluna]), coluna

    # ...mas a2, que é da Etapa 2 e está presente, fica intocado.
    assert resultado.loc[1, "a2"] == 7.0
