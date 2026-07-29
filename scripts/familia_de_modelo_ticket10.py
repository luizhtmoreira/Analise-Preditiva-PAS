"""Ticket 10 — família de modelo: qual, com quais hiperparâmetros, decidido por medição.

Compara, sobre a mesma régua do ticket 06 e o mesmo conjunto de features do ticket 09
(`FEATURES_CANONICAS`), três famílias candidatas:

- **Ridge** — linear regularizado, o piso não-trivial que o ticket pede;
- **LightGBM único** — o padrão de mercado para dado tabular, com tuning honesto;
- **Ensemble por volatilidade reimplementado** — o mecanismo de `ensemble.py` (linear × LightGBM,
  ponderados pela sigmoide do CV de `[EB_PAS1, EB_PAS2]`), **sem** o meta-modelo roteador, que o
  ticket 07 já mostrou pior que qualquer componente sozinho.

O quarto candidato do ticket ("modelo multi-saída, se o ticket 04 escolher prever as 3 notas")
não se aplica: o ticket 04 decidiu que o alvo é só `A3` (ADR-0009), não as 3 notas.

Tuning honesto: os hiperparâmetros de Ridge e LightGBM são escolhidos numa dobra de ajuste
dedicada — treina em 2016/2018, valida em 2017/2019 — que é disjunta das 5 dobras de medição
(`gerar_dobras` só testa a partir do terceiro triênio disponível) e do lacre. Escolhidos uma vez,
os hiperparâmetros ficam fixos para as 5 dobras que produzem o número reportado.

Restrição do ticket 14 sobre este ticket: "'aceita valor faltante nativamente' é critério com
peso, não desempate". Medido em separado (§ missing-value nativo): um único LightGBM enxergando
`NaN` nas colunas da Etapa 1 do Aluno sem Etapa 1 (deixa a árvore rotear pelo padrão de falta) vs.
dois modelos treinados por classe (`etapa_1_ausente`).

Roda com:

    .venv/bin/python scripts/familia_de_modelo_ticket10.py

Privacidade: lê `resultado_final.csv`/`notas_corte.csv` só para o erro de decisão (mesmo uso do
ticket 07); nada além de agregado sai daqui.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from pas_intelligence.dataset_pas3 import (  # noqa: E402
    FEATURES_CANONICAS,
    adicionar_derivadas_trajetoria,
    carregar_dataset as _carregar_dataset_base,
    com_faltante_nativo_etapa1 as com_faltante_nativo,
)
from pas_intelligence.ensemble import _sigmoid_weight  # noqa: E402
from pas_intelligence.validation import (  # noqa: E402
    TRIENIO_LACRADO,
    ResultadoValidacao,
    avaliar,
    erro_de_decisao,
)

from baseline_honesto import menor_corte_por_aluno  # noqa: E402

warnings.filterwarnings("ignore")

SEMENTE = 20260728

# O Portão 1 (ticket 07 §8) — o que qualquer candidato precisa bater para não regredir.
PORTAO_1 = dict(geral=5.167, majoritaria=5.038, minoritaria=6.028, vies_abs=0.500)

# A faixa de decisão, congelada no ticket 07 (§7): 1 RMSE de Argumento Final do melhor baseline
# trivial. Não se recalcula por modelo — senão a métrica vira auto-referente.
LARGURA_FAIXA_CONGELADA = 15.500


# ─── A dobra de ajuste — disjunta das 5 de medição e do lacre ─────────────────────────────────


def carregar_dataset() -> pd.DataFrame:
    """As 6 legadas mais as 3 derivadas de trajetória do ticket 09 — `FEATURES_CANONICAS`."""
    return adicionar_derivadas_trajetoria(_carregar_dataset_base())


def trienios_disponiveis(df: pd.DataFrame) -> list[str]:
    return sorted(t for t in df["trienio"].unique() if t != TRIENIO_LACRADO)


def fold_de_ajuste(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina em 2016/2018, valida em 2017/2019.

    Nenhum dos dois é triênio de teste em `gerar_dobras` (ela só testa a partir do terceiro
    triênio disponível em diante) — escolher hiperparâmetro aqui não vaza para nenhum número que
    a tabela final reporta.
    """
    disponiveis = trienios_disponiveis(df)
    treino = df[df["trienio"] == disponiveis[0]]
    validacao = df[df["trienio"] == disponiveis[1]]
    return treino, validacao


def _rmse(real: np.ndarray, previsto: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(previsto) - np.asarray(real)))))


def ajustar_ridge(treino: pd.DataFrame, validacao: pd.DataFrame, features: list[str]) -> dict:
    from sklearn.linear_model import Ridge  # type: ignore

    grade = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    melhor = None
    for alpha in grade:
        modelo = Ridge(alpha=alpha, random_state=SEMENTE)
        modelo.fit(treino[features], treino["a3"])
        rmse = _rmse(validacao["a3"].to_numpy(), modelo.predict(validacao[features]))
        if melhor is None or rmse < melhor[0]:
            melhor = (rmse, {"alpha": alpha})
    return melhor[1] | {"_rmse_ajuste": melhor[0]}


def ajustar_lgbm(treino: pd.DataFrame, validacao: pd.DataFrame, features: list[str]) -> dict:
    from lightgbm import LGBMRegressor  # type: ignore

    melhor = None
    for n_estimators in (100, 200, 400):
        for learning_rate in (0.01, 0.05, 0.1):
            for num_leaves in (15, 31):
                params = dict(
                    n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves
                )
                modelo = LGBMRegressor(random_state=SEMENTE, verbose=-1, **params)
                modelo.fit(treino[features], treino["a3"])
                rmse = _rmse(validacao["a3"].to_numpy(), modelo.predict(validacao[features]))
                if melhor is None or rmse < melhor[0]:
                    melhor = (rmse, params)
    return melhor[1] | {"_rmse_ajuste": melhor[0]}


# ─── Candidatos ────────────────────────────────────────────────────────────────────────────────


def fabrica_ridge(alpha: float):
    from sklearn.linear_model import Ridge  # type: ignore

    return lambda semente: Ridge(alpha=alpha, random_state=semente)


def fabrica_lgbm(params: dict):
    from lightgbm import LGBMRegressor  # type: ignore

    return lambda semente: LGBMRegressor(random_state=semente, verbose=-1, **params)


class EnsemblePorVolatilidade:
    """Reimplementação do mecanismo de `ensemble.py` — linear × LightGBM, ponderados pela
    sigmoide do CV de `[EB_PAS1, EB_PAS2]` — sobre o alvo e a régua novos, **sem** o meta-modelo
    roteador (ticket 07 §5: nem ele nem o roteador batem o melhor componente sozinho).

    Os limiares da sigmoide (`low_cv=10, high_cv=20`) ficam **os mesmos** de `ensemble.py` — este
    candidato testa a arquitetura herdada, não uma redesenhada.
    """

    def __init__(self, alpha_ridge: float, params_lgbm: dict, semente: int):
        self.alpha_ridge = alpha_ridge
        self.params_lgbm = params_lgbm
        self.semente = semente

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from lightgbm import LGBMRegressor  # type: ignore
        from sklearn.linear_model import Ridge  # type: ignore

        self.linear = Ridge(alpha=self.alpha_ridge, random_state=self.semente).fit(X, y)
        self.lgbm = LGBMRegressor(random_state=self.semente, verbose=-1, **self.params_lgbm).fit(
            X, y
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        eb1 = X["EB_PAS1"].to_numpy(dtype=float)
        eb2 = X["EB_PAS2"].to_numpy(dtype=float)
        media = (eb1 + eb2) / 2
        cv = np.where(media != 0, np.abs(eb2 - eb1) / 2 / media * 100, np.nan)
        peso = np.array([_sigmoid_weight(v) if np.isfinite(v) else 0.5 for v in cv])
        pred_linear = np.asarray(self.linear.predict(X), dtype=float)
        pred_lgbm = np.asarray(self.lgbm.predict(X), dtype=float)
        return (1 - peso) * pred_linear + peso * pred_lgbm


class DoisModelosPorClasse:
    """Um submodelo treinado só na classe majoritária, outro só na minoritária
    (`etapa_1_ausente`) — a alternativa ao valor faltante nativo que o ticket 14 pede para medir.

    Espera `etapa_1_ausente` como a última coluna de `X` (roteamento), removida antes de repassar
    aos submodelos — eles nunca veem a própria classe como feature.
    """

    def __init__(self, fabrica_submodelo, semente: int = 0):
        self.fabrica_submodelo = fabrica_submodelo
        self.semente = semente

    def fit(self, X: pd.DataFrame, y: pd.Series):
        classe = X["etapa_1_ausente"].to_numpy(dtype=bool)
        Xf = X.drop(columns="etapa_1_ausente")
        yf = np.asarray(y, dtype=float)

        self.n_treino_minoria = int(classe.sum())
        self.maioria = self.fabrica_submodelo(self.semente)
        self.maioria.fit(Xf[~classe], yf[~classe])
        self.minoria = self.fabrica_submodelo(self.semente)
        self.minoria.fit(Xf[classe], yf[classe])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        classe = X["etapa_1_ausente"].to_numpy(dtype=bool)
        Xf = X.drop(columns="etapa_1_ausente")
        previsto = np.empty(len(X), dtype=float)
        if (~classe).any():
            previsto[~classe] = self.maioria.predict(Xf[~classe])
        if classe.any():
            previsto[classe] = self.minoria.predict(Xf[classe])
        return previsto


# ─── Saída ──────────────────────────────────────────────────────────────────────────────────


def linha(nome: str, r: ResultadoValidacao) -> str:
    geral = r.agrupado
    maj = r.agrupado_majoritaria
    minoria = r.agrupado_minoritaria
    return (
        f"{nome:<42} {geral.rmse:>8.3f} {maj.rmse:>9.3f} "
        f"{(minoria.rmse if minoria else float('nan')):>9.3f} {geral.vies:>+8.3f}"
    )


CABECALHO = f"{'candidato':<42} {'RMSE':>8} {'RMSE maj':>9} {'RMSE min':>9} {'viés':>8}"


def contra_portao(r: ResultadoValidacao) -> str:
    geral = r.agrupado
    maj = r.agrupado_majoritaria
    minoria = r.agrupado_minoritaria
    ok = (
        geral.rmse <= PORTAO_1["geral"]
        and maj.rmse <= PORTAO_1["majoritaria"]
        and (minoria is None or minoria.rmse <= PORTAO_1["minoritaria"])
        and abs(geral.vies) <= PORTAO_1["vies_abs"]
    )
    return "bate o Portão 1" if ok else "NÃO bate o Portão 1"


def main() -> None:
    df = carregar_dataset()
    print(f"dataset: {len(df)} linhas, {df['trienio'].nunique()} triênios")

    treino_ajuste, val_ajuste = fold_de_ajuste(df)
    print(
        f"dobra de ajuste: treina {treino_ajuste['trienio'].iloc[0]} "
        f"({len(treino_ajuste)} linhas), valida {val_ajuste['trienio'].iloc[0]} "
        f"({len(val_ajuste)} linhas) — disjunta das 5 dobras de medição e do lacre\n"
    )

    # ─── Tuning honesto, uma vez, fora das dobras de medição ───────────────────────────────
    print("═══ Tuning (dobra de ajuste 2016/2018 → 2017/2019) ═══")
    params_ridge = ajustar_ridge(treino_ajuste, val_ajuste, FEATURES_CANONICAS)
    print(f"Ridge: alpha={params_ridge['alpha']} (RMSE na dobra de ajuste: "
          f"{params_ridge['_rmse_ajuste']:.3f})")
    params_lgbm = ajustar_lgbm(treino_ajuste, val_ajuste, FEATURES_CANONICAS)
    print(f"LightGBM: {({k: v for k, v in params_lgbm.items() if not k.startswith('_')})} "
          f"(RMSE na dobra de ajuste: {params_lgbm['_rmse_ajuste']:.3f})\n")

    alpha = params_ridge["alpha"]
    lgbm_params = {k: v for k, v in params_lgbm.items() if not k.startswith("_")}

    # ─── Os três candidatos, sobre as 5 dobras de medição ──────────────────────────────────
    resultados: dict[str, ResultadoValidacao] = {}
    resultados["Ridge (regularizado)"] = avaliar(
        df, fabrica_ridge(alpha), nome="Ridge", features=FEATURES_CANONICAS, semente=SEMENTE
    )
    resultados["LightGBM único"] = avaliar(
        df, fabrica_lgbm(lgbm_params), nome="LightGBM", features=FEATURES_CANONICAS,
        semente=SEMENTE,
    )
    resultados["Ensemble por volatilidade (sem roteador)"] = avaliar(
        df,
        lambda semente: EnsemblePorVolatilidade(alpha, lgbm_params, semente),
        nome="Ensemble",
        features=FEATURES_CANONICAS,
        semente=SEMENTE,
    )

    print("═══ As três famílias, sobre as mesmas 5 dobras (baseline ticket 07: RMSE 5,167) ═══")
    print(CABECALHO)
    for nome, r in resultados.items():
        print(linha(nome, r) + f"   [{contra_portao(r)}]")

    print("\n═══ Ganho relativo contra o baseline do ticket 07 (5,167) ═══")
    for nome, r in resultados.items():
        ganho = (PORTAO_1["geral"] - r.agrupado.rmse) / PORTAO_1["geral"] * 100
        print(f"{nome:<42} {ganho:>+7.2f}%")

    # ─── Erro de decisão ────────────────────────────────────────────────────────────────────
    cortes = menor_corte_por_aluno(df)
    largura = LARGURA_FAIXA_CONGELADA
    print(f"\n═══ Erro de decisão (faixa congelada em {largura:.3f}, ticket 07) ═══")
    print(f"{'candidato':<42} {'erra':>7} {'na faixa':>9} {'erra lá':>8}")
    for nome, r in resultados.items():
        d = erro_de_decisao(
            r.previsoes.merge(cortes, on=["id_pseudonimo", "trienio"], how="left"),
            largura_faixa=largura,
        )
        print(f"{nome:<42} {d.taxa_erro * 100:>6.1f}% {d.n_na_faixa:>9} "
              f"{d.taxa_erro_na_faixa * 100:>7.1f}%")

    # ─── Veredito do ensemble + volatilidade como feature ──────────────────────────────────
    melhor_componente = min(
        ("Ridge (regularizado)", "LightGBM único"), key=lambda n: resultados[n].agrupado.rmse
    )
    print(
        f"\n═══ Veredito do ensemble ═══\n"
        f"melhor componente sozinho: {melhor_componente} "
        f"(RMSE {resultados[melhor_componente].agrupado.rmse:.3f}) vs. "
        f"ensemble {resultados['Ensemble por volatilidade (sem roteador)'].agrupado.rmse:.3f}"
    )
    ensemble_ganha = (
        resultados["Ensemble por volatilidade (sem roteador)"].agrupado.rmse
        < resultados[melhor_componente].agrupado.rmse * 0.99  # >1% relativo = ganho material
    )
    print(
        "veredito: ENSEMBLE MANTIDO (ganho material sobre o melhor componente)"
        if ensemble_ganha
        else "veredito: ENSEMBLE APOSENTADO (não ganha 1% do melhor componente sozinho)"
    )

    print("\n═══ Volatilidade como feature (se o ensemble for aposentado) ═══")
    df_vol = df.copy()
    eb1 = df_vol["EB_PAS1"].to_numpy(dtype=float)
    eb2 = df_vol["EB_PAS2"].to_numpy(dtype=float)
    media = (eb1 + eb2) / 2
    df_vol["cv_volatilidade"] = np.where(media != 0, np.abs(eb2 - eb1) / 2 / media * 100, np.nan)
    features_com_vol = [*FEATURES_CANONICAS, "cv_volatilidade"]
    fabrica_vencedora = fabrica_ridge(alpha) if melhor_componente.startswith("Ridge") \
        else fabrica_lgbm(lgbm_params)
    r_sem_vol = resultados[melhor_componente]
    r_com_vol = avaliar(
        df_vol, fabrica_vencedora, nome="com CV", features=features_com_vol, semente=SEMENTE
    )
    ganho_vol = (r_sem_vol.agrupado.rmse - r_com_vol.agrupado.rmse) / r_sem_vol.agrupado.rmse * 100
    print(f"{melhor_componente} sem CV: {r_sem_vol.agrupado.rmse:.3f} · "
          f"com CV de volatilidade: {r_com_vol.agrupado.rmse:.3f} (ganho {ganho_vol:+.2f}%)")

    # ─── Valor faltante nativo × dois modelos (restrição do ticket 14 sobre o 10) ──────────
    print("\n═══ Valor faltante nativo × dois modelos por classe ═══")
    print(f"{'candidato':<42} {'RMSE min':>9} {'RMSE maj':>9}")
    base_lgbm = resultados["LightGBM único"]
    print(f"{'LightGBM único, zero literal (base)':<42} "
          f"{(base_lgbm.agrupado_minoritaria.rmse if base_lgbm.agrupado_minoritaria else float('nan')):>9.3f} "
          f"{base_lgbm.agrupado_majoritaria.rmse:>9.3f}")

    r_nan = avaliar(
        com_faltante_nativo(df), fabrica_lgbm(lgbm_params), nome="LightGBM NaN",
        features=FEATURES_CANONICAS, semente=SEMENTE,
    )
    minoria_nan = r_nan.agrupado_minoritaria
    print(f"{'LightGBM único, NaN nativo':<42} "
          f"{(minoria_nan.rmse if minoria_nan else float('nan')):>9.3f} "
          f"{r_nan.agrupado_majoritaria.rmse:>9.3f}")

    r_dois = avaliar(
        df,
        lambda semente: DoisModelosPorClasse(fabrica_lgbm(lgbm_params), semente),
        nome="dois modelos",
        features=[*FEATURES_CANONICAS, "etapa_1_ausente"], semente=SEMENTE,
    )
    minoria_dois = r_dois.agrupado_minoritaria
    print(f"{'dois modelos (LightGBM por classe)':<42} "
          f"{(minoria_dois.rmse if minoria_dois else float('nan')):>9.3f} "
          f"{r_dois.agrupado_majoritaria.rmse:>9.3f}")

    print("\n— treino da minoria por dobra (contexto de tamanho, trava 1 da §2.2) —")
    for d in resultados["LightGBM único"].dobras:
        m = d.minoritaria
        print(f"  dobra {d.dobra.indice} ({d.dobra.trienio_teste}): treino {m.n_treino_classe} "
              f"exemplos da classe")

    print(
        "\n═══ Resumo — o que o ticket 10 decide ═══\n"
        f"Melhor RMSE geral: {min(r.agrupado.rmse for r in resultados.values()):.3f} "
        f"contra o baseline do ticket 07 (5,167)."
    )


if __name__ == "__main__":
    main()
