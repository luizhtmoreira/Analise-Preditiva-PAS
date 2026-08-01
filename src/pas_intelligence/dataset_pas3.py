"""Carregamento do dataset canônico de treino (ticket 05) com as 6 features legadas embutidas.

Extraído de `scripts/baseline_honesto.py` (ticket 07) porque `scripts/janela_de_dados.py`
(ticket 08) precisa do mesmo carregamento — e o próprio `validation.py` argumenta contra escrever
o mesmo preparo de dado em cada script que consome a régua.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

DATASET_PAS3 = Path(__file__).resolve().parent.parent.parent / "data" / "training" / "pas3_dataset.parquet"

# A ordem real do vetor de features legadas, lida de `booster.feature_name()` dos próprios
# artefatos — não da documentação. `scripts/baseline_avaliacao.py:55` declarava outra, e foi a
# causa dos `R² = -83` do ADR-0007.
FEATURES_LEGADAS = ["EB_PAS1", "Red_PAS1", "EB_PAS2", "Red_PAS2", "Cresc_EB", "Cresc_Red"]

# As três razões do ticket 09 — as mesmas que `meta_scaler.joblib` já usa para rotear o ensemble
# atual, testadas ali como feature de regressão direta. Foi o único bloco de feature candidato
# que pagou o próprio custo (+2,13% de RMSE em `A3`, grátis).
FEATURES_TRAJETORIA = ["cresc_eb_pct", "cresc_red_pct", "sinal_cresc_eb"]

# O conjunto de features que fechou o ticket 09 (relatório `09-conjunto-de-features.md` §2):
# as 6 legadas + (A1, A2) + as 3 derivadas de trajetória. RMSE 5,057 em `A3` — bate o Portão 1
# do ticket 07 nas três pernas (geral, majoritária, minoritária).
FEATURES_CANONICAS = ["a1", "a2", *FEATURES_LEGADAS, *FEATURES_TRAJETORIA]


def adicionar_features_legadas(df: pd.DataFrame) -> pd.DataFrame:
    """As 6 features legadas, nos nomes que os artefatos `.joblib` atuais carregam dentro de si.

    Extraída de `carregar_dataset` (ticket 12) porque o pipeline de treino monta essas colunas
    em cima do dataset canônico já em memória (saído de `training_dataset.build_training_dataset`),
    sem passar por `pas3_dataset.parquet`.
    """
    df = df.copy()
    df["EB_PAS1"] = df["eb_pas1"]
    df["Red_PAS1"] = df["red_e1"]
    df["EB_PAS2"] = df["eb_pas2"]
    df["Red_PAS2"] = df["red_e2"]
    df["Cresc_EB"] = df["eb_pas2"] - df["eb_pas1"]
    df["Cresc_Red"] = df["red_e2"] - df["red_e1"]
    return df


def carregar_dataset(caminho: Path = DATASET_PAS3) -> pd.DataFrame:
    """`pas3_dataset.parquet` com as 6 features legadas reconstruídas."""
    return adicionar_features_legadas(pd.read_parquet(caminho))


# Colunas derivadas da Etapa 1 do Aluno — as que ficam sem sentido, e não zero real, quando ele
# não tem Etapa 1 (ticket 14). `FEATURES_ETAPA1` porque o pipeline (ticket 12) precisa da mesma
# lista que `scripts/familia_de_modelo_ticket10.py` mediu para decidir a família de modelo.
FEATURES_ETAPA1 = [
    "a1",
    "EB_PAS1",
    "Red_PAS1",
    "Cresc_EB",
    "Cresc_Red",
    "cresc_eb_pct",
    "cresc_red_pct",
    "sinal_cresc_eb",
]


def com_faltante_nativo_etapa1(df: pd.DataFrame) -> pd.DataFrame:
    """Troca o zero estrutural do Aluno sem Etapa 1 por `NaN` nas colunas de `FEATURES_ETAPA1`.

    Decisão do ticket 10: `NaN` nativo (deixa o LightGBM rotear pelo padrão de falta em cada nó)
    ganha do zero literal e de um segundo modelo dedicado à classe minoritária. Requer que
    `adicionar_derivadas_trajetoria` já tenha rodado — precisa de `cresc_eb_pct` etc.
    """
    df = df.copy()
    ausente = df["etapa_1_ausente"]
    for coluna in FEATURES_ETAPA1:
        df.loc[ausente, coluna] = np.nan
    return df


def adicionar_derivadas_trajetoria(df: pd.DataFrame) -> pd.DataFrame:
    """As três razões do ticket 09 (`FEATURES_TRAJETORIA`), sobre um `df` já com as 6 legadas.

    Normalizam o tamanho do salto `Cresc_EB`/`Cresc_Red` pelo ponto de partida — subir 5 pontos
    a partir de 10 não é o mesmo salto que subir 5 a partir de 40.
    """
    df = df.copy()
    df["cresc_eb_pct"] = df["Cresc_EB"].abs() / (df["EB_PAS1"].abs() + 0.01)
    df["cresc_red_pct"] = df["Cresc_Red"].abs() / (df["Red_PAS1"].abs() + 0.01)
    df["sinal_cresc_eb"] = np.sign(df["Cresc_EB"])
    return df


def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Do dataset canônico do ticket 05 até o vetor de features do ticket 09, com o valor
    faltante nativo do ticket 10.

    A ordem é a que `scripts/familia_de_modelo_ticket10.py` mediu (legadas → derivadas de
    trajetória → `NaN` nativo), e ela mora **aqui** — não no pipeline de treino — porque o
    runtime (`model_package`) precisa exatamente da mesma montagem. Duas montagens é o
    *train/serve skew*: não levanta exceção nenhuma, devolve número errado com cara de certo.
    """
    df = adicionar_features_legadas(df)
    df = adicionar_derivadas_trajetoria(df)
    df = com_faltante_nativo_etapa1(df)
    return df
