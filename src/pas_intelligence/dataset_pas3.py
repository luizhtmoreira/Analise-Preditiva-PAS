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


def carregar_dataset(caminho: Path = DATASET_PAS3) -> pd.DataFrame:
    """`pas3_dataset.parquet` com as 6 features legadas reconstruídas, nos nomes que os
    artefatos `.joblib` atuais carregam dentro de si."""
    df = pd.read_parquet(caminho)
    df["EB_PAS1"] = df["eb_pas1"]
    df["Red_PAS1"] = df["red_e1"]
    df["EB_PAS2"] = df["eb_pas2"]
    df["Red_PAS2"] = df["red_e2"]
    df["Cresc_EB"] = df["eb_pas2"] - df["eb_pas1"]
    df["Cresc_Red"] = df["red_e2"] - df["red_e1"]
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
