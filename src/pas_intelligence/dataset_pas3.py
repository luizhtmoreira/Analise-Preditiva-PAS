"""Carregamento do dataset canônico de treino (ticket 05) com as 6 features legadas embutidas.

Extraído de `scripts/baseline_honesto.py` (ticket 07) porque `scripts/janela_de_dados.py`
(ticket 08) precisa do mesmo carregamento — e o próprio `validation.py` argumenta contra escrever
o mesmo preparo de dado em cada script que consome a régua.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore

DATASET_PAS3 = Path(__file__).resolve().parent.parent.parent / "data" / "training" / "pas3_dataset.parquet"

# A ordem real do vetor de features legadas, lida de `booster.feature_name()` dos próprios
# artefatos — não da documentação. `scripts/baseline_avaliacao.py:55` declarava outra, e foi a
# causa dos `R² = -83` do ADR-0007.
FEATURES_LEGADAS = ["EB_PAS1", "Red_PAS1", "EB_PAS2", "Red_PAS2", "Cresc_EB", "Cresc_Red"]


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
