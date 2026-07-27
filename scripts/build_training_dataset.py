#!/usr/bin/env python3
"""Regenera o dataset de treino canônico do PAS 3 (ticket 05 do mapa treino-modelos-pas3).

Uso:
    python scripts/build_training_dataset.py

Lê `.scratch/pdf-extraction/saida-nova/resultado_final.csv` (fora do git — dado de aluno real)
e escreve `data/training/pas3_dataset.parquet` (também fora do git: `data/` está no
.gitignore). Regerar é rodar este script de novo; a saída é determinística, sem passo aleatório.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pas_intelligence.training_dataset import (  # noqa: E402
    REQUIRED_SOURCE_COLUMNS,
    build_training_dataset,
)

DEFAULT_SOURCE = Path(".scratch/pdf-extraction/saida-nova/resultado_final.csv")
DEFAULT_OUTPUT = Path("data/training/pas3_dataset.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = pd.read_csv(args.source, usecols=REQUIRED_SOURCE_COLUMNS)
    df = build_training_dataset(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    repetidos = int(df["inscricao_repetida_entre_trienios"].sum())
    ausentes = int(df["etapa_1_ausente"].sum())
    print(f"{len(df)} linhas escritas em {args.output}")
    print(f"etapa_1_ausente: {ausentes} ({ausentes / len(df):.2%})")
    print(f"inscricao_repetida_entre_trienios: {repetidos} ({repetidos / len(df):.2%})")

    bruto_por_trienio = source["trienio"].value_counts()
    incluido_por_trienio = df["trienio"].value_counts()
    descarte = pd.DataFrame(
        {
            "bruto": bruto_por_trienio,
            "incluido": incluido_por_trienio,
        }
    ).fillna(0).astype(int)
    descarte["descartado"] = descarte["bruto"] - descarte["incluido"]
    print("\ninclusão/descarte por triênio (motivo: checksum_fecha == False — ticket 01):")
    print(descarte.sort_index().to_string())


if __name__ == "__main__":
    main()
