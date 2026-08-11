"""Erro de decisão estratificado pela distância até a Nota de Corte.

    .venv/bin/python scripts/erro_de_decisao_perto_do_corte.py \\
        .scratch/pdf-extraction/saida-nova/resultado_final.csv \\
        --pacote models/pas3

A pergunta que o `5,41%` agregado do ticket 13 não responde: **o sistema serve para quem está
na dúvida?** Dois em cada três Alunos recebem probabilidade saturada (`<1%` ou `>99%`) porque
estão longe do corte — e são eles que carregam a taxa de acerto. Quem está em cima da linha é
quem precisa da resposta e é onde ela é difícil.

Este script **não decide nada e não toca no artefato**. Ele relê o triênio lacrado, que o
ticket 13 já abriu em 2026-07-28, e reparte o mesmo número por faixa de distância. Não chama
`holdout_final_use_uma_vez` de propósito: o lacre é gasto uma vez e já foi: aqui é releitura de
um resultado publicado, não uma segunda abertura.

Duas leituras da "distância", que respondem a perguntas diferentes:

- **Distância real** (`AF real − corte`) — retrospectiva: *"entre os Alunos que de fato estavam
  em cima da linha, quantos o sistema classificou errado?"*
- **Distância prevista** (`AF previsto − corte`) — prospectiva, e é a que vale para o produto:
  *"quando o app mostra um número perto do corte, quanto ele acerta?"* É a única das duas que o
  Aluno pode ver na hora, porque a real só existe depois do resultado.

A unidade é a **Largura de Incerteza** do próprio Aluno (`3 × σ(A3)` da classe dele), não pontos
absolutos — uma distância de 10 pontos é perto ou longe dependendo da largura que o pacote
promete para aquela classe.

Privacidade: só agregados e contagens. Nenhuma linha individual é impressa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.stats import norm  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pas_intelligence.dataset_pas3 import FEATURES_CANONICAS, montar_features  # noqa: E402
from pas_intelligence.training_dataset import load_and_build  # noqa: E402
from pas_intelligence.validation import COLUNA_CLASSE, TRIENIO_LACRADO  # noqa: E402

FAIXAS = [
    (0.0, 0.5, "≤ 0,5 largura"),
    (0.5, 1.0, "0,5 – 1 largura"),
    (1.0, 2.0, "1 – 2 larguras"),
    (2.0, np.inf, "> 2 larguras"),
]
"""Faixas de distância até o corte, em múltiplos da Largura de Incerteza do Aluno."""


def _carregar_booster(pacote: Path):
    import lightgbm as lgb  # type: ignore

    return lgb.Booster(model_file=str(pacote / "modelo_pas3.txt"))


def _cortes_do_lacre(caminho: Path, af_min: float, af_max: float) -> pd.DataFrame:
    """Mesma limpeza do `lacre_ticket13.py::_cortes_do_lacre` — repetida aqui de propósito, para
    que os dois scripts casem linha a linha e a comparação com o `5,41%` seja legítima."""
    cortes = pd.read_csv(caminho)
    cortes = cortes[~cortes["curso"].astype(str).str.contains("SUB JUDICE", case=False, na=False)]
    cortes = cortes[
        (cortes["trienio"] == TRIENIO_LACRADO)
        & (cortes["sistema_nome"] == "Universal")
        & (cortes["parcial"] == False)  # noqa: E712
        & cortes["nota_corte"].between(af_min, af_max)
    ]
    return (
        cortes.groupby(["campus", "curso", "turno"], as_index=False)["nota_corte"]
        .min()
        .rename(columns={"nota_corte": "corte"})
    )


def _tabela(df: pd.DataFrame, coluna_distancia: str, titulo: str) -> None:
    print(f"\n─── {titulo} ───")
    print(
        f"{'faixa':>18} {'n':>7} {'% base':>7} {'erro dec.':>10} "
        f"{'chutar maioria':>15} {'Brier':>7} {'passou':>7}"
    )
    distancia_relativa = (df[coluna_distancia].abs() / df["largura"]).to_numpy()
    for minimo, maximo, rotulo in FAIXAS:
        dentro = (distancia_relativa >= minimo) & (distancia_relativa < maximo)
        faixa = df[dentro]
        if faixa.empty:
            continue
        passou = faixa["passou"].to_numpy()
        errou = faixa["errou_decisao"].to_numpy()
        # "Chutar maioria": o acerto de um sistema burro que responde sempre o mais comum
        # *daquela faixa*. É o piso honesto contra o qual o erro de decisão precisa ser lido.
        chutar_maioria = max(passou.mean(), 1 - passou.mean())
        brier = float(np.mean((faixa["prob"].to_numpy() - passou.astype(float)) ** 2))
        print(
            f"{rotulo:>18} {len(faixa):>7} {len(faixa) / len(df):>6.1%} "
            f"{errou.mean():>9.2%} {1 - chutar_maioria:>14.2%} "
            f"{brier:>7.4f} {passou.mean():>6.1%}"
        )
    print(
        f"{'TODOS':>18} {len(df):>7} {1.0:>6.1%} "
        f"{df['errou_decisao'].mean():>9.2%} "
        f"{1 - max(df['passou'].mean(), 1 - df['passou'].mean()):>14.2%} "
        f"{float(np.mean((df['prob'] - df['passou'].astype(float)) ** 2)):>7.4f} "
        f"{df['passou'].mean():>6.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--pacote", type=Path, default=RAIZ / "models/pas3")
    parser.add_argument(
        "--cortes",
        type=Path,
        default=RAIZ / ".scratch/pdf-extraction/saida-nova/notas_corte.csv",
    )
    args = parser.parse_args()

    manifesto = json.loads((args.pacote / "manifest.json").read_text(encoding="utf-8"))
    modelo = manifesto["modelos"][0]
    if modelo["treinado_excluindo_trienio"] != TRIENIO_LACRADO:
        raise SystemExit(
            f"O pacote foi treinado excluindo {modelo['treinado_excluindo_trienio']}, não "
            f"{TRIENIO_LACRADO}. Medir com ele seria medir memória, não previsão."
        )
    booster = _carregar_booster(args.pacote)

    canonico = load_and_build(args.csv)
    df = montar_features(canonico)
    df["a1_real"] = canonico["a1"].to_numpy()

    lacre = df[df["trienio"] == TRIENIO_LACRADO].copy()
    lacre["previsto_a3"] = booster.predict(lacre[FEATURES_CANONICAS])
    lacre["af_novo"] = lacre["a1_real"] + 2 * lacre["a2"] + 3 * lacre["previsto_a3"]
    lacre["af_real"] = lacre["argumento_final"]

    incerteza = manifesto["incerteza"]["sigma_por_classe"]
    lacre["largura"] = 3 * np.where(
        lacre[COLUNA_CLASSE].to_numpy(), incerteza["sem_etapa_1"], incerteza["com_etapa_1"]
    )

    cortes = _cortes_do_lacre(
        args.cortes, float(lacre["af_real"].min()), float(lacre["af_real"].max())
    )
    casado = lacre.merge(cortes, on=["campus", "curso", "turno"], how="left")
    com_corte = casado[casado["corte"].notna()].copy()

    com_corte["prob"] = 1 - norm.cdf(
        com_corte["corte"], loc=com_corte["af_novo"], scale=com_corte["largura"]
    )
    com_corte["passou"] = com_corte["af_real"] >= com_corte["corte"]
    com_corte["errou_decisao"] = (com_corte["prob"] >= 0.5) != com_corte["passou"]
    com_corte["dist_real"] = com_corte["af_real"] - com_corte["corte"]
    com_corte["dist_prevista"] = com_corte["af_novo"] - com_corte["corte"]

    print(f"═══ Erro de decisão por distância do corte — lacre {TRIENIO_LACRADO} ═══")
    print(f"Alunos com corte casado: {len(com_corte)} de {len(casado)}")
    print(
        f"Largura de Incerteza (Argumento Final): "
        f"{3 * incerteza['com_etapa_1']:.2f} com Etapa 1 · "
        f"{3 * incerteza['sem_etapa_1']:.2f} sem"
    )
    print(
        "\n'erro dec.' = em que fração o sistema disse a coisa errada sobre passar.\n"
        "'chutar maioria' = o erro de um sistema burro que responde sempre o mais comum daquela\n"
        "faixa. A diferença entre as duas colunas é o que o modelo de fato acrescenta."
    )

    _tabela(
        com_corte,
        "dist_real",
        "Por distância REAL até o corte (retrospectiva: quem estava mesmo na linha)",
    )
    _tabela(
        com_corte,
        "dist_prevista",
        "Por distância PREVISTA até o corte (o que o app mostra na hora)",
    )

    saturada = (com_corte["prob"] < 0.01) | (com_corte["prob"] > 0.99)
    print(f"\nprobabilidades saturadas (<1% ou >99%): {saturada.mean():.1%}")
    print(f"erro de decisão só nas saturadas:        {com_corte.loc[saturada, 'errou_decisao'].mean():.2%}")
    print(f"erro de decisão só nas NÃO saturadas:    {com_corte.loc[~saturada, 'errou_decisao'].mean():.2%}")


if __name__ == "__main__":
    main()
