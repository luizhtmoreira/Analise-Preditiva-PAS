"""Ticket 08 — a janela de dados: os Alunos desde 2018 ainda ajudam?

Mede, dentro da régua do ticket 06 (`avaliar`), se treinar com mais triênios antigos melhora ou
piora a previsão da Etapa 3, e separa três hipóteses para qualquer quebra encontrada: buraco
pandêmico, mudança de fórmula (já descartada pelo ticket 02) ou deriva gradual. Roda com:

    .venv/bin/python scripts/janela_de_dados.py

Modelo usado: `linear em (A1, A2) + as 6 legadas` — o melhor baseline trivial do ticket 07
(RMSE 5,167 em A3, agrupado), para que a curva da janela seja comparável ao Portão 1. Só a
classe majoritária entra na curva (restrição do ticket 06 §2.2: a minoritária não tem série).

Privacidade: só lê `pas3_dataset.parquet` (sem PII) e imprime agregados.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pas_intelligence.validation import (  # noqa: E402
    Erro,
    avaliar,
    gerar_dobras,
)

warnings.filterwarnings("ignore")

SEMENTE = 20260728
DATASET = RAIZ / "data" / "training" / "pas3_dataset.parquet"

FEATURES_LEGADAS = ["EB_PAS1", "Red_PAS1", "EB_PAS2", "Red_PAS2", "Cresc_EB", "Cresc_Red"]
FEATURES = ["a1", "a2", *FEATURES_LEGADAS]

# Mapa de coortes pandêmicas do ticket 02 — quais Etapas de cada triênio caíram em ano
# pandêmico, para que a deriva de distribuição não seja lida como surpresa.
COORTES_PANDEMICAS = {
    "2018/2020": "E3",
    "2019/2021": "E2 + E3",
    "2020/2022": "E1 + E2",
    "2021/2023": "E1",
}


def carregar_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    df["EB_PAS1"] = df["eb_pas1"]
    df["Red_PAS1"] = df["red_e1"]
    df["EB_PAS2"] = df["eb_pas2"]
    df["Red_PAS2"] = df["red_e2"]
    df["Cresc_EB"] = df["eb_pas2"] - df["eb_pas1"]
    df["Cresc_Red"] = df["red_e2"] - df["red_e1"]
    return df


def regressao_linear(semente: int):
    from sklearn.linear_model import LinearRegression  # type: ignore

    return LinearRegression()


# ─── 1. Curva de erro contra número de triênios de treino ─────────────────────────────────────


def curva_por_janela(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE agrupado (classe majoritária) para cada janela de 1 a 8, mais a expansiva (None)."""
    linhas = []
    trienios_disponiveis = sorted(t for t in df["trienio"].unique())
    n_max_treino = len(trienios_disponiveis) - 1  # exclui o lacre

    for janela in (*range(1, n_max_treino), None):
        resultado = avaliar(
            df,
            regressao_linear,
            nome=f"janela={janela}",
            features=FEATURES,
            semente=SEMENTE,
            janela=janela,
        )
        erro = resultado.agrupado_majoritaria
        rotulo = "expansiva (todas)" if janela is None else str(janela)
        linhas.append(
            {"janela": rotulo, "n_treino_max": max(len(d.dobra.trienios_treino) for d in resultado.dobras),
             "n_teste": erro.n, "rmse": erro.rmse, "mae": erro.mae, "vies": erro.vies}
        )
    return pd.DataFrame(linhas)


# ─── 2. Corte por janela × ponderação por idade ────────────────────────────────────────────────


def fabrica_pesos(base: float):
    """Peso decrescente geométrico pela distância (em triênios) até o mais recente do treino.

    `base=1.0` reproduz o treino sem peso (todo mundo pesa 1) — serve de referência dentro da
    própria comparação ponderada, sem precisar de uma segunda chamada a `avaliar`.
    """

    def pesos(treino: pd.DataFrame) -> np.ndarray:
        trienios = sorted(treino["trienio"].unique())
        idade = {t: len(trienios) - 1 - i for i, t in enumerate(trienios)}
        return treino["trienio"].map(idade).map(lambda d: base**d).to_numpy(dtype=float)

    return pesos


def corte_vs_peso(df: pd.DataFrame) -> pd.DataFrame:
    """Mesma pergunta por dois caminhos: truncar o treino em N, ou treinar em tudo com peso.

    Comparação pareada dentro da dobra (restrição do ticket 06 §5): os triênios de teste não
    mudam entre as duas linhas, só o que entra no treino e o peso de cada linha.
    """
    linhas = []

    truncada = avaliar(
        df, regressao_linear, nome="corte em 3", features=FEATURES, semente=SEMENTE, janela=3
    )
    linhas.append({"estrategia": "corte: só os 3 mais recentes", "rmse": truncada.agrupado_majoritaria.rmse})

    expansiva_sem_peso = avaliar(
        df, regressao_linear, nome="tudo, sem peso", features=FEATURES, semente=SEMENTE
    )
    linhas.append({"estrategia": "tudo, sem peso (janela expansiva)",
                    "rmse": expansiva_sem_peso.agrupado_majoritaria.rmse})

    for base in (0.5, 0.7, 0.85):
        ponderada = avaliar(
            df, regressao_linear, nome=f"peso base={base}", features=FEATURES, semente=SEMENTE,
            pesos=fabrica_pesos(base),
        )
        linhas.append({"estrategia": f"tudo, peso geométrico (base={base})",
                        "rmse": ponderada.agrupado_majoritaria.rmse})

    return pd.DataFrame(linhas)


# ─── 3. Deriva de distribuição por triênio ─────────────────────────────────────────────────────


def deriva_por_trienio(df: pd.DataFrame) -> pd.DataFrame:
    """Média/desvio de A1, A2, A3 e a relação (correlação) entre (A1,A2) e A3, por triênio.

    Separa as três hipóteses do ticket: **pandemia** aparece como buraco isolado num triênio do
    meio da série (n baixo, média/desvio fora da vizinhança, correlação preservada); **mudança de
    fórmula** já foi descartada pelo ticket 02; **deriva gradual** aparece como tendência
    monotônica na correlação ou na média ao longo dos 8 triênios, sem buraco isolado.
    """
    linhas = []
    for trienio in sorted(df["trienio"].unique()):
        bloco = df[df["trienio"] == trienio]
        corr = np.corrcoef((bloco["a1"] + bloco["a2"]) / 2, bloco["a3"])[0, 1]
        linhas.append(
            {
                "trienio": trienio,
                "n": len(bloco),
                "media_a1": bloco["a1"].mean(),
                "media_a2": bloco["a2"].mean(),
                "media_a3": bloco["a3"].mean(),
                "desvio_a3": bloco["a3"].std(),
                "corr((a1+a2)/2, a3)": corr,
                "etapas_pandemicas": COORTES_PANDEMICAS.get(trienio, "—"),
            }
        )
    return pd.DataFrame(linhas)


# ─── Saída ──────────────────────────────────────────────────────────────────────────────────


def main() -> None:
    df = carregar_dataset()
    print(f"dataset: {len(df)} linhas, {df['trienio'].nunique()} triênios (o lacre 2023/2025 fica de fora)\n")

    print("═══ 1. Curva de erro × número de triênios de treino (classe majoritária) ═══")
    curva = curva_por_janela(df)
    print(curva.to_string(index=False, formatters={
        "rmse": "{:.3f}".format, "mae": "{:.3f}".format, "vies": "{:+.3f}".format,
    }))

    print("\n═══ 2. Corte por janela × ponderação por idade (mesmos 5 triênios de teste) ═══")
    comparacao = corte_vs_peso(df)
    print(comparacao.to_string(index=False, formatters={"rmse": "{:.3f}".format}))

    print("\n═══ 1b. Série por dobra, janelas 1 / 3 / expansiva (onde a curva melhora) ═══")
    for janela, rotulo in ((1, "janela=1"), (3, "janela=3"), (None, "expansiva")):
        resultado = avaliar(
            df, regressao_linear, nome=rotulo, features=FEATURES, semente=SEMENTE, janela=janela
        )
        serie = " ".join(f"{d.dobra.trienio_teste}:{e.rmse:.3f}"
                          for d, e in zip(resultado.dobras, resultado.serie_majoritaria))
        print(f"    {rotulo:<12} {serie}")

    print("\n═══ 3. Deriva de distribuição por triênio ═══")
    deriva = deriva_por_trienio(df)
    print(deriva.to_string(index=False, formatters={
        "media_a1": "{:.2f}".format, "media_a2": "{:.2f}".format, "media_a3": "{:.2f}".format,
        "desvio_a3": "{:.2f}".format, "corr((a1+a2)/2, a3)": "{:.3f}".format,
    }))

    print("\n═══ Portão 1 (ticket 07 §8) — para conferência ═══")
    print("RMSE A3 agrupado, classe majoritária, precisa bater ≤ 5,038")


if __name__ == "__main__":
    main()
