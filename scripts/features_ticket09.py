"""Ticket 09 — ganho de cada bloco de feature candidato, sobre a régua do ticket 06.

Mede, isoladamente, o quanto cada bloco de feature listado no ticket melhora (ou não) a linha de
base do ticket 07 — **"linear em (A1, A2) + as 6 legadas"**, RMSE 5,167 em `A3` — e monta o
conjunto final combinando os blocos que pagam o próprio custo.

Roda com:

    .venv/bin/python scripts/features_ticket09.py

Reusa a mesma régua (`validation.avaliar`), o mesmo dataset e a mesma semente do ticket 07, para
que o número de cada bloco seja comparável linha a linha com a tabela de referência.

Privacidade: `perfil_cota` e as 5 booleanas de cota vêm de `resultado_final.csv` (tem PII) só
para o merge por `id_pseudonimo`+`trienio`; nada além de agregado sai daqui.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pas_intelligence.validation import Erro, avaliar, gerar_dobras  # noqa: E402

warnings.filterwarnings("ignore")

SEMENTE = 20260728

DATASET = RAIZ / "data" / "training" / "pas3_dataset.parquet"
RESULTADO_FINAL = RAIZ / ".scratch" / "pdf-extraction" / "saida-nova" / "resultado_final.csv"
CORTES = RAIZ / ".scratch" / "pdf-extraction" / "saida-nova" / "notas_corte.csv"

FEATURES_LEGADAS = ["EB_PAS1", "Red_PAS1", "EB_PAS2", "Red_PAS2", "Cresc_EB", "Cresc_Red"]
BASE = ["a1", "a2", *FEATURES_LEGADAS]

COLUNAS_COTA = ["sistema_negros", "escola_publica", "renda_baixa", "ppi", "pcd"]


# ─── Preparo ────────────────────────────────────────────────────────────────────────────────


def carregar_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    df["EB_PAS1"] = df["eb_pas1"]
    df["Red_PAS1"] = df["red_e1"]
    df["EB_PAS2"] = df["eb_pas2"]
    df["Red_PAS2"] = df["red_e2"]
    df["Cresc_EB"] = df["eb_pas2"] - df["eb_pas1"]
    df["Cresc_Red"] = df["red_e2"] - df["red_e1"]

    # Derivadas de trajetória — as mesmas razões que `meta_scaler` já usa (ticket 09, "grátis").
    df["cresc_eb_pct"] = df["Cresc_EB"].abs() / (df["EB_PAS1"].abs() + 0.01)
    df["cresc_red_pct"] = df["Cresc_Red"].abs() / (df["Red_PAS1"].abs() + 0.01)
    df["sinal_cresc_eb"] = np.sign(df["Cresc_EB"])

    # `trienio` como efeito temporal contínuo, nunca categoria (armadilha do ticket 09).
    df["ano_inicio"] = df["trienio"].str.slice(0, 4).astype(int)

    return df


def anexar_cota(df: pd.DataFrame) -> pd.DataFrame:
    """`perfil_cota` e as 5 booleanas, casadas por `id_pseudonimo`+`trienio`.

    Não estão no `pas3_dataset.parquet` (ticket 05 não as levou) — vêm direto do
    `resultado_final.csv`, com o mesmo hash de pseudonimização do dataset de treino.
    """
    r = pd.read_csv(
        RESULTADO_FINAL,
        usecols=["inscricao", "trienio", "perfil_cota", *COLUNAS_COTA],
    )
    r["id_pseudonimo"] = (
        r["inscricao"].astype(str).map(lambda v: hashlib.sha256(v.encode()).hexdigest()[:16])
    )
    r = r.drop(columns="inscricao").drop_duplicates(subset=["id_pseudonimo", "trienio"])

    antes = len(df)
    df = df.merge(r, on=["id_pseudonimo", "trienio"], how="left")
    sem_match = df["perfil_cota"].isna().sum()
    print(f"cota: {antes - sem_match} de {antes} linhas casadas ({(antes - sem_match) / antes:.1%})")

    df["perfil_cota"] = df["perfil_cota"].fillna("desconhecido")
    for c in COLUNAS_COTA:
        df[c] = df[c].fillna(False).astype(bool)
    return df


# ─── Modelos ────────────────────────────────────────────────────────────────────────────────


class LinearComCategoricas:
    """`LinearRegression` sobre numéricas + one-hot das categóricas.

    `handle_unknown="ignore"` porque a régua treina em triênios passados e testa no seguinte —
    uma categoria nova no teste (curso que só existe no ano seguinte) vira tudo-zero, não erro.
    """

    def __init__(self, numericas: list[str], categoricas: list[str]):
        self.numericas = numericas
        self.categoricas = categoricas

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.linear_model import LinearRegression  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import OneHotEncoder  # type: ignore

        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categoricas),
                ("num", "passthrough", self.numericas),
            ]
        )
        self.pipeline = Pipeline([("pre", pre), ("reg", LinearRegression())])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(X), dtype=float)


def regressao_linear(semente: int):
    from sklearn.linear_model import LinearRegression  # type: ignore

    return LinearRegression()


# ─── Saída ──────────────────────────────────────────────────────────────────────────────────


def linha(nome: str, erro: Erro, referencia: Erro) -> str:
    ganho = (referencia.rmse - erro.rmse) / referencia.rmse * 100
    return (
        f"{nome:<44} {erro.rmse:>8.3f} {erro.mae:>8.3f} {erro.vies:>+8.3f} "
        f"{ganho:>+7.2f}%"
    )


CABECALHO = f"{'bloco':<44} {'RMSE':>8} {'MAE':>8} {'viés':>8} {'ganho':>8}"


def main() -> None:
    df = carregar_dataset()
    df = anexar_cota(df)
    print(f"dataset: {len(df)} linhas, {df['trienio'].nunique()} triênios\n")

    def rodar(nome, fabrica, numericas, categoricas=()):
        if categoricas:
            fab = lambda s: LinearComCategoricas(list(numericas), list(categoricas))
        else:
            fab = fabrica
        return avaliar(
            df, fab, nome=nome, features=[*numericas, *categoricas], semente=SEMENTE
        )

    referencia = rodar("BASE (a1,a2 + 6 legadas)", regressao_linear, BASE)

    blocos = {
        "+ curso": rodar("+ curso", None, BASE, ["curso"]),
        "+ campus + turno": rodar("+ campus + turno", None, BASE, ["campus", "turno"]),
        "+ lingua_e1/e2/e3": rodar(
            "+ lingua_e1/e2/e3", None, BASE, ["lingua_e1", "lingua_e2", "lingua_e3"]
        ),
        "+ perfil_cota + 5 booleanas": rodar(
            "+ perfil_cota + 5 booleanas",
            None,
            BASE,
            ["perfil_cota", *COLUNAS_COTA],
        ),
        "+ ano_inicio (temporal)": rodar(
            "+ ano_inicio (temporal)", regressao_linear, [*BASE, "ano_inicio"]
        ),
        "+ derivadas de trajetória": rodar(
            "+ derivadas de trajetória",
            regressao_linear,
            [*BASE, "cresc_eb_pct", "cresc_red_pct", "sinal_cresc_eb"],
        ),
    }

    print("═══ Ganho de cada bloco, isolado, contra a BASE (ticket 07) ═══")
    print(CABECALHO)
    print(linha("BASE", referencia.agrupado, referencia.agrupado))
    for nome, r in blocos.items():
        print(linha(nome, r.agrupado, referencia.agrupado))

    print("\n═══ Por classe (ticket 14) ═══")
    print(f"{'bloco':<44} {'RMSE maj':>9} {'RMSE min':>9}")
    print(f"{'BASE':<44} {referencia.agrupado_majoritaria.rmse:>9.3f} "
          f"{(referencia.agrupado_minoritaria.rmse if referencia.agrupado_minoritaria else float('nan')):>9.3f}")
    for nome, r in blocos.items():
        minoria = r.agrupado_minoritaria
        print(f"{nome:<44} {r.agrupado_majoritaria.rmse:>9.3f} "
              f"{(f'{minoria.rmse:.3f}' if minoria else '—'):>9}")

    # ─── curso como proxy da Nota de Corte ─────────────────────────────────────────────
    print("\n═══ `curso` é proxy da Nota de Corte? ═══")
    verificar_curso_proxy_corte(df)

    # ─── Conjunto final: combina os blocos que pagaram o próprio custo ────────────────
    print("\n═══ Conjunto final (blocos com ganho > 0,5% combinados) ═══")
    numericas_finais = list(BASE)
    categoricas_finais: list[str] = []
    for nome, r in blocos.items():
        ganho = (referencia.agrupado.rmse - r.agrupado.rmse) / referencia.agrupado.rmse * 100
        if ganho > 0.5:
            print(f"  incluído: {nome} (ganho isolado {ganho:+.2f}%)")
            if nome == "+ ano_inicio (temporal)":
                numericas_finais.append("ano_inicio")
            elif nome == "+ derivadas de trajetória":
                numericas_finais += ["cresc_eb_pct", "cresc_red_pct", "sinal_cresc_eb"]
            elif nome == "+ curso":
                categoricas_finais.append("curso")
            elif nome == "+ campus + turno":
                categoricas_finais += ["campus", "turno"]
            elif nome == "+ lingua_e1/e2/e3":
                categoricas_finais += ["lingua_e1", "lingua_e2", "lingua_e3"]
            elif nome == "+ perfil_cota + 5 booleanas":
                categoricas_finais += ["perfil_cota", *COLUNAS_COTA]
        else:
            print(f"  descartado: {nome} (ganho isolado {ganho:+.2f}%)")

    if categoricas_finais:
        final = rodar("conjunto final", None, numericas_finais, categoricas_finais)
    else:
        final = rodar("conjunto final", regressao_linear, numericas_finais)
    print(f"\n{CABECALHO}")
    print(linha("conjunto final", final.agrupado, referencia.agrupado))

    # ─── Kitchen sink: todos os blocos juntos, para checar se compoem ──────────────────
    print("\n═══ Todos os blocos juntos (checa se blocos fracos compõem) ═══")
    tudo = rodar(
        "tudo junto",
        None,
        [*BASE, "cresc_eb_pct", "cresc_red_pct", "sinal_cresc_eb"],
        ["curso", "campus", "turno", "lingua_e1", "lingua_e2", "lingua_e3",
         "perfil_cota", *COLUNAS_COTA],
    )
    print(CABECALHO)
    print(linha("tudo junto", tudo.agrupado, referencia.agrupado))


def verificar_curso_proxy_corte(df: pd.DataFrame) -> None:
    """`curso` correlaciona com o corte, ou carrega sinal do Aluno além disso?

    Regride `A3` em (`a1`,`a2`) por regressão linear simples, tira o resíduo por curso (a média
    do curso que **não** é explicada pela trajetória do Aluno), e correlaciona esse resíduo com a
    Nota de Corte média do curso. Se a correlação for alta, o que `curso` adiciona sobre a BASE
    é, em boa parte, o corte do curso reaparecendo pela porta dos fundos — não sinal novo sobre o
    Aluno.
    """
    from sklearn.linear_model import LinearRegression  # type: ignore

    if not CORTES.exists():
        print("  notas_corte.csv não encontrado — pulando verificação")
        return

    reg = LinearRegression().fit(df[["a1", "a2"]], df["a3"])
    residuo = df["a3"].to_numpy() - reg.predict(df[["a1", "a2"]])
    tmp = pd.DataFrame({"curso": df["curso"], "campus": df["campus"], "residuo": residuo})
    residuo_por_curso = tmp.groupby("curso")["residuo"].mean()

    c = pd.read_csv(CORTES)
    c = c[c["checksum_fecha"] & ~c["parcial"] & (c["sistema_nome"] == "Universal")]
    corte_por_curso = c.groupby("curso")["nota_corte"].mean()

    comum = residuo_por_curso.index.intersection(corte_por_curso.index)
    x = corte_por_curso.loc[comum]
    y = residuo_por_curso.loc[comum]
    correlacao = np.corrcoef(x, y)[0, 1]
    print(f"  {len(comum)} cursos com corte casado")
    print(f"  correlação(resíduo médio de A3 por curso, corte médio do curso) = {correlacao:.3f}")


if __name__ == "__main__":
    main()
