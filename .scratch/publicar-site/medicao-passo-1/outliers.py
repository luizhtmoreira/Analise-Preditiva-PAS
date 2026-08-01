"""Quanto do CSV esta corrompido, e as flags de qualidade pegam?"""
import sys
import pandas as pd

CSV = ".scratch/pdf-extraction/saida-nova/resultado_final.csv"
df = pd.read_csv(CSV, dtype=str)

NOTAS = ["eb_p1_e1", "eb_p2_e1", "red_e1", "eb_p1_e2", "eb_p2_e2", "red_e2",
         "eb_p1_e3", "eb_p2_e3", "red_e3"]
num = df[NOTAS].apply(pd.to_numeric, errors="coerce")

print("=== percentis ===")
print(num.describe(percentiles=[0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]).T.to_string())

print("\n=== flags de qualidade ===")
for c in ["campos_formato_invalido", "checksum_fecha", "checksum_delta", "classificacao_buracos"]:
    vc = df[c].fillna("<vazio>").value_counts()
    print(f"{c}: {dict(list(vc.items())[:5])}  (distintos={len(vc)})")

# Faixa plausivel grosseira, so para contar contaminacao
suspeito = (
    (num[["eb_p1_e1", "eb_p1_e2", "eb_p1_e3"]].abs() > 25).any(axis=1)
    | (num[["eb_p2_e1", "eb_p2_e2", "eb_p2_e3"]].abs() > 150).any(axis=1)
    | (num[["red_e1", "red_e2", "red_e3"]].abs() > 12).any(axis=1)
)
print(f"\nlinhas com nota fora de faixa plausivel: {suspeito.sum()} ({suspeito.mean():.2%})")

print("\ncruzamento com checksum_fecha:")
print(pd.crosstab(suspeito, df["checksum_fecha"].fillna("<vazio>")).to_string())

print("\ncruzamento com campos_formato_invalido (vazio = sem defeito):")
tem_fmt = df["campos_formato_invalido"].notna() & (df["campos_formato_invalido"] != "")
print(pd.crosstab(suspeito, tem_fmt).to_string())

print("\nsuspeitos por trienio:")
print(df.assign(s=suspeito).groupby("trienio")["s"].agg(["sum", "mean", "size"]).to_string())

print("\n=== exemplos de linha suspeita ===")
print(df.loc[suspeito, ["nome", "trienio"] + NOTAS + ["checksum_fecha", "checksum_delta"]].head(8).to_string())
