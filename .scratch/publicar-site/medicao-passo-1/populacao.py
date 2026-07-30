"""Qual recorte da lista do Edital de Etapa reproduz a media oficial?

O Edital de Etapa 1 de 2022 tem 18.381 candidatos; a media oficial (20.406) fica
ACIMA da media da lista inteira (19.059), logo o Cebraspe usa um recorte menor.
Testa candidatos a recorte contra o gabarito.
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".scratch/publicar-site/medicao-passo-1")
from extrair_etapa import registros
from pas_intelligence.pas_constants import OFFICIAL_STATS

PDF = "data/pdfs/editais-de-etapa/ED_8_PAS_1_2022-2024_Retificação_Res_Final_tipo_D_redação.pdf"
of = OFFICIAL_STATS[(2022, 1)]

df, _ = registros(PDF)
print(f"lista completa: {len(df)} candidatos")
print(f"gabarito oficial: m_p2={of.m_p2:.3f} dp_p2={of.dp_p2:.3f} "
      f"m_red={of.m_red:.3f} dp_red={of.dp_red:.3f}\n")

soma = df["eb_p1"] + df["eb_p2"]
recortes = {
    "lista inteira": df,
    "tirando faltoso (P1=P2=0 e red=0)": df[~((soma == 0) & (df["red"] == 0))],
    "tirando P1=P2=0": df[soma != 0],
    "tirando red=0": df[df["red"] != 0],
    "tirando red=0 E P1=P2=0": df[(df["red"] != 0) & (soma != 0)],
    "tirando tipo_d=0": df[df["tipo_d"] != 0],
    "so quem tem tudo > 0": df[(df["red"] > 0) & (df["eb_p2"] > 0) & (df["eb_p1"] > 0)],
}

linhas = []
for nome, b in recortes.items():
    if b.empty:
        continue
    linhas.append({
        "recorte": nome, "n": len(b), "%": len(b) / len(df),
        "m_p2": b["eb_p2"].mean(), "d_m_p2": b["eb_p2"].mean() - of.m_p2,
        "dp_p2": b["eb_p2"].std(ddof=0), "d_dp_p2": b["eb_p2"].std(ddof=0) - of.dp_p2,
        "m_red": b["red"].mean(), "d_m_red": b["red"].mean() - of.m_red,
        "dp_red": b["red"].std(ddof=0), "d_dp_red": b["red"].std(ddof=0) - of.dp_red,
    })

r = pd.DataFrame(linhas)
r["erro_total"] = r[["d_m_p2", "d_dp_p2", "d_m_red", "d_dp_red"]].abs().sum(axis=1)
pd.set_option("display.width", 220)
print(r.round(3).to_string(index=False))
