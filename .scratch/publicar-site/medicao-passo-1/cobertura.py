"""A retificacao de 2022 cobre todos os candidatos, ou so uma parte?

Teste: todo Aluno do triênio 2022/2024 que tem Etapa 1 no resultado_final.csv
precisa estar na lista do Edital de Etapa 1 de 2022. Se faltar muita gente,
o Edital e parcial e a media dele nao vale como estimativa.

Depois: traduz o erro medido em pontos de Argumento Final.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".scratch/publicar-site/medicao-passo-1")
from extrair_etapa import registros
from pas_intelligence.argument_calculator import PESO_P1, PESO_P2, PESO_REDACAO
from pas_intelligence.pas_constants import OFFICIAL_STATS

PDF = "data/pdfs/editais-de-etapa/ED_8_PAS_1_2022-2024_Retificação_Res_Final_tipo_D_redação.pdf"
CSV = ".scratch/pdf-extraction/saida-nova/resultado_final.csv"

edital, _ = registros(PDF)
edital["inscricao"] = edital["inscricao"].str.replace(r"\s+", "", regex=True)

csv = pd.read_csv(CSV, dtype=str)
csv = csv[(csv["trienio"] == "2022/2024") & (csv["checksum_fecha"] == "True")].copy()
for c in ["eb_p1_e1", "eb_p2_e1", "red_e1"]:
    csv[c] = pd.to_numeric(csv[c], errors="coerce")
com_e1 = csv[~((csv["eb_p1_e1"] == 0) & (csv["eb_p2_e1"] == 0) & (csv["red_e1"] == 0))]

no_edital = com_e1["inscricao"].isin(set(edital["inscricao"]))
print(f"Alunos do trienio 2022/2024 com Etapa 1 no CSV : {len(com_e1)}")
print(f"   destes, presentes no Edital de Etapa 1 2022 : {no_edital.sum()} ({no_edital.mean():.2%})")
print(f"Candidatos no Edital de Etapa 1 2022           : {len(edital)}")

# As notas batem para quem esta nos dois lados? (confirma que e a mesma prova)
j = com_e1[no_edital].merge(edital, on="inscricao", how="inner")
for a, b in [("eb_p1_e1", "eb_p1"), ("eb_p2_e1", "eb_p2"), ("red_e1", "red")]:
    d = (j[a] - j[b]).abs()
    print(f"   {a:10s} vs {b:6s}: iguais em {(d < 0.005).mean():.2%}  (|dif| media {d.mean():.4f})")

# ---- erro em pontos de Argumento Final ----
of = OFFICIAL_STATS[(2022, 1)]
est = {"m_p2": edital["eb_p2"].mean(), "dp_p2": edital["eb_p2"].std(ddof=0),
       "m_red": edital["red"].mean(), "dp_red": edital["red"].std(ddof=0),
       "m_p1": edital["eb_p1"].mean(), "dp_p1": edital["eb_p1"].std(ddof=0)}

b = com_e1[com_e1["lingua_e1"].isin(of.parte_1.keys())]
m_lg = b["lingua_e1"].map(lambda k: of.parte_1[k].media).to_numpy()
dp_lg = b["lingua_e1"].map(lambda k: of.parte_1[k].desvio_padrao).to_numpy()
p1, p2, red = b["eb_p1_e1"].to_numpy(), b["eb_p2_e1"].to_numpy(), b["red_e1"].to_numpy()


def arg(n, m, dp, peso):
    return ((n - m) / dp) * peso


verdade = (arg(p1, m_lg, dp_lg, PESO_P1) + arg(p2, of.m_p2, of.dp_p2, PESO_P2)
           + arg(red, of.m_red, of.dp_red, PESO_REDACAO))
estimado = (arg(p1, est["m_p1"], est["dp_p1"], PESO_P1)
            + arg(p2, est["m_p2"], est["dp_p2"], PESO_P2)
            + arg(red, est["m_red"], est["dp_red"], PESO_REDACAO))
d = estimado - verdade

print(f"\n=== Erro em A1, com as estatisticas do Edital isolado ({len(b)} Alunos) ===")
print(f"media={d.mean():+.3f}  |media|={np.abs(d).mean():.3f}  "
      f"p95|.|={np.percentile(np.abs(d),95):.3f}  max|.|={np.abs(d).max():.3f}")
print(f"No Argumento Final, A1 entra com peso 1  ->  os mesmos numeros.")
print(f"Limiar (3 x RMSE 5.009) = 15.03  ->  acima dele: {(np.abs(d) > 15.03).mean():.3%}")

comp = pd.DataFrame({
    "fonte": ["Edital isolado (18.381)", "Sobreviventes do CSV (8.499)"],
    "m_p2": [est["m_p2"], b["eb_p2_e1"].mean()],
    "dp_p2": [est["dp_p2"], b["eb_p2_e1"].std(ddof=0)],
    "m_red": [est["m_red"], b["red_e1"].mean()],
    "dp_red": [est["dp_red"], b["red_e1"].std(ddof=0)],
})
comp["|erro| m_p2"] = (comp["m_p2"] - of.m_p2).abs()
comp["|erro| dp_p2"] = (comp["dp_p2"] - of.dp_p2).abs()
print(f"\n=== Qual fonte chega mais perto do oficial? (m_p2={of.m_p2}, dp_p2={of.dp_p2}) ===")
print(comp.round(3).to_string(index=False))
