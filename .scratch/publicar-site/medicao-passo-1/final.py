"""Erro no Argumento Final usando SO os Editais isolados, medido no triênio 2023/2025,
o unico onde temos Etapa 1 e Etapa 2 isoladas mais o gabarito oficial das duas.

Argumento Final = A1 + 2*A2 + 3*A3  ->  o erro de A2 entra dobrado.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".scratch/publicar-site/medicao-passo-1")
from extrair_etapa import registros
from pas_intelligence.argument_calculator import PESO_P1, PESO_P2, PESO_REDACAO
from pas_intelligence.pas_constants import OFFICIAL_STATS

EDITAIS = {
    1: ("data/pdfs/editais-de-etapa/Ed_7_PAS_1_2023_2025_Res_final_tipo_D_redação.pdf", 2023),
    2: ("data/pdfs/editais-de-etapa/Ed_15_PAS_2_2023-2025_Res_final_tipo_D_redacao.pdf", 2024),
}
RMSE_A3, TRIENIO = 5.009, "2023/2025"
LIMIAR = 3 * RMSE_A3

csv = pd.read_csv(".scratch/pdf-extraction/saida-nova/resultado_final.csv", dtype=str)
csv = csv[(csv["trienio"] == TRIENIO) & (csv["checksum_fecha"] == "True")].copy()
for e in (1, 2):
    for c in [f"eb_p1_e{e}", f"eb_p2_e{e}", f"red_e{e}"]:
        csv[c] = pd.to_numeric(csv[c], errors="coerce")


def arg(n, m, dp, peso):
    return ((n - m) / dp) * peso


deltas = {}
for etapa, (pdf, ano) in EDITAIS.items():
    ed, _ = registros(pdf)
    of = OFFICIAL_STATS[(ano, etapa)]
    est = {"m_p1": ed["eb_p1"].mean(), "dp_p1": ed["eb_p1"].std(ddof=0),
           "m_p2": ed["eb_p2"].mean(), "dp_p2": ed["eb_p2"].std(ddof=0),
           "m_red": ed["red"].mean(), "dp_red": ed["red"].std(ddof=0)}

    p1c, p2c, redc, lgc = (f"eb_p1_e{etapa}", f"eb_p2_e{etapa}",
                           f"red_e{etapa}", f"lingua_e{etapa}")
    b = csv.dropna(subset=[p1c, p2c, redc])
    b = b[~((b[p1c] == 0) & (b[p2c] == 0) & (b[redc] == 0))]
    b = b[b[lgc].isin(of.parte_1.keys())]

    m_lg = b[lgc].map(lambda k: of.parte_1[k].media).to_numpy()
    dp_lg = b[lgc].map(lambda k: of.parte_1[k].desvio_padrao).to_numpy()
    p1, p2, red = b[p1c].to_numpy(), b[p2c].to_numpy(), b[redc].to_numpy()

    verdade = (arg(p1, m_lg, dp_lg, PESO_P1) + arg(p2, of.m_p2, of.dp_p2, PESO_P2)
               + arg(red, of.m_red, of.dp_red, PESO_REDACAO))
    estimado = (arg(p1, est["m_p1"], est["dp_p1"], PESO_P1)
                + arg(p2, est["m_p2"], est["dp_p2"], PESO_P2)
                + arg(red, est["m_red"], est["dp_red"], PESO_REDACAO))

    d = pd.Series(estimado - verdade, index=b["inscricao"].to_numpy())
    deltas[etapa] = d
    print(f"Etapa {etapa} ({ano}): n={len(d)}  dA media={d.mean():+.3f}  "
          f"|media|={d.abs().mean():.3f}  p95={d.abs().quantile(.95):.3f}  max={d.abs().max():.3f}")

j = pd.DataFrame({"dA1": deltas[1], "dA2": deltas[2]}).dropna()
j["dArgFinal"] = j["dA1"] + 2 * j["dA2"]
a = j["dArgFinal"].abs()
print(f"\n=== Argumento Final (1*dA1 + 2*dA2), {len(j)} Alunos ===")
print(f"media={j['dArgFinal'].mean():+.3f}  |media|={a.mean():.3f}  p50={a.quantile(.5):.3f}  "
      f"p95={a.quantile(.95):.3f}  p99={a.quantile(.99):.3f}  max={a.max():.3f}")
print(f"acima do limiar {LIMIAR:.2f}: {(a > LIMIAR).mean():.2%}")

corr = (j["dArgFinal"] - j["dArgFinal"].mean()).abs()
print(f"\nse corrigido pelo deslocamento medio ({j['dArgFinal'].mean():+.3f}):")
print(f"  |erro| media={corr.mean():.3f}  p95={corr.quantile(.95):.3f}  max={corr.max():.3f}  "
      f"acima do limiar: {(corr > LIMIAR).mean():.2%}")
