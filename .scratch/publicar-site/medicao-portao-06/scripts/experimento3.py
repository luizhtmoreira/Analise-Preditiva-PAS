"""Terceira rodada: ONDE mora o residuo que sobra, e em que direcao ele erra.

Sob S1 (constante) o ticket 06 mostrou que o residuo maximo estava no TOPO -- exatamente
onde o produto precisa de precisao. Sob S2 (afim, nos stats) isso muda? E o erro e otimista
(perigoso) ou pessimista (seguro)?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
exec(open(SCRATCH / "experimento.py").read().split("# --- esquemas")[0])

idx = par.set_index(["etapa", "ano"])
anos_por_etapa = {e: sorted(par[par.etapa == e].ano.tolist()) for e in (1, 2)}


def agg_loo(etapa, ano, coluna, como):
    outros = [a for a in anos_por_etapa[etapa] if a != ano]
    vals = [idx.loc[(etapa, a), coluna] for a in outros]
    return float(np.median(vals)) if como == "median" else float(np.mean(vals))


def esquema(como, corrigir_desvio=True):
    total = 0.0
    for c in COMPONENTES:
        dm = np.array([agg_loo(e, a, f"dm_{c}", como) for e, a in zip(aluno.etapa, aluno.ano)])
        rs = np.array([agg_loo(e, a, f"rs_{c}", como) for e, a in zip(aluno.etapa, aluno.ano)]) if corrigir_desvio else 1.0
        total = total + PESOS[c] * (aluno[f"x_{c}"].to_numpy() - (aluno[f"emp_m_{c}"].to_numpy() - dm)) / (
            aluno[f"emp_s_{c}"].to_numpy() / rs)
    return pd.Series(total, index=aluno.index) - aluno["A_true"]


aluno["e_S2"] = esquema("median")
aluno["e_S2mean"] = esquema("mean")
aluno["e_S1"] = aluno["A_emp"] - aluno["A_true"]
d1 = par[par.etapa == 1]["dA"].mean(); d2 = par[par.etapa == 2]["dA"].mean()
aluno["e_S1"] = aluno["e_S1"] - aluno["etapa"].map({1: d1, 2: d2})

t = aluno[["trienio", "inscricao", "etapa", "e_S2", "e_S2mean", "e_S1", "A_true"]]
p = t.pivot_table(index=["trienio", "inscricao"], columns="etapa",
                  values=["e_S2", "e_S2mean", "e_S1", "A_true"], aggfunc="first").dropna()
af = pd.DataFrame({
    "S1_const": 1 * p[("e_S1", 1)] + 2 * p[("e_S1", 2)],
    "S2_mediana": 1 * p[("e_S2", 1)] + 2 * p[("e_S2", 2)],
    "S2_media": 1 * p[("e_S2mean", 1)] + 2 * p[("e_S2mean", 2)],
    "AF_verdadeiro": 1 * p[("A_true", 1)] + 2 * p[("A_true", 2)],
}).reset_index()

af["faixa"] = pd.qcut(af["AF_verdadeiro"], [0, .25, .50, .75, .90, .99, 1.0],
                      labels=["0-25%", "25-50%", "50-75%", "75-90%", "90-99%", "top 1%"])

print("\n=== residuo por faixa do Argumento Final VERDADEIRO (1*A1+2*A2) ===")
for col in ("S1_const", "S2_mediana"):
    g = af.groupby("faixa", observed=True)[col]
    print(f"\n-- {col} --")
    print(pd.DataFrame({
        "n": g.size(), "vies": g.mean(), "|med|": g.apply(lambda s: s.abs().mean()),
        "p99": g.apply(lambda s: s.abs().quantile(.99)), "max": g.apply(lambda s: s.abs().max()),
        "otimista>0": g.apply(lambda s: f"{(s > 0).mean():.1%}"),
    }).to_string(float_format=lambda v: f"{v:7.3f}"))

print("\n=== so o TOPO (top 1% do Argumento Final), por trienio ===")
topo = af[af.faixa == "top 1%"]
print(topo.groupby("trienio")[["S1_const", "S2_mediana", "S2_media"]].agg(
    ["mean", lambda s: s.abs().max()]).to_string(float_format=lambda v: f"{v:7.3f}"))

print("\n=== direcao do erro (positivo = OTIMISTA, diz ao Aluno que ele esta melhor) ===")
for col in ("S1_const", "S2_mediana", "S2_media"):
    s = af[col]
    print(f"  {col:12s} vies={s.mean():+7.3f}  max otimista={s.max():7.3f}  max pessimista={s.min():8.3f}")

print("\n=== o portao, nas duas leituras ===")
for col in ("S1_const", "S2_media", "S2_mediana"):
    s = af[col].abs()
    print(f"  {col:12s} max={s.max():7.3f}  p99.9={s.quantile(.999):7.3f}  RMSE={np.sqrt((af[col]**2).mean()):7.3f}"
          f"   {'APROVA' if s.max() < 5.009 else 'reprova'}")
af.to_csv(SCRATCH / "residuos_r3.csv", index=False)
