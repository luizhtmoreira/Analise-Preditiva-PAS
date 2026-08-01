"""Segunda rodada: refina o esquema S2 e questiona a METRICA do portao.

- S2 com mediana (robusto ao ano anomalo de 2021)
- S2 so na Etapa 2 / so no p2
- quem e o Aluno do residuo maximo
- o portao lido em RMSE e em quadratura com a incerteza do A3 (3 x 5,009)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/luizhenrique/Documents/Vetor PAS/Analise-Preditiva-PAS")
sys.path.insert(0, str(REPO / "src"))
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

exec(open(SCRATCH / "experimento.py").read().split("# --- esquemas")[0])  # reaproveita o setup

RMSE_A3 = 5.009
SIGMA_AF = 3 * RMSE_A3  # a incerteza que o Argumento Final JA carrega, do A3 previsto

idx = par.set_index(["etapa", "ano"])
anos_por_etapa = {e: sorted(par[par.etapa == e].ano.tolist()) for e in (1, 2)}


def agg_loo(etapa, ano, coluna, como="mean", excluir=()):
    outros = [a for a in anos_por_etapa[etapa] if a != ano and a not in excluir]
    vals = [idx.loc[(etapa, a), coluna] for a in outros]
    if not vals:
        return 1.0 if coluna.startswith("rs_") else 0.0
    return float(np.median(vals)) if como == "median" else float(np.mean(vals))


def esquema_stats(como="mean", corrigir_desvio=True, componentes=COMPONENTES, etapas=(1, 2), excluir=()):
    total = 0.0
    for c in COMPONENTES:
        ativo = (c in componentes)
        dm = np.array([agg_loo(e, a, f"dm_{c}", como, excluir) if (ativo and e in etapas) else 0.0
                       for e, a in zip(aluno.etapa, aluno.ano)])
        rs = np.array([agg_loo(e, a, f"rs_{c}", como, excluir) if (ativo and corrigir_desvio and e in etapas) else 1.0
                       for e, a in zip(aluno.etapa, aluno.ano)])
        total = total + PESOS[c] * (aluno[f"x_{c}"].to_numpy() - (aluno[f"emp_m_{c}"].to_numpy() - dm)) / (
            aluno[f"emp_s_{c}"].to_numpy() / rs)
    return pd.Series(total, index=aluno.index) - aluno["A_true"]


esquemas = {
    "S2 media (LOO)": esquema_stats("mean"),
    "S2 mediana (LOO)": esquema_stats("median"),
    "S2 so p2": esquema_stats("mean", componentes=("p2",)),
    "S2 sem corrigir desvio": esquema_stats("mean", corrigir_desvio=False),
    "S2 so Etapa 2": esquema_stats("mean", etapas=(2,)),
    "S2 mediana, sem 2021 no pool": esquema_stats("median", excluir=(2021,)),
    "PISO lingua": aluno["A_piso"] - aluno["A_true"],
}

res = {}
for nome, serie in esquemas.items():
    t = aluno[["trienio", "inscricao", "etapa"]].copy()
    t["v"] = serie.to_numpy()
    p = t.pivot_table(index=["trienio", "inscricao"], columns="etapa", values="v", aggfunc="first").dropna()
    res[nome] = 1 * p[1] + 2 * p[2]
af = pd.DataFrame(res).reset_index()


def linha(s):
    a = s.abs()
    rmse = float(np.sqrt((s ** 2).mean()))
    return {"vies": s.mean(), "|med|": a.mean(), "RMSE": rmse, "p99": a.quantile(.99),
            "p99.9": a.quantile(.999), "max": a.max(),
            "sigma_total": np.sqrt(SIGMA_AF ** 2 + rmse ** 2),
            "+% sigma": 100 * (np.sqrt(SIGMA_AF ** 2 + rmse ** 2) / SIGMA_AF - 1)}


print("\n=== esquemas (Argumento Final = 1*A1 + 2*A2), 34.812 Alunos, 5 trienios ===")
print(pd.DataFrame({k: linha(af[k]) for k in res}).T.to_string(float_format=lambda v: f"{v:8.3f}"))

print("\n=== max por trienio ===")
print(af.groupby("trienio")[list(res)].agg(lambda s: s.abs().max()).to_string(float_format=lambda v: f"{v:7.3f}"))

print(f"\nSigma do Argumento Final so pelo A3 previsto: 3 x {RMSE_A3} = {SIGMA_AF:.3f}")

# quem carrega o maximo do melhor esquema
melhor = "S2 mediana (LOO)"
piores = af.reindex(af[melhor].abs().sort_values(ascending=False).index).head(8)
print(f"\n=== 8 piores em '{melhor}' ===")
notas = aluno.pivot_table(index=["trienio", "inscricao"], columns="etapa",
                          values=["x_p1", "x_p2", "x_red"], aggfunc="first")
notas.columns = [f"{a}_e{b}" for a, b in notas.columns]
print(piores.merge(notas.reset_index(), on=["trienio", "inscricao"])[
    ["trienio", melhor, "PISO lingua", "x_p2_e1", "x_red_e1", "x_p2_e2", "x_red_e2"]
].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

print(f"\n=== quantos Alunos acima de cada limiar, em '{melhor}' ===")
for lim in (3, 4, 5.009, 6):
    n = (af[melhor].abs() > lim).sum()
    print(f"  > {lim:6.3f} : {n:5d} de {len(af)}  ({n/len(af):.4%})")

af.to_csv(SCRATCH / "residuos_r2.csv", index=False)
