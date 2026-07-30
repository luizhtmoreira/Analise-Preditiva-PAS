"""Decompoe o residuo do ticket 06 e testa esquemas alternativos de correcao.

Pergunta: o portao reprovou com residuo max 5,751 (limiar 5,009). Quanto disso e:
  (a) degrau sistematico ja corrigido,
  (b) variacao ano a ano do degrau (irredutivel com media global),
  (c) MISMATCH DE DESVIO -- a correcao constante nao mexe no desvio, e o erro de z-score
      cresce linear com a distancia a media, entao o topo carrega o residuo,
  (d) lingua misturada (irredutivel, ruido puro).

Esquemas testados (todos LOO -- leave-one-year-out -- exceto onde marcado):
  S0  bruto
  S1  deslocamento constante por Etapa, media global (o do ticket 06)
  S1L S1 com leave-one-year-out
  S2  correcao AFIM nos stats: corrige media E desvio de cada componente, LOO
  S2m S2 so na media (sem tocar no desvio), LOO
  S3  deslocamento constante do ano mais recente disponivel (LOO)
  PISO piso irredutivel: stats oficiais MISTURADAS por lingua (so o ruido de lingua)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/luizhenrique/Documents/Vetor PAS/Analise-Preditiva-PAS")
sys.path.insert(0, str(REPO / "src"))
SCRATCH = Path(__file__).parent

from pas_intelligence.argument_calculator import PESO_P1, PESO_P2, PESO_REDACAO  # noqa: E402
from pas_intelligence.training_dataset import (  # noqa: E402
    REQUIRED_SOURCE_COLUMNS, _STATS_POR_ANO_ETAPA_LINGUA, anos_do_trienio, etapa_1_ausente,
)
from pas_intelligence.pas_constants import OFFICIAL_STATS  # noqa: E402

CSV = REPO / ".scratch" / "pdf-extraction" / "saida-nova" / "resultado_final.csv"
LIMIAR = 5.009
COMPONENTES = ("p1", "p2", "red")
PESOS = {"p1": PESO_P1, "p2": PESO_P2, "red": PESO_REDACAO}

# --- stats empiricos (Edital isolado), do cache ------------------------------------------
emp_raw = json.loads((SCRATCH / "stats_empiricos.json").read_text())
EMP = {}
for k, v in emp_raw.items():
    ano, etapa = (int(x) for x in k.split("-"))
    EMP[(ano, etapa)] = {
        "p1": (v["m_p1"], v["dp_p1"]), "p2": (v["m_p2"], v["dp_p2"]), "red": (v["m_red"], v["dp_red"]),
        "n": v["n"],
    }

# --- dados -------------------------------------------------------------------------------
df = pd.read_csv(CSV, usecols=REQUIRED_SOURCE_COLUMNS)
df = df[df["checksum_fecha"] == True].copy()  # noqa: E712
anos = df["trienio"].map(anos_do_trienio)
df["_ano_e1"] = anos.map(lambda t: t[0])
df["_ano_e2"] = anos.map(lambda t: t[1])


def stats_oficiais(ano, etapa, lingua):
    h = _STATS_POR_ANO_ETAPA_LINGUA[(ano, etapa, lingua)]
    return {"p1": (h.mean_p1, h.std_p1), "p2": (h.mean_p2, h.std_p2), "red": (h.mean_red, h.std_red)}


def oficial_misturado(ano, etapa, shares):
    """Media/desvio oficiais da Parte 1 na forma MISTURADA (mistura de 3 normais, ponderada
    pelo share real de lingua daquele (ano, Etapa)). p2/red nao dependem de lingua."""
    es = OFFICIAL_STATS[(ano, etapa)]
    m = sum(shares[l] * es.parte_1[l].media for l in shares)
    var = sum(shares[l] * (es.parte_1[l].desvio_padrao ** 2 + es.parte_1[l].media ** 2) for l in shares) - m ** 2
    return (m, float(np.sqrt(var)))


# --- monta as tabelas por (ano, etapa) ---------------------------------------------------
linhas = []
for etapa in (1, 2):
    col = lambda c: f"{c}_e{etapa}" if c != "p1" else f"eb_p1_e{etapa}"
    p1c, p2c, redc = f"eb_p1_e{etapa}", f"eb_p2_e{etapa}", f"red_e{etapa}"
    lc, ac = f"lingua_e{etapa}", f"_ano_e{etapa}"
    presente = ~etapa_1_ausente(df[p1c], df[p2c], df[redc])
    sub = df[presente & df[ac].isin([a for (a, e) in EMP if e == etapa])].copy()
    for ano, bloco in sub.groupby(ac):
        shares = bloco[lc].value_counts(normalize=True).to_dict()
        of_mix = oficial_misturado(int(ano), etapa, shares)
        for _, r in bloco.iterrows():
            linhas.append({
                "trienio": r["trienio"], "inscricao": r["inscricao"], "etapa": etapa, "ano": int(ano),
                "lingua": r[lc], "x_p1": r[p1c], "x_p2": r[p2c], "x_red": r[redc],
                "of_mix_p1": of_mix,
            })
aluno = pd.DataFrame(linhas)
print(f"linhas aluno-etapa: {len(aluno)}")

# medias/desvios oficiais por lingua, vetorizado
for c in COMPONENTES:
    aluno[f"of_m_{c}"] = [stats_oficiais(a, e, l)[c][0] for a, e, l in zip(aluno.ano, aluno.etapa, aluno.lingua)]
    aluno[f"of_s_{c}"] = [stats_oficiais(a, e, l)[c][1] for a, e, l in zip(aluno.ano, aluno.etapa, aluno.lingua)]
    aluno[f"emp_m_{c}"] = [EMP[(a, e)][c][0] for a, e in zip(aluno.ano, aluno.etapa)]
    aluno[f"emp_s_{c}"] = [EMP[(a, e)][c][1] for a, e in zip(aluno.ano, aluno.etapa)]
# oficial misturado (p1 muda; p2/red iguais ao oficial por lingua)
aluno["ofmix_m_p1"] = [t[0] for t in aluno.of_mix_p1]
aluno["ofmix_s_p1"] = [t[1] for t in aluno.of_mix_p1]
aluno.drop(columns=["of_mix_p1"], inplace=True)


def argumento(prefix_m, prefix_s, df_=None, override=None):
    d = aluno if df_ is None else df_
    total = 0.0
    for c in COMPONENTES:
        m = override[c][0] if override and c in override else d[f"{prefix_m}_{c}"]
        s = override[c][1] if override and c in override else d[f"{prefix_s}_{c}"]
        total = total + PESOS[c] * (d[f"x_{c}"] - m) / s
    return total


aluno["A_true"] = argumento("of_m", "of_s")
aluno["A_emp"] = argumento("emp_m", "emp_s")
# PISO: oficial, mas com a Parte 1 misturada (unico erro = nao saber a lingua)
piso_m = aluno[["ofmix_m_p1", "of_m_p2", "of_m_red"]].rename(
    columns={"ofmix_m_p1": "piso_m_p1", "of_m_p2": "piso_m_p2", "of_m_red": "piso_m_red"})
piso_s = aluno[["ofmix_s_p1", "of_s_p2", "of_s_red"]].rename(
    columns={"ofmix_s_p1": "piso_s_p1", "of_s_p2": "piso_s_p2", "of_s_red": "piso_s_red"})
aluno = pd.concat([aluno, piso_m, piso_s], axis=1)
aluno["A_piso"] = argumento("piso_m", "piso_s")

# --- parametros de correcao por (etapa, componente, ano) ---------------------------------
# delta de media e razao de desvio, medidos contra o oficial MISTURADO (que e o alvo honesto:
# a Turma viva nunca vai saber a lingua de ninguem)
par = (aluno.groupby(["etapa", "ano"])
       .apply(lambda g: pd.Series({
           **{f"dm_{c}": g[f"emp_m_{c}"].iloc[0] - (g["ofmix_m_p1"].iloc[0] if c == "p1" else g[f"of_m_{c}"].iloc[0])
              for c in COMPONENTES},
           **{f"rs_{c}": g[f"emp_s_{c}"].iloc[0] / (g["ofmix_s_p1"].iloc[0] if c == "p1" else g[f"of_s_{c}"].iloc[0])
              for c in COMPONENTES},
           "dA": (g["A_emp"] - g["A_true"]).mean(),
       }), include_groups=False)
       .reset_index())
print("\n=== parametros por (etapa, ano) ===")
print(par.round(4).to_string(index=False))

anos_por_etapa = {e: sorted(par[par.etapa == e].ano.tolist()) for e in (1, 2)}

# --- esquemas ----------------------------------------------------------------------------
idx = par.set_index(["etapa", "ano"])


def loo(etapa, ano, coluna, agg="mean", janela=None):
    outros = [a for a in anos_por_etapa[etapa] if a != ano]
    if janela == "recente":
        outros = [max(outros)] if outros else []
    elif isinstance(janela, int):
        outros = sorted(outros)[-janela:]
    vals = [idx.loc[(etapa, a), coluna] for a in outros]
    return float(np.mean(vals)) if agg == "mean" else float(np.median(vals))


esquemas = {}
esquemas["S0 bruto"] = aluno["A_emp"] - aluno["A_true"]

# S1 global (o do ticket): deslocamento = media de TODOS os anos, inclusive o proprio
desl_global = {e: par[par.etapa == e]["dA"].mean() for e in (1, 2)}
esquemas["S1 const global (ticket)"] = aluno["A_emp"] - aluno["A_true"] - aluno["etapa"].map(desl_global)

# S1L: leave-one-year-out
d_loo = np.array([loo(e, a, "dA") for e, a in zip(aluno.etapa, aluno.ano)])
esquemas["S1L const LOO"] = aluno["A_emp"] - aluno["A_true"] - d_loo

# S3: deslocamento do ano mais recente disponivel (LOO)
d_rec = np.array([loo(e, a, "dA", janela="recente") for e, a in zip(aluno.etapa, aluno.ano)])
esquemas["S3 const ano+recente LOO"] = aluno["A_emp"] - aluno["A_true"] - d_rec

# S2 / S2m: correcao nos STATS
for nome, corrigir_desvio in (("S2m stats so media LOO", False), ("S2 stats media+desvio LOO", True)):
    total = 0.0
    for c in COMPONENTES:
        dm = np.array([loo(e, a, f"dm_{c}") for e, a in zip(aluno.etapa, aluno.ano)])
        rs = np.array([loo(e, a, f"rs_{c}") for e, a in zip(aluno.etapa, aluno.ano)]) if corrigir_desvio else 1.0
        m_corr = aluno[f"emp_m_{c}"].to_numpy() - dm
        s_corr = aluno[f"emp_s_{c}"].to_numpy() / rs
        total = total + PESOS[c] * (aluno[f"x_{c}"].to_numpy() - m_corr) / s_corr
    esquemas[nome] = pd.Series(total, index=aluno.index) - aluno["A_true"]

esquemas["PISO lingua (irredutivel)"] = aluno["A_piso"] - aluno["A_true"]

# --- agrega para Argumento Final: 1*e1 + 2*e2 --------------------------------------------
res = {}
for nome, serie in esquemas.items():
    t = aluno[["trienio", "inscricao", "etapa"]].copy()
    t["v"] = serie.to_numpy()
    p = t.pivot_table(index=["trienio", "inscricao"], columns="etapa", values="v", aggfunc="first").dropna()
    res[nome] = (1 * p[1] + 2 * p[2])
af = pd.DataFrame(res).reset_index()
print(f"\nAlunos com as duas Etapas: {len(af)}  | trienios: {sorted(af.trienio.unique())}")


def resumo(s):
    a = s.abs()
    return {"media": s.mean(), "|med|": a.mean(), "p95": a.quantile(.95), "p99": a.quantile(.99),
            "p99.9": a.quantile(.999), "max": a.max(), f">{LIMIAR}": f"{(a > LIMIAR).mean():.4%}"}


print("\n=== TODOS os trienios juntos ===")
print(pd.DataFrame({k: resumo(af[k]) for k in res}).T.to_string(float_format=lambda v: f"{v:8.3f}"))

print("\n=== max |residuo| por trienio (o que o portao le) ===")
tab = af.groupby("trienio")[list(res)].agg(lambda s: s.abs().max())
print(tab.to_string(float_format=lambda v: f"{v:7.3f}"))
print("\nPORTAO (max global < 5.009):")
for k in res:
    m = af[k].abs().max()
    print(f"  {k:30s} max={m:7.3f}  {'APROVA' if m < LIMIAR else 'reprova'}")

print("\n=== p99.9 por trienio ===")
print(af.groupby("trienio")[list(res)].agg(lambda s: s.abs().quantile(.999)).to_string(float_format=lambda v: f"{v:7.3f}"))

af.to_csv(SCRATCH / "residuos_por_esquema.csv", index=False)
par.to_csv(SCRATCH / "parametros_por_ano.csv", index=False)
print(f"\nsalvo: {SCRATCH/'residuos_por_esquema.csv'}")
