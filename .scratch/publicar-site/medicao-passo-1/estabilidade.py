"""Medicao 3: o erro e um deslocamento estavel (logo corrigivel) ou e ruido?"""
import pandas as pd

BASE = "/private/tmp/claude-501/-Users-luizhenrique-Documents-Vetor-PAS-Analise-Preditiva-PAS/8dd696b3-0c1f-46de-bff5-9b2499876b6d/scratchpad"
pd.set_option("display.width", 200)

stats = pd.read_csv(f"{BASE}/stats_por_etapa.csv")

print("=== Estatisticas inferidas vs oficiais, por (ano, etapa) ===")
cols = ["ano", "etapa", "n", "m_p2_of", "m_p2_emp", "d_m_p2", "dp_p2_of", "dp_p2_emp",
        "d_dp_p2", "m_red_of", "m_red_emp", "d_m_red", "d_dp_red"]
print(stats[cols].round(3).to_string(index=False))

print("\n=== Direcao e magnitude do erro de populacao ===")
for c in ["d_m_p2", "d_dp_p2", "d_m_red", "d_dp_red"]:
    s = stats[c]
    print(f"{c:10s} media={s.mean():+7.3f}  dp={s.std():6.3f}  "
          f"positivos={int((s > 0).sum())}/{len(s)}  min={s.min():+7.3f}  max={s.max():+7.3f}")

print("\n=== Parte 1: a media misturada fica a que distancia de cada lingua? ===")
for lg, col in [("inglesa", "m_p1_ing"), ("espanhola", "m_p1_esp"), ("francesa", "m_p1_fra")]:
    d = stats["m_p1_mix_of"] - stats[col]
    print(f"mistura - {lg:10s} media={d.mean():+7.3f}  dp={d.std():6.3f}  "
          f"min={d.min():+7.3f}  max={d.max():+7.3f}")

print("\n=== Proporcao de lingua por ano (a mistura muda de composicao?) ===")
print(stats[["ano", "etapa", "share_ing", "share_esp", "share_fra"]].round(4).to_string(index=False))

print("\n=== Parte 1 empirica vs mistura oficial (efeito populacao na P1) ===")
d = stats["m_p1_emp"] - stats["m_p1_mix_of"]
print(f"m_p1_emp - m_p1_mix_of : media={d.mean():+7.3f} dp={d.std():6.3f} "
      f"positivos={int((d > 0).sum())}/{len(d)}")

# --- o deslocamento no Argumento Final e estavel por trienio? ---
af = pd.read_csv(f"{BASE}/delta_argumento_final.csv")
print("\n=== Delta no Argumento Final por trienio ===")
r = af.groupby("trienio")[["dA_L", "dA_P", "dA_LP"]].agg(["mean", "std"]).round(3)
print(r.to_string())

print("\n=== Se corrigirmos pelo deslocamento medio global, o que sobra? ===")
for c in ["dA_L", "dA_P", "dA_LP"]:
    bruto = af[c].abs()
    corr = (af[c] - af[c].mean()).abs()
    print(f"{c}:  |erro| bruto p95={bruto.quantile(0.95):.3f} max={bruto.max():.3f}"
          f"   ->  corrigido p95={corr.quantile(0.95):.3f} max={corr.max():.3f}")
