"""Passo 1: quanto custa, em pontos de Argumento Final, inferir media e desvio
das Etapas 1 e 2 a partir do Edital isolado, em vez de le-los do Edital do PAS 3.

Tres efeitos, medidos separadamente sobre os 8 trienios fechados, onde o valor
oficial e conhecido e serve de gabarito:

  (L)  lingua     - o Edital isolado nao diz a lingua estrangeira de cada Aluno,
                    entao a Parte 1 so pode ser padronizada por uma media unica,
                    misturando as tres linguas.
  (P)  populacao  - a media inferida sai da populacao do Edital, nao da populacao
                    completa que o Cebraspe usa (viés de sobrevivencia).
  (LP) os dois juntos - o cenario realista.

Saida: distribuicao de  Delta Argumento Final = 1*dA1 + 2*dA2  contra a
incerteza propria do modelo, 3*sigma(A3).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pas_intelligence.argument_calculator import PESO_P1, PESO_P2, PESO_REDACAO
from pas_intelligence.pas_constants import OFFICIAL_STATS

CSV = ".scratch/pdf-extraction/saida-nova/resultado_final.csv"
RMSE_A3 = 5.009  # manifest do pacote promovido (ticket 13)
LIMIAR = 3 * RMSE_A3  # sigma(Argumento Final) = 3 * sigma(A3)


def ano_da_etapa(trienio: str, etapa: int) -> int:
    """'2023/2025' + etapa 2 -> 2024. Etapa 1 no ano inicial, uma por ano."""
    return int(trienio.split("/")[0]) + (etapa - 1)


def momentos_agrupados(pesos, medias, desvios):
    """Media e desvio da mistura das tres linguas, dados os momentos de cada uma.

    A variancia da mistura NAO e a media das variancias: inclui a dispersao
    entre os grupos. E exatamente o que a coluna de Parte 1 de um Edital
    isolado produziria, se a populacao fosse a mesma.
    """
    pesos = np.asarray(pesos, dtype=float)
    pesos = pesos / pesos.sum()
    medias = np.asarray(medias, dtype=float)
    desvios = np.asarray(desvios, dtype=float)
    media = float((pesos * medias).sum())
    var = float((pesos * (desvios**2 + medias**2)).sum() - media**2)
    return media, float(np.sqrt(var))


def z(nota, media, desvio, peso):
    return ((nota - media) / desvio) * peso


def carregar():
    df = pd.read_csv(CSV, dtype=str)
    df = df[df["checksum_fecha"] == "True"].copy()  # 510 linhas corrompidas ficam fora
    for c in ["eb_p1_e1", "eb_p2_e1", "red_e1", "eb_p1_e2", "eb_p2_e2", "red_e2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def medir(df, apenas_lingua_certa: bool):
    linhas_stats, por_aluno = [], []

    for trienio in sorted(df["trienio"].unique()):
        bloco_trienio = df[df["trienio"] == trienio]

        for etapa in (1, 2):
            ano = ano_da_etapa(trienio, etapa)
            of = OFFICIAL_STATS[(ano, etapa)]
            p1, p2, red = f"eb_p1_e{etapa}", f"eb_p2_e{etapa}", f"red_e{etapa}"
            lingua = f"lingua_e{etapa}"

            b = bloco_trienio.dropna(subset=[p1, p2, red])
            # Aluno sem esta Etapa: as tres notas zeradas (ADR-0011)
            b = b[~((b[p1] == 0) & (b[p2] == 0) & (b[red] == 0))]
            b = b[b[lingua].isin(of.parte_1.keys())]
            if apenas_lingua_certa:
                b = b[b["lingua_ambigua"] == "False"]
            if b.empty:
                continue

            # --- estatisticas empiricas (o que se infere do Edital isolado) ---
            emp = {
                "m_p1": b[p1].mean(), "dp_p1": b[p1].std(ddof=0),
                "m_p2": b[p2].mean(), "dp_p2": b[p2].std(ddof=0),
                "m_red": b[red].mean(), "dp_red": b[red].std(ddof=0),
            }

            # --- Parte 1 misturada, a partir dos valores OFICIAIS por lingua ---
            share = b[lingua].value_counts(normalize=True)
            ordem = list(of.parte_1.keys())
            m_mix, dp_mix = momentos_agrupados(
                [share.get(k, 0.0) for k in ordem],
                [of.parte_1[k].media for k in ordem],
                [of.parte_1[k].desvio_padrao for k in ordem],
            )

            m_lg = b[lingua].map(lambda k: of.parte_1[k].media).to_numpy()
            dp_lg = b[lingua].map(lambda k: of.parte_1[k].desvio_padrao).to_numpy()
            n_p1, n_p2, n_red = b[p1].to_numpy(), b[p2].to_numpy(), b[red].to_numpy()

            arg_p2_of = z(n_p2, of.m_p2, of.dp_p2, PESO_P2)
            arg_red_of = z(n_red, of.m_red, of.dp_red, PESO_REDACAO)
            arg_p2_emp = z(n_p2, emp["m_p2"], emp["dp_p2"], PESO_P2)
            arg_red_emp = z(n_red, emp["m_red"], emp["dp_red"], PESO_REDACAO)

            arg_p1_verdade = z(n_p1, m_lg, dp_lg, PESO_P1)
            arg_p1_mix = z(n_p1, m_mix, dp_mix, PESO_P1)          # so lingua
            arg_p1_emp = z(n_p1, emp["m_p1"], emp["dp_p1"], PESO_P1)  # lingua + populacao

            verdade = arg_p1_verdade + arg_p2_of + arg_red_of
            cen_L = arg_p1_mix + arg_p2_of + arg_red_of
            cen_P = arg_p1_verdade + arg_p2_emp + arg_red_emp
            cen_LP = arg_p1_emp + arg_p2_emp + arg_red_emp

            por_aluno.append(pd.DataFrame({
                "trienio": trienio, "etapa": etapa, "inscricao": b["inscricao"].to_numpy(),
                "dA_L": cen_L - verdade, "dA_P": cen_P - verdade, "dA_LP": cen_LP - verdade,
            }))

            linhas_stats.append({
                "trienio": trienio, "ano": ano, "etapa": etapa, "n": len(b),
                "m_p2_of": of.m_p2, "m_p2_emp": emp["m_p2"], "d_m_p2": emp["m_p2"] - of.m_p2,
                "dp_p2_of": of.dp_p2, "dp_p2_emp": emp["dp_p2"], "d_dp_p2": emp["dp_p2"] - of.dp_p2,
                "m_red_of": of.m_red, "m_red_emp": emp["m_red"], "d_m_red": emp["m_red"] - of.m_red,
                "dp_red_of": of.dp_red, "dp_red_emp": emp["dp_red"], "d_dp_red": emp["dp_red"] - of.dp_red,
                "m_p1_mix_of": m_mix, "dp_p1_mix_of": dp_mix,
                "m_p1_emp": emp["m_p1"], "dp_p1_emp": emp["dp_p1"],
                "m_p1_ing": of.parte_1["inglesa"].media,
                "m_p1_fra": of.parte_1["francesa"].media,
                "m_p1_esp": of.parte_1["espanhola"].media,
                "share_ing": share.get("inglesa", 0.0),
                "share_esp": share.get("espanhola", 0.0),
                "share_fra": share.get("francesa", 0.0),
            })

    return pd.DataFrame(linhas_stats), pd.concat(por_aluno, ignore_index=True)


def argumento_final(por_aluno):
    """dArgumentoFinal = 1*dA1 + 2*dA2, so para quem tem as duas Etapas."""
    p = por_aluno.pivot_table(
        index=["trienio", "inscricao"], columns="etapa",
        values=["dA_L", "dA_P", "dA_LP"], aggfunc="first",
    ).dropna()
    saida = pd.DataFrame(index=p.index)
    for cen in ["dA_L", "dA_P", "dA_LP"]:
        saida[cen] = 1 * p[(cen, 1)] + 2 * p[(cen, 2)]
    return saida.reset_index()


def resumo(s: pd.Series) -> dict:
    a = s.abs()
    return {
        "media": s.mean(), "|media|": a.mean(), "p50|.|": a.quantile(0.50),
        "p95|.|": a.quantile(0.95), "p99|.|": a.quantile(0.99), "max|.|": a.max(),
        f">{LIMIAR:.1f}": f"{(a > LIMIAR).mean():.3%}",
    }


if __name__ == "__main__":
    df = carregar()
    print(f"linhas apos filtro de checksum: {len(df)}")

    for apenas_certa in (False, True):
        rotulo = "SEM linguas ambiguas" if apenas_certa else "TODAS as linguas"
        stats, aluno = medir(df, apenas_certa)
        af = argumento_final(aluno)
        print(f"\n{'='*78}\n{rotulo}  |  {len(af)} Alunos com Etapa 1 e 2\n{'='*78}")

        print("\n--- Delta por Etapa (pontos de A) ---")
        print(aluno.groupby("etapa")[["dA_L", "dA_P", "dA_LP"]]
              .apply(lambda g: pd.DataFrame({c: resumo(g[c]) for c in g.columns}).T).to_string())

        print(f"\n--- Delta no Argumento Final (limiar 3*RMSE = {LIMIAR:.2f}) ---")
        print(pd.DataFrame({c: resumo(af[c]) for c in ["dA_L", "dA_P", "dA_LP"]}).T.to_string())

        if not apenas_certa:
            stats.to_csv("/private/tmp/claude-501/-Users-luizhenrique-Documents-Vetor-PAS-Analise-Preditiva-PAS/8dd696b3-0c1f-46de-bff5-9b2499876b6d/scratchpad/stats_por_etapa.csv", index=False)
            af.to_csv("/private/tmp/claude-501/-Users-luizhenrique-Documents-Vetor-PAS-Analise-Preditiva-PAS/8dd696b3-0c1f-46de-bff5-9b2499876b6d/scratchpad/delta_argumento_final.csv", index=False)
