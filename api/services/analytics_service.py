"""
Analytics — Análise Temporal (pública), Escola vs. População e Comparação A/B.
Migra as seções correspondentes do Streamlit (streamlit_app.py:1624, 3537, 4028).
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pas_intelligence.pas_constants import OFFICIAL_STATS
from pas_intelligence.ab_testing import compare_groups

from api.services import gestao_service
from api.schemas.analytics import (
    EtapaStat,
    TemporalResponse,
    CorteEvolucao,
    EscolaResponse,
    EtapaComparativo,
    HistBin,
    ComparacaoResponse,
    BoxStats,
)

# ---------------------------------------------------------------------------
# Banco populacional completo (lazy, com colunas além das usadas na gestão)
# ---------------------------------------------------------------------------

_df_pop: Optional[pd.DataFrame] = None


def _load_population() -> Optional[pd.DataFrame]:
    global _df_pop
    if _df_pop is not None:
        return _df_pop

    path = _ROOT / "data" / "banco_alunos_pas_final.csv"
    if not path.exists():
        return None

    cols = [
        "Inscricao", "Ano_Trienio", "Arg_Final",
        "P1_PAS1", "P2_PAS1", "P1_PAS2", "P2_PAS2", "P1_PAS3", "P2_PAS3",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols, dtype={"Inscricao": str})
    df["Inscricao"] = df["Inscricao"].astype(str).str.strip()
    df["EB_PAS1"] = df["P1_PAS1"] + df["P2_PAS1"]
    df["EB_PAS2"] = df["P1_PAS2"] + df["P2_PAS2"]
    df["EB_PAS3"] = df["P1_PAS3"] + df["P2_PAS3"]
    _df_pop = df
    return _df_pop


# ---------------------------------------------------------------------------
# Análise Temporal (pública)
# ---------------------------------------------------------------------------


def get_temporal() -> TemporalResponse:
    etapas = [
        EtapaStat(
            ano=ano,
            etapa=etapa,
            p1_medio=round(s.m_p1, 2),
            p2_medio=round(s.m_p2, 2),
            eb_medio=round(s.m_p1 + s.m_p2, 2),
            red_media=round(s.m_red, 2),
        )
        for (ano, etapa), s in sorted(OFFICIAL_STATS.items())
    ]

    cursos: list[str] = []
    df_corte = gestao_service._df_corte
    if df_corte is not None:
        mask = (df_corte["Sistema_Nome"] == "Sistema Universal") & (df_corte["Semestre"] == "1°")
        sub = df_corte[mask]
        keys = (
            sub["Curso_Limpo"].astype(str) + " - " + sub["Turno"].astype(str)
            + " (" + sub["Campus"].astype(str) + ")"
        )
        cursos = sorted(keys.unique().tolist())

    return TemporalResponse(etapas=etapas, cursos=cursos)


def get_corte_evolucao(curso: str) -> list[CorteEvolucao]:
    """Evolução da nota de corte de um curso (Sistema Universal) através dos triênios."""
    df_corte = gestao_service._df_corte
    if df_corte is None or not curso:
        return []

    df = df_corte[df_corte["Sistema_Nome"] == "Sistema Universal"].copy()
    keys = (
        df["Curso_Limpo"].astype(str) + " - " + df["Turno"].astype(str)
        + " (" + df["Campus"].astype(str) + ")"
    )
    df = df[keys == curso]
    if df.empty:
        return []

    out: list[CorteEvolucao] = []
    for trienio in sorted(df["Trienio"].unique()):
        grp = df[df["Trienio"] == trienio]
        # 1º semestre: menor corte entre as chamadas; 2º semestre: idem
        c1 = grp[grp["Semestre"] == "1°"]["Min"].min()
        c2 = grp[grp["Semestre"] == "2°"]["Min"].min()
        out.append(CorteEvolucao(
            trienio=trienio,
            corte_1sem=round(float(c1), 3) if pd.notna(c1) else None,
            corte_2sem=round(float(c2), 3) if pd.notna(c2) else None,
        ))
    return out


# ---------------------------------------------------------------------------
# Escola vs. População
# ---------------------------------------------------------------------------


def _hist_bins(values: np.ndarray, n_bins: int = 30) -> list[HistBin]:
    counts, edges = np.histogram(values, bins=n_bins)
    return [
        HistBin(x0=round(float(edges[i]), 2), x1=round(float(edges[i + 1]), 2), count=int(counts[i]))
        for i in range(len(counts))
    ]


def analyze_escola(inscricoes: list[str], trienio: str) -> EscolaResponse:
    df_pop = _load_population()
    if df_pop is None:
        raise ValueError("Banco populacional não disponível no servidor.")

    trienios = sorted(df_pop["Ano_Trienio"].dropna().unique().tolist(), reverse=True)
    if not trienio or trienio not in trienios:
        trienio = trienios[0] if trienios else ""

    df_trienio = df_pop[df_pop["Ano_Trienio"] == trienio]

    enviados = [str(i).strip() for i in inscricoes if str(i).strip() and str(i).strip().lower() != "none"]
    df_escola = df_trienio[df_trienio["Inscricao"].isin(set(enviados))]

    n_enviados = len(enviados)
    n_encontrados = min(int(df_escola["Inscricao"].nunique()), n_enviados)
    taxa = (n_encontrados / n_enviados * 100) if n_enviados else 0.0

    if n_encontrados < 5:
        # Sem amostra suficiente: devolve estrutura mínima para o front exibir aviso
        return EscolaResponse(
            n_enviados=n_enviados, n_encontrados=n_encontrados, taxa_match=round(taxa, 1),
            trienio=trienio, trienios_disponiveis=trienios,
            media_escola=0, media_geral=0, std_escola=0, std_geral=0, diff=0,
            percentil=0, p_value=1.0, significativo=False, etapas=[], hist_arg=[],
        )

    media_escola = float(df_escola["Arg_Final"].mean())
    media_geral = float(df_trienio["Arg_Final"].mean())
    diff = media_escola - media_geral
    percentil = float((df_trienio["Arg_Final"] < media_escola).mean() * 100)

    try:
        result = compare_groups(
            group_a=df_escola["Arg_Final"].values,
            group_b=df_trienio["Arg_Final"].values,
            group_a_name="Sua Escola",
            group_b_name="População Geral",
            metric_name="Argumento Final",
        )
        p_value = float(result["p_value"])
        significativo = bool(result["statistically_significant"])
    except Exception:
        p_value, significativo = 1.0, False

    etapas = [
        EtapaComparativo(
            etapa=f"PAS {i}",
            escola=round(float(df_escola[f"EB_PAS{i}"].mean()), 1),
            geral=round(float(df_trienio[f"EB_PAS{i}"].mean()), 1),
        )
        for i in (1, 2, 3)
    ]

    return EscolaResponse(
        n_enviados=n_enviados,
        n_encontrados=n_encontrados,
        taxa_match=round(taxa, 1),
        trienio=trienio,
        trienios_disponiveis=trienios,
        media_escola=round(media_escola, 2),
        media_geral=round(media_geral, 2),
        std_escola=round(float(df_escola["Arg_Final"].std()), 2),
        std_geral=round(float(df_trienio["Arg_Final"].std()), 2),
        diff=round(diff, 2),
        percentil=round(percentil, 1),
        p_value=p_value,
        significativo=significativo,
        etapas=etapas,
        hist_arg=_hist_bins(df_trienio["Arg_Final"].dropna().values),
    )


# ---------------------------------------------------------------------------
# Comparação entre grupos (A/B)
# ---------------------------------------------------------------------------


def _box_stats(values: np.ndarray) -> BoxStats:
    return BoxStats(
        min=round(float(np.min(values)), 2),
        q1=round(float(np.percentile(values, 25)), 2),
        med=round(float(np.percentile(values, 50)), 2),
        q3=round(float(np.percentile(values, 75)), 2),
        max=round(float(np.max(values)), 2),
        mean=round(float(np.mean(values)), 2),
        n=len(values),
    )


def compare(group_a: list[float], group_b: list[float], name_a: str, name_b: str, metric: str) -> ComparacaoResponse:
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    result = compare_groups(
        group_a=a, group_b=b,
        group_a_name=name_a, group_b_name=name_b,
        metric_name=metric or "média",
    )

    return ComparacaoResponse(
        mean_a=round(float(result["group_a_mean"]), 2),
        mean_b=round(float(result["group_b_mean"]), 2),
        diff=round(float(result["difference"]), 2),
        p_value=float(result["p_value"]),
        t_statistic=round(float(result["t_statistic"]), 4),
        effect_size=round(float(result["effect_size"]), 2),
        significativo=bool(result["statistically_significant"]),
        box_a=_box_stats(a),
        box_b=_box_stats(b),
    )
