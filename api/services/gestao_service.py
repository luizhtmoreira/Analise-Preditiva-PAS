"""
Gestão de Ativos — lógica de análise por aluno.
Extrai a lógica da seção 'Gestão de Ativos' do Streamlit (streamlit_app.py:1945).
"""
import sys
import difflib
import logging
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Permite importar src/pas_intelligence/ quando rodando do root do projeto
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pas_intelligence.statistics import (
    calculate_approval_probability,
    calculate_cohort_evolution_probability,
)
from pas_intelligence.argument_calculator import HistoricalStats
from pas_intelligence.model_package import (
    EntradaDePrevisao,
    EstatisticasIndisponiveisError,
    NotasDeEtapa,
    PacoteDeModelo,
    PacoteIndisponivelError,
)

from api.schemas.gestao import StudentInput, StudentResult, GestaoKpis, GestaoResponse

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# `ARG_FINAL_MAE = 13.49` morava aqui. Era o resíduo de um modelo aposentado, igual para todo
# Aluno, e ficava errado por construção a cada troca de modelo sem que nada avisasse. A Largura
# de Incerteza agora viaja dentro do pacote (`PacoteDeModelo.largura_de_incerteza`) e vem por
# classe de Aluno — ADR-0012.

SEM_PACOTE = (
    "Nenhum pacote de modelo carregado. Sem pacote não há previsão nem Largura de Incerteza — "
    "o estado 'previsão sim, largura não' não é representável (ADR-0012)."
)

# TODO(ticket 04 §9, defeito 1): estes números não são os do Cebraspe. Em 2022-2024 o P2 da
# Etapa 1 aqui é 20,7094 contra 20,406 do Edital — têm cara de calculados de uma amostra. Hoje
# eles só alimentam o Reality Check (abaixo), que é opcional; o Argumento previsto não passa
# mais por aqui. Os valores certos já estão em `pas_constants.OFFICIAL_STATS`.
TRIENNIUM_STATS = {
    "2024-2026": {
        "PAS1": HistoricalStats(mean_p1=2.9552, std_p1=2.4909, mean_p2=25.4272, std_p2=11.3914, mean_red=6.9051, std_red=1.8409),
        "PAS2": HistoricalStats(mean_p1=3.1320, std_p1=3.0884, mean_p2=28.0855, std_p2=14.6654, mean_red=6.5109, std_red=1.9882),
        "PAS3": None,
    },
    "2023-2025": {
        "PAS1": HistoricalStats(mean_p1=2.2175, std_p1=2.4766, mean_p2=25.3330, std_p2=14.6860, mean_red=6.0345, std_red=2.4790),
        "PAS2": HistoricalStats(mean_p1=3.1496, std_p1=3.2475, mean_p2=29.2750, std_p2=14.2913, mean_red=6.8770, std_red=2.0050),
        "PAS3": HistoricalStats(mean_p1=3.8200, std_p1=2.1000, mean_p2=30.6750, std_p2=13.7520, mean_red=7.6500, std_red=1.8500),
    },
    "2022-2024": {
        "PAS1": HistoricalStats(mean_p1=3.6037, std_p1=3.0053, mean_p2=20.7094, std_p2=13.5819, mean_red=5.8878, std_red=2.7796),
        "PAS2": HistoricalStats(mean_p1=3.7393, std_p1=2.2378, mean_p2=30.3477, std_p2=13.2532, mean_red=6.9370, std_red=1.9723),
        "PAS3": HistoricalStats(mean_p1=3.7679, std_p1=2.1778, mean_p2=32.0862, std_p2=14.1289, mean_red=7.5791, std_red=1.7304),
    },
}

STATS_PAS3_TREND = HistoricalStats(
    mean_p1=3.82, std_p1=2.1,
    mean_p2=33.74, std_p2=14.5,
    mean_red=7.65, std_red=1.85,
)

# ---------------------------------------------------------------------------
# Singleton: recursos carregados uma vez no startup
# ---------------------------------------------------------------------------

_pacote: Optional[PacoteDeModelo] = None
"""O pacote de modelo promovido — um único modelo (`A3`) mais aritmética, no lugar dos oito
`.joblib` do ensemble aposentado (ADR-0011). `None` quando não há pacote em disco: aí a API
responde `modelo_disponivel: False` em vez de inventar zeros."""

_df_corte: Optional[pd.DataFrame] = None
_df_cohort: Optional[pd.DataFrame] = None


def load_resources() -> None:
    global _pacote, _df_corte, _df_cohort

    data_dir = _ROOT / "data"

    try:
        _pacote = PacoteDeModelo.carregar()
    except PacoteIndisponivelError:
        # Falha no startup vira log e `modelo_disponivel: False`, nunca previsão degradada em
        # silêncio — que é o defeito de `target_calculator.py:66`, o `print` que escondeu por
        # meses que dois modelos nem carregavam.
        logger.exception("Nenhum pacote de modelo carregado; a API responderá sem previsão.")
        _pacote = None

    # Banco de notas de corte
    corte_path = data_dir / "notas_corte_pas.csv"
    if corte_path.exists():
        _df_corte = pd.read_csv(corte_path)
        if "Min" in _df_corte.columns:
            _df_corte["Min"] = pd.to_numeric(_df_corte["Min"], errors="coerce")
        if "Curso_Limpo" not in _df_corte.columns and "Curso" in _df_corte.columns:
            _df_corte["Curso_Limpo"] = _df_corte["Curso"]

    # Banco histórico de alunos (Reality Check)
    cohort_path = data_dir / "banco_alunos_pas_final.csv"
    if cohort_path.exists():
        cols = ["P1_PAS1", "P2_PAS1", "P1_PAS2", "P2_PAS2", "P1_PAS3", "P2_PAS3", "Arg_Final"]
        _df_cohort = pd.read_csv(cohort_path, usecols=lambda c: c in cols)
        if "Arg_Final" in _df_cohort.columns:
            _df_cohort = _df_cohort.rename(columns={"Arg_Final": "ARG_FINAL_REAL"})
        if "ARG_FINAL_REAL" in _df_cohort.columns:
            _df_cohort = _df_cohort[_df_cohort["ARG_FINAL_REAL"] != 0]
        for col1, col2, dest in [("P1_PAS1", "P2_PAS1", "EB_PAS1"), ("P1_PAS2", "P2_PAS2", "EB_PAS2")]:
            if col1 in _df_cohort.columns and col2 in _df_cohort.columns:
                _df_cohort[dest] = _df_cohort[col1] + _df_cohort[col2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_match(query: str, choices: list[str], cutoff: float = 0.6) -> str:
    if not query or not choices:
        return query

    matches = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    try:
        q_norm = unicodedata.normalize("NFKD", str(query)).encode("ASCII", "ignore").decode().lower().strip()

        candidates = []
        for c in choices:
            c_norm = unicodedata.normalize("NFKD", str(c)).encode("ASCII", "ignore").decode().lower()
            if q_norm in c_norm:
                candidates.append(c)
        if candidates:
            return min(candidates, key=len)

        q_clean = q_norm.replace("(", " ").replace(")", " ").replace("-", " ")
        q_tokens = set(q_clean.split())
        best, best_score = None, 0.0
        for c in choices:
            c_norm = unicodedata.normalize("NFKD", str(c)).encode("ASCII", "ignore").decode().lower()
            c_clean = c_norm.replace("(", " ").replace(")", " ").replace("-", " ")
            common = q_tokens.intersection(set(c_clean.split()))
            if not common:
                continue
            score = len(common) / len(q_tokens)
            if score > best_score:
                best_score = score
                best = c
        if best and best_score >= 0.5:
            return best
    except Exception:
        pass

    return query


def _build_cutoff_maps(trienio_sel: str):
    """Retorna (corte_1sem_map, corte_2sem_map, trienio_ref) para o triênio selecionado."""
    if _df_corte is None:
        return {}, {}, "N/A"

    try:
        start, end = map(int, trienio_sel.split("-"))
        trienio_ref = f"{start - 1}-{end - 1}"
        if trienio_ref not in _df_corte["Trienio"].unique():
            trienio_ref = "2022-2024"
    except Exception:
        trienio_ref = "2022-2024"

    def _build_map(semestre: str, ascending: bool) -> dict:
        df = _df_corte[(_df_corte["Trienio"] == trienio_ref) & (_df_corte["Semestre"] == semestre)]
        df = df.sort_values("Min", ascending=ascending).drop_duplicates(
            subset=["Sistema_Nome", "Curso_Limpo", "Campus", "Turno"], keep="first"
        )
        result: dict = {}
        for sistema, grp in df.groupby("Sistema_Nome"):
            result[sistema] = {}
            for _, row in grp.iterrows():
                key = f"{row['Curso_Limpo']} - {row['Turno']} ({row['Campus']})"
                result[sistema][key] = row["Min"]
        return result

    c1 = _build_map("1°", ascending=True)
    # 2nd semester: check fallback if not found
    has_2sem = not _df_corte[(_df_corte["Trienio"] == trienio_ref) & (_df_corte["Semestre"] == "2°")].empty
    ref_2sem = trienio_ref if has_2sem else "2022-2024"
    c2 = _build_map("2°", ascending=False)

    return c1, c2, trienio_ref


def _classify_risk(prob_1: float, prob_2: float):
    if prob_1 >= 50 or prob_2 >= 75:
        return "green", "Baixo Risco"
    if prob_2 >= 30:
        return "yellow", "Oportunidade (2º Sem)"
    return "red", "Alto Risco"


def _prever(s: StudentInput, trienio_padrao: str) -> tuple[object | None, str | None]:
    """A previsão de um Aluno, ou `(None, motivo)` quando ela não é possível para o triênio dele.

    `None` e não zero: um Aluno sem previsão que aparece com Argumento `0,0` e probabilidade
    `0,0%` vira "Alto Risco" na tela da coordenação — uma afirmação sobre ele que ninguém mediu.

    O motivo sobe junto porque a coordenação precisa saber **qual** dos dois problemas ela tem:
    "o modelo não carregou" e "o Edital da Etapa desta turma ainda não saiu" pedem ações
    diferentes, de pessoas diferentes.
    """
    if _pacote is None:
        return None, SEM_PACOTE
    try:
        return _pacote.prever(
            EntradaDePrevisao(
                etapa_1=NotasDeEtapa(p1=s.p1_pas1, p2=s.p2_pas1, redacao=s.red_pas1),
                etapa_2=NotasDeEtapa(p1=s.p1_pas2, p2=s.p2_pas2, redacao=s.red_pas2),
                lingua=s.lingua,
                trienio=s.ano_trienio or trienio_padrao,
            )
        ), None
    except (EstatisticasIndisponiveisError, ValueError) as erro:
        logger.info("Sem previsão para um Aluno do triênio %s: %s", s.ano_trienio, erro)
        return None, str(erro)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_students(students: list[StudentInput], trienio: str, cenario: str) -> GestaoResponse:
    c1_map, c2_map, trienio_ref = _build_cutoff_maps(trienio)

    stats_p3 = STATS_PAS3_TREND if cenario == "tendencia" else (
        (TRIENNIUM_STATS.get(trienio) or {}).get("PAS3") or STATS_PAS3_TREND
    )

    results: list[StudentResult] = []
    motivo_sem_previsao: Optional[str] = None

    for s in students:
        eb_p1 = s.p1_pas1 + s.p2_pas1
        eb_p2 = s.p1_pas2 + s.p2_pas2

        # Normaliza sistema de concorrência via fuzzy match
        available_systems = list(set(c1_map.keys()) | set(c2_map.keys()))
        sistema = _find_best_match(s.cota, available_systems, cutoff=0.6) if available_systems else s.cota

        m1 = c1_map.get(sistema, c1_map.get("Sistema Universal", {}))
        m2 = c2_map.get(sistema, c2_map.get("Sistema Universal", {}))

        available_courses = list(m1.keys()) or list(m2.keys())
        if not available_courses:
            for cm in c1_map.values():
                available_courses.extend(cm.keys())

        curso_matched = _find_best_match(s.curso_alvo, available_courses, cutoff=0.4) if available_courses else s.curso_alvo

        nota_corte_1sem: float = m1.get(curso_matched) or 0.0
        nota_corte_2sem: Optional[float] = m2.get(curso_matched)

        # Predição — o Argumento Final sai de `A1 + 2·A2 + 3·Â3`, com A1 e A2 exatos (ADR-0009)
        previsao, motivo = _prever(s, trienio)
        if motivo and motivo_sem_previsao is None:
            motivo_sem_previsao = motivo
        arg_pred = previsao.argumento_final if previsao else 0.0
        largura = previsao.largura_argumento_final if previsao else 0.0

        gap = round(arg_pred - nota_corte_1sem, 1) if (previsao and nota_corte_1sem) else 0.0

        prob_1 = (
            calculate_approval_probability(arg_pred, nota_corte_1sem, largura_incerteza=largura) * 100
            if (previsao and nota_corte_1sem) else 0.0
        )
        prob_2 = (
            calculate_approval_probability(arg_pred, nota_corte_2sem, largura_incerteza=largura) * 100
            if (previsao and nota_corte_2sem) else 0.0
        )

        # Reality Check (cohort)
        historico_pct = 0.0
        if previsao and _df_cohort is not None and nota_corte_1sem:
            try:
                from pas_intelligence.target_calculator import TargetCalculator
                stats_ciclo = TRIENNIUM_STATS.get(s.ano_trienio or trienio)
                if stats_ciclo:
                    calc = TargetCalculator()
                    notas = {
                        "P1_PAS1": s.p1_pas1, "P2_PAS1": s.p2_pas1, "Red_PAS1": s.red_pas1,
                        "P1_PAS2": s.p1_pas2, "P2_PAS2": s.p2_pas2, "Red_PAS2": s.red_pas2,
                    }
                    cutoff_h = nota_corte_1sem if (prob_1 >= 50 or prob_2 >= 75) else (nota_corte_2sem or nota_corte_1sem)
                    path = calc.calculate_required_score(notas, cutoff_h, stats_ciclo["PAS1"], stats_ciclo["PAS2"], stats_p3 or STATS_PAS3_TREND)
                    eb_nec = path.p1_estimado + path.p2_necessario
                    prob_h, _ = calculate_cohort_evolution_probability({"eb_pas1": eb_p1, "eb_pas2": eb_p2}, eb_nec, _df_cohort)
                    historico_pct = round(prob_h, 1)
            except Exception:
                # O reality check é opcional: sem ele o resto da análise segue válido.
                # Mas a falha vai para o log — o silêncio anterior escondeu por meses que
                # os modelos de P1/Redação nem carregavam.
                logger.exception("Reality check (cohort) falhou para o aluno; seguindo sem ele.")

        # Risco. Sem previsão não há risco a classificar — `grey` diz "não sei", que é diferente
        # de "Alto Risco", e é o que zerar o Argumento faria a tela afirmar.
        if previsao is None:
            status, status_label = "grey", "Sem previsão"
        else:
            status, status_label = _classify_risk(prob_1, prob_2)

        # Sugestão (apenas para não-verdes)
        sugestao = ""
        if previsao and status not in ("green", "grey"):
            for curso_alt, corte_alt in sorted(m1.items(), key=lambda x: x[1], reverse=True):
                if curso_alt == curso_matched:
                    continue
                p = calculate_approval_probability(arg_pred, corte_alt, largura_incerteza=largura)
                if p >= 0.8:
                    sugestao = curso_alt.split(" (")[0]
                    break

        if previsao is None:
            chance_display = "—"
        elif nota_corte_2sem:
            chance_display = f"1º: {prob_1:.1f}% | 2º: {prob_2:.1f}%"
        else:
            chance_display = f"{prob_1:.1f}%"

        results.append(StudentResult(
            nome=s.nome,
            turma=s.turma,
            unidade=s.unidade,
            curso_alvo=curso_matched or s.curso_alvo,
            sistema_concorrencia=sistema,
            arg_previsto=round(arg_pred, 1),
            gap=gap,
            chance_display=chance_display,
            historico_pct=historico_pct,
            sugestao=sugestao or "—",
            status=status,
            status_label=status_label,
            prob_1_sem=round(prob_1, 1),
            prob_2_sem=round(prob_2, 1),
        ))

    # Ordena: sem previsão → red → yellow → green. O `grey` vem primeiro porque é o único que
    # pede uma ação de quem opera o sistema, e não do Aluno.
    order = {"grey": -1, "red": 0, "yellow": 1, "green": 2}
    results.sort(key=lambda r: order[r.status])

    kpis = GestaoKpis(
        total=len(results),
        n_red=sum(1 for r in results if r.status == "red"),
        n_yellow=sum(1 for r in results if r.status == "yellow"),
        n_green=sum(1 for r in results if r.status == "green"),
        n_sem_previsao=sum(1 for r in results if r.status == "grey"),
    )

    return GestaoResponse(
        results=results,
        kpis=kpis,
        trienio_ref=trienio_ref,
        modelo_disponivel=_pacote is not None,
        motivo_sem_previsao=motivo_sem_previsao,
    )
