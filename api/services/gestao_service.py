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
from pas_intelligence.derivado_deploy import COLUNAS_CHAMADAS, COLUNAS_NOTAS_CORTE, COLUNAS_RESULTADO_FINAL
from pas_intelligence.pas_constants import OFFICIAL_STATS
from pas_intelligence.training_dataset import (
    EstatisticaOficialAusenteError,
    anos_do_trienio,
    stats_da_prova,
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

# O Cebraspe reformula alguns nomes de curso entre triênios (ex.: Mecatrônica perdeu o
# qualificador "– Controle e Automação" em 2023/2025) e algumas extrações antigas saíram sem o
# sufixo "(BACHARELADO)" que os demais anos têm. Sem esta tabela, o mesmo curso vira duas chaves
# distintas — `_build_cutoff_maps` (que só enxerga o triênio de referência) resolve para uma
# grafia, e `get_corte_evolucao` (que casa string exata contra todos os triênios) perde o
# histórico do lado que não bateu. Mapeia sempre para a grafia mais recente/completa.
CURSO_ALIASES = {
    "ENGENHARIA MECATRÔNICA – CONTROLE E AUTOMAÇÃO (BACHARELADO)": "ENGENHARIA MECATRÔNICA (BACHARELADO)",
    "ENGENHARIA AMBIENTAL": "ENGENHARIA AMBIENTAL (BACHARELADO)",
    "ENGENHARIAS – AEROESPACIAL / AUTOMOTIVA / ELETRÔNICA / ENERGIA / SOFTWARE (BACHARELADOS)":
        "ENGENHARIAS – AEROESPACIAL / AUTOMOTIVA / ELETRÔNICA / ENERGIA / SOFTWARE",
    "ESTATÍSTICA": "ESTATÍSTICA (BACHARELADO)",
    "LÍNGUA ESTRANGEIRA APLICADA – MULTILINGUISMO E SOCIEDADE DA INFORMAÇÃO":
        "LÍNGUA ESTRANGEIRA APLICADA – MULTILINGUISMO E SOCIEDADE DA INFORMAÇÃO (BACHARELADO)",
}

# Até o ticket 16, o seletor de cota da tela usava a nomenclatura "L1/L2/L9/L10" — um empréstimo
# solto do SiSU que não vinha do Edital do PAS e nem batia com a própria ordem dele. O `COTA_ALIASES`
# que existia aqui traduzia esses rótulos para o `Sistema_Nome` do CSV ("EP / Baixa Renda / PPI"
# etc.); a fuzzy match sozinha não bastava porque as duas descrições têm poucas palavras em comum.
# O ticket 16 trocou o seletor para usar o próprio `MAPA_SISTEMAS` (a fonte que gera o `Sistema_Nome`
# do CSV) como rótulo, então a tela já manda a chave exata — sem alias para resolver.
def _resolver_sistema(cota: str, available_systems: list[str]) -> str:
    """A chave de `Sistema_Nome` que corresponde à cota escolhida na tela.

    `_find_best_match` (fuzzy) cobre variações de grafia; a tela manda o rótulo de
    `MAPA_SISTEMAS` diretamente, que já bate exato com `Sistema_Nome` na maioria dos casos.
    """
    if not available_systems:
        return cota
    return _find_best_match(cota, available_systems, cutoff=0.6)

# ---------------------------------------------------------------------------
# Singleton: recursos carregados uma vez no startup
# ---------------------------------------------------------------------------

_pacote: Optional[PacoteDeModelo] = None
"""O pacote de modelo promovido — um único modelo (`A3`) mais aritmética, no lugar dos oito
`.joblib` do ensemble aposentado (ADR-0011). `None` quando não há pacote em disco: aí a API
responde `modelo_disponivel: False` em vez de inventar zeros."""

_df_corte: Optional[pd.DataFrame] = None
_df_chamadas: Optional[pd.DataFrame] = None
_df_cohort: Optional[pd.DataFrame] = None


def _normaliza_campus(campus: str) -> str:
    """"UnB CEILÂNDIA (FCE)" → "CEILÂNDIA"; "DARCY RIBEIRO" fica como está."""
    base = str(campus).split(" (")[0].strip()
    for prefixo in ("UnB ", "UNB "):
        if base.upper().startswith(prefixo.upper()):
            base = base[len(prefixo):].strip()
            break
    return base.upper()


def _normalizar_cortes(df: pd.DataFrame) -> pd.DataFrame:
    """Traduz o schema minúsculo que a frente de extração escreve (`trienio`, `sistema_nome`,
    `curso`, ...) para o schema que o resto do serviço espera (`Trienio`, `Semestre`,
    `Sistema_Nome`, `Curso_Limpo`, `Min`). Compartilhada por `notas_corte.csv` (Nota de Corte
    oficial, só a maior chamada) e `chamadas.csv` (histórico, todas as chamadas) — mesmo schema
    de origem, mesma sujeira a normalizar.
    """
    # `checksum_fecha == True` é o único filtro necessário: a contaminação de escala (ex.
    # MEDICINA/Darcy Ribeiro/2020-2022 saindo com corte=199.162,872) sempre falha o checksum.
    df = df[df["checksum_fecha"] == True].drop(columns=["checksum_fecha"])  # noqa: E712
    # "Sub judice" é uma rodada de convocação por decisão judicial, à parte da concorrência
    # normal do curso — sua nota de corte não é comparável à do curso e não deve aparecer
    # como se fosse (item 2 do relatório de 2026-07-31).
    df = df[~df["curso"].astype(str).str.contains("SUB JUDICE", case=False, na=False)]
    df = df.rename(columns={
        "curso": "Curso_Limpo",
        "campus": "Campus",
        "turno": "Turno",
        "sistema_nome": "Sistema_Nome",
        "nota_corte": "Min",
        "chamada": "Chamada",
    })
    df["Curso_Limpo"] = df["Curso_Limpo"].replace(CURSO_ALIASES)
    df["Trienio"] = df["trienio"].astype(str).str.replace("/", "-", regex=False)
    # `semestre` vem como "1"/"2"/"desconhecido"; sem saber o semestre a linha não pode ser
    # roteada para nenhum dos dois mapas de corte, então ela sai (perda medida: ~427 de 4.986).
    df["Semestre"] = df["semestre"].map({"1": "1°", "2": "2°"})
    df = df.dropna(subset=["Semestre"]).drop(columns=["trienio", "semestre"])
    # Turno ausente não é sempre "curso de período integral" (ex. Enfermagem, Farmácia,
    # que nunca reportam Turno em trienio nenhum): alguns cursos com Turno normal só
    # passaram a discriminá-lo no Edital a partir de um certo ano (ex. as Engenharias do
    # Gama, sem Turno até 2021/2023 e "DIURNO" a partir de 2022/2024). Preencher tudo com
    # "Integral" cria, para esses cursos, uma chave de curso que não bate com a grafia mais
    # recente do mesmo curso — mesmo defeito do `CURSO_ALIASES`, agora na dimensão Turno.
    # Herda o Turno mais comum que o próprio curso já reportou em algum trienio; só cai
    # para "Integral" quando o curso nunca reportou Turno.
    turno_modal = df.dropna(subset=["Turno"]).groupby("Curso_Limpo")["Turno"].agg(lambda s: s.mode().iat[0])
    df["Turno"] = df.apply(
        lambda r: r["Turno"] if pd.notna(r["Turno"]) else turno_modal.get(r["Curso_Limpo"], "Integral"),
        axis=1,
    )
    # O nome do Sistema Universal mudou de rótulo na extração nova; o resto do serviço (e o
    # default de `cota` na API) ainda espera o rótulo antigo.
    df["Sistema_Nome"] = df["Sistema_Nome"].replace({"Universal": "Sistema Universal"})
    # Campus veio com a sigla da faculdade entre parênteses (ex. "UnB CEILÂNDIA (FCE)"), e a
    # chave de curso (`_build_cutoff_maps`) já usa parênteses pra embutir o Campus —
    # `_parse_course_key` faz `rfind("(")` e quebra com parênteses aninhados. Normaliza para
    # o nome de Campus simples que o resto do sistema sempre assumiu.
    df["Campus"] = df["Campus"].map(_normaliza_campus)
    df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
    return df


def load_resources() -> None:
    global _pacote, _df_corte, _df_chamadas, _df_cohort

    data_dir = _ROOT / "data"

    try:
        _pacote = PacoteDeModelo.carregar()
    except PacoteIndisponivelError:
        # Falha no startup vira log e `modelo_disponivel: False`, nunca previsão degradada em
        # silêncio — que é o defeito de `target_calculator.py:66`, o `print` que escondeu por
        # meses que dois modelos nem carregavam.
        logger.exception("Nenhum pacote de modelo carregado; a API responderá sem previsão.")
        _pacote = None

    # Banco de notas de corte — a frente de extração escreve schema minúsculo
    # (`trienio`, `sistema_nome`, `curso`, ...) e carrega `inscricao`/`nome` de Alunos reais junto
    # (ticket 09). A tradução para o schema que o resto do serviço espera acontece só em
    # `_normalizar_cortes` — o único ponto de carga — e as duas colunas de PII nunca chegam a
    # ser lidas (`usecols`).
    corte_path = data_dir / "notas_corte.csv"
    if corte_path.exists():
        df = pd.read_csv(corte_path, usecols=lambda c: c in COLUNAS_NOTAS_CORTE)
        _df_corte = _normalizar_cortes(df)

    # Histórico de chamadas — mesmo schema de `notas_corte.csv`, mas uma linha por chamada em
    # vez de só a última (gerado por `scripts/gerar_historico_chamadas.py`). `notas_corte.csv`
    # nunca guardou isso: desde o ticket 10 ele é colapsado para a maior chamada de cada grupo
    # por definição de produto (ver `notas_corte.py`), então a tela "Histórico de Chamadas"
    # (`get_course_chamadas`) lia esse CSV e nunca via mais de uma linha por curso.
    chamadas_path = data_dir / "chamadas.csv"
    if chamadas_path.exists():
        df = pd.read_csv(chamadas_path, usecols=lambda c: c in COLUNAS_CHAMADAS)
        _df_chamadas = _normalizar_cortes(df)

    # Banco histórico de alunos (Reality Check) — mesma frente de extração, mesma família de
    # sujeira e o mesmo filtro único (`checksum_fecha == True` deixa 64.298 de 66.313 linhas).
    cohort_path = data_dir / "resultado_final.csv"
    if cohort_path.exists():
        df = pd.read_csv(cohort_path, usecols=lambda c: c in COLUNAS_RESULTADO_FINAL)
        df = df[df["checksum_fecha"] == True].drop(columns=["checksum_fecha"])  # noqa: E712
        df = df.rename(columns={
            "eb_p1_e1": "P1_PAS1", "eb_p2_e1": "P2_PAS1",
            "eb_p1_e2": "P1_PAS2", "eb_p2_e2": "P2_PAS2",
            "eb_p1_e3": "P1_PAS3", "eb_p2_e3": "P2_PAS3",
            "argumento_final": "ARG_FINAL_REAL",
        })
        df = df[df["ARG_FINAL_REAL"] != 0]
        for col1, col2, dest in [("P1_PAS1", "P2_PAS1", "EB_PAS1"), ("P1_PAS2", "P2_PAS2", "EB_PAS2")]:
            df[dest] = df[col1] + df[col2]
        _df_cohort = df


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


def _stats_pas3_ancora(lingua: str) -> HistoricalStats:
    """A Etapa 3 real e já publicada mais recente do `OFFICIAL_STATS`, usada como Ano-Âncora do
    Reality Check quando o triênio do Aluno ainda não tem a própria (turma viva).

    Substitui a regressão que `STATS_PAS3_TREND` fazia sobre uma prova que ainda não aconteceu
    — o ticket 04 do treino decidiu contra projetar (decisão 3: Ano-Âncora). Esta é a versão de
    um ano só; o ticket 12 desta rodada constrói os cinco Anos-Âncora na tela.
    """
    ano_mais_recente = max(ano for (ano, etapa) in OFFICIAL_STATS if etapa == 3)
    return stats_da_prova(ano_mais_recente, 3, lingua)


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
                lingua_e1=s.lingua_e1,
                lingua_e2=s.lingua_e2,
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
    # `cenario` ("padrao" | "tendencia") escolhia entre `TRIENNIUM_STATS` e a regressão de
    # `STATS_PAS3_TREND`. As duas saíram (este ticket); o Reality Check agora sempre lê o
    # `OFFICIAL_STATS` pela porta única (`_stats_pas3_ancora`/`stats_da_prova`), então o
    # parâmetro fica sem efeito até o ticket 12 (Ano-Âncora) lhe dar um uso de novo.
    c1_map, c2_map, trienio_ref = _build_cutoff_maps(trienio)

    results: list[StudentResult] = []
    motivo_sem_previsao: Optional[str] = None

    for s in students:
        eb_p1 = s.p1_pas1 + s.p2_pas1
        eb_p2 = s.p1_pas2 + s.p2_pas2

        # Normaliza sistema de concorrência via fuzzy match
        available_systems = list(set(c1_map.keys()) | set(c2_map.keys()))
        sistema = _resolver_sistema(s.cota, available_systems)

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

        # Reality Check (cohort) — mesma porta única do OFFICIAL_STATS que o Preditor usa
        # (`stats_da_prova`), para que A1 e A2 aqui sejam os mesmos que `previsao` já tem.
        historico_pct = 0.0
        if previsao and _df_cohort is not None and nota_corte_1sem:
            try:
                from pas_intelligence.target_calculator import TargetCalculator
                ano_e1, ano_e2, ano_e3 = anos_do_trienio(s.ano_trienio or trienio)
                stats_p1 = stats_da_prova(ano_e1, 1, s.lingua_e1)
                stats_p2 = stats_da_prova(ano_e2, 2, s.lingua_e2)
                # A Etapa 3 não tem campo próprio (`StudentInput` não a pergunta); a língua da
                # Etapa 2 é a melhor aproximação disponível, por ser a mais recente que o Aluno
                # declarou — a mesma lógica do Ano-Âncora logo abaixo.
                try:
                    stats_p3 = stats_da_prova(ano_e3, 3, s.lingua_e2)
                except EstatisticaOficialAusenteError:
                    # Turma viva: a Etapa 3 dela ainda não aconteceu. Ano-Âncora de um ano real
                    # e já publicado no lugar da regressão que `STATS_PAS3_TREND` fazia.
                    stats_p3 = _stats_pas3_ancora(s.lingua_e2)

                calc = TargetCalculator()
                notas = {
                    "P1_PAS1": s.p1_pas1, "P2_PAS1": s.p2_pas1, "Red_PAS1": s.red_pas1,
                    "P1_PAS2": s.p1_pas2, "P2_PAS2": s.p2_pas2, "Red_PAS2": s.red_pas2,
                }
                cutoff_h = nota_corte_1sem if (prob_1 >= 50 or prob_2 >= 75) else (nota_corte_2sem or nota_corte_1sem)
                path = calc.calculate_required_score(notas, cutoff_h, stats_p1, stats_p2, stats_p3)
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
