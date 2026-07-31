"""
Preditor PAS 3 — previsão individual do Argumento Final.

Uma previsão só (`Â3`, o Alvo Canônico do ADR-0009); todo o resto da resposta sai dela por
aritmética. Antes deste ticket a tela mostrava **dois** números previstos por modelos
independentes — Argumento Final e EB PAS 3 — que discordavam sobre "passa ou não passa" em 11%
dos Alunos (relatório 04 §3.2). Agora existe um número e ele não pode se contradizer.

O EB PAS 3 saiu da resposta: derivá-lo do `A3` exige o Estimador Auxiliar e o Ano-Âncora do
ticket 04 §7.1, que ainda não têm ticket; ressuscitar o modelo aposentado para preenchê-lo
traria de volta exatamente a contradição acima.
"""
import logging

from api.schemas.predict import PredictInput, PredictResponse, CourseResult, StrategyInput, StrategyResponse
from api.services import gestao_service          # referência ao módulo, não ao valor
from api.services.gestao_service import SEM_PACOTE, _find_best_match, _build_cutoff_maps
from pas_intelligence.model_package import (
    EntradaDePrevisao,
    EstatisticasIndisponiveisError,
    NotasDeEtapa,
)
from pas_intelligence.statistics import calculate_approval_probability
from pas_intelligence.training_dataset import (
    EstatisticaOficialAusenteError,
    anos_do_trienio,
    stats_da_prova,
)

logger = logging.getLogger(__name__)

# Limite e piso vindos do lado do portal: 10 cursos a partir de 20% de chance. O corte antigo
# (8 a partir de 30%) devolvia lista vazia para o Aluno que ainda está longe do curso que quer —
# exatamente quem mais precisa ver a distância.
TOP_CURSOS_LIMIT = 10
MIN_PROB_THRESHOLD = 0.20

# Sem login a lista é uma amostra, não a ferramenta: três cursos bastam para o Aluno se situar.
TOP_CURSOS_DESLOGADO = 3


def entrada_de_previsao(inp: PredictInput) -> EntradaDePrevisao:
    return EntradaDePrevisao(
        etapa_1=NotasDeEtapa(p1=inp.p1_pas1, p2=inp.p2_pas1, redacao=inp.red_pas1),
        etapa_2=NotasDeEtapa(p1=inp.p1_pas2, p2=inp.p2_pas2, redacao=inp.red_pas2),
        lingua=inp.lingua,
        trienio=inp.trienio,
    )


def _semestres_a_buscar(semestre: str | None, m1: dict, m2: dict) -> list[tuple[str, dict]]:
    """Os mapas de corte que a busca varre, na ordem em que o Aluno os vê.

    O seletor de semestre é filtro de consulta, não dado do Aluno (ele não escolhe em qual
    semestre entra): "1°" e "2°" olham um mapa só, e qualquer outro valor — inclusive `None`,
    que é o Aluno deslogado, que não tem o seletor — olha os dois.
    """
    if semestre == "1°":
        return [("1°", m1)]
    if semestre == "2°":
        return [("2°", m2)]
    return [("1°", m1), ("2°", m2)]


def _selecionar_cursos(
    candidatos: list[tuple[str, float, str]],
    arg_previsto: float,
    largura: float,
    ordem,
    limite: int,
) -> list[CourseResult]:
    """Os `limite` primeiros cursos por `ordem`, sem repetir curso entre os dois semestres.

    A deduplicação é por chave de curso e vem **depois** da ordenação: um curso que abre nos dois
    semestres entra uma vez só, pelo semestre que a ordem escolheu primeiro.
    """
    escolhidos: list[CourseResult] = []
    vistos: set[str] = set()
    for chave, nota, semestre in sorted(candidatos, key=ordem):
        if chave in vistos:
            continue
        vistos.add(chave)
        partes = _parse_course_key(chave)
        prob = calculate_approval_probability(arg_previsto, nota, largura_incerteza=largura)
        escolhidos.append(CourseResult(
            curso=partes["curso"],
            turno=partes["turno"],
            campus=partes["campus"],
            nota_corte=round(nota, 3),
            prob=round(prob * 100, 1),
            semestre=semestre,
        ))
        if len(escolhidos) >= limite:
            break
    return escolhidos


def predict_student(inp: PredictInput) -> PredictResponse:
    entrada = entrada_de_previsao(inp)
    c1_map, c2_map, trienio_ref = _build_cutoff_maps(inp.trienio)

    previsao = None
    motivo_indisponivel = None
    if gestao_service._pacote is None:
        motivo_indisponivel = SEM_PACOTE
    else:
        try:
            previsao = gestao_service._pacote.prever(entrada)
        except EstatisticasIndisponiveisError as erro:
            # Esperado no triênio vivo: enquanto o Edital de média e desvio daquela Etapa não for
            # extraído, `A1` e `A2` não são exatos — e aproximá-los destruiria a parte exata da
            # conta, que é a fundação do ADR-0009. Recusar é a resposta certa, não um bug.
            motivo_indisponivel = str(erro)
            logger.info("Previsão recusada por Edital de Etapa ausente: %s", erro)
        except ValueError as erro:
            # Triênio malformado. O schema já barra língua fora das três (422), então o que chega
            # aqui é `trienio`, que é texto livre por causa dos clientes antigos. Um endpoint
            # público não pode responder 500 a entrada ruim.
            motivo_indisponivel = str(erro)
            logger.info("Previsão recusada por entrada malformada: %s", erro)

    if previsao is None:
        return PredictResponse(
            arg_previsto=0.0,
            a1=0.0,
            a2=0.0,
            a3_previsto=0.0,
            largura_incerteza=0.0,
            etapa_1_ausente=entrada.etapa_1_ausente,
            curso_alvo_result=None,
            top_cursos=[],
            trienio_ref=trienio_ref,
            modelo_disponivel=False,
            motivo_indisponivel=motivo_indisponivel,
        )

    arg_previsto = previsao.argumento_final
    largura = previsao.largura_argumento_final

    available_systems = list(set(c1_map.keys()) | set(c2_map.keys()))
    sistema = _find_best_match(inp.cota, available_systems, cutoff=0.6) if available_systems else inp.cota

    m1 = c1_map.get(sistema, c1_map.get("Sistema Universal", {}))
    m2 = c2_map.get(sistema, c2_map.get("Sistema Universal", {}))

    # Curso alvo (opcional)
    curso_alvo_result: CourseResult | None = None
    if inp.curso_alvo:
        all_courses = list(set(list(m1.keys()) + list(m2.keys())))
        curso_matched = _find_best_match(inp.curso_alvo, all_courses, cutoff=0.4) if all_courses else inp.curso_alvo

        # Sem login não há seletor de semestre na tela, então a busca varre os dois mapas.
        semestre_filtro = inp.semestre if inp.is_logged_in else None

        for semestre, m in _semestres_a_buscar(semestre_filtro, m1, m2):
            nota = m.get(curso_matched)
            if nota:
                prob = calculate_approval_probability(arg_previsto, nota, largura_incerteza=largura)
                parts = _parse_course_key(curso_matched)
                curso_alvo_result = CourseResult(
                    curso=parts["curso"],
                    turno=parts["turno"],
                    campus=parts["campus"],
                    nota_corte=round(nota, 3),
                    prob=round(prob * 100, 1),
                    semestre=semestre,
                )
                break

    if inp.is_logged_in:
        # Com login, a lista é aspiracional: dos cursos que ainda estão ao alcance
        # (chance >= MIN_PROB_THRESHOLD), os TOP_CURSOS_LIMIT **mais difíceis** — ordenar pela
        # menor probabilidade e cortar é o que faz o Aluno ver até onde dá para mirar, em vez de
        # uma lista de cursos que ele já passaria com folga. O `reverse` final só põe o mais
        # difícil no topo da tela.
        candidatos = [
            (chave, nota, semestre)
            for semestre, mapa in _semestres_a_buscar(inp.semestre, m1, m2)
            for chave, nota in mapa.items()
            if calculate_approval_probability(arg_previsto, nota, largura_incerteza=largura) >= MIN_PROB_THRESHOLD
        ]
        top_cursos = _selecionar_cursos(
            candidatos, arg_previsto, largura,
            ordem=lambda c: calculate_approval_probability(arg_previsto, c[1], largura_incerteza=largura),
            limite=TOP_CURSOS_LIMIT,
        )
        top_cursos.reverse()
    else:
        # Sem login, a lista é de referência: os TOP_CURSOS_DESLOGADO cursos cujo corte cai mais
        # perto do Argumento previsto, para os dois semestres. Sem piso de probabilidade — a
        # graça é mostrar onde o Aluno está, não onde ele já ganhou.
        candidatos = [
            (chave, nota, semestre)
            for semestre, mapa in _semestres_a_buscar(None, m1, m2)
            for chave, nota in mapa.items()
        ]
        top_cursos = _selecionar_cursos(
            candidatos, arg_previsto, largura,
            ordem=lambda c: abs(c[1] - arg_previsto),
            limite=TOP_CURSOS_DESLOGADO,
        )

    return PredictResponse(
        arg_previsto=round(arg_previsto, 1),
        a1=round(previsao.a1, 3),
        a2=round(previsao.a2, 3),
        a3_previsto=round(previsao.a3, 3),
        largura_incerteza=round(largura, 3),
        etapa_1_ausente=previsao.etapa_1_ausente,
        curso_alvo_result=curso_alvo_result,
        top_cursos=top_cursos,
        trienio_ref=trienio_ref,
        modelo_disponivel=True,
    )


def get_courses(cota: str, trienio: str) -> list[str]:
    """Retorna lista de cursos disponíveis para o combo de seleção."""
    c1_map, c2_map, _ = _build_cutoff_maps(trienio)
    available_systems = list(set(c1_map.keys()) | set(c2_map.keys()))
    sistema = _find_best_match(cota, available_systems, cutoff=0.6) if available_systems else cota
    m1 = c1_map.get(sistema, c1_map.get("Sistema Universal", {}))
    m2 = c2_map.get(sistema, c2_map.get("Sistema Universal", {}))
    all_keys = sorted(set(list(m1.keys()) + list(m2.keys())))
    return all_keys


def _parse_course_key(key: str) -> dict:
    """'Engenharia Civil - Diurno (Darcy Ribeiro)' → {curso, turno, campus}"""
    campus = ""
    if "(" in key and key.endswith(")"):
        campus = key[key.rfind("(") + 1:-1]
        key = key[:key.rfind("(")].strip()

    if " - " in key:
        parts = key.rsplit(" - ", 1)
        return {"curso": parts[0].strip(), "turno": parts[1].strip(), "campus": campus}

    return {"curso": key.strip(), "turno": "", "campus": campus}


def get_course_chamadas(curso_key: str, cota: str, trienio: str, semestre: str) -> list[dict]:
    import pandas as pd
    df_corte = gestao_service._df_corte
    if df_corte is None or not curso_key:
        return []

    # Parse course key
    parts = _parse_course_key(curso_key)
    curso_nome = parts["curso"]
    turno = parts["turno"].upper()
    campus = parts["campus"].upper()

    # Resolve reference triennium if not present in database (e.g. future triennium)
    trienio_ref = trienio
    if trienio not in df_corte["Trienio"].dropna().unique():
        try:
            start, end = map(int, trienio.split("-"))
            trienio_ref = f"{start - 1}-{end - 1}"
            if trienio_ref not in df_corte["Trienio"].dropna().unique():
                trienio_ref = "2022-2024"
        except Exception:
            trienio_ref = "2022-2024"

    # Filter system name
    available_systems = list(df_corte["Sistema_Nome"].dropna().unique())
    sistema = _find_best_match(cota, available_systems, cutoff=0.6) if available_systems else cota

    # Filter mask
    mask = (
        (df_corte["Curso_Limpo"].str.upper() == curso_nome.upper()) &
        (df_corte["Turno"].str.upper() == turno) &
        (df_corte["Campus"].str.upper() == campus) &
        (df_corte["Sistema_Nome"] == sistema) &
        (df_corte["Trienio"] == trienio_ref)
    )
    
    # Semestre filter (1° ou 2°)
    sem_db = "1°" if semestre.startswith("1") else "2°"
    mask = mask & (df_corte["Semestre"] == sem_db)

    sub = df_corte[mask].copy()
    if sub.empty:
        return []

    # Extract digits to sort calls (e.g. "1ª" -> 1, "2ª" -> 2)
    sub["chamada_num"] = sub["Chamada"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
    sub = sub.sort_values("chamada_num")

    out = []
    for _, row in sub.iterrows():
        out.append({
            "chamada": str(row.get("Chamada", "")),
            "campus": str(row.get("Campus", "")),
            "turno": str(row.get("Turno", "")),
            "nota_corte": float(row.get("Min", 0.0)) if pd.notna(row.get("Min")) else 0.0
        })
    return out


def _stats_do_ciclo(ciclo_aluno: str, lingua: str):
    """Média e desvio oficiais das três Etapas do triênio do Aluno, pela porta única.

    `stats_da_prova` é a mesma função que o treino e o Preditor usam (ticket 05): o
    `TRIENNIUM_STATS` que esta função substitui era uma cópia paralela do `OFFICIAL_STATS`,
    que podia divergir dele sem que nada avisasse.

    A Etapa 3 do triênio vivo ainda não aconteceu, e a projeção por regressão foi descartada
    pelo ADR-0009. No lugar dela vai o Ano-Âncora: a Etapa 3 real e publicada mais recente.
    """
    ano_e1, ano_e2, ano_e3 = anos_do_trienio(ciclo_aluno)
    stats_p1 = stats_da_prova(ano_e1, 1, lingua)
    stats_p2 = stats_da_prova(ano_e2, 2, lingua)
    try:
        stats_p3 = stats_da_prova(ano_e3, 3, lingua)
    except EstatisticaOficialAusenteError:
        stats_p3 = gestao_service._stats_pas3_ancora(lingua)
    return stats_p1, stats_p2, stats_p3


def predict_strategy(inp: StrategyInput) -> StrategyResponse:
    """Calculadora de Estratégia: o P2 que o Aluno precisa no PAS 3 para bater a `nota_alvo`.

    `base_projecao` chega do cliente antigo e **não tem efeito**: escolhia entre o
    `TRIENNIUM_STATS` e a regressão do `STATS_PAS3_TREND`, e as duas saíram no ticket 05. Fica
    aceito para não quebrar a tela; o ticket 12 lhe dá um uso de novo com os cinco Anos-Âncora.

    Quando o Edital de média e desvio de uma das Etapas já feitas ainda não saiu — o caso do
    triênio vivo — a rota não é calculável, e a resposta diz isso em `status`/`mensagem` em vez
    de 500 ou de números inventados. Mesma decisão do Preditor (`motivo_indisponivel`).
    """
    from pas_intelligence.target_calculator import TargetCalculator
    from pas_intelligence.statistics import calculate_cohort_evolution_probability

    notas_validas = {
        'P1_PAS1': inp.p1_pas1,
        'P2_PAS1': inp.p2_pas1,
        'Red_PAS1': inp.red_pas1,
        'P1_PAS2': inp.p1_pas2,
        'P2_PAS2': inp.p2_pas2,
        'Red_PAS2': inp.red_pas2
    }

    calc = TargetCalculator()

    try:
        stats_p1, stats_p2, stats_p3 = _stats_do_ciclo(inp.ciclo_aluno, inp.lingua)
    except (EstatisticaOficialAusenteError, ValueError) as erro:
        logger.info("Calculadora recusada por estatística oficial ausente: %s", erro)
        return StrategyResponse(
            p1_estimado=0.0, p2_necessario=0.0, red_estimada=0.0,
            total_pas3=0.0, arg_pas3_necessario=0.0,
            status="indisponivel",
            mensagem=(
                "Ainda não dá para calcular a rota do seu triênio: o Argumento das Etapas que "
                "você já fez depende da média e do desvio-padrão que o Cebraspe publica no "
                "Edital de cada Etapa, e o do seu triênio ainda não saiu."
            ),
            prob_hist=0.0, amostra=0, p1_ia=0.0, red_ia=0.0,
        )

    # Previsões automáticas do Estimador Auxiliar (p1_ia, red_ia) — precisam das três
    # estatísticas, por isso só dá para calcular depois que `_stats_do_ciclo` responde.
    previsao_ia = calc.predict_stable_components(notas_validas, stats_p1, stats_p2, stats_p3)
    p1_ia = float(previsao_ia.get('p1_pred', 0.0))
    red_ia = float(previsao_ia.get('red_pred', 0.0))

    # Executa predição de rota de aprovação
    result = calc.calculate_required_score(
        notas_validas,
        inp.nota_alvo,
        stats_p1,
        stats_p2,
        stats_p3,
        p1_override=inp.p1_override,
        red_override=inp.red_override,
    )

    # Reality Check (coorte histórica)
    prob_hist = 0.0
    amostra = 0
    eb_pas3_necessario = result.p1_estimado + result.p2_necessario

    df_cohort = gestao_service._df_cohort
    if df_cohort is not None and not df_cohort.empty:
        try:
            eb_pas1 = inp.p1_pas1 + inp.p2_pas1
            eb_pas2 = inp.p1_pas2 + inp.p2_pas2
            aluno_dados = {'eb_pas1': eb_pas1, 'eb_pas2': eb_pas2}
            prob_hist, amostra = calculate_cohort_evolution_probability(aluno_dados, eb_pas3_necessario, df_cohort)
        except Exception:
            # O reality check é opcional: sem ele a rota calculada segue válida. Mas a falha vai
            # para o log — o silêncio anterior escondeu por meses que os modelos nem carregavam.
            logger.exception("Reality check (coorte) falhou na Calculadora; seguindo sem ele.")

    return StrategyResponse(
        p1_estimado=result.p1_estimado,
        p2_necessario=result.p2_necessario,
        red_estimada=result.red_estimada,
        total_pas3=result.total_pas3,
        arg_pas3_necessario=result.arg_pas3_necessario,
        status=result.status,
        mensagem=result.mensagem,
        prob_hist=round(prob_hist, 1),
        amostra=amostra,
        p1_ia=p1_ia,
        red_ia=red_ia
    )

