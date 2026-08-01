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

from api.schemas.predict import (
    PredictInput,
    PredictResponse,
    CourseResult,
    StrategyInput,
    StrategyResponse,
    AnoAncoraResultado,
)
from api.services import gestao_service          # referência ao módulo, não ao valor
from api.services.gestao_service import SEM_PACOTE, _find_best_match, _resolver_sistema, _build_cutoff_maps
from pas_intelligence.model_package import (
    EntradaDePrevisao,
    EstatisticasIndisponiveisError,
    NotasDeEtapa,
)
from pas_intelligence.pas_constants import anos_ancora as pas_anos_ancora
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
        lingua_e1=inp.lingua_e1,
        lingua_e2=inp.lingua_e2,
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
    sistema = _resolver_sistema(inp.cota, available_systems)

    m1 = c1_map.get(sistema, c1_map.get("Sistema Universal", {}))
    m2 = c2_map.get(sistema, c2_map.get("Sistema Universal", {}))

    # Curso alvo (opcional)
    curso_alvo_result: CourseResult | None = None
    curso_alvo_sem_dados_cota = False
    if inp.curso_alvo:
        # Resolve o curso contra o universo de TODOS os cursos ofertados (todas as cotas), não
        # contra `m1`/`m2` (já filtrados pela cota escolhida): algumas cotas não têm corte
        # publicado para todo curso (ex. cota L1 não teve candidato em Engenharia Civil no
        # triênio 2023/2025). Casar o fuzzy match direto contra esse mapa reduzido "chutava" o
        # curso mais parecido disponível NAQUELA cota — podia devolver um curso sem relação
        # nenhuma com o que o Aluno pediu. Resolvido o curso contra o universo completo, uma
        # cota sem corte publicado para ele vira "sem dados", nunca um curso errado.
        universo_cursos: set[str] = set()
        for m in list(c1_map.values()) + list(c2_map.values()):
            universo_cursos.update(m.keys())
        curso_matched = _find_best_match(inp.curso_alvo, list(universo_cursos), cutoff=0.4) if universo_cursos else inp.curso_alvo

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

        curso_alvo_sem_dados_cota = curso_alvo_result is None

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
        curso_alvo_sem_dados_cota=curso_alvo_sem_dados_cota,
        top_cursos=top_cursos,
        trienio_ref=trienio_ref,
        modelo_disponivel=True,
        usa_estatistica_derivada=previsao.usa_estatistica_derivada,
    )


def get_courses(cota: str, trienio: str) -> list[str]:
    """Retorna lista de cursos disponíveis para o combo de seleção."""
    c1_map, c2_map, _ = _build_cutoff_maps(trienio)
    available_systems = list(set(c1_map.keys()) | set(c2_map.keys()))
    sistema = _resolver_sistema(cota, available_systems)
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
    df_corte = gestao_service._df_chamadas
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
    sistema = _resolver_sistema(cota, available_systems)

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

    # `chamadas.csv` já traz uma linha por chamada (ver `scripts/gerar_historico_chamadas.py`);
    # o groupby aqui só garante 1 linha por `chamada_num` mesmo se o filtro acima (curso/turno/
    # campus/sistema/trienio/semestre) ainda deixar duplicata por outra dimensão não filtrada.
    grouped = (
        sub.sort_values("chamada_num")
        .groupby("chamada_num", sort=True)
        .agg(Chamada=("Chamada", "first"), Campus=("Campus", "first"), Turno=("Turno", "first"), Min=("Min", "min"))
        .reset_index()
    )

    out = []
    for _, row in grouped.iterrows():
        out.append({
            "chamada": str(row.get("Chamada", "")),
            "campus": str(row.get("Campus", "")),
            "turno": str(row.get("Turno", "")),
            "nota_corte": float(row.get("Min", 0.0)) if pd.notna(row.get("Min")) else 0.0
        })
    return out


def _stats_do_ciclo(ciclo_aluno: str, lingua_e1: str, lingua_e2: str):
    """Média e desvio oficiais das três Etapas do triênio do Aluno, pela porta única.

    `stats_da_prova` é a mesma função que o treino e o Preditor usam (ticket 05): o
    `TRIENNIUM_STATS` que esta função substitui era uma cópia paralela do `OFFICIAL_STATS`,
    que podia divergir dele sem que nada avisasse.

    A Parte 1 é normalizada por língua **por Etapa** (13,9% da base troca de língua entre a
    Etapa 1 e a Etapa 2) — daí `lingua_e1` reger a Etapa 1 e `lingua_e2` reger a Etapa 2.

    A Etapa 3 do triênio vivo ainda não aconteceu, e a projeção por regressão foi descartada
    pelo ADR-0009. No lugar dela vai o Ano-Âncora — o quarto valor de retorno (`e3_e_ancora`)
    diz a `predict_strategy` se a Etapa 3 usada aqui (a mais recente publicada) é o cenário
    único que sobra quando o Aluno não pediu os cinco (ticket 12), ou a Etapa 3 real do
    triênio dele. Usa `lingua_e2` como escala por ser a língua mais recente que o Aluno
    declarou — a mesma aproximação do Ano-Âncora logo abaixo.
    """
    ano_e1, ano_e2, ano_e3 = anos_do_trienio(ciclo_aluno)
    stats_p1 = stats_da_prova(ano_e1, 1, lingua_e1)
    stats_p2 = stats_da_prova(ano_e2, 2, lingua_e2)
    try:
        stats_p3 = stats_da_prova(ano_e3, 3, lingua_e2)
        e3_e_ancora = False
    except EstatisticaOficialAusenteError:
        stats_p3 = gestao_service._stats_pas3_ancora(lingua_e2)
        e3_e_ancora = True
    return stats_p1, stats_p2, stats_p3, e3_e_ancora


def get_course_cutoff(curso: str, cota: str, trienio: str, semestre: str) -> float:
    """A Nota de Corte publicada de um curso num triênio específico.

    Porta única para `/api/courses/cutoff` e para os cinco cenários do Ano-Âncora (ticket 12) —
    os dois precisam da mesma regra (última chamada do 1º semestre é o piso mais baixo que
    passou; 1ª chamada do 2º semestre é o teto mais alto que ainda não passou).
    """
    calls = get_course_chamadas(curso, cota, trienio, semestre)
    # `!= 0`, não `> 0`: o Argumento Final é legitimamente negativo (~49% da base, ver
    # CLAUDE.md) — filtrar por positivo descartava a nota de corte real de cotas como L9/L10
    # e caía no sentinela de "sem dado" (`get_course_chamadas` usa `0.0` só para NaN, linha
    # 328 acima), fazendo o Argumento de Corte Alvo da Calculadora parecer travado em 0.
    scores = [c["nota_corte"] for c in calls if c["nota_corte"] != 0]
    if not scores:
        return 0.0
    return min(scores) if semestre == "1°" else max(scores)


def _resultados_ano_ancora(
    calc,
    notas_validas: dict,
    stats_p1,
    stats_p2,
    lingua: str,
    arg_alvo_fallback: float,
    curso_alvo: str | None,
    cota: str,
    semestre: str,
    p1_override: float | None,
    red_override: float | None,
) -> list["AnoAncoraResultado"]:
    """Os cinco cenários do Ano-Âncora (ticket 12) — cada um varia **junto** a estatística da
    Etapa 3 daquele ano (dificuldade) e a Nota de Corte do curso no triênio correspondente
    (concorrência): Ano-Âncora 2025 → triênio 2023-2025. Nunca uma combinação que não aconteceu.

    Sem `curso_alvo` (cliente anterior ao ticket 12), cai de volta a comparar os cinco contra o
    único `nota_alvo` que o cliente já resolveu — a mesma resposta de antes deste ticket, só que
    replicada nas cinco estatísticas de Etapa 3 em vez de uma só.
    """
    resultados = []
    for ano in pas_anos_ancora():
        stats_p3_ano = stats_da_prova(ano, 3, lingua)
        trienio_corte = f"{ano - 2}-{ano}"
        nota_corte = arg_alvo_fallback
        if curso_alvo:
            achada = get_course_cutoff(curso_alvo, cota, trienio_corte, semestre)
            if achada > 0:
                nota_corte = achada
        resultado = calc.calculate_required_score(
            notas_validas, nota_corte, stats_p1, stats_p2, stats_p3_ano,
            p1_override=p1_override, red_override=red_override,
        )
        resultados.append(AnoAncoraResultado(
            ano=ano,
            trienio_corte=trienio_corte,
            nota_corte=round(nota_corte, 3),
            p1_estimado=resultado.p1_estimado,
            p2_necessario=resultado.p2_necessario,
            red_estimada=resultado.red_estimada,
            total_pas3=resultado.total_pas3,
            arg_pas3_necessario=resultado.arg_pas3_necessario,
            status=resultado.status,
            mensagem=resultado.mensagem,
        ))
    return resultados


def predict_strategy(inp: StrategyInput) -> StrategyResponse:
    """Calculadora de Estratégia: o P2 que o Aluno precisa no PAS 3 para bater a `nota_alvo`.

    `base_projecao` chega do cliente antigo e **não tem efeito**: escolhia entre o
    `TRIENNIUM_STATS` e a regressão do `STATS_PAS3_TREND`, e as duas saíram no ticket 05.
    Nenhuma projeção linear de prova futura sobrevive no caminho — o Ano-Âncora (ticket 12) é
    aritmética sobre anos reais e já publicados, nunca uma reta ajustada sobre eles.

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
        stats_p1, stats_p2, stats_p3, e3_e_ancora = _stats_do_ciclo(inp.ciclo_aluno, inp.lingua_e1, inp.lingua_e2)
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
    # estatísticas, por isso só dá para calcular depois que `_stats_do_ciclo` responde. Usa o
    # Ano-Âncora mais recente como escala; os cinco cenários abaixo recalculam P1/Redação com a
    # escala do próprio ano de cada um.
    previsao_ia = calc.predict_stable_components(notas_validas, stats_p1, stats_p2, stats_p3)
    p1_ia = float(previsao_ia.get('p1_pred', 0.0))
    red_ia = float(previsao_ia.get('red_pred', 0.0))

    anos_ancora_resultados: list[AnoAncoraResultado] = []
    if e3_e_ancora:
        # A Etapa 3 do próprio triênio ainda não aconteceu: os cinco cenários substituem o
        # cenário único que `_stats_do_ciclo` devolveria sozinho.
        anos_ancora_resultados = _resultados_ano_ancora(
            calc, notas_validas, stats_p1, stats_p2, inp.lingua_e2,
            arg_alvo_fallback=inp.nota_alvo,
            curso_alvo=inp.curso_alvo, cota=inp.cota, semestre=inp.semestre,
            p1_override=inp.p1_override, red_override=inp.red_override,
        )
        principal = anos_ancora_resultados[0]
        p1_estimado = principal.p1_estimado
        p2_necessario = principal.p2_necessario
        red_estimada = principal.red_estimada
        total_pas3 = principal.total_pas3
        status = principal.status
        mensagem = principal.mensagem
        # Réplica do Ano-Âncora mais recente, pela mesma razão dos demais campos únicos: um
        # cliente que ainda não sabe dos cinco cenários continua lendo um número, não um zero.
        arg_pas3_necessario = principal.arg_pas3_necessario
    else:
        # A Etapa 3 do próprio triênio já foi publicada: a resposta é exata, nenhum cenário
        # precisa ser simulado.
        resultado = calc.calculate_required_score(
            notas_validas, inp.nota_alvo, stats_p1, stats_p2, stats_p3,
            p1_override=inp.p1_override, red_override=inp.red_override,
        )
        p1_estimado = resultado.p1_estimado
        p2_necessario = resultado.p2_necessario
        red_estimada = resultado.red_estimada
        total_pas3 = resultado.total_pas3
        status = resultado.status
        mensagem = resultado.mensagem
        arg_pas3_necessario = resultado.arg_pas3_necessario

    # Reality Check (coorte histórica) — um só, sobre o cenário principal (o mais recente, quando
    # há cinco). Ticket 12 não pediu cinco reality checks, só cinco rotas.
    prob_hist = 0.0
    amostra = 0
    eb_pas3_necessario = p1_estimado + p2_necessario

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
        p1_estimado=p1_estimado,
        p2_necessario=p2_necessario,
        red_estimada=red_estimada,
        total_pas3=total_pas3,
        arg_pas3_necessario=arg_pas3_necessario,
        status=status,
        mensagem=mensagem,
        prob_hist=round(prob_hist, 1),
        amostra=amostra,
        p1_ia=p1_ia,
        red_ia=red_ia,
        anos_ancora=anos_ancora_resultados,
    )

