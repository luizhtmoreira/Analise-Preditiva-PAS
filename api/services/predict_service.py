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

from api.schemas.predict import PredictInput, PredictResponse, CourseResult
from api.services import gestao_service          # referência ao módulo, não ao valor
from api.services.gestao_service import _find_best_match, _build_cutoff_maps
from pas_intelligence.model_package import (
    EntradaDePrevisao,
    EstatisticasIndisponiveisError,
    NotasDeEtapa,
)
from pas_intelligence.statistics import calculate_approval_probability

logger = logging.getLogger(__name__)

TOP_CURSOS_LIMIT = 8
MIN_PROB_THRESHOLD = 0.30

SEM_PACOTE = (
    "Nenhum pacote de modelo carregado. Sem pacote não há previsão nem Largura de Incerteza — "
    "o estado 'previsão sim, largura não' não é representável (ADR-0012)."
)


def entrada_de_previsao(inp: PredictInput) -> EntradaDePrevisao:
    return EntradaDePrevisao(
        etapa_1=NotasDeEtapa(p1=inp.p1_pas1, p2=inp.p2_pas1, redacao=inp.red_pas1),
        etapa_2=NotasDeEtapa(p1=inp.p1_pas2, p2=inp.p2_pas2, redacao=inp.red_pas2),
        lingua=inp.lingua,
        trienio=inp.trienio,
    )


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

        for semestre, m in [("1°", m1), ("2°", m2)]:
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

    # Top cursos acessíveis
    seen: set[str] = set()
    top_cursos: list[CourseResult] = []

    for semestre, m in [("1°", m1), ("2°", m2)]:
        for course_key, nota in sorted(m.items(), key=lambda x: x[1], reverse=True):
            if course_key in seen:
                continue
            prob = calculate_approval_probability(arg_previsto, nota, largura_incerteza=largura)
            if prob < MIN_PROB_THRESHOLD:
                continue
            seen.add(course_key)
            parts = _parse_course_key(course_key)
            top_cursos.append(CourseResult(
                curso=parts["curso"],
                turno=parts["turno"],
                campus=parts["campus"],
                nota_corte=round(nota, 3),
                prob=round(prob * 100, 1),
                semestre=semestre,
            ))
            if len(top_cursos) >= TOP_CURSOS_LIMIT:
                break
        if len(top_cursos) >= TOP_CURSOS_LIMIT:
            break

    top_cursos.sort(key=lambda c: c.prob, reverse=True)

    return PredictResponse(
        arg_previsto=round(arg_previsto, 1),
        a1=round(previsao.a1, 3),
        a2=round(previsao.a2, 3),
        a3_previsto=round(previsao.a3, 3),
        largura_incerteza=round(largura, 3),
        etapa_1_ausente=previsao.etapa_1_ausente,
        curso_alvo_result=curso_alvo_result,
        top_cursos=top_cursos[:TOP_CURSOS_LIMIT],
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
