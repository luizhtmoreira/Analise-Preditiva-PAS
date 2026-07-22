"""
Preditor PAS 3 — previsão individual de EB e Argumento Final.
"""
import numpy as np

from api.schemas.predict import PredictInput, PredictResponse, CourseResult
from api.services import gestao_service          # referência ao módulo, não ao valor
from api.services.gestao_service import (
    _find_best_match,
    _build_cutoff_maps,
    ARG_FINAL_MAE,
)
from pas_intelligence.statistics import calculate_approval_probability

TOP_CURSOS_LIMIT = 10
MIN_PROB_THRESHOLD = 0.30
EB_MAE = 5.0  # margem aproximada para EB PAS 3


def predict_student(inp: PredictInput) -> PredictResponse:
    print(f"--- API PREDICT --- is_logged_in: {inp.is_logged_in}, cota: {inp.cota}, trienio: {inp.trienio}")
    eb_p1 = inp.p1_pas1 + inp.p2_pas1
    eb_p2 = inp.p1_pas2 + inp.p2_pas2
    c_eb  = eb_p2 - eb_p1
    c_red = inp.red_pas2 - inp.red_pas1

    features = np.array([[eb_p1, inp.red_pas1, eb_p2, inp.red_pas2, c_eb, c_red]])

    # EB PAS 3 previsto (LightGBM)
    eb_pas3_previsto = 0.0
    if gestao_service._eb_model is not None:
        eb_pas3_previsto = float(gestao_service._eb_model.predict(features)[0])

    # Argumento Final previsto
    arg_previsto = 0.0
    if gestao_service._arg_model is not None:
        arg_previsto = float(gestao_service._arg_model.predict(features)[0])

    modelo_disponivel = gestao_service._arg_model is not None

    c1_map, c2_map, trienio_ref = _build_cutoff_maps(inp.trienio)

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
                prob = calculate_approval_probability(arg_previsto, nota, rmse=ARG_FINAL_MAE)
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

    top_cursos: list[CourseResult] = []
    seen: set[str] = set()

    if inp.is_logged_in:
        # Com login: top 10 cursos com prob >= 30% ordenados por maior probabilidade
        candidates = []
        for semestre, m in [("1°", m1), ("2°", m2)]:
            for course_key, nota in m.items():
                prob = calculate_approval_probability(arg_previsto, nota, rmse=ARG_FINAL_MAE)
                if prob >= MIN_PROB_THRESHOLD:
                    candidates.append((course_key, nota, prob, semestre))
        
        # Ordenar por maior probabilidade
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Selecionar sem duplicar o course_key
        for course_key, nota, prob, semestre in candidates:
            if course_key in seen:
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
    else:
        # Sem login: 3 cursos com corte mais próximo de arg_previsto (abs(nota - arg_previsto))
        candidates = []
        for semestre, m in [("1°", m1), ("2°", m2)]:
            for course_key, nota in m.items():
                diff = abs(nota - arg_previsto)
                candidates.append((course_key, nota, diff, semestre))
        
        # Ordenar pela menor diferença absoluta
        candidates.sort(key=lambda x: x[2])
        
        # Selecionar sem duplicar o course_key
        for course_key, nota, diff, semestre in candidates:
            if course_key in seen:
                continue
            seen.add(course_key)
            prob = calculate_approval_probability(arg_previsto, nota, rmse=ARG_FINAL_MAE)
            parts = _parse_course_key(course_key)
            top_cursos.append(CourseResult(
                curso=parts["curso"],
                turno=parts["turno"],
                campus=parts["campus"],
                nota_corte=round(nota, 3),
                prob=round(prob * 100, 1),
                semestre=semestre,
            ))
            if len(top_cursos) >= 3:
                break

    return PredictResponse(
        eb_pas3_previsto=round(eb_pas3_previsto, 3),
        arg_previsto=round(arg_previsto, 1),
        arg_min=round(arg_previsto - ARG_FINAL_MAE, 1),
        arg_max=round(arg_previsto + ARG_FINAL_MAE, 1),
        curso_alvo_result=curso_alvo_result,
        top_cursos=top_cursos,
        trienio_ref=trienio_ref,
        modelo_disponivel=modelo_disponivel,
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
