"""
PAS Intelligence - Sistema de Inteligência para o PAS UnB

Módulos:
- ensemble: Seletor de modelo dinâmico baseado em volatilidade
- argument_calculator: Calculadora do Argumento Final com projeção temporal
- recommender: Sistema de recomendação de cursos (KNN vetorial)
- ab_testing: Validação estatística de grupos (Teste A/B)
"""
from typing import TYPE_CHECKING

# Se for apenas o linter verificando, importa tudo (para ele parar de reclamar)
if TYPE_CHECKING:
    from .ensemble import predict_with_dynamic_ensemble, calculate_volatility # type: ignore
    from .argument_calculator import calculate_argument_final, calculate_argument_part, project_historical_stats # type: ignore
    from .recommender import CourseRecommender, recommend_courses # type: ignore
    from .ab_testing import compare_groups # type: ignore
    from .target_calculator import TargetCalculator, get_reverse_prediction # type: ignore

__version__ = "1.0.0"

# Lazy imports - carrega apenas quando acessado
def __getattr__(name):
    if name == "predict_with_dynamic_ensemble":
        from .ensemble import predict_with_dynamic_ensemble # type: ignore
        return predict_with_dynamic_ensemble
    elif name == "calculate_volatility":
        from .ensemble import calculate_volatility # type: ignore
        return calculate_volatility
    elif name == "calculate_argument_final":
        from .argument_calculator import calculate_argument_final # type: ignore
        return calculate_argument_final
    elif name == "calculate_argument_part":
        from .argument_calculator import calculate_argument_part # type: ignore
        return calculate_argument_part
    elif name == "project_historical_stats":
        from .argument_calculator import project_historical_stats # type: ignore
        return project_historical_stats
    elif name == "CourseRecommender":
        from .recommender import CourseRecommender # type: ignore
        return CourseRecommender
    elif name == "recommend_courses":
        from .recommender import recommend_courses # type: ignore
        return recommend_courses
    elif name == "compare_groups":
        from .ab_testing import compare_groups # type: ignore
        return compare_groups
    elif name == "TargetCalculator":
        from .target_calculator import TargetCalculator # type: ignore
        return TargetCalculator
    elif name == "get_reverse_prediction":
        from .target_calculator import get_reverse_prediction # type: ignore
        return get_reverse_prediction
    raise AttributeError(f"module 'pas_intelligence' has no attribute '{name}'")

__all__ = [
    "predict_with_dynamic_ensemble",
    "calculate_volatility",
    "calculate_argument_final",
    "calculate_argument_part",
    "project_historical_stats",
    "CourseRecommender",
    "recommend_courses",
    "compare_groups",
    "TargetCalculator",
    "get_reverse_prediction",
]
