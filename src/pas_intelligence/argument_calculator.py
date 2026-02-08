

from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import numpy as np # type: ignore


# Pesos oficiais do PAS/UnB
PESO_P1 = 0.72 
PESO_P2 = 8.28 
PESO_REDACAO = 1.00


@dataclass
class HistoricalStats:
    """Estatísticas históricas de uma parte da prova."""
    mean_p1: float # Média Parte 1 (língua estrangeira)
    std_p1: float # Desvio padrão Parte 1
    mean_p2: float # Média Parte 2 (demais disciplinas)
    std_p2: float # Desvio padrão Parte 2
    mean_red: float # Média Redação
    std_red: float  # Desvio padrão Redação


def project_historical_stats(
    historical_means: List[float],
    historical_stds: List[float],
    years: Optional[List[int]] = None,
) -> Tuple[float, float]:
    """
    Projeta média e desvio-padrão para o próximo ano usando regressão linear.
    
    Usa numpy.polyfit (grau 1) para capturar a tendência temporal e projetar
    os valores para o ano seguinte. Útil quando a prova está ficando mais
    difícil/fácil ao longo do tempo.
    
    Args:
        historical_means: Lista de médias dos últimos N anos (cronológica)
        historical_stds: Lista de desvios-padrão dos últimos N anos
        years: Anos correspondentes (opcional, usa índices se não fornecido)
    
    Returns:
        Tuple (mean_projected, std_projected) para o próximo ano
    
    Raises:
        ValueError: Se listas tiverem tamanhos diferentes ou menos de 2 pontos
    
    Example:
        >>> means = [28.5, 27.8, 27.2, 26.9, 26.5]  # Tendência de queda
        >>> stds = [10.1, 10.3, 10.5, 10.8, 11.0]   # Tendência de aumento
        >>> mean_proj, std_proj = project_historical_stats(means, stds)
        >>> print(f"Média projetada: {mean_proj:.2f}")  # ~26.1
    """
    if len(historical_means) != len(historical_stds):
        raise ValueError("Listas de médias e desvios devem ter mesmo tamanho")
    
    if len(historical_means) < 2:
        raise ValueError("Necessário pelo menos 2 pontos históricos para projeção")
    
    # Se anos não fornecidos, usa índices (0, 1, 2, ...)
    if years is None:
        years = list(range(len(historical_means)))
    
    years = np.array(years, dtype=np.float64)
    means = np.array(historical_means, dtype=np.float64)
    stds = np.array(historical_stds, dtype=np.float64)
    
    # Regressão linear (grau 1)
    coef_mean = np.polyfit(years, means, 1)
    coef_std = np.polyfit(years, stds, 1)
    
    # Projeta para o próximo "ano" (índice máximo + 1)
    next_year = years[-1] + 1
    mean_projected = float(np.polyval(coef_mean, next_year))
    std_projected = float(np.polyval(coef_std, next_year))
    
    # Garante que std projetado seja positivo
    std_projected = max(std_projected, 0.1)
    
    return mean_projected, std_projected


def calculate_argument_part(
    nota: float,
    media: float,
    desvio_padrao: float,
    peso: float,
) -> float:
    """
    Calcula o argumento padronizado de uma parte da prova.
    
    Fórmula: A = [(Nota - Média) / Desvio_Padrão] * peso
    
    Args:
        nota: Nota obtida pelo aluno
        media: Média da parte da prova
        desvio_padrao: Desvio padrão da parte da prova
        peso: Peso da parte (0.72 para P1, 8.28 para P2, 1.00 para Redação)
    
    Returns:
        Argumento padronizado (pode ser negativo)
    
    Raises:
        ValueError: Se desvio_padrao for zero ou negativo
    """
    if desvio_padrao <= 0:
        raise ValueError(f"Desvio padrão deve ser positivo, recebido: {desvio_padrao}")
    
    return ((nota - media) / desvio_padrao) * peso


def calculate_argument_etapa(
    nota_p1: float,
    nota_p2: float,
    nota_red: float,
    stats: HistoricalStats,
) -> float:
    """
    Calcula o argumento total de uma etapa (PAS 1, 2 ou 3).
    
    Argumento Etapa = Arg_P1 + Arg_P2 + Arg_Redação
    
    Args:
        nota_p1: Nota na Parte 1 (língua estrangeira)
        nota_p2: Nota na Parte 2 (demais disciplinas)
        nota_red: Nota na Redação
        stats: Estatísticas históricas da etapa
    
    Returns:
        Argumento total da etapa
    """
    arg_p1 = calculate_argument_part(nota_p1, stats.mean_p1, stats.std_p1, PESO_P1)
    arg_p2 = calculate_argument_part(nota_p2, stats.mean_p2, stats.std_p2, PESO_P2)
    arg_red = calculate_argument_part(nota_red, stats.mean_red, stats.std_red, PESO_REDACAO)
    
    return arg_p1 + arg_p2 + arg_red


def calculate_argument_final(
    notas: Dict[str, float],
    stats_pas1: HistoricalStats,
    stats_pas2: HistoricalStats,
    stats_pas3: HistoricalStats,
) -> Dict[str, Any]:
    """
    Calcula o Argumento Final completo do PAS/UnB.
    
    Fórmula: AF = 1*AP1 + 2*AP2 + 3*AP3
    
    Onde cada APn é calculado como a soma dos argumentos padronizados
    de cada parte da prova (P1, P2, Redação).
    
    Args:
        notas: Dicionário com as notas do aluno:
            - P1_PAS1, P2_PAS1, Red_PAS1: Notas do PAS 1
            - P1_PAS2, P2_PAS2, Red_PAS2: Notas do PAS 2
            - P1_PAS3, P2_PAS3, Red_PAS3: Notas do PAS 3
        stats_pas1: Estatísticas históricas do PAS 1
        stats_pas2: Estatísticas históricas do PAS 2
        stats_pas3: Estatísticas históricas do PAS 3
    
    Returns:
        Dicionário contendo:
        - arg_pas1: Argumento do PAS 1
        - arg_pas2: Argumento do PAS 2
        - arg_pas3: Argumento do PAS 3
        - arg_final: Argumento Final ponderado
        - eb_total: Escore Bruto total (soma de todas as partes)
    
    Example:
        >>> notas = {
        ...     'P1_PAS1': 7.5, 'P2_PAS1': 30.0, 'Red_PAS1': 7.0,
        ...     'P1_PAS2': 8.0, 'P2_PAS2': 35.0, 'Red_PAS2': 7.5,
        ...     'P1_PAS3': 8.5, 'P2_PAS3': 40.0, 'Red_PAS3': 8.0,
        ... }
        >>> result = calculate_argument_final(notas, stats1, stats2, stats3)
        >>> print(f"Argumento Final: {result['arg_final']:.3f}")
    """
    # Calcula Escore Bruto de cada etapa
    eb_pas1 = notas['P1_PAS1'] + notas['P2_PAS1']
    eb_pas2 = notas['P1_PAS2'] + notas['P2_PAS2']
    eb_pas3 = notas['P1_PAS3'] + notas['P2_PAS3']
    
    # Calcula argumento de cada etapa
    arg_pas1 = calculate_argument_etapa(
        notas['P1_PAS1'], notas['P2_PAS1'], notas['Red_PAS1'], stats_pas1
    )
    arg_pas2 = calculate_argument_etapa(
        notas['P1_PAS2'], notas['P2_PAS2'], notas['Red_PAS2'], stats_pas2
    )
    arg_pas3 = calculate_argument_etapa(
        notas['P1_PAS3'], notas['P2_PAS3'], notas['Red_PAS3'], stats_pas3
    )
    
    # Argumento Final com pesos (1, 2, 3)
    arg_final = 1 * arg_pas1 + 2 * arg_pas2 + 3 * arg_pas3
    
    # Arredondamento para 3 casas decimais (conforme edital)
    return {
        'arg_pas1': round(arg_pas1, 3),  # type: ignore
        'arg_pas2': round(arg_pas2, 3), # type: ignore
        'arg_pas3': round(arg_pas3, 3), # type: ignore
        'arg_final': round(arg_final, 3), # type: ignore
        'eb_pas1': round(eb_pas1, 3), # type: ignore
        'eb_pas2': round(eb_pas2, 3), # type: ignore
        'eb_pas3': round(eb_pas3, 3), # type: ignore
        'eb_total': round(eb_pas1 + eb_pas2 + eb_pas3, 3), # type: ignore
    }


def predict_argument_from_eb(
    eb_pas3_predicted: float,
    notas_existentes: Dict[str, float],
    stats_pas1: HistoricalStats,
    stats_pas2: HistoricalStats,
    stats_pas3: HistoricalStats,
    proporcao_p1_p2: float = 0.25,
    proporcao_redacao: float = 0.7,
) -> Dict[str, Any]:
    """
    Estima o Argumento Final a partir de uma predição de Escore Bruto do PAS 3.
    
    Usado quando temos apenas a predição do EB_PAS3 e precisamos estimar
    o Argumento Final completo. Assume proporções típicas entre P1, P2 e Redação.
    
    Args:
        eb_pas3_predicted: Escore Bruto previsto para o PAS 3
        notas_existentes: Notas já conhecidas (PAS 1 e 2)
        stats_pas*: Estatísticas históricas de cada etapa
        proporcao_p1_p2: Proporção típica de P1 no EB (padrão 0.25)
        proporcao_redacao: Proporção típica da nota redação (padrão 0.7)
    
    Returns:
        Mesmo formato de calculate_argument_final()
    """
    # Estima P1 e P2 do PAS 3 baseado em proporções típicas
    p1_pas3_est = eb_pas3_predicted * proporcao_p1_p2
    p2_pas3_est = eb_pas3_predicted * (1 - proporcao_p1_p2)
    
    # Estima redação como proporção da nota máxima (10)
    red_pas3_est = 10.0 * proporcao_redacao
    
    notas_completas = {
        **notas_existentes,
        'P1_PAS3': p1_pas3_est,
        'P2_PAS3': p2_pas3_est,
        'Red_PAS3': red_pas3_est,
    }
    
    result = calculate_argument_final(
        notas_completas, stats_pas1, stats_pas2, stats_pas3
    )
    result['is_estimated'] = True
    
    return result


