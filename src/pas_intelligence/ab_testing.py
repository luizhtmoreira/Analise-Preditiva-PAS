"""
Feature 4: Validação Estatística de Grupos (Teste A/B)

Este módulo implementa comparação estatística entre dois grupos de alunos
usando teste t de Student para amostras independentes.

Usado para relatórios gerenciais como:
- "Turma A vs Turma B: houve diferença significativa?"
- "Escola X vs Escola Y: desempenho diferente?"
- "Antes vs Depois de intervenção pedagógica"
"""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np # type: ignore
from scipy import stats # type: ignore


@dataclass  
class ABTestResult:
    """Resultado estruturado de um teste A/B."""
    statistically_significant: bool
    p_value: float
    t_statistic: float
    group_a_mean: float
    group_b_mean: float
    group_a_std: float
    group_b_std: float
    group_a_n: int
    group_b_n: int
    difference: float
    interpretation: str
    effect_size: float  # Cohen's d


def compare_groups(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alpha: float = 0.05,
    group_a_name: str = "Grupo A",
    group_b_name: str = "Grupo B",
    metric_name: str = "média",
) -> Dict:
    """
    Compara estatisticamente as médias de dois grupos usando teste t independente.
    
    O teste t de Student para amostras independentes verifica se a diferença
    observada entre as médias dos grupos é estatisticamente significativa
    ou pode ser atribuída ao acaso.
    
    Args:
        group_a: Array numpy com valores do Grupo A
        group_b: Array numpy com valores do Grupo B
        alpha: Nível de significância (padrão 0.05 = 95% confiança)
        group_a_name: Nome descritivo do Grupo A (para mensagem)
        group_b_name: Nome descritivo do Grupo B (para mensagem)
        metric_name: Nome da métrica comparada (para mensagem)
    
    Returns:
        Dicionário contendo:
        - statistically_significant: bool - Se p < alpha
        - p_value: float - Valor p do teste
        - t_statistic: float - Estatística t calculada
        - group_a_mean/std/n: Estatísticas do Grupo A
        - group_b_mean/std/n: Estatísticas do Grupo B
        - difference: float - Diferença entre médias (A - B)
        - effect_size: float - Tamanho do efeito (Cohen's d)
        - interpretation: str - Texto interpretativo em português
    
    Raises:
        ValueError: Se grupos tiverem menos de 2 elementos
    
    Example:
        >>> turma_a = np.array([30.5, 32.1, 28.7, 31.2, 29.8])
        >>> turma_b = np.array([25.3, 27.8, 24.1, 26.5, 25.9])
        >>> result = compare_groups(turma_a, turma_b, group_a_name="Turma A")
        >>> print(result['interpretation'])
        'A diferença é estatisticamente significante (p = 0.003). 
         Turma A tem média superior (30.06 vs 25.92).'
    """
    group_a = np.asarray(group_a, dtype=np.float64)
    group_b = np.asarray(group_b, dtype=np.float64)
    
    # Validações
    if len(group_a) < 2:
        raise ValueError(f"Grupo A precisa de pelo menos 2 elementos, tem {len(group_a)}")
    if len(group_b) < 2:
        raise ValueError(f"Grupo B precisa de pelo menos 2 elementos, tem {len(group_b)}")
    
    # Estatísticas descritivas
    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    std_a = float(np.std(group_a, ddof=1))  # Amostra
    std_b = float(np.std(group_b, ddof=1))
    n_a = len(group_a)
    n_b = len(group_b)
    
    difference = mean_a - mean_b
    
    # Teste t para amostras independentes
    # equal_var=False usa a correção de Welch (mais robusta)
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    significant = p_value < alpha
    
    # Tamanho do efeito (Cohen's d)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    if pooled_std > 0:
        cohens_d = difference / pooled_std
    else:
        cohens_d = 0.0
    
    # Interpretação do tamanho do efeito
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligenciável"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "pequeno"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "médio"
    else:
        effect_interpretation = "grande"
    
    # Gera interpretação textual
    if significant:
        if difference > 0:
            comparison = f"{group_a_name} tem {metric_name} superior"
        else:
            comparison = f"{group_b_name} tem {metric_name} superior"
        
        p_val_str = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
        interpretation = (
            f"A diferença é estatisticamente significante (p = {p_val_str}). "
            f"{comparison} ({mean_a:.2f} vs {mean_b:.2f}). "
            f"O tamanho do efeito é {effect_interpretation} (d = {cohens_d:.2f})."
        )
    else:
        p_val_str = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
        interpretation = (
            f"A diferença NÃO é estatisticamente significante (p = {p_val_str}). "
            f"A diferença observada ({mean_a:.2f} vs {mean_b:.2f}) pode ser aleatória. "
            f"Não há evidência suficiente para afirmar que os grupos diferem."
        )
    
    return {
        'statistically_significant': significant,
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'group_a_mean': mean_a,
        'group_b_mean': mean_b,
        'group_a_std': std_a,
        'group_b_std': std_b,
        'group_a_n': n_a,
        'group_b_n': n_b,
        'difference': difference,
        'effect_size': float(cohens_d),
        'effect_interpretation': effect_interpretation,
        'interpretation': interpretation,
        'alpha': alpha,
    }


def compare_multiple_groups(
    groups: Dict[str, np.ndarray],
    control_group: Optional[str] = None,
    alpha: float = 0.05,
    metric_name: str = "média",
) -> Dict[str, Dict]:
    """
    Compara múltiplos grupos contra um grupo controle (ou todos entre si).
    
    Args:
        groups: Dicionário {nome_grupo: array_valores}
        control_group: Nome do grupo controle. Se None, usa o primeiro.
        alpha: Nível de significância
        metric_name: Nome da métrica
    
    Returns:
        Dicionário de resultados, um para cada comparação
    
    Example:
        >>> groups = {
        ...     'Turma A': np.array([30, 32, 28]),
        ...     'Turma B': np.array([25, 27, 24]),
        ...     'Turma C': np.array([28, 29, 27]),
        ... }
        >>> results = compare_multiple_groups(groups, control_group='Turma A')
    """
    if len(groups) < 2:
        raise ValueError("Necessário pelo menos 2 grupos para comparação")
    
    group_names = list(groups.keys())
    
    if control_group is None:
        control_group = group_names[0]
    
    if control_group not in groups:
        raise ValueError(f"Grupo controle '{control_group}' não encontrado")
    
    results = {}
    control_data = groups[control_group]
    
    for name, data in groups.items():
        if name == control_group:
            continue
        
        comparison_key = f"{control_group} vs {name}"
        results[comparison_key] = compare_groups(
            group_a=control_data,
            group_b=data,
            alpha=alpha,
            group_a_name=control_group,
            group_b_name=name,
            metric_name=metric_name,
        )
    
    return results
