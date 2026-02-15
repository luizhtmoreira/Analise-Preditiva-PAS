import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_approval_probability(predicted_arg: float, cutoff_score: float, rmse: float = 13.49) -> float:
    """
    Calcula a probabilidade de aprovação baseada na distribuição normal dos erros do modelo.
    
    A incerteza do modelo é modelada como uma distribuição normal centrada no argumento previsto
    com desvio padrão igual ao RMSE do modelo.
    
    Args:
        predicted_arg: Argumento final previsto pelo modelo
        cutoff_score: Nota de corte do curso desejado
        rmse: Root Mean Squared Error do modelo (padrão 13.49)
        
    Returns:
        float: Probabilidade de aprovação (0.0 a 1.0)
    """
    # Z-score: quão longe a nota de corte está da nossa previsão, em unidades de desvio padrão
    # Queremos P(X > cutoff), que é 1 - CDF(cutoff)
    # X ~ N(predicted_arg, rmse^2)
    
    probability = 1 - norm.cdf(cutoff_score, loc=predicted_arg, scale=rmse)
    return float(probability)


def calculate_cohort_evolution_probability(
    current_student_data: dict, 
    target_arg: float, 
    historical_df: pd.DataFrame,
    tolerance_percent: float = 0.10
) -> tuple[float, int]:
    """
    Calcula a probabilidade histórica de um aluno atingir a nota necessária no PAS 3.
    baseada em alunos com desempenho similar no PAS 1 e 2.
    
    Args:
        current_student_data: Dict com 'eb_pas1' e 'eb_pas2'
        target_arg: EB PAS 3 necessário (P1 + P2 do PAS 3)
        historical_df: DataFrame com histórico (banco_alunos_pas_final.csv)
        tolerance_percent: Margem de tolerância para encontrar vizinhos (default 10%)
        
    Returns:
        tuple: (probabilidade_sucesso (%), tamanho_amostra)
    """
    if historical_df is None or historical_df.empty:
        return 0.0, 0
        
    hist_data = historical_df.copy()
    
    # 1. Normaliza nomes de colunas (Case-insensitive e mapeamento)
    cols_norm = {c.upper(): c for c in hist_data.columns}
    
    # Identifica colunas necessárias
    p1_eb_col = cols_norm.get('EB_PAS1')
    p2_eb_col = cols_norm.get('EB_PAS2')
    
    # Se não tem EB, tenta calcular de P1+P2
    if not p1_eb_col and 'P1_PAS1' in cols_norm and 'P2_PAS1' in cols_norm:
        hist_data['EB_PAS1'] = hist_data[cols_norm['P1_PAS1']] + hist_data[cols_norm['P2_PAS1']]
        p1_eb_col = 'EB_PAS1'
        
    if not p2_eb_col and 'P1_PAS2' in cols_norm and 'P2_PAS2' in cols_norm:
        hist_data['EB_PAS2'] = hist_data[cols_norm['P1_PAS2']] + hist_data[cols_norm['P2_PAS2']]
        p2_eb_col = 'EB_PAS2'
    
    # Calcula EB_PAS3 do histórico (P1_PAS3 + P2_PAS3)
    p1_pas3_col = cols_norm.get('P1_PAS3')
    p2_pas3_col = cols_norm.get('P2_PAS3')
    
    if not p1_pas3_col or not p2_pas3_col:
        return 0.0, 0
    
    hist_data['EB_PAS3'] = hist_data[p1_pas3_col] + hist_data[p2_pas3_col]
        
    if not all([p1_eb_col, p2_eb_col]):
        return 0.0, 0
        
    # 2. Calcula proxy de similaridade
    user_score_proxy = (current_student_data['eb_pas1'] * 1 + current_student_data['eb_pas2'] * 2)
    hist_data['Score_Proxy'] = hist_data[p1_eb_col] * 1 + hist_data[p2_eb_col] * 2
    
    # 3. Define faixa de similaridade
    lower_bound = user_score_proxy * (1 - tolerance_percent)
    upper_bound = user_score_proxy * (1 + tolerance_percent)
    
    # 4. Filtra vizinhos
    neighbors = hist_data[
        (hist_data['Score_Proxy'] >= lower_bound) & 
        (hist_data['Score_Proxy'] <= upper_bound)
    ]
    
    sample_size = len(neighbors)
    
    if sample_size == 0:
        return 0.0, 0
        
    # 5. Verifica sucesso (compara EB_PAS3 real com EB_PAS3 necessário)
    successes = neighbors[neighbors['EB_PAS3'] >= target_arg]
    success_rate = (len(successes) / sample_size) * 100
    
    return success_rate, sample_size
