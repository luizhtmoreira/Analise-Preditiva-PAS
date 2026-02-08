"""
Feature 1: Seletor de Modelo via Ensemble Dinâmico (Baseado em Incerteza)

Este módulo implementa um sistema de ensemble que pondera dinamicamente entre
um modelo conservador (Regressão Linear) e um modelo arrojado (LightGBM/RandomForest)
baseado na volatilidade do histórico do aluno.

Lógica:
- Volatilidade baixa (CV < 10%): 80% peso para Linear (conservador)
- Volatilidade alta (CV > 20%): 80% peso para modelo de árvore (arrojado)
- Transição suave via função sigmoide
"""

from typing import Tuple, Dict, Any, Protocol, Union, Optional
import numpy as np # type: ignore


class PredictorProtocol(Protocol):
    """Protocolo para modelos que implementam predict()."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def calculate_volatility(scores: np.ndarray) -> float:
    """
    Calcula o Coeficiente de Variação (CV) de uma série de notas.
    
    O CV é uma medida de dispersão relativa, calculado como:
    CV = (desvio_padrão / média) * 100
    
    Args:
        scores: Array numpy com as notas do aluno (ex: [EB_PAS1, EB_PAS2])
    
    Returns:
        Coeficiente de Variação em porcentagem (0-100+)
    
    Raises:
        ValueError: Se scores tiver menos de 2 elementos ou média zero
    
    Example:
        >>> calculate_volatility(np.array([30.0, 35.0]))
        10.88...
    """
    scores = np.asarray(scores, dtype=np.float64)
    
    if len(scores) < 2:
        raise ValueError("Necessário pelo menos 2 notas para calcular volatilidade")
    
    mean = np.mean(scores)
    std = np.std(scores, ddof=0)  # População, não amostra
    
    if mean == 0:
        raise ValueError("Média zero: impossível calcular CV")
    
    return float((std / mean) * 100)


def _sigmoid_weight(cv: float, low_cv: float = 10.0, high_cv: float = 20.0) -> float:
    """
    Calcula o peso para o modelo arrojado usando uma função sigmoide suave.
    
    A sigmoide transiciona suavemente entre 0.2 (CV baixo) e 0.8 (CV alto),
    evitando saltos bruscos na ponderação.
    
    Args:
        cv: Coeficiente de Variação
        low_cv: Limiar inferior (CV abaixo disso → peso 0.2)
        high_cv: Limiar superior (CV acima disso → peso 0.8)
    
    Returns:
        Peso para o modelo arrojado (0.2 a 0.8)
    """
    # Normaliza CV para escala -6 a +6 (para sigmoid ter transição suave)
    midpoint = (low_cv + high_cv) / 2
    scale = (high_cv - low_cv) / 2
    
    x = (cv - midpoint) / scale * 3  # Multiplica por 3 para transição mais acentuada
    
    # Sigmoid: 1 / (1 + e^-x)
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Escala de 0.2 (conservador) a 0.8 (arrojado)
    weight_aggressive = 0.2 + 0.6 * sigmoid
    
    return float(weight_aggressive)


def predict_with_dynamic_ensemble(
    eb_pas1: float,
    eb_pas2: float,
    model_conservative: PredictorProtocol,
    model_aggressive: PredictorProtocol,
    features: Optional[np.ndarray] = None,
    scaler: Any = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Prediz o escore bruto do PAS 3 usando ensemble dinâmico baseado em volatilidade.
    
    A função calcula a volatilidade do aluno (CV entre PAS 1 e PAS 2) e
    pondera dinamicamente entre um modelo conservador (Linear) e um arrojado
    (LightGBM/RandomForest).
    
    Args:
        eb_pas1: Escore Bruto do PAS 1
        eb_pas2: Escore Bruto do PAS 2
        model_conservative: Modelo conservador (ex: LinearRegression).
                           Se usar scaler, recebe features já escaladas.
        model_aggressive: Modelo arrojado (ex: LGBMRegressor).
                         Recebe features não escaladas.
        features: Array com todas as features para predição.
                  Se None, usa [eb_pas1, eb_pas2, eb_pas2-eb_pas1] como padrão.
        scaler: StandardScaler para o modelo conservador (opcional)
    
    Returns:
        Tuple contendo:
        - prediction: float - Predição ponderada do ensemble
        - metadata: dict - Informações sobre a decisão:
            - volatility_cv: Coeficiente de Variação
            - weight_conservative: Peso do modelo conservador
            - weight_aggressive: Peso do modelo arrojado
            - pred_conservative: Predição do modelo conservador
            - pred_aggressive: Predição do modelo arrojado
    
    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> from lightgbm import LGBMRegressor
        >>> # Assumindo modelos treinados
        >>> pred, meta = predict_with_dynamic_ensemble(
        ...     eb_pas1=30.0, eb_pas2=35.0,
        ...     model_conservative=lr_model,
        ...     model_aggressive=lgbm_model
        ... )
        >>> print(f"Predição: {pred:.2f}, CV: {meta['volatility_cv']:.2f}%")
    """
    # Calcula volatilidade
    scores = np.array([eb_pas1, eb_pas2])
    cv = calculate_volatility(scores)
    
    # Calcula pesos via sigmoide
    weight_aggressive = _sigmoid_weight(cv)
    weight_conservative = 1.0 - weight_aggressive
    
    # Prepara features se não fornecidas
    if features is None:
        features = np.array([[eb_pas1, eb_pas2, eb_pas2 - eb_pas1]])
    else:
        features = np.atleast_2d(features)
    
    # Predição conservador (pode precisar de scaler)
    if scaler is not None:
        features_scaled = scaler.transform(features)
        pred_conservative = float(model_conservative.predict(features_scaled)[0])
    else:
        pred_conservative = float(model_conservative.predict(features)[0])
    
    # Predição arrojado (sem scaler)
    pred_aggressive = float(model_aggressive.predict(features)[0])
    
    # Ensemble ponderado
    prediction = (weight_conservative * pred_conservative + 
                  weight_aggressive * pred_aggressive)
    
    metadata = {
        "volatility_cv": cv,
        "weight_conservative": weight_conservative,
        "weight_aggressive": weight_aggressive,
        "pred_conservative": pred_conservative,
        "pred_aggressive": pred_aggressive,
        "strategy": "conservador" if weight_conservative > 0.5 else "arrojado",
    }
    
    return prediction, metadata
