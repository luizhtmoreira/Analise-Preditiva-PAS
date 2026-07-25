"""
Simulador de Itens (PAS/Cebraspe)

Módulo responsável pela conversão e simulação de acertos por tipo de item
(Tipo A, Tipo B, Tipo C e Tipo D) e Redação no PAS 3 para atingir a pontuação-alvo (X).
"""

from typing import Dict, Any
from dataclasses import dataclass

PESO_TIPO_A = 1.0
PESO_TIPO_B = 2.0
PESO_TIPO_C = 2.0
PESO_TIPO_D = 3.0

MAX_TIPO_A = 120
MAX_TIPO_B = 10
MAX_TIPO_C = 10
MAX_TIPO_D = 5
MAX_REDACAO = 10.0


@dataclass
class ItemSimulationResult:
    """Resultado da simulação por itens."""
    p2_simulado: float
    meta_p2: float
    percentual_meta: float
    atingiu_meta: bool
    diferenca: float
    pontos_a: float
    pontos_b: float
    pontos_c: float
    pontos_d: float
    pontos_redacao: float


def calculate_item_simulation(
    itens_a: int,
    itens_b: int,
    itens_c: int,
    itens_d: int,
    redacao: float,
    meta_p2: float
) -> ItemSimulationResult:
    """
    Calcula a pontuação da Parte 2 (P2) e Redação simulada com base na quantidade
    de acertos de cada tipo de item do Cebraspe.

    Args:
        itens_a: Quantidade de itens Tipo A (peso 1.0, max 120)
        itens_b: Quantidade de itens Tipo B (peso 2.0)
        itens_c: Quantidade de itens Tipo C (peso 2.0)
        itens_d: Quantidade de itens Tipo D (peso 3.0)
        redacao: Nota estimada da Redação (0.0 a 10.0)
        meta_p2: Nota alvo X (P2 necessária)

    Returns:
        ItemSimulationResult com os detalhes do cálculo e status de atingimento da meta.
    """
    pontos_a = itens_a * PESO_TIPO_A
    pontos_b = itens_b * PESO_TIPO_B
    pontos_c = itens_c * PESO_TIPO_C
    pontos_d = itens_d * PESO_TIPO_D
    pontos_red = float(redacao)

    p2_simulado = pontos_a + pontos_b + pontos_c + pontos_d

    percentual = (p2_simulado / meta_p2 * 100.0) if meta_p2 > 0 else 100.0
    diferenca = p2_simulado - meta_p2
    atingiu = p2_simulado >= meta_p2

    return ItemSimulationResult(
        p2_simulado=round(p2_simulado, 2),
        meta_p2=round(meta_p2, 2),
        percentual_meta=round(percentual, 1),
        atingiu_meta=atingiu,
        diferenca=round(diferenca, 2),
        pontos_a=round(pontos_a, 2),
        pontos_b=round(pontos_b, 2),
        pontos_c=round(pontos_c, 2),
        pontos_d=round(pontos_d, 2),
        pontos_redacao=round(pontos_red, 2)
    )
