"""
Testes unitários para o Simulador de Itens.
"""

import pytest
from pas_intelligence.simulador_itens import calculate_item_simulation


def test_simulador_itens_basic_calculation():
    # 50 Tipo A (50), 2 Tipo B (4), 3 Tipo C (6), 1 Tipo D (3), Redacao 8.0 -> P2 = 63.0
    res = calculate_item_simulation(
        itens_a=50,
        itens_b=2,
        itens_c=3,
        itens_d=1,
        redacao=8.0,
        meta_p2=60.0
    )

    assert res.p2_simulado == 63.0
    assert res.pontos_a == 50.0
    assert res.pontos_b == 4.0
    assert res.pontos_c == 6.0
    assert res.pontos_d == 3.0
    assert res.atingiu_meta is True
    assert res.diferenca == 3.0
    assert res.percentual_meta == 105.0


def test_simulador_itens_below_target():
    res = calculate_item_simulation(
        itens_a=30,
        itens_b=1,
        itens_c=1,
        itens_d=0,
        redacao=5.0,
        meta_p2=50.0
    )

    # 30*1 + 1*2 + 1*2 + 0*3 = 34
    assert res.p2_simulado == 34.0
    assert res.atingiu_meta is False
    assert res.diferenca == -16.0
    assert res.percentual_meta == 68.0
