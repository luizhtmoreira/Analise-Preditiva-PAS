"""Testes de `statistics.py` — dado sintético, sem CSV nem modelo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_intelligence.statistics import calculate_approval_probability


def test_largura_de_incerteza_e_obrigatoria():
    """Relatório 11 §7.2 / ADR-0012: sem valor padrão, esquecer a largura é erro imediato, não o
    `13.49` de um modelo aposentado voltando em silêncio."""
    with pytest.raises(TypeError):
        calculate_approval_probability(predicted_arg=100.0, cutoff_score=90.0)


def test_calculo_com_largura_explicita_continua_funcionando():
    probabilidade = calculate_approval_probability(
        predicted_arg=100.0, cutoff_score=100.0, largura_incerteza=15.0
    )
    assert probabilidade == pytest.approx(0.5)


def test_largura_maior_aproxima_a_probabilidade_de_meio():
    """A largura é a única coisa que separa "quase certeza" de "não faço ideia" para a mesma
    distância até a Nota de Corte — o parâmetro que o ticket 11 tirou do código e pôs no
    manifesto."""
    estreita = calculate_approval_probability(110.0, 100.0, largura_incerteza=5.0)
    larga = calculate_approval_probability(110.0, 100.0, largura_incerteza=50.0)
    assert estreita > larga > 0.5
