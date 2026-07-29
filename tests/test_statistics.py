"""Testes de `statistics.py` — dado sintético, sem CSV nem modelo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_intelligence.statistics import calculate_approval_probability


def test_rmse_e_obrigatorio():
    """Relatório 11 §7.2 / ADR-0012: sem valor padrão, esquecer o `rmse` é erro imediato, não o
    `13.49` de um modelo aposentado voltando em silêncio."""
    with pytest.raises(TypeError):
        calculate_approval_probability(predicted_arg=100.0, cutoff_score=90.0)


def test_calculo_com_rmse_explicito_continua_funcionando():
    probabilidade = calculate_approval_probability(
        predicted_arg=100.0, cutoff_score=100.0, rmse=15.0
    )
    assert probabilidade == pytest.approx(0.5)
