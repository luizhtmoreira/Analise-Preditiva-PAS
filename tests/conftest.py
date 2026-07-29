"""Fixtures compartilhadas — nenhum artefato de produção e nenhuma linha de Aluno.

O pacote destes testes é um LightGBM de brinquedo treinado sobre ruído, escrito em disco no mesmo
formato que `training_pipeline.treinar` produz. Ele existe para que os testes exercitem o
**carregamento e a montagem de features de verdade** (que é onde moram os erros silenciosos) sem
depender de `models/`, que fica fora do git ([[project_parser_privacy]]).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from pas_intelligence.dataset_pas3 import FEATURES_CANONICAS

# Valores **arbitrários**, escolhidos só por serem distinguíveis um do outro: os testes provam que
# cada classe recebe a sua largura, não que a largura vale isto. As larguras reais vivem no
# `manifest.json` do pacote promovido e mudam a cada retreino — fixá-las aqui faria um teste
# quebrar de mentira no próximo treino.
SIGMA_COM_ETAPA_1 = 4.0
SIGMA_SEM_ETAPA_1 = 6.0


@pytest.fixture
def diretorio_do_pacote(tmp_path: Path) -> Path:
    """Um pacote no formato do `training_pipeline`, com booster de brinquedo."""
    from lightgbm import LGBMRegressor  # type: ignore

    aleatorio = np.random.default_rng(20260728)
    treino = pd.DataFrame(
        aleatorio.normal(size=(200, len(FEATURES_CANONICAS))), columns=FEATURES_CANONICAS
    )
    modelo = LGBMRegressor(n_estimators=5, num_leaves=3, verbose=-1)
    modelo.fit(treino, aleatorio.normal(size=200))
    modelo.booster_.save_model(str(tmp_path / "modelo_pas3.txt"))

    manifesto = {
        "modelos": [
            {
                "arquivo": "modelo_pas3.txt",
                "alvo": "a3",
                "features": list(FEATURES_CANONICAS),
                "treinado_excluindo_trienio": "2023/2025",
            }
        ],
        "incerteza": {
            "forma": "normal",
            "escala": "a3",
            "sigma_por_classe": {
                "com_etapa_1": SIGMA_COM_ETAPA_1,
                "sem_etapa_1": SIGMA_SEM_ETAPA_1,
            },
        },
        "codigo": {"commit": "abc123"},
        "criado_em": "2026-07-28T00:00:00+00:00",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifesto), encoding="utf-8")
    return tmp_path


@pytest.fixture
def pacote(diretorio_do_pacote: Path):
    from pas_intelligence.model_package import PacoteDeModelo

    return PacoteDeModelo.carregar(diretorio_do_pacote)
