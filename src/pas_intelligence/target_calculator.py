"""
Calculadora de Meta: Reverse Prediction (Curso → Nota Necessária)

Este módulo implementa a lógica reversa do preditor PAS:
dado o curso-alvo do aluno, calcula a nota mínima no PAS 3 necessária.
"""

from typing import Any, Dict, TypedDict, Optional, List
from dataclasses import dataclass
import logging
import os
import numpy as np  # type: ignore
import joblib
import pandas as pd
from pathlib import Path

from .argument_calculator import (
    HistoricalStats,
    calculate_argument_part,
    calculate_argument_etapa,
    PESO_P1,
    PESO_P2,
    PESO_REDACAO,
)


logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Um modelo esperado em disco não pôde ser carregado ou usado."""


def _modo_estrito() -> bool:
    """
    Em modo estrito, falha de modelo derruba em vez de degradar para média ponderada.

    Produção liga `PAS_STRICT_MODELS=1`: lá o pacote de modelos é assado na imagem e a
    ausência ou incompatibilidade dele é defeito de build, não condição de operação.
    Fora dela o padrão é degradar, porque a máquina de quem desenvolve pode legitimamente
    não ter `models/` (o diretório é gitignored).
    """
    return os.getenv("PAS_STRICT_MODELS", "").strip().lower() in {"1", "true", "yes"}


@dataclass
class ReverseResult:
    """Resultado do cálculo de nota necessária."""
    p1_estimado: float
    p2_necessario: float
    red_estimada: float
    total_pas3: float
    arg_pas3_necessario: float
    status: str  # 'possivel', 'impossivel', 'garantido', 'improvavel'
    mensagem: str


class TargetCalculator:
    """
    Calculadora que determina a nota necessária no PAS 3 para atingir
    um curso-alvo, dado o histórico do aluno.
    
    Estratégia:
    - P1 (Língua Estrangeira) e Redação: Predição via ML (Random Forest) ou Média Ponderada
    - P2 (Conhecimentos Gerais): Cálculo algébrico reverso
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Inicializa a calculadora e carrega os modelos ML.

        Se um modelo não puder ser carregado, a degradação para média ponderada é
        registrada em `self.model_load_error` e sai no log em nível ERROR — nunca em
        silêncio. Em modo estrito (ver `_modo_estrito`), levanta `ModelLoadError`.

        Args:
            models_dir: diretório dos artefatos. Por padrão, `models/` na raiz do
                projeto. Injetável para teste.
        """
        self.model_load_error: Optional[str] = None

        if models_dir is None:
            # Caminho relativo: src/pas_intelligence/target_calculator.py -> ... -> models/
            models_dir = Path(__file__).parent.parent.parent / "models"

        self.model_p1 = self._carregar_modelo(models_dir / "p1_pas3_model.joblib")
        self.model_red = self._carregar_modelo(models_dir / "red_pas3_model.joblib")

    def _carregar_modelo(self, caminho: Path):
        """Carrega um modelo do disco, registrando o motivo se não der."""
        if not caminho.exists():
            self._registrar_degradacao(f"{caminho.name}: arquivo não encontrado em {caminho.parent}")
            return None

        try:
            return joblib.load(caminho)
        except Exception as e:
            # Causa mais comum: o artefato foi serializado com outra versão de
            # scikit-learn e a receita de remontagem aponta para um módulo que mudou de
            # lugar (ex.: ModuleNotFoundError: No module named '_loss').
            self._registrar_degradacao(f"{caminho.name}: {type(e).__name__}: {e}")
            return None

    def _registrar_degradacao(self, motivo: str) -> None:
        """Acumula o motivo da degradação, grita no log e derruba em modo estrito."""
        self.model_load_error = (
            motivo if self.model_load_error is None else f"{self.model_load_error}; {motivo}"
        )

        if _modo_estrito():
            raise ModelLoadError(
                f"PAS_STRICT_MODELS ativo e modelo indisponível — {self.model_load_error}"
            )

        logger.error(
            "Modelo ML indisponível (%s). A calculadora reversa vai responder por média "
            "ponderada, NÃO por ML.",
            motivo,
        )

    def predict_stable_components(
        self,
        notas: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Prevê P1 e Redação do PAS 3.

        Prioridade:
        1. Modelo ML (Random Forest) se disponível.
        2. Média Ponderada (Fallback).

        Args:
            notas: Dicionário com notas do PAS 1 e 2.

        Returns:
            Dict com 'p1_pred', 'red_pred', 'method' ('ml' ou 'weighted_avg') e
            'fallback_reason' — `None` quando veio de ML, e o motivo da degradação
            quando veio da média ponderada. Quem consome o número precisa poder saber
            de onde ele veio.
        """
        # Extrai notas para fallback ou uso geral
        p1_pas1 = notas.get('P1_PAS1', 0.0)
        p1_pas2 = notas.get('P1_PAS2', 0.0)
        red_pas1 = notas.get('Red_PAS1', 0.0)
        red_pas2 = notas.get('Red_PAS2', 0.0)
        
        # Tenta usar ML
        if self.model_p1 and self.model_red:
            try:
                # Features esperadas pelo modelo (Base + Engenheiradas)
                features_base = [
                    'P1_PAS1', 'Red_PAS1', 'P2_PAS1',
                    'P1_PAS2', 'Red_PAS2', 'P2_PAS2'
                ]
                
                # Garante que todas as features existem (pelo menos como 0.0)
                input_data = {feat: [notas.get(feat, 0.0)] for feat in features_base}
                df_input = pd.DataFrame(input_data)
                
                # Feature Engineering (Reproduz lógica do treino)
                # Deltas
                df_input['delta_p1'] = df_input['P1_PAS2'] - df_input['P1_PAS1']
                df_input['delta_red'] = df_input['Red_PAS2'] - df_input['Red_PAS1']
                df_input['delta_p2'] = df_input['P2_PAS2'] - df_input['P2_PAS1']
                
                # Médias
                df_input['mean_p1'] = (df_input['P1_PAS1'] + df_input['P1_PAS2']) / 2
                df_input['mean_red'] = (df_input['Red_PAS1'] + df_input['Red_PAS2']) / 2
                
                # A ordem das colunas no predict deve ser a mesma do treino
                # O HistGradientBoosting geralmente é robusto a ordem se tiver nomes, mas vamos garantir
                # A lista de colunas usadas no treino foi:
                # ['P1_PAS1', 'Red_PAS1', 'P2_PAS1', 'P1_PAS2', 'Red_PAS2', 'P2_PAS2', 
                #  'delta_p1', 'delta_red', 'delta_p2', 'mean_p1', 'mean_red']
                
                p1_pred = float(self.model_p1.predict(df_input)[0])
                red_pred = float(self.model_red.predict(df_input)[0])
                
                # Limites
                p1_pred = max(-20.0, min(20.0, p1_pred))
                red_pred = max(0.0, min(10.0, red_pred))
                
                return {
                    'p1_pred': round(p1_pred, 3),
                    'red_pred': round(red_pred, 3),
                    'method': 'ml',
                    'fallback_reason': None,
                }
            except Exception as e:
                # Os modelos carregaram mas a predição falhou — anomalia real (vetor de
                # features fora do contrato, por exemplo), não condição de operação.
                self._registrar_degradacao(f"predição ML falhou: {type(e).__name__}: {e}")

        # Fallback: Média Ponderada
        # (p1_pas1 * 1 + p1_pas2 * 2) / 3
        p1_pred = (p1_pas1 * 1 + p1_pas2 * 2) / 3
        p1_pred = max(-20.0, min(20.0, p1_pred))
        
        red_pred = (red_pas1 * 1 + red_pas2 * 2) / 3
        red_pred = max(0.0, min(10.0, red_pred))
        
        return {
            'p1_pred': round(p1_pred, 3),
            'red_pred': round(red_pred, 3),
            'method': 'weighted_avg',
            'fallback_reason': self.model_load_error or 'modelos ML não carregados',
        }
    
    def calculate_required_score(
        self,
        notas_existentes: Dict[str, float],
        arg_alvo: float,
        stats_pas1: HistoricalStats,
        stats_pas2: HistoricalStats,
        stats_pas3: HistoricalStats,
        p1_override: Optional[float] = None,
        red_override: Optional[float] = None,
    ) -> ReverseResult:
        """
        Calcula a nota necessária na Parte 2 do PAS 3 para atingir o argumento-alvo.

        Suporta overrides manuais para P1 e Redação (simulação de cenários), **independentes
        entre si**: passar só `red_override` mantém o P1 vindo da previsão, e vice-versa.


        
        Fórmulas:
        - Arg_Final = 1*A1 + 2*A2 + 3*A3
        - A3 = Arg_P1 + Arg_P2 + Arg_Red
        - Arg = [(Nota - Média) / Desvio] × Peso
        - Reverso: Nota = (Arg × Desvio / Peso) + Média
        
        Args:
            notas_existentes: Dict com P1_PAS1, P2_PAS1, Red_PAS1, 
                             P1_PAS2, P2_PAS2, Red_PAS2
            arg_alvo: Argumento de corte do curso desejado
            stats_pas*: Estatísticas históricas de cada etapa
            
        Returns:
            ReverseResult com a nota necessária e status
        """
        # 1. Calcula argumentos das etapas já realizadas
        arg_pas1 = calculate_argument_etapa(
            notas_existentes['P1_PAS1'],
            notas_existentes['P2_PAS1'],
            notas_existentes['Red_PAS1'],
            stats_pas1,
        )
        
        arg_pas2 = calculate_argument_etapa(
            notas_existentes['P1_PAS2'],
            notas_existentes['P2_PAS2'],
            notas_existentes['Red_PAS2'],
            stats_pas2,
        )
        
        # 2. Calcula o "Gap" de argumento necessário
        # Arg_Final = 1*A1 + 2*A2 + 3*A3
        # A3_necessario = (Arg_Alvo - A1 - 2*A2) / 3
        arg_pas3_necessario = (arg_alvo - arg_pas1 - 2 * arg_pas2) / 3
        
        # 3. Determina P1 e Redação do PAS 3 (Predição ou Override)
        # Cada override vale por si: quem mexe só na Redação — o caso mais provável, já que ela
        # é o componente mais sensível (1 ponto de erro move o P2 necessário em ~0,95, contra
        # ~0,60 do P1) — não pode ter o valor digitado descartado por causa do outro campo.
        if p1_override is None or red_override is None:
            previsao = self.predict_stable_components(notas_existentes)

        p1_pred = p1_override if p1_override is not None else previsao['p1_pred']
        red_pred = red_override if red_override is not None else previsao['red_pred']
        
        # 4. Converte P1 e Redação previstos em argumentos
        arg_p1_pred = calculate_argument_part(
            p1_pred, stats_pas3.mean_p1, stats_pas3.std_p1, PESO_P1
        )
        arg_red_pred = calculate_argument_part(
            red_pred, stats_pas3.mean_red, stats_pas3.std_red, PESO_REDACAO
        )
        
        # 5. Calcula argumento necessário da P2
        arg_p2_necessario = arg_pas3_necessario - arg_p1_pred - arg_red_pred
        
        # 6. Fórmula reversa: Arg → Nota
        # Arg = [(Nota - Média) / Desvio] × Peso
        # Nota = (Arg × Desvio / Peso) + Média
        p2_necessario = (arg_p2_necessario * stats_pas3.std_p2 / PESO_P2) + stats_pas3.mean_p2
        
        # 7. Calcula total do PAS 3
        total_pas3 = p1_pred + p2_necessario
        
        # 8. Determina status
        if p2_necessario > 100:
            status = 'impossivel'
            mensagem = f"Nota necessária ({p2_necessario:.1f}) ultrapassa o máximo da prova (100 pts). Este curso pode ser estatisticamente inalcançável com seu histórico atual."
        elif p2_necessario < -100:
            status = 'garantido'
            mensagem = f":material/celebration: Seu histórico já é suficiente! Mesmo se você zerar ou tiver um desempenho extremamente baixo na P2, você provavelmente passará."
            p2_necessario = -100.0
            total_pas3 = p1_pred + p2_necessario
        elif p2_necessario > 80:
            status = 'improvavel'
            mensagem = f"Nota necessária ({p2_necessario:.1f}) é muito alta. Estatisticamente improvável, mas não impossível."
        else:
            status = 'possivel'
            mensagem = f"Meta alcançável! Você precisa de {p2_necessario:.1f} pts na Parte 2."
        
        return ReverseResult(
            p1_estimado=round(p1_pred, 2),
            p2_necessario=round(p2_necessario, 2),
            red_estimada=round(red_pred, 2),
            total_pas3=round(total_pas3, 2),
            arg_pas3_necessario=round(arg_pas3_necessario, 3),
            status=status,
            mensagem=mensagem,
        )


def get_reverse_prediction(
    notas: Dict[str, float],
    curso_nota_corte: float,
    stats_pas1: HistoricalStats,
    stats_pas2: HistoricalStats,
    stats_pas3: HistoricalStats,
) -> ReverseResult:
    """
    Função de conveniência para calcular a nota necessária.
    
    Args:
        notas: Dicionário com todas as notas do PAS 1 e 2
        curso_nota_corte: Argumento de corte mínimo do curso
        stats_pas*: Estatísticas de cada etapa
        
    Returns:
        ReverseResult com a análise completa
    """
    calculator = TargetCalculator()
    return calculator.calculate_required_score(
        notas, curso_nota_corte, stats_pas1, stats_pas2, stats_pas3
    )

def get_strategy_prediction(
    notas_existentes: Dict[str, float],
    nota_alvo: float,
    ciclo_aluno: str,
    triennium_stats: Dict[str, Dict[str, HistoricalStats]],
    stats_pas3_trend: HistoricalStats,
    p1_override: Optional[float] = None,
    red_override: Optional[float] = None,
    base_projecao: str = "Utilizar Projeção Tendência"
) -> ReverseResult:
    """
    Encapsula a lógica completa da Calculadora de Estratégia para 
    resolver e definir quais estatísticas (stats_pas1, stats_pas2, stats_p3_usado)
    serão utilizadas com base no ciclo do aluno e configuração de projeção.
    """
    calc = TargetCalculator()
    
    # 1. Resolve stats for PAS 1 and PAS 2
    stats_ciclo = triennium_stats.get(ciclo_aluno, triennium_stats.get("2023-2025"))
    stats_p1 = stats_ciclo["PAS1"]
    stats_p2 = stats_ciclo["PAS2"]
    
    # 2. Resolve stats for PAS 3
    stats_p3_usado = None
    if ciclo_aluno == "2024-2026":
        if base_projecao == "Replicar Padrão 2023-2025":
            stats_p3_usado = triennium_stats.get("2023-2025", stats_ciclo)["PAS3"]
        else:
            stats_p3_usado = stats_pas3_trend
    elif ciclo_aluno == "2023-2025":
        stats_p3_usado = stats_ciclo["PAS3"]
    else:
        # Fallback para outros ciclos ou missing
        stats_p3_usado = stats_ciclo.get("PAS3", stats_pas3_trend)
        
    return calc.calculate_required_score(
        notas_existentes, nota_alvo,
        stats_p1, stats_p2, stats_p3_usado,
        p1_override=p1_override, red_override=red_override
    )

