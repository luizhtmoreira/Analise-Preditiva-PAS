"""
Calculadora de Meta: Reverse Prediction (Curso → Nota Necessária)

Este módulo implementa a lógica reversa do preditor PAS:
dado o curso-alvo do aluno, calcula a nota mínima no PAS 3 necessária.
"""

from typing import Dict, Optional
from dataclasses import dataclass

from .argument_calculator import (
    HistoricalStats,
    calculate_argument_part,
    calculate_argument_etapa,
    PESO_P1,
    PESO_P2,
    PESO_REDACAO,
)


# Faixa medida da Parte 2 na Etapa 3 — 8 triênios, ~64 mil Alunos (ticket 11). Decide sozinha
# quando os quatro status abaixo disparam (a fronteira de 'impossível' não é uma destas duas
# constantes: é `100 − P1̂`, ver `calculate_required_score`).
#
# Piso: 0,24 — em 8 triênios de Etapa 3, 0% dos candidatos tiveram P2 negativa (contra 2,3% na
# Etapa 2, pior caso −19,6). É por isso que a faixa é medida *por Etapa*, e a Calculadora — que
# resolve para a Etapa 3 — usa a da Etapa 3.
# Teto: 85,6 — o maior P2 já registrado entre os 64 mil Alunos (percentil 99,9 fica em ~78). O
# teto teórico da prova continua 100, mas é de `P1 + P2` **juntos** (o fator de normalização
# existe para que acertar tudo dê 100); a P1 sozinha já come até 8,5 pontos dele.
P2_MAXIMO = 85.6
P2_MINIMO = 0.24

# Peso do Estimador Auxiliar (Etapa 1 : Etapa 2) — o mesmo 1:2 da média ponderada que ele
# substitui (ver `predict_stable_components`).
_PESO_ETAPA1 = 1
_PESO_ETAPA2 = 2


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
    - P1 (Língua Estrangeira) e Redação: Estimador Auxiliar — aritmética pura, sem modelo
      serializado (ver `predict_stable_components`).
    - P2 (Conhecimentos Gerais): cálculo algébrico reverso a partir do argumento-alvo.
    """

    def predict_stable_components(
        self,
        notas: Dict[str, float],
        stats_pas1: HistoricalStats,
        stats_pas2: HistoricalStats,
        stats_pas3: HistoricalStats,
    ) -> Dict[str, float]:
        """
        Estimador Auxiliar: prevê P1 e Redação da Etapa 3 por média ponderada de z-scores.

        Um z-score expressa a nota em desvios-padrão em relação à média daquela prova — é o que
        permite comparar a Etapa 1 e a Etapa 2 sem herdar a dificuldade de cada uma. A previsão
        pondera 1 para a Etapa 1 e 2 para a Etapa 2 (mesmo peso da média ponderada anterior),
        soma na escala padronizada e só então reconverte para a escala da Etapa 3:

            z_e1 = (nota_e1 − média_e1) / desvio_e1
            z_e2 = (nota_e2 − média_e2) / desvio_e2
            z_e3 = (1·z_e1 + 2·z_e2) / 3
            nota_e3 = z_e3 · desvio_e3 + média_e3

        `stats_pas3` precisa vir resolvido pelo chamador com a estatística da Parte 1 da língua
        da **Etapa 2** — não a da Etapa 1. O Aluno ainda não sentou a Etapa 3, a troca de língua
        no PAS é de mão única (72% das trocas vão de inglesa para espanhola) e a última língua
        declarada é a melhor evidência disponível de qual será a da Etapa 3. Assunção explícita,
        para revisitar se algum dia medirmos `língua_e2 → língua_e3` diretamente.

        Args:
            notas: Dict com P1_PAS1, Red_PAS1, P1_PAS2, Red_PAS2 (as notas já realizadas).
            stats_pas1, stats_pas2, stats_pas3: média/desvio oficiais de cada Etapa.

        Returns:
            Dict com 'p1_pred' e 'red_pred'.
        """
        z_p1 = (
            _PESO_ETAPA1 * self._z(notas['P1_PAS1'], stats_pas1.mean_p1, stats_pas1.std_p1)
            + _PESO_ETAPA2 * self._z(notas['P1_PAS2'], stats_pas2.mean_p1, stats_pas2.std_p1)
        ) / (_PESO_ETAPA1 + _PESO_ETAPA2)

        z_red = (
            _PESO_ETAPA1 * self._z(notas['Red_PAS1'], stats_pas1.mean_red, stats_pas1.std_red)
            + _PESO_ETAPA2 * self._z(notas['Red_PAS2'], stats_pas2.mean_red, stats_pas2.std_red)
        ) / (_PESO_ETAPA1 + _PESO_ETAPA2)

        p1_pred = z_p1 * stats_pas3.std_p1 + stats_pas3.mean_p1
        red_pred = z_red * stats_pas3.std_red + stats_pas3.mean_red

        # Limites físicos da prova: P1 pode ser negativa (desconto por erro), Redação não.
        p1_pred = max(-20.0, min(20.0, p1_pred))
        red_pred = max(0.0, min(10.0, red_pred))

        return {
            'p1_pred': round(p1_pred, 3),
            'red_pred': round(red_pred, 3),
        }

    @staticmethod
    def _z(nota: float, media: float, desvio: float) -> float:
        return (nota - media) / desvio

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

        # 3. Determina P1 e Redação do PAS 3 (Estimador Auxiliar ou Override)
        # Cada override vale por si: quem mexe só na Redação — o caso mais provável, já que ela
        # é o componente mais sensível (1 ponto de erro move o P2 necessário em ~0,95, contra
        # ~0,60 do P1) — não pode ter o valor digitado descartado por causa do outro campo.
        if p1_override is None or red_override is None:
            previsao = self.predict_stable_components(
                notas_existentes, stats_pas1, stats_pas2, stats_pas3
            )

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
        # 'impossível' não é uma constante: é aritmética. O EB (P1+P2) não pode passar de 100,
        # então a P2 necessária não pode passar de `100 − P1̂` — e esse limite cai abaixo de
        # `P2_MAXIMO` sempre que a P1 estimada passa de 14,4 pontos.
        limite_impossivel = 100.0 - p1_pred
        if p2_necessario > limite_impossivel:
            status = 'impossivel'
            mensagem = (
                f"Nota necessária ({p2_necessario:.1f}) ultrapassa o máximo possível na Parte 2 "
                f"— com sua P1 estimada em {p1_pred:.1f}, o teto é {limite_impossivel:.1f} pts "
                "(P1 + P2 não pode passar de 100). Este curso é aritmeticamente inalcançável com "
                "seu histórico atual."
            )
        elif p2_necessario > P2_MAXIMO:
            status = 'improvavel'
            mensagem = (
                f"Nota necessária ({p2_necessario:.1f}) supera o recorde histórico da Parte 2 na "
                f"Etapa 3 ({P2_MAXIMO} pts, entre ~64 mil Alunos). Estatisticamente improvável, "
                "mas não impossível."
            )
        elif p2_necessario < P2_MINIMO:
            status = 'garantido'
            mensagem = ":material/celebration: Seu histórico já é suficiente! Mesmo com o pior desempenho possível na Parte 2, você já bate a meta."
            p2_necessario = P2_MINIMO
            total_pas3 = p1_pred + p2_necessario
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

# `get_strategy_prediction` morava aqui. Resolvia média e desvio a partir do `TRIENNIUM_STATS`
# (cópia paralela do `OFFICIAL_STATS`) e da regressão do `STATS_PAS3_TREND` — as duas saíram no
# ticket 05, e o ADR-0009 descartou projetar a Etapa 3. Quem chama agora resolve as três
# `HistoricalStats` pela porta única (`stats_da_prova`) e usa `calculate_required_score`
# direto: ver `api/services/predict_service._stats_do_ciclo`.
#
# `p1_pas3_model.joblib` / `red_pas3_model.joblib`, o carregamento de `.joblib`, o registro de
# degradação (`model_load_error`), `ModelLoadError` e o modo estrito (`PAS_STRICT_MODELS`)
# saíram no ticket 11: os dois artefatos não carregavam sob o sklearn atual
# (`ModuleNotFoundError: No module named '_loss'`, defeito 3 de `defeitos-pendentes.md`, agora
# resolvido por remoção) e o Estimador Auxiliar acima é aritmética — não há mais nada para
# degradar.
