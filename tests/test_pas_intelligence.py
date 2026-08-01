
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest # type: ignore
import numpy as np # type: ignore

from pas_intelligence.argument_calculator import ( # type: ignore
    project_historical_stats,
    calculate_argument_part,
    calculate_argument_etapa,
    HistoricalStats,
    PESO_P1,
    PESO_P2,
    PESO_REDACAO,
)
from pas_intelligence.ab_testing import compare_groups # type: ignore


# `ensemble.py` foi removido no ticket 13: o ADR-0011 aposentou o ensemble por Volatilidade
# (ganho de 0,10% sobre o melhor componente sozinho, dentro do ruído entre dobras) e o ADR-0009
# tirou o Coeficiente de Variação de circulação junto. Os testes de `calculate_volatility` e
# `_sigmoid_weight` saíram com ele; o mecanismo continua reproduzível em
# `scripts/familia_de_modelo_ticket10.py`, que é a medição que o aposentou, com testes próprios
# em `tests/test_familia_de_modelo_ticket10.py`.

# =============================================================================
# TESTES: argument_calculator.py
# =============================================================================

class TestHistoricalProjection:
    """Testes para projeção de estatísticas históricas."""
    
    def test_projection_trend_down(self):
        """Tendência de queda deve projetar valor menor."""
        means = [30.0, 29.0, 28.0, 27.0, 26.0]
        stds = [10.0, 10.0, 10.0, 10.0, 10.0]
        mean_proj, std_proj = project_historical_stats(means, stds)
        assert mean_proj < 26.0  # Deve continuar caindo
    
    def test_projection_trend_up(self):
        """Tendência de subida deve projetar valor maior."""
        means = [26.0, 27.0, 28.0, 29.0, 30.0]
        stds = [10.0, 10.0, 10.0, 10.0, 10.0]
        mean_proj, _ = project_historical_stats(means, stds)
        assert mean_proj > 30.0  # Deve continuar subindo
    
    def test_projection_stable(self):
        """Valores estáveis devem projetar similar."""
        means = [28.0, 28.0, 28.0, 28.0, 28.0]
        stds = [10.0, 10.0, 10.0, 10.0, 10.0]
        mean_proj, _ = project_historical_stats(means, stds)
        assert 27.5 < mean_proj < 28.5
    
    def test_projection_minimum_points(self):
        """Deve exigir pelo menos 2 pontos."""
        with pytest.raises(ValueError):
            project_historical_stats([28.0], [10.0])
    
    def test_projection_std_positive(self):
        """STD projetado deve ser sempre positivo."""
        stds = [5.0, 4.0, 3.0, 2.0, 1.0]  # Tendência de queda
        means = [28.0] * 5
        _, std_proj = project_historical_stats(means, stds)
        assert std_proj > 0


class TestArgumentPart:
    """Testes para cálculo de argumento por parte."""
    
    def test_argument_above_average(self):
        """Nota acima da média deve dar argumento positivo."""
        arg = calculate_argument_part(nota=35.0, media=30.0, desvio_padrao=5.0, peso=1.0)
        assert arg == 1.0  # (35-30)/5 * 1 = 1
    
    def test_argument_below_average(self):
        """Nota abaixo da média deve dar argumento negativo."""
        arg = calculate_argument_part(nota=25.0, media=30.0, desvio_padrao=5.0, peso=1.0)
        assert arg == -1.0  # (25-30)/5 * 1 = -1
    
    def test_argument_at_average(self):
        """Nota igual à média deve dar argumento zero."""
        arg = calculate_argument_part(nota=30.0, media=30.0, desvio_padrao=5.0, peso=1.0)
        assert arg == 0.0
    
    def test_argument_weight_applied(self):
        """Peso deve ser aplicado corretamente."""
        arg = calculate_argument_part(nota=35.0, media=30.0, desvio_padrao=5.0, peso=8.28)
        assert arg == pytest.approx(8.28)  # (35-30)/5 * 8.28
    
    def test_argument_zero_std_error(self):
        """Desvio zero deve levantar erro."""
        with pytest.raises(ValueError):
            calculate_argument_part(nota=35.0, media=30.0, desvio_padrao=0.0, peso=1.0)


# =============================================================================
# TESTES: ab_testing.py
# =============================================================================

class TestABTesting:
    """Testes para comparação de grupos."""
    
    def test_compare_significant_difference(self):
        """Grupos muito diferentes devem ser estatisticamente significantes."""
        group_a = np.array([30, 31, 32, 33, 34, 35])
        group_b = np.array([20, 21, 22, 23, 24, 25])
        
        result = compare_groups(group_a, group_b)
        
        assert result['statistically_significant']
        assert result['p_value'] < 0.05
        assert result['group_a_mean'] > result['group_b_mean']
    
    def test_compare_no_significant_difference(self):
        """Grupos similares não devem ser significantes."""
        np.random.seed(42)
        group_a = np.random.normal(30, 5, 100)
        group_b = np.random.normal(30, 5, 100)
        
        result = compare_groups(group_a, group_b)
        
        assert result['p_value'] > 0.05  # Não significante
    
    def test_compare_interpretation_present(self):
        """Resultado deve conter interpretação."""
        group_a = np.array([30, 32, 34])
        group_b = np.array([20, 22, 24])
        
        result = compare_groups(group_a, group_b)
        
        assert 'interpretation' in result
        assert len(result['interpretation']) > 0
    
    def test_compare_effect_size(self):
        """Tamanho do efeito deve ser calculado."""
        group_a = np.array([30, 31, 32, 33, 34])
        group_b = np.array([20, 21, 22, 23, 24])
        
        result = compare_groups(group_a, group_b)
        
        assert 'effect_size' in result
        assert result['effect_size'] > 0  # A > B
    
    def test_compare_minimum_samples(self):
        """Deve exigir pelo menos 2 amostras por grupo."""
        with pytest.raises(ValueError):
            compare_groups(np.array([30]), np.array([20, 22]))


# =============================================================================
# TESTES DE INTEGRAÇÃO
# =============================================================================

class TestIntegration:
    """Testes de integração entre módulos."""
    
    def test_full_argument_calculation(self):
        """Teste completo do cálculo de argumento final."""
        from pas_intelligence.argument_calculator import calculate_argument_final # type: ignore
        
        stats_pas1 = HistoricalStats(
            mean_p1=6.5, std_p1=2.0,
            mean_p2=25.0, std_p2=8.0,
            mean_red=6.0, std_red=1.5,
        )
        stats_pas2 = HistoricalStats(
            mean_p1=7.0, std_p1=1.8,
            mean_p2=27.0, std_p2=7.5,
            mean_red=7.0, std_red=1.3,
        )
        stats_pas3 = HistoricalStats(
            mean_p1=7.5, std_p1=1.5,
            mean_p2=30.0, std_p2=7.0,
            mean_red=7.5, std_red=1.2,
        )
        
        notas = {
            'P1_PAS1': 7.0, 'P2_PAS1': 28.0, 'Red_PAS1': 6.5,
            'P1_PAS2': 8.0, 'P2_PAS2': 32.0, 'Red_PAS2': 7.5,
            'P1_PAS3': 9.0, 'P2_PAS3': 38.0, 'Red_PAS3': 8.5,
        }
        
        result = calculate_argument_final(notas, stats_pas1, stats_pas2, stats_pas3)
        
        # Verifica que todos os campos esperados estão presentes
        assert 'arg_pas1' in result
        assert 'arg_pas2' in result
        assert 'arg_pas3' in result
        assert 'arg_final' in result
        
        # Argumento final deve ser 1*AP1 + 2*AP2 + 3*AP3
        expected_final = result['arg_pas1'] + 2*result['arg_pas2'] + 3*result['arg_pas3']
        assert result['arg_final'] == pytest.approx(expected_final, rel=0.01)



# =============================================================================
# TESTES: target_calculator.py
# =============================================================================

class TestTargetCalculator:
    """Testes para a calculadora de meta (Reverse Prediction)."""
    
    def test_estimador_auxiliar_pondera_z_scores_1_para_2(self):
        """Estimador Auxiliar: média ponderada (1:2) dos z-scores de Etapa 1 e 2, reconvertida
        para a escala da Etapa 3 — não mais média ponderada de notas cruas (ticket 11)."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore

        calc = TargetCalculator()

        notas = {
            'P1_PAS1': 5.0, 'P2_PAS1': 0.0, 'Red_PAS1': 6.0,
            'P1_PAS2': 6.0, 'P2_PAS2': 0.0, 'Red_PAS2': 7.0,
        }
        stats_pas1 = HistoricalStats(mean_p1=4.0, std_p1=2.0, mean_p2=25, std_p2=10, mean_red=5.0, std_red=1.0)
        stats_pas2 = HistoricalStats(mean_p1=5.0, std_p1=2.0, mean_p2=25, std_p2=10, mean_red=6.0, std_red=1.0)
        stats_pas3 = HistoricalStats(mean_p1=6.0, std_p1=3.0, mean_p2=25, std_p2=10, mean_red=7.0, std_red=1.5)

        # z_p1_e1 = (5-4)/2 = 0.5 | z_p1_e2 = (6-5)/2 = 0.5 -> z_p1 = (0.5+2*0.5)/3 = 0.5
        # p1_pred = 0.5*3 + 6 = 7.5
        # z_red_e1 = (6-5)/1 = 1.0 | z_red_e2 = (7-6)/1 = 1.0 -> z_red = 1.0
        # red_pred = 1.0*1.5 + 7 = 8.5
        result = calc.predict_stable_components(notas, stats_pas1, stats_pas2, stats_pas3)

        assert result['p1_pred'] == pytest.approx(7.5, abs=0.001)
        assert result['red_pred'] == pytest.approx(8.5, abs=0.001)

    def test_estimador_auxiliar_pesa_etapa_2_o_dobro_da_etapa_1(self):
        """Um desvio isolado na Etapa 2 move o z-score final o dobro do mesmo desvio na Etapa 1
        — é o peso 1:2 que a média ponderada anterior já usava (ticket 11 mantém a proporção)."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore

        calc = TargetCalculator()
        stats = HistoricalStats(mean_p1=5.0, std_p1=2.0, mean_p2=25, std_p2=10, mean_red=6.0, std_red=1.0)

        notas_desvio_e1 = {
            'P1_PAS1': 7.0, 'P2_PAS1': 0.0, 'Red_PAS1': 6.0,
            'P1_PAS2': 5.0, 'P2_PAS2': 0.0, 'Red_PAS2': 6.0,
        }
        notas_desvio_e2 = {
            'P1_PAS1': 5.0, 'P2_PAS1': 0.0, 'Red_PAS1': 6.0,
            'P1_PAS2': 7.0, 'P2_PAS2': 0.0, 'Red_PAS2': 6.0,
        }

        base = calc.predict_stable_components(
            {'P1_PAS1': 5.0, 'P2_PAS1': 0.0, 'Red_PAS1': 6.0, 'P1_PAS2': 5.0, 'P2_PAS2': 0.0, 'Red_PAS2': 6.0},
            stats, stats, stats,
        )
        efeito_e1 = calc.predict_stable_components(notas_desvio_e1, stats, stats, stats)['p1_pred'] - base['p1_pred']
        efeito_e2 = calc.predict_stable_components(notas_desvio_e2, stats, stats, stats)['p1_pred'] - base['p1_pred']

        assert efeito_e2 == pytest.approx(2 * efeito_e1, rel=0.01)

    def test_override_parcial_e_respeitado(self):
        """Um override sozinho vale — não é descartado por o outro campo estar vazio.

        Regressão do defeito 7 (ticket 04): a condição era `and`, então quem mexia só na
        Redação tinha os dois overrides ignorados em silêncio e via na tela um P2 necessário
        que não correspondia ao número digitado.
        """
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore

        calc = TargetCalculator()

        notas = {
            'P1_PAS1': 5.0, 'P2_PAS1': 20.0, 'Red_PAS1': 6.0,
            'P1_PAS2': 6.0, 'P2_PAS2': 25.0, 'Red_PAS2': 7.0,
        }
        stats = HistoricalStats(
            mean_p1=3.8, std_p1=2.1, mean_p2=30.7, std_p2=13.8, mean_red=7.6, std_red=1.9
        )
        # As três Etapas usam a mesma `stats`, então o Estimador Auxiliar (média ponderada de
        # z-scores) reduz à mesma aritmética da antiga média ponderada de notas:
        # P1 = (5+2·6)/3 = 5.667, Red = (6+2·7)/3 = 6.667
        kwargs = dict(notas_existentes=notas, arg_alvo=50.0,
                      stats_pas1=stats, stats_pas2=stats, stats_pas3=stats)

        base = calc.calculate_required_score(**kwargs)
        assert base.p1_estimado == pytest.approx(5.667, abs=0.01)
        assert base.red_estimada == pytest.approx(6.667, abs=0.01)

        # Só a Redação sobrescrita: ela muda, o P1 continua vindo da previsão.
        so_red = calc.calculate_required_score(**kwargs, red_override=9.0)
        assert so_red.red_estimada == pytest.approx(9.0)
        assert so_red.p1_estimado == pytest.approx(5.667, abs=0.01)

        # Só o P1 sobrescrito: simétrico.
        so_p1 = calc.calculate_required_score(**kwargs, p1_override=8.0)
        assert so_p1.p1_estimado == pytest.approx(8.0)
        assert so_p1.red_estimada == pytest.approx(6.667, abs=0.01)

        # E o override tem que efetivamente mover o P2 necessário, não só o texto da tela.
        # Subir a Redação alivia a P2 (o Argumento da Etapa 3 alvo é fixo).
        assert so_red.p2_necessario < base.p2_necessario

        # Os dois juntos continuam funcionando como antes.
        ambos = calc.calculate_required_score(**kwargs, p1_override=8.0, red_override=9.0)
        assert ambos.p1_estimado == pytest.approx(8.0)
        assert ambos.red_estimada == pytest.approx(9.0)

    def test_calculadora_nao_usa_joblib(self):
        """`TargetCalculator` não tem mais superfície de modelo serializado: nada de
        `model_p1`/`model_red`/`model_load_error`/`ModelLoadError`/`PAS_STRICT_MODELS` — o
        Estimador Auxiliar é aritmética (ticket 11, resolve por remoção o defeito 3 de
        `defeitos-pendentes.md`)."""
        import pas_intelligence.target_calculator as tc  # type: ignore

        calc = tc.TargetCalculator()

        assert not hasattr(calc, 'model_p1')
        assert not hasattr(calc, 'model_red')
        assert not hasattr(calc, 'model_load_error')
        assert not hasattr(tc, 'ModelLoadError')
        assert not hasattr(tc, 'joblib')

    def test_predict_stable_components_bounds(self):
        """Previsão deve respeitar limites físicos (P1: [-20,20], Red: [0,10]) mesmo com
        z-scores extremos."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore

        calc = TargetCalculator()
        stats = HistoricalStats(mean_p1=5.0, std_p1=2.0, mean_p2=25, std_p2=10, mean_red=6.0, std_red=2.0)

        # Notas muito acima da média das Etapas 1 e 2 -> z-score alto -> estoura o teto.
        notas = {
            'P1_PAS1': 25.0, 'P2_PAS1': 0.0, 'Red_PAS1': 12.0,
            'P1_PAS2': 25.0, 'P2_PAS2': 0.0, 'Red_PAS2': 12.0,
        }

        result = calc.predict_stable_components(notas, stats, stats, stats)

        assert result['p1_pred'] == 20.0
        assert result['red_pred'] == 10.0

    def test_reverse_formula_consistency(self):
        """Reverse deve ser o inverso do Forward (aproximadamente)."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats, calculate_argument_part, PESO_P2 # type: ignore
        
        # Cenário
        stats = HistoricalStats(
            mean_p1=0, std_p1=1,
            mean_p2=25.0, std_p2=12.0,
            mean_red=0, std_red=1
        )
        nota_p2_original = 40.0
        
        # 1. Forward: Nota -> Argumento
        arg_p2 = calculate_argument_part(nota_p2_original, stats.mean_p2, stats.std_p2, PESO_P2)
        
        # 2. Reverse: Argumento -> Nota (Lógica interna do TargetCalculator)
        # Nota = (Arg * Desvio / Peso) + Média
        nota_p2_reversa = (arg_p2 * stats.std_p2 / PESO_P2) + stats.mean_p2
        
        assert nota_p2_reversa == pytest.approx(nota_p2_original, abs=0.001)

    def test_calculate_required_score_impl(self):
        """Teste ponta a ponta do cálculo de meta."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore
        
        calc = TargetCalculator()

        # Estatísticas dummy
        stats = HistoricalStats(mean_p1=5, std_p1=2, mean_p2=25, std_p2=10, mean_red=6, std_red=2)
        
        # Notas dummy
        notas = {
            'P1_PAS1': 5.0, 'P2_PAS1': 25.0, 'Red_PAS1': 6.0, # Args ~ 0
            'P1_PAS2': 5.0, 'P2_PAS2': 25.0, 'Red_PAS2': 6.0, # Args ~ 0
        }
        
        # Se aluno teve média em tudo, Args PAS1 e PAS2 ~ 0.
        # Se o curso exige Arg Final = 0, ele precisa de Arg PAS3 ~ 0.
        # Logo, P2 PAS3 deve ser próxima da média (25.0).
        
        res = calc.calculate_required_score(
            notas_existentes=notas,
            arg_alvo=0.0,
            stats_pas1=stats, stats_pas2=stats, stats_pas3=stats
        )
        
        # Previsão P1/Red deve ser 5.0 e 6.0 (estável)
        assert res.p1_estimado == 5.0
        assert res.red_estimada == 6.0
        
        # P2 deve ser próxima de 25.0
        assert res.p2_necessario == pytest.approx(25.0, abs=1.0)
        assert res.status == 'possivel'

    def test_impossible_scenario(self):
        """Deve identificar meta impossível (>100)."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore
        
        calc = TargetCalculator()

        stats = HistoricalStats(mean_p1=5, std_p1=2, mean_p2=25, std_p2=10, mean_red=6, std_red=2)

        notas = {'P1_PAS1': 0, 'P2_PAS1': 0, 'Red_PAS1': 0, 'P1_PAS2': 0, 'P2_PAS2': 0, 'Red_PAS2': 0}
        
        # Arg alvo muito alto para quem tirou zeros
        res = calc.calculate_required_score(notas, arg_alvo=200.0, stats_pas1=stats, stats_pas2=stats, stats_pas3=stats)
        
        assert res.status == 'impossivel'
        
    def test_guaranteed_scenario(self):
        """Meta garantida: a P2 necessária cai abaixo do piso da prova, e é truncada nele.

        No PAS **nota de prova pode ser negativa** — o Escore Bruto desconta erro, então zero não
        é o mínimo. "Garantido" não é `p2_necessario == 0`; é `p2_necessario` abaixo do piso da
        faixa, ou seja, nem o pior desempenho possível derruba o Aluno.
        """
        from pas_intelligence.target_calculator import TargetCalculator, P2_MINIMO  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore

        calc = TargetCalculator()

        stats = HistoricalStats(mean_p1=5, std_p1=2, mean_p2=25, std_p2=10, mean_red=6, std_red=2)

        notas = {'P1_PAS1': 10, 'P2_PAS1': 100, 'Red_PAS1': 10, 'P1_PAS2': 10, 'P2_PAS2': 100, 'Red_PAS2': 10}

        res = calc.calculate_required_score(notas, arg_alvo=-500.0, stats_pas1=stats, stats_pas2=stats, stats_pas3=stats)

        assert res.status == 'garantido'
        assert res.p2_necessario == P2_MINIMO

    def test_alvo_baixo_mas_dentro_da_faixa_ainda_e_possivel_nao_garantido(self):
        """A fronteira do 'garantido' com a faixa medida (ticket 11) — mudou de valor porque a
        faixa mudou. Com o piso antigo (−100), `arg_alvo=-100` dava p2≈−99,4 e ainda cabia dentro
        da faixa ('possivel'). Com o piso medido (`P2_MINIMO=0,24`), esse mesmo cenário agora cai
        em 'garantido' — é exatamente a correção de comunicação que o ticket 11 documenta (ver
        `test_guaranteed_scenario`). Este teste fixa a nova fronteira: um `arg_alvo` alto o
        bastante para deixar a P2 necessária pouco acima do piso medido, ainda 'possivel'.
        """
        from pas_intelligence.target_calculator import TargetCalculator, P2_MINIMO, P2_MAXIMO  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore

        calc = TargetCalculator()

        stats = HistoricalStats(mean_p1=5, std_p1=2, mean_p2=25, std_p2=10, mean_red=6, std_red=2)
        notas = {'P1_PAS1': 10, 'P2_PAS1': 100, 'Red_PAS1': 10, 'P1_PAS2': 10, 'P2_PAS2': 100, 'Red_PAS2': 10}

        # Vizinho do piso: p2_necessario ≈ 1,2 — acima de P2_MINIMO (0,24), abaixo de P2_MAXIMO.
        res = calc.calculate_required_score(notas, arg_alvo=150.0, stats_pas1=stats, stats_pas2=stats, stats_pas3=stats)

        assert res.status == 'possivel'
        assert P2_MINIMO < res.p2_necessario < P2_MAXIMO


# =============================================================================
# TESTES: pas_constants.py (ticket 12 — OFFICIAL_STATS com valores oficiais)
# =============================================================================

class TestExamStats:
    """Testes para o ExamStats com Parte 1 por língua (ticket 12)."""

    def test_m_p1_e_dp_p1_sao_media_das_tres_linguas(self):
        """m_p1/dp_p1 continuam existindo, mas como média simples das três línguas."""
        from pas_intelligence.pas_constants import (  # type: ignore
            ExamStats, Parte1PorLingua, ValorLingua,
        )

        stats = ExamStats(
            m_p2=25.0, dp_p2=13.0, m_red=6.0, dp_red=2.0,
            parte_1=Parte1PorLingua({
                "inglesa": ValorLingua(3.0, 2.0),
                "francesa": ValorLingua(6.0, 2.6),
                "espanhola": ValorLingua(3.0, 2.0),
            }),
        )

        assert stats.m_p1 == pytest.approx(4.0)  # (3 + 6 + 3) / 3
        assert stats.dp_p1 == pytest.approx(2.2)  # (2.0 + 2.6 + 2.0) / 3

    def test_official_stats_tem_26_entradas(self):
        """As 24 entradas de Edital mais as duas `DERIVADA` da Turma viva (ticket 07):
        `(2024, 1)` e `(2025, 2)`, ainda sem Edital de médias e desvios do Cebraspe.
        """
        from pas_intelligence.pas_constants import OFFICIAL_STATS  # type: ignore

        assert len(OFFICIAL_STATS) == 26
        assert (2023, 1) in OFFICIAL_STATS and (2024, 2) in OFFICIAL_STATS and (2025, 3) in OFFICIAL_STATS
        assert (2024, 1) in OFFICIAL_STATS and (2025, 2) in OFFICIAL_STATS

    def test_as_24_entradas_de_edital_tem_parte_1_por_lingua_completa(self):
        """A checagem é pelo **tipo**, não pelo conjunto de chaves: desde o ticket 01 a forma
        misturada também tem as três, então contar chaves deixou de discriminar. As duas
        entradas `DERIVADA` do ticket 07 ficam de fora — a delas é misturada, de propósito.
        """
        from pas_intelligence.pas_constants import (  # type: ignore
            LINGUAS_OFICIAIS, OFFICIAL_STATS, Origem, Parte1PorLingua,
        )

        entradas_de_edital = {
            chave: stats for chave, stats in OFFICIAL_STATS.items() if stats.origem is Origem.EDITAL
        }
        assert len(entradas_de_edital) == 24
        for chave, stats in entradas_de_edital.items():
            assert isinstance(stats.parte_1, Parte1PorLingua), chave
            assert set(stats.parte_1) == set(LINGUAS_OFICIAIS), chave

    def test_consumidor_analytics_service_continua_funcionando(self):
        """api/services/analytics_service.py lê s.m_p1 e s.m_p1 + s.m_p2 — a interface não
        pode quebrar mesmo com a mudança de forma da Parte 1."""
        from pas_intelligence.pas_constants import OFFICIAL_STATS  # type: ignore

        s = OFFICIAL_STATS[(2022, 1)]
        assert isinstance(s.m_p1, float)
        assert isinstance(s.m_p1 + s.m_p2, float)


# =============================================================================
# TESTES: Parte 1 misturada e procedência (ticket 01 — publicar-site)
# =============================================================================

def _etapa_misturada(**overrides):
    """Uma `(Ano, Etapa)` sintética vinda de Edital isolado de Etapa: Parte 1 misturada."""
    from pas_intelligence.pas_constants import (  # type: ignore
        ExamStats, Parte1Misturada, ValorLingua,
    )

    campos = dict(
        m_p2=25.0, dp_p2=13.0, m_red=6.0, dp_red=2.0,
        parte_1=Parte1Misturada(ValorLingua(4.0, 2.2)),
    )
    campos.update(overrides)
    return ExamStats(**campos)


class TestParte1Misturada:
    """O Edital isolado de Etapa não diz a língua de ninguém: só dá a Parte 1 misturada."""

    def test_forma_misturada_e_distinguivel_da_forma_por_lingua_pelo_tipo(self):
        """Sem contar chaves de dicionário — as duas formas são tipos diferentes."""
        from pas_intelligence.pas_constants import (  # type: ignore
            OFFICIAL_STATS, Parte1Misturada, Parte1PorLingua,
        )

        misturada = _etapa_misturada()
        por_lingua = OFFICIAL_STATS[(2022, 1)]

        assert isinstance(misturada.parte_1, Parte1Misturada)
        assert misturada.parte_1.misturada is True
        assert isinstance(por_lingua.parte_1, Parte1PorLingua)
        assert por_lingua.parte_1.misturada is False

    def test_m_p1_e_dp_p1_na_forma_misturada_sao_o_proprio_valor(self):
        stats = _etapa_misturada()

        assert stats.m_p1 == pytest.approx(4.0)
        assert stats.dp_p1 == pytest.approx(2.2)

    def test_as_tres_linguas_devolvem_a_estatistica_misturada(self):
        from pas_intelligence.pas_constants import LINGUAS_OFICIAIS, ValorLingua  # type: ignore

        parte_1 = _etapa_misturada().parte_1

        assert set(parte_1) == set(LINGUAS_OFICIAIS)
        for lingua in LINGUAS_OFICIAIS:
            assert parte_1[lingua] == ValorLingua(4.0, 2.2)

    def test_lingua_desconhecida_continua_sendo_erro_na_forma_misturada(self):
        """"Devolve o mesmo valor para as três" não é "devolve para qualquer coisa"."""
        with pytest.raises(KeyError):
            _etapa_misturada().parte_1["alemã"]

    def test_parte_1_por_lingua_recusa_conjunto_de_linguas_incompleto(self):
        from pas_intelligence.pas_constants import Parte1PorLingua, ValorLingua  # type: ignore

        with pytest.raises(ValueError):
            Parte1PorLingua({"inglesa": ValorLingua(3.0, 2.0)})


class TestProcedencia:
    """Quando o Edital de verdade sair, esses números serão substituídos e as previsões vão
    mexer — a origem precisa estar registrada, não descoberta depois."""

    def test_as_24_entradas_de_edital_sao_de_edital(self):
        from pas_intelligence.pas_constants import OFFICIAL_STATS, Origem  # type: ignore

        entradas_de_edital = [
            chave for chave, s in OFFICIAL_STATS.items()
            if chave not in {(2024, 1), (2025, 2)}
        ]
        assert len(entradas_de_edital) == 24
        assert all(OFFICIAL_STATS[chave].origem is Origem.EDITAL for chave in entradas_de_edital)

    def test_entrada_pode_declarar_se_derivada(self):
        from pas_intelligence.pas_constants import Origem  # type: ignore

        assert _etapa_misturada(origem=Origem.DERIVADA).origem is Origem.DERIVADA

    def test_misturada_e_derivada_sao_eixos_independentes(self):
        """O Edital isolado de Etapa é um Edital: dá Parte 1 misturada de origem `EDITAL`."""
        from pas_intelligence.pas_constants import Origem  # type: ignore

        stats = _etapa_misturada()

        assert stats.parte_1.misturada is True
        assert stats.origem is Origem.EDITAL

    def test_turma_viva_2024_1_e_2025_2_sao_derivadas_e_misturadas(self):
        """As duas entradas que o ticket 07 escreveu — a Turma viva 2024-2026 ainda não tem
        Edital de médias e desvios de nenhuma Etapa. Quando ele sair, `origem` muda para
        `EDITAL` e a Parte 1 deixa de ser misturada; até lá a API precisa saber que é derivada."""
        from pas_intelligence.pas_constants import (  # type: ignore
            OFFICIAL_STATS, Origem, Parte1Misturada,
        )

        for chave in [(2024, 1), (2025, 2)]:
            stats = OFFICIAL_STATS[chave]
            assert stats.origem is Origem.DERIVADA, chave
            assert isinstance(stats.parte_1, Parte1Misturada), chave


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
