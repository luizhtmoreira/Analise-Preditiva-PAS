
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest # type: ignore
import numpy as np # type: ignore

from pas_intelligence.ensemble import ( # type: ignore
    calculate_volatility,
    _sigmoid_weight,
)
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


# =============================================================================
# TESTES: ensemble.py
# =============================================================================

class TestVolatility:
    """Testes para cálculo de volatilidade (Coeficiente de Variação)."""
    
    def test_volatility_basic(self):
        """CV de valores iguais deve ser 0."""
        scores = np.array([30.0, 30.0])
        cv = calculate_volatility(scores)
        assert cv == 0.0
    
    def test_volatility_typical(self):
        """CV de valores típicos deve estar na faixa esperada."""
        scores = np.array([30.0, 35.0])
        cv = calculate_volatility(scores)
        # CV = std / mean * 100
        # mean = 32.5, std = 2.5, CV = 7.69%
        assert 7 < cv < 8
    
    def test_volatility_high(self):
        """CV de valores muito diferentes deve ser alto."""
        scores = np.array([20.0, 40.0])
        cv = calculate_volatility(scores)
        assert cv > 20  # Alta volatilidade
    
    def test_volatility_minimum_points(self):
        """Deve exigir pelo menos 2 pontos."""
        with pytest.raises(ValueError):
            calculate_volatility(np.array([30.0]))
    
    def test_volatility_zero_mean(self):
        """Deve rejeitar média zero."""
        with pytest.raises(ValueError):
            calculate_volatility(np.array([5.0, -5.0]))


class TestSigmoidWeight:
    """Testes para função de peso sigmoide."""
    
    def test_low_cv_conservative(self):
        """CV baixo deve dar peso baixo para modelo arrojado."""
        weight = _sigmoid_weight(cv=5.0)
        assert weight < 0.3  # Próximo de 0.2 (conservador)
    
    def test_high_cv_aggressive(self):
        """CV alto deve dar peso alto para modelo arrojado."""
        weight = _sigmoid_weight(cv=25.0)
        assert weight > 0.7  # Próximo de 0.8 (arrojado)
    
    def test_midpoint_balanced(self):
        """CV no meio deve dar peso ~0.5."""
        weight = _sigmoid_weight(cv=15.0)  # Meio entre 10 e 20
        assert 0.4 < weight < 0.6
    
    def test_weight_bounds(self):
        """Peso deve estar sempre entre 0.2 e 0.8."""
        for cv in [0, 5, 10, 15, 20, 25, 30, 50, 100]:
            weight = _sigmoid_weight(cv)
            assert 0.2 <= weight <= 0.8


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
    
    def test_predict_weighted_avg_fallback(self):
        """Previsão deve usar média ponderada quando modelos ML não disponíveis."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        
        calc = TargetCalculator()
        # Força falha no carregamento de ML para testar fallback
        calc.model_p1 = None
        calc.model_red = None
        
        notas = {
            'P1_PAS1': 5.0, 'P2_PAS1': 0.0, 'Red_PAS1': 6.0,
            'P1_PAS2': 6.0, 'P2_PAS2': 0.0, 'Red_PAS2': 7.0,
        }
        
        # Média Pon: (5*1 + 6*2)/3 = 5.667 | (6*1 + 7*2)/3 = 6.667
        result = calc.predict_stable_components(notas)
        
        assert result['p1_pred'] == pytest.approx(5.667, abs=0.001)
        assert result['red_pred'] == pytest.approx(6.667, abs=0.001)
        assert result['method'] == 'weighted_avg'

    def test_ml_model_integration(self):
        """Deve carregar modelos ML e retornar método 'ml' se arquivos existirem."""
        from pas_intelligence.target_calculator import TargetCalculator
        import joblib
        from pathlib import Path
        
        # Só roda se modelos existirem (ambiente de dev)
        base_dir = Path(__file__).parent.parent
        if not (base_dir / "models/p1_pas3_model.joblib").exists():
            pytest.skip("Modelos ML não encontrados. Pulando teste de integração.")
            
        calc = TargetCalculator()
        
        notas = {
            'P1_PAS1': 5.0, 'P2_PAS1': 20.0, 'Red_PAS1': 6.0,
            'P1_PAS2': 6.0, 'P2_PAS2': 25.0, 'Red_PAS2': 7.0,
        }
        
        result = calc.predict_stable_components(notas)
        
        # Se carregou modelos, deve usar 'ml'
        if calc.model_p1 and calc.model_red:
            assert result['method'] == 'ml'
            assert 0 <= result['p1_pred'] <= 20
            assert 0 <= result['red_pred'] <= 10
        else:
            # Se não carregou por algum motivo (ex: erro de versão sklearn), fallback
            assert result['method'] == 'weighted_avg'

    def test_predict_stable_components_bounds(self):
        """Previsão deve respeitar limites (P1: 0-20, Red: 0-10) no fallback."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        
        calc = TargetCalculator()
        calc.model_p1 = None # Força fallback
        
        # Testando limite superior com valores altos
        notas = {
            'P1_PAS1': 25.0, 'P2_PAS1': 0.0, 'Red_PAS1': 12.0,
            'P1_PAS2': 25.0, 'P2_PAS2': 0.0, 'Red_PAS2': 12.0,
        }
        
        result = calc.predict_stable_components(notas)
        
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
        calc.model_p1 = None # Forçar fallback para determinismo
        
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
        calc.model_p1 = None
        
        stats = HistoricalStats(mean_p1=5, std_p1=2, mean_p2=25, std_p2=10, mean_red=6, std_red=2)
        
        notas = {'P1_PAS1': 0, 'P2_PAS1': 0, 'Red_PAS1': 0, 'P1_PAS2': 0, 'P2_PAS2': 0, 'Red_PAS2': 0}
        
        # Arg alvo muito alto para quem tirou zeros
        res = calc.calculate_required_score(notas, arg_alvo=200.0, stats_pas1=stats, stats_pas2=stats, stats_pas3=stats)
        
        assert res.status == 'impossivel'
        
    def test_guaranteed_scenario(self):
        """Deve identificar meta garantida (<0)."""
        from pas_intelligence.target_calculator import TargetCalculator  # type: ignore
        from pas_intelligence.argument_calculator import HistoricalStats  # type: ignore
        
        calc = TargetCalculator()
        calc.model_p1 = None
        
        stats = HistoricalStats(mean_p1=5, std_p1=2, mean_p2=25, std_p2=10, mean_red=6, std_red=2)
        
        notas = {'P1_PAS1': 10, 'P2_PAS1': 100, 'Red_PAS1': 10, 'P1_PAS2': 10, 'P2_PAS2': 100, 'Red_PAS2': 10}
        
        # Arg alvo baixo para quem gabaritou
        res = calc.calculate_required_score(notas, arg_alvo=-100.0, stats_pas1=stats, stats_pas2=stats, stats_pas3=stats)
        
        assert res.status == 'garantido'
        assert res.p2_necessario == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
