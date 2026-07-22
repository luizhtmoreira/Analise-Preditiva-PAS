# Motor de Inteligência e Cálculos

O diretório `src/pas_intelligence/` abriga o "cérebro" do Vetor PAS. É aqui que todos os dados brutos se tornam predições acionáveis.

## Estrutura do Módulo

### `ensemble.py`
Nós utilizamos uma abordagem de **Ensemble Dinâmico**. Ao invés de um modelo global único, o sistema pondera as previsões de 4 algoritmos diferentes de acordo com o histórico individual do aluno.

- Se a variância de notas do aluno for **estável** -> A *Regressão Linear* assume o peso maior da predição.
- Se a variância for **instável (volátil)** -> Os algoritmos baseados em árvores, como *LightGBM* e *Random Forest*, ganham protagonismo.

### `target_calculator.py`
A lógica de engenharia reversa. Recebe o Arg Final Alvo (nota de corte) e realiza o retrocesso pela fórmula de pesos da Universidade de Brasília para encontrar exatamente qual a nota bruta necessária no PAS 3.

### `argument_calculator.py`
Implementa a conversão rígida das notas padronizadas conforme o subitem de cálculo estipulado no Edital do Cebraspe.

### `statistics.py` e `recommender.py`
Calculam a probabilidade percentual de aprovação (usando distribuições estatísticas sobre os cortes antigos) e, baseados no desempenho atual, podem sugerir cursos em que a aprovação do aluno seja mais garantida.

### `ab_testing.py`
Fornece utilitários para simulação e teste estatístico comparativo de notas. Permite validar mudanças nos pesos ou na acurácia do modelo comparando novos resultados contra o baseline estabelecido.

!!! note "Base de Treinamento"
    Todos os modelos foram exaustivamente treinados e validados por meio de um *Backtest Temporal* em uma base contendo **48.758 alunos** espalhados por sete triênios (2016 - 2024).
