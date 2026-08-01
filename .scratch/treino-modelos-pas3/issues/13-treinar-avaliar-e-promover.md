# 13 — Treinar, avaliar e promover

**Type:** task
**Status:** resolvido em 2026-07-28
**Relatório:** `relatorios/13-treinar-avaliar-e-promover.md`

## Question

Rodar o pipeline, produzir os artefatos definitivos, e colocá-los em produção sem quebrar quem
consome. É o ticket que fecha o mapa.

**Nada se decide aqui.** Se surgir uma decisão durante este ticket, ela é sinal de que um ticket
anterior ficou incompleto — volta para lá, não se resolve aqui.

**O que precisa continuar funcionando:**

- `api/services/` — os consumidores dos modelos no backend FastAPI;
- `src/pas_intelligence/target_calculator.py` — o caminho reverso (dado um corte, qual P2 o
  aluno precisa), que hoje depende de `p1_pas3_model.joblib` e `red_pas3_model.joblib`. Se o
  ticket 04 mudou o alvo canônico, o encaixe muda aqui, e a forma disso ainda é névoa no mapa;
- `src/pas_intelligence/statistics.py` — passa a ler a incerteza do artefato (ticket 11) em vez
  da constante `13.49`;
- `src/pas_intelligence/ensemble.py` — se o ticket 10 aposentou o ensemble por volatilidade,
  este módulo muda ou sai;
- `tests/test_pas_intelligence.py` — continua passando;
- o app Streamlit legado (`app/streamlit_app.py`, gitignored) — verificar se ainda é usado
  localmente antes de quebrá-lo em silêncio.

**Comparação lado a lado antes de promover.** Um punhado de alunos reais do holdout, modelo
antigo contra modelo novo, previsão e probabilidade de aprovação de cada. É a última chance de
ver que o número melhorou mas a saída ficou absurda — e o número agregado não mostra isso.

- [x] Pipeline do ticket 12 executado; artefatos definitivos produzidos
- [x] Critério de aceite do ticket 06 batido e registrado
- [x] Comparação lado a lado antigo vs. novo revisada pelo dono do produto antes da promoção
- [x] `models/` atualizado no formato e no domicílio decididos pelo ticket 03, com o anterior
      preservado para reverter
- [x] Todos os consumidores listados acima verificados; `tests/` passando
- [x] `statistics.py` sem constante de RMSE cravada
- [x] `CLAUDE.md` e `DEVELOPER_HANDBOOK.md` atualizados: a tabela de modelos, o feature vector e
      o caminho de retreino refletem a realidade nova
- [x] Relatório em `relatorios/13-treinar-avaliar-e-promover.md`

---

## Resolução

Ver `relatorios/13-treinar-avaliar-e-promover.md`. Em uma linha por item:

- Pipeline rodou de primeira sobre o CSV real: RMSE `5,009` em `A3`, Portão 1 batido nas quatro
  pernas. Pacote em `models/pas3/`, anterior em `models/aposentados-2026-07-28/`.
- Lacre aberto uma vez: `σ = 4,624` em `A3` — dentro da banda `[4,5; 5,5]` da regra assimétrica,
  **nenhuma ação**. Cobertura a 80% saiu 83,6%: a largura promovida é conservadora.
- Lado a lado no lacre: RMSE de Argumento Final `17,942 → 13,871`, viés `+8,658 → +0,517`, erro de
  decisão `7,81% → 5,41%`, Brier `0,0564 → 0,0391`. Revisado e autorizado pelo dono do produto.
- `ensemble.py`, `calculate.py`, `baseline_avaliacao.py`, `ARG_FINAL_MAE` e `ARG_MARGEM` removidos.
- `model_package.py` é a porta única entre `api/` e o artefato, montando features pelas funções do
  treino (teste de paridade incluído). Formulário ganhou língua estrangeira e o botão
  *"Não fiz o PAS 1"*. Gestão ganhou o estado `grey` / *Sem previsão*.
- **Aberto:** a turma viva (2024-2026) não recebe previsão enquanto `(2024, Etapa 1)` e
  `(2025, Etapa 2)` não forem extraídos dos Editais de Etapa — §8 do relatório. Nenhuma mudança de
  código será necessária quando existirem.
- **Dívida declarada:** o `eb_pas3_previsto` saiu da tela até o Estimador Auxiliar e o Ano-Âncora
  existirem (ticket 04 §7.1). §6 do relatório.
