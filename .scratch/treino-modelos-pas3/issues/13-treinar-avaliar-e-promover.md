# 13 — Treinar, avaliar e promover

**Type:** task
**Status:** open
**Blocked by:** 11, 12

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

- [ ] Pipeline do ticket 12 executado; artefatos definitivos produzidos
- [ ] Critério de aceite do ticket 06 batido e registrado
- [ ] Comparação lado a lado antigo vs. novo revisada pelo dono do produto antes da promoção
- [ ] `models/` atualizado no formato e no domicílio decididos pelo ticket 03, com o anterior
      preservado para reverter
- [ ] Todos os consumidores listados acima verificados; `tests/` passando
- [ ] `statistics.py` sem constante de RMSE cravada
- [ ] `CLAUDE.md` e `DEVELOPER_HANDBOOK.md` atualizados: a tabela de modelos, o feature vector e
      o caminho de retreino refletem a realidade nova
- [ ] Relatório em `relatorios/13-treinar-avaliar-e-promover.md`
