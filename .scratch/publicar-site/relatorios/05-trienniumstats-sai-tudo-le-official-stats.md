# Relatório — Ticket 05: `TRIENNIUM_STATS`/`STATS_PAS3_TREND` saem; tudo lê `OFFICIAL_STATS`

**Ticket:** `.scratch/publicar-site/issues/05-trienniumstats-sai-tudo-le-official-stats.md`
**Status:** concluído
**Onde vive o código:** `api/services/gestao_service.py` (o essencial do diff acabou absorvido
pelo commit `dd8d211`, de uma sessão concorrente que fechou o ticket 09 no mesmo arquivo — ver
§4); os testes de regressão e a baixa do defeito ficaram no commit `245ff12`.

---

## 1. O que foi pedido

`api/services/gestao_service.py` carregava `TRIENNIUM_STATS` e `STATS_PAS3_TREND`, dois
dicionários próprios de média/desvio que divergiam do `pas_constants.OFFICIAL_STATS` (defeito 6
de `defeitos-pendentes.md`). Com a Calculadora de Estratégia entrando nesta rodada, publicar duas
telas que calculam o mesmo `A1`/`A2` com números diferentes deixaria de ser um defeito
silencioso e viraria uma contradição visível para o mesmo Aluno.

Critérios de aceite (todos atendidos):

- [x] `TRIENNIUM_STATS` e `STATS_PAS3_TREND` não existem mais no código
- [x] Todo consumidor de média e desvio oficiais lê `OFFICIAL_STATS`, chaveado por `(ano, etapa)`,
      pela porta única que já existe (`anos_do_trienio` + `stats_da_prova`, de
      `training_dataset.py`)
- [x] O Reality Check da Gestão de Ativos continua funcionando, agora com os números do Edital
- [x] Um Aluno de teste produz o mesmo `A1` e o mesmo `A2` pelo caminho do Preditor e pelo
      caminho da Gestão — os dois chamam `stats_da_prova` com o mesmo `(ano, etapa, língua)`
- [x] O defeito 6 de `defeitos-pendentes.md` está marcado como corrigido
- [x] `pytest tests/` continua verde (356 passam; a única falha de coleção,
      `test_pas_extraction_etapa.py`, é pré-existente e não relacionada — módulo de outro ticket
      ainda não mesclado nesta árvore)

---

## 2. O que foi entregue

```
api/services/gestao_service.py
  − TRIENNIUM_STATS, STATS_PAS3_TREND        (removidos)
  + _stats_pas3_ancora(lingua) -> HistoricalStats
        a Etapa 3 real e já publicada mais recente do OFFICIAL_STATS
  Reality Check (bloco dentro de analyze_students):
        ano_e1, ano_e2, ano_e3 = anos_do_trienio(...)
        stats_p1 = stats_da_prova(ano_e1, 1, lingua)
        stats_p2 = stats_da_prova(ano_e2, 2, lingua)
        stats_p3 = stats_da_prova(ano_e3, 3, lingua)  — ou _stats_pas3_ancora(lingua) se
                   EstatisticaOficialAusenteError (turma viva, Etapa 3 ainda não aconteceu)

tests/test_api_predict.py
  + test_trienniumstats_e_stats_pas3_trend_nao_existem_mais
  + test_reality_check_le_official_stats_pela_porta_unica

.scratch/treino-modelos-pas3/relatorios/defeitos-pendentes.md
  defeito 6 riscado e marcado "CORRIGIDO em 2026-07-30"
```

---

## 3. Decisões tomadas e o porquê

### 3.1 Reaproveitar a porta única, não recalcular à parte

**Decisão:** o Reality Check chama exatamente `anos_do_trienio` e `stats_da_prova` —as mesmas
funções que `model_package.py:178-183` já usa para o Preditor — em vez de ler `OFFICIAL_STATS`
direto ou escrever uma segunda tradução trienio→ano.

**Porquê:** é o próprio texto do ticket ("a tradução... já existe... não escreva uma segunda") e
é o que garante o critério de aceite mais importante: como as duas chamadas usam o mesmo
`(ano, etapa, língua)`, `A1`/`A2` do Reality Check são aritmeticamente idênticos aos que
`previsao.a1`/`previsao.a2` já carregam — não apenas "parecidos", garantidos pela mesma função.

### 3.2 Ano-Âncora de um ano só, não os cinco do ticket 12

**Decisão:** `_stats_pas3_ancora` pega só o `(ano, etapa=3)` mais recente do `OFFICIAL_STATS`
(hoje, 2025) como estatística de substituição quando a Etapa 3 do próprio triênio do Aluno ainda
não foi publicada (a turma viva, 2024-2026).

**Porquê:** o ticket 04 do treino decidiu contra projetar a prova futura (decisão 3: Ano-Âncora),
e o ticket 12 desta rodada é quem constrói a versão completa — cinco anos na tela, um por
resultado. Construir aqui uma versão com mais de um ano seria antecipar um ticket que ainda não
rodou; a versão de um ano só resolve o que o ticket 05 pede (o consumidor da Calculadora recebe
"um ano real e já publicado" em vez da regressão) sem inventar UI que o ticket 12 vai decidir.

### 3.3 `cenario` fica no contrato, mas sem efeito

**Decisão:** o parâmetro `cenario` ("padrao"/"tendencia") de `analyze_students` — e o campo
correspondente em `GestaoRequest`, o router e `landing-page/lib/api.ts` — não foi removido, só
comentado como inerte.

**Porquê:** ele existia só para escolher entre `TRIENNIUM_STATS` e a regressão de
`STATS_PAS3_TREND`; sem as duas, não há mais o que escolher. Apagar o parâmetro tocaria schema,
router e o client TypeScript por um ganho que não está no checklist deste ticket, e o ticket 12 é
candidato natural a lhe dar uso de novo (escolher *qual* Ano-Âncora). Um comentário explícito no
código evita que o parâmetro pareça esquecido por descuido.

### 3.4 Bloco de Reality Check continua opcional e silencioso na falha

**Decisão:** mantido o `try/except Exception` amplo em volta do bloco inteiro, com
`logger.exception` — o mesmo comportamento de antes do ticket.

**Porquê:** o Reality Check é um dado auxiliar da tela da coordenação; se ele falhar (Edital
faltando, cohort ausente, etc.) o resto da análise do Aluno continua válido. O que mudou foi
**quem** o bloco chama para pegar estatística, não a política de resiliência dele.

---

## 4. Nota sobre a sessão concorrente (ticket 09)

Durante a implementação, outra sessão do Claude Code editava `api/services/gestao_service.py` e
`analytics_service.py` ao mesmo tempo (troca dos CSVs de origem, ticket 09). O commit `dd8d211`
dessa sessão acabou incluindo, no mesmo arquivo, o diff inteiro do ticket 05 — confirmado
conferindo que `TRIENNIUM_STATS`/`STATS_PAS3_TREND` já não existem e que `_stats_pas3_ancora` e o
bloco reescrito do Reality Check estão presentes em `dd8d211`. O commit `245ff12` fechou o que
sobrou fora dele: os dois testes novos e a baixa do defeito 6.

---

## 5. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê |
|---|---|
| Os cinco Anos-Âncora na tela da Calculadora | Ticket 12 do mapa `publicar-site` |
| Reaproveitar `cenario` para escolher o Ano-Âncora | Também ticket 12 — hoje o parâmetro só fica documentado como inerte |
| Teste de paridade automatizado comparando `previsao.a1`/`a2` byte-a-byte com o Reality Check | A garantia vem de os dois caminhos chamarem a mesma função (`stats_da_prova`) com os mesmos argumentos — não de um teste dedicado; nenhum critério de aceite pedia esse teste específico |

---

## 6. Como foi verificado

- **2 testes novos** em `tests/test_api_predict.py`: um documenta que `TRIENNIUM_STATS`/
  `STATS_PAS3_TREND` não existem mais como atributos do módulo; o outro monta um cohort e um mapa
  de corte sintéticos e confere que o Reality Check produz `historico_pct > 0` sem nenhum
  "falhou" no log — ou seja, sem cair no `except` silencioso.
- **Suíte inteira** (`pytest tests/ --ignore=tests/test_pas_extraction_etapa.py`): 356 passam (346
  linha de base + 2 deste ticket + 8 do ticket 09 concorrente). A exclusão é de um módulo
  (`pas_extraction.etapa`) que não existe ainda nesta árvore — falha de coleção pré-existente,
  confirmada rodando o mesmo teste isolado antes de qualquer mudança deste ticket.
- **Manual, via REPL:** `analyze_students` para o triênio `2023-2025` (com Edital completo)
  produz `historico_pct` não-zero; para `2024-2026` (turma viva, sem Etapa 3 publicada), o
  fallback `_stats_pas3_ancora` é acionado sem lançar exceção.

---

## 7. Glossário — termos necessários para entender este relatório

| Termo | Significado |
|---|---|
| **Porta única (`stats_da_prova`)** | A única função que lê `OFFICIAL_STATS` para quem vai calcular um Argumento de Etapa — treino e runtime pela mesma porta, para que o `A1` mostrado seja o `A1` com que o modelo foi treinado. |
| **Ano-Âncora** | Um ano real e já publicado usado como estatística de cenário para uma Etapa que ainda não aconteceu (*"e se a minha Etapa 3 for como a de 2025?"*), no lugar de projetar a prova futura por regressão. |
| **Reality Check (cohort)** | Comparação opcional na Gestão de Ativos: usa o banco histórico de Alunos reais para estimar a chance do Aluno atual seguir uma trajetória parecida com quem já passou. |
| **Turma viva** | O triênio em andamento (hoje, 2024-2026), cujas Etapas mais recentes ainda não têm Edital de média/desvio publicado. |
