# 05 — `TRIENNIUM_STATS` e `STATS_PAS3_TREND` saem; tudo lê `OFFICIAL_STATS`

**What to build:** existe **um** lugar no produto de onde sai média e desvio oficiais, e ele é o
`OFFICIAL_STATS`, que vem dos Editais do Cebraspe.

Hoje existem dois. `api/services/gestao_service.py` carrega um `TRIENNIUM_STATS` próprio cujos
números **divergem dos Editais**:

| triênio / etapa | `TRIENNIUM_STATS` (P2) | Edital | |
|---|---|---|---|
| 2023-2025 PAS2 | 29,2750 / 14,2913 | 29,275 / 14,604 | média bate, **desvio não** |
| 2022-2024 PAS1 | 20,7094 / 13,5819 | 20,406 / 13,533 | **não bate** |
| 2022-2024 PAS2 | 30,3477 / 13,2532 | 29,980 / 13,213 | **não bate** |
| 2022-2024 PAS3 | 32,0862 / 14,1289 | 31,740 / 14,063 | **não bate** |

As de 2022-2024 têm quatro casas decimais e desvio sistematicamente maior que o oficial — cara de
**calculadas de uma amostra de Alunos**, não copiadas do Edital. É o defeito 6 de
`defeitos-pendentes.md`, severidade **alta** com conserto barato: a fonte certa já está em disco.

**Por que agora, e não como faxina opcional.** O `get_strategy_prediction` — a Calculadora de
Estratégia, que entra nesta rodada (ticket 11) — consome `triennium_stats`. Publicar a Calculadora
sobre esse dicionário significa pôr no ar **duas telas que calculam o mesmo `A1` e o mesmo `A2` com
números diferentes**, no mesmo site, para o mesmo Aluno.

E a divergência é maior justamente no **desvio**, que é o denominador de todo z-score. Enquanto
`A1` e `A2` eram só entrada de um modelo que previa o Argumento inteiro, um erro ali se diluía. Na
rota canônica do ADR-0009 eles são a parte **exata** da conta — `Argumento Final = A1 + 2·A2 + 3·Â3`
— e um desvio errado contamina ⅗ do peso, sem nada para compensar.

**O `STATS_PAS3_TREND` sai junto.** Ele é uma regressão linear que extrapola a média e o desvio de
uma prova que **ainda não aconteceu**. O ticket 04 do treino já decidiu contra isso (decisão 3:
Ano-Âncora — nada de projetar a prova futura), e o ticket 12 desta rodada constrói a substituição.
Neste ticket ele sai do caminho da Gestão; o consumidor da Calculadora passa a receber a
estatística de um ano real e já publicado.

**A chave muda de forma.** `TRIENNIUM_STATS` é chaveado por string de triênio (`"2023-2025"`) mais
rótulo de etapa (`"PAS1"`). `OFFICIAL_STATS` é chaveado por `(ano, etapa)`. A tradução entre os dois
já existe e é a função que converte triênio em três anos, uma por Etapa — não escreva uma segunda.

**Blocked by:** 01 (`ExamStats` com Parte 1 misturada e procedência) — é ele que fixa a forma final
do dado que este ticket passa a consumir em todo lugar.

**Status:** ready-for-agent

- [ ] `TRIENNIUM_STATS` e `STATS_PAS3_TREND` não existem mais no código
- [ ] Todo consumidor de média e desvio oficiais lê `OFFICIAL_STATS`, chaveado por `(ano, etapa)`,
      pela porta única que já existe
- [ ] O Reality Check da Gestão de Ativos continua funcionando, agora com os números do Edital
- [ ] Um Aluno de teste produz o **mesmo** `A1` e o **mesmo** `A2` pelo caminho do Preditor e pelo
      caminho da Gestão — é o que este ticket existe para garantir
- [ ] O defeito 6 de `defeitos-pendentes.md` é marcado como corrigido
- [ ] `pytest tests/` continua verde
