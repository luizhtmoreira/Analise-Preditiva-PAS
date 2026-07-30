# 12 — Ano-Âncora: cinco anos reais na tela

**What to build:** a Calculadora de Estratégia mostra o resultado contra **cinco anos reais** de
Etapa 3 em vez de um único ano extrapolado, e a faixa entre eles **é** a incerteza do Aluno sobre a
prova que ele ainda não fez.

Um **Ano-Âncora** é um ano real e já publicado usado como cenário: *"e se a minha Etapa 3 for como a
de 2023?"*.

## O que ele substitui

Hoje a Calculadora usa `STATS_PAS3_TREND` — uma regressão linear que **extrapola** a média e o
desvio de uma prova que ainda não aconteceu, e devolve um número único com uma precisão que não
tem. É a decisão 3 do relatório 04 (Alvo Canônico): *nada de projetar a prova futura*. O
`STATS_PAS3_TREND` já sai do caminho no ticket 05; este ticket constrói o que fica no lugar.

## A forma

A resposta passa a carregar **cinco resultados**, um por Ano-Âncora, com o mais recente em destaque:

```
┌──────────────────────────────┐
│ Se a Etapa 3 for como…       │
│  2025 →  P2 necessária 41,2  │
│  2024 →  P2 necessária 43,8  │
│  2023 →  P2 necessária 38,1  │
│  2022 →  P2 necessária 39,4  │
│  2021 →  P2 necessária 36,7  │
└──────────────────────────────┘
```

Os Anos-Âncora são as cinco chaves `(ano, Etapa 3)` mais recentes do `OFFICIAL_STATS` — hoje 2025,
2024, 2023, 2022 e 2021. A lista é **derivada do dado**, não uma constante: quando o Edital de 2026
entrar, o quinto ano cai fora sozinho.

## Cada Ano-Âncora varia duas coisas juntas

Separá-las produziria um cenário que nunca existiu. Um Ano-Âncora carrega:

- a **média e o desvio da Etapa 3 daquele ano** — o que muda quanto vale cada ponto de P2;
- a **Nota de Corte do curso no triênio correspondente** — Ano-Âncora 2025 → triênio 2023-2025.

Isso é o que o relatório 04 §7.2 registra: *"a Nota de Corte comparada é a do Ano-Âncora, e são
cinco comparações, não uma"*.

## Por que isso é honesto e o número único não era

O Aluno não sabe se a prova dele vai ser fácil ou difícil, e nós também não. A dispersão entre cinco
anos reais é uma declaração medida dessa ignorância. Um número extrapolado é a mesma ignorância,
escondida atrás de uma casa decimal.

**Blocked by:** 11 (Calculadora sem `.joblib` — o Estimador Auxiliar e a faixa medida vêm antes; sem
eles os cinco números estariam todos errados por igual).

**Status:** ready-for-agent

- [ ] A resposta da Calculadora carrega uma lista de cinco resultados, um por Ano-Âncora
- [ ] Os Anos-Âncora saem das cinco chaves `(ano, Etapa 3)` mais recentes do `OFFICIAL_STATS`, não
      de uma lista cravada
- [ ] Cada Ano-Âncora usa a estatística da Etapa 3 **e** a Nota de Corte do triênio correspondente
- [ ] A tela mostra os cinco com o mais recente em destaque, e a faixa entre eles é legível como
      incerteza
- [ ] Nenhuma projeção linear de prova futura sobra no caminho
- [ ] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
