# 01 — Semântica das flags de qualidade: descartar, reparar ou ignorar?

**Type:** research
**Status:** resolved (2026-07-26)
**Blocked by:** nenhum

## Question

Uma linha do `resultado_final.csv` marcada com `campos_formato_invalido` ou
`checksum_fecha=False` tem valor de nota **confiável ou não**? A resposta define quantos dos
66.313 registros entram no treino.

Medido em 2026-07-26 sobre `.scratch/pdf-extraction/saida-nova/resultado_final.csv`:

| Flag | Registros | Distribuição |
|---|---|---|
| `campos_formato_invalido` não-vazio | 6.476 (9,8%) | uniforme: 8,8%–10,6% em **todos** os 8 triênios |
| `checksum_fecha=False` | 2.015 (3,0%) | concentrado: 7,6% e 9,9% nos dois mais antigos, **0%** nos três mais recentes |
| `cota_padrao_suspeito=True` | 8 | irrelevante para nota |

Campo mais marcado: `argumento_final` (2.181), depois os 9 campos de nota bem distribuídos
entre si (376–570 cada).

**A hipótese a confirmar ou derrubar:** o relatório do ticket 04 do `pdf-extraction` afirma que
*"todos os 758 registros reparados fecham o checksum"* — e 2021/2023 tem exatamente 758 linhas
com `campos_formato_invalido` e **zero** falha de checksum. Isso sugere que a flag marca um campo
que **foi corrompido e reparado**, com o checksum confirmando o reparo — não um campo ainda
errado. Se for isso, descartar 9,8% da base seria jogar fora dado bom.

A uniformidade da flag entre triênios (8,8%–10,6%) reforça a hipótese: é uma taxa de corrupção
de extração de PDF, não uma propriedade dos dados de um ano.

- [x] Lido em `src/pas_extraction/resultado_final.py` o que exatamente escreve
      `campos_formato_invalido` e em que momento o reparo acontece em relação à marcação
- [x] Respondido: campo marcado é campo reparado, campo descartado, ou campo entregue como veio?
- [x] Confirmado o que `checksum_fecha` compara e com que tolerância (`checksum_delta` aparece
      como `0.0`, `0.001`, `0.002` nas amostras — qual o limiar de "fecha")
- [x] Verificado se `checksum_delta` é utilizável como **medida contínua de confiança por linha**
      em vez de um booleano — isso permitiria ponderar em vez de descartar
- [x] Recomendação explícita para o ticket 05: quais linhas entram, quais saem, e a contagem
      resultante por triênio
- [x] Relatório em `relatorios/01-semantica-das-flags-de-qualidade.md`

## Answer

Detalhe completo em [`relatorios/01-semantica-das-flags-de-qualidade.md`](../relatorios/01-semantica-das-flags-de-qualidade.md).

**`campos_formato_invalido` marca REPARO, não erro.** Provado pela ordem de operações em
`resultado_final.py::_montar_registro`: `_tentar_float` repara removendo todo espaço interno
(l. 193), o registro inteiro é descartado se o reparo falha (l. 196), e só então a flag é
computada contra o texto **bruto** (l. 198-202). Não existe no CSV linha com campo "corrompido
e deixado assim" — a flag é memória de que houve conserto, e o checksum confirma o conserto.

**Descartar as linhas marcadas seria pior que inútil: enviesaria o treino.** A corrupção de
extração atinge sobretudo o sinal de menos — `P(Argumento Final < 0 | flag no argumento_final)
= 90,8%` contra 54,4% de base — então a taxa de flag cai monotonicamente de 11,3% no decil
inferior de Argumento Final para 6,1% no superior. Jogar fora 9,8% da base removeria
preferencialmente o aluno de nota baixa.

**`checksum_delta` não serve como peso contínuo.** Dentro da tolerância assume só 6 valores
(múltiplos de 0,001), com distribuição idêntica entre linhas com e sem flag — é grade de
arredondamento, não informação. Fora da tolerância é bimodal com vazio no meio. Serve como
rótulo de 3 categorias, nunca como medida contínua.

**Emenda à hipótese do ticket — as falhas de checksum são DUAS populações, não uma:**

- **Pop. B (569 linhas, `delta > 5`)** — corrupção real: 568 têm valor fisicamente impossível
  (ex. um `eb_p2_e1` na casa dos 39 mil), todas com flag.
- **Pop. A (1.446 linhas, `delta ≤ 5`)** — **não é corrupção**: zero valores impossíveis, taxa
  de flag igual à da base, e **100% são alunos com a Etapa 1 inteira zerada**, confinados aos
  dois triênios mais antigos — os únicos cuja tabela oficial vem de Edital avulso.

A Pop. A é achado direto para o ticket 02: parte da concentração de checksum nos triênios
antigos não é layout velho nem fórmula diferente, é um subgrupo específico de aluno.

**Regra recomendada para o ticket 05:** `checksum_fecha == True`, **sem filtrar por flag** →
**64.298 linhas (96,96%)**, incluindo deliberadamente 5.740 reparadas. Nenhum valor
fisicamente impossível sobrevive ao filtro. Por triênio: 8.877 / 8.874 / 5.804 / 8.392 /
7.130 / 8.019 / 8.499 / 8.703. Variante opcional (`ou delta ≤ 5`) recupera 65.744.

**Dois avisos herdados pelo ticket 05:**

- **4.285 linhas incluídas têm a Etapa 1 zerada.** `eb_pas1 = 0` estoura o CV que roteia o
  ensemble atual (`std/mean` com média perto de zero). Excluí-las daria 60.013.
- **146 inscrições aparecem em mais de um triênio** (0,22%) — vazamento entre treino e teste.
