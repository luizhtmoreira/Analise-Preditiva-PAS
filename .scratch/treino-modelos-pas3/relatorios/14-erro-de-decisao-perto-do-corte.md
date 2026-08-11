# Relatório — Erro de decisão perto do corte

**Tipo:** medição, releitura do lacre já aberto. **Nenhuma decisão de modelagem.**
**Script:** `scripts/erro_de_decisao_perto_do_corte.py`
**Data:** 2026-08-01 (números abaixo corrigidos em 2026-08-11)
**Motivo:** antes de divulgar o produto, o dono do produto perguntou se o `94%` da landing se
sustenta. Ele se sustenta, mas o agregado esconde de quem ele vale.
**Privacidade:** só agregados e contagens; nenhuma linha individual sai do script.

**Correção de 2026-08-11:** `_cortes_do_lacre` (copiado de `lacre_ticket13.py`) não excluía as
linhas "SUB JUDICE" de `notas_corte.csv` — convocação por decisão judicial, à parte da
concorrência normal do curso, cuja nota não é comparável (mesmo filtro que
`api/services/gestao_service.py::_normalizar_cortes` já aplicava na API ao vivo desde o ticket do
dia 01/08, mas que nunca tinha sido replicado aqui nem em `lacre_ticket13.py` /
`baseline_honesto.py`). 11 dos 7.449 Alunos "com corte casado" originais eram, na verdade, casados
contra um corte de rodada judicial. Números abaixo já refletem o filtro corrigido; a mudança é
pequena (5,41% → 5,36% de erro agregado) e não muda nenhuma leitura da §4.

---

## 1. A pergunta

O ticket 13 mediu **5,41% de erro de decisão** no triênio lacrado (7.449 Alunos com corte
casado). O que ele não mediu: **como esse erro se distribui entre quem está longe e quem está
perto da Nota de Corte.** Como 67,9% das probabilidades saturam (`<1%` ou `>99%`), havia a
suspeita de que a taxa agregada fosse carregada por casos triviais.

Era. E o tamanho do efeito é maior do que o esperado.

## 2. Como foi medido

Releitura do mesmo triênio 2023/2025, mesmo pacote (`models/pas3`, commit `355b6e4`), mesma
limpeza de cortes do `lacre_ticket13.py::_cortes_do_lacre` (agora também excluindo "SUB JUDICE",
ver correção acima). O script **não chama** `holdout_final_use_uma_vez`: o lacre é gasto uma vez
e já foi gasto em 2026-07-28 — isto aqui é releitura de um resultado publicado, não uma segunda
abertura.

A distância é medida em múltiplos da **Largura de Incerteza do próprio Aluno**
(`3 × σ(A3)` da classe dele: 14,97 com Etapa 1, 15,48 sem), não em pontos absolutos.

Duas leituras, que respondem a perguntas diferentes:
- **real** (`AF real − corte`): quem *estava* na linha. Retrospectiva.
- **prevista** (`AF previsto − corte`): o que o app mostra na hora. É a que vale para o produto,
  porque a real só existe depois do resultado.

`chutar maioria` = o erro de um sistema burro que responde sempre o mais comum **daquela faixa**.
É o piso honesto: a diferença entre as duas colunas é o que o modelo acrescenta.

## 3. Resultado

### Por distância REAL até o corte

| faixa | n | % base | erro decisão | chutar maioria | Brier | taxa aprovação |
|---|---:|---:|---:|---:|---:|---:|
| ≤ 0,5 largura | 580 | 7,8% | **36,72%** | 46,72% | 0,2523 | 53,3% |
| 0,5 – 1 largura | 486 | 6,5% | **19,55%** | 47,12% | 0,1446 | 47,1% |
| 1 – 2 larguras | 984 | 13,2% | 8,54% | 42,68% | 0,0643 | 42,7% |
| > 2 larguras | 5.388 | 72,4% | **0,13%** | 27,99% | 0,0016 | 28,0% |
| **TODOS** | 7.438 | 100% | **5,36%** | 33,15% | 0,0388 | 33,2% |

### Por distância PREVISTA até o corte (o que o app mostra)

| faixa | n | % base | erro decisão | chutar maioria | Brier | taxa aprovação |
|---|---:|---:|---:|---:|---:|---:|
| ≤ 0,5 largura | 560 | 7,5% | **37,32%** | 48,75% | 0,2291 | 51,2% |
| 0,5 – 1 largura | 527 | 7,1% | **20,30%** | 49,15% | 0,1601 | 50,9% |
| 1 – 2 larguras | 986 | 13,3% | 7,61% | 44,02% | 0,0688 | 44,0% |
| > 2 larguras | 5.365 | 72,1% | **0,15%** | 27,53% | 0,0015 | 27,5% |

### Saturação

| | valor |
|---|---:|
| probabilidades saturadas | 68,0% |
| erro de decisão só nas saturadas | **0,04%** |
| erro de decisão só nas NÃO saturadas | **16,66%** |

## 4. Leitura

**4.1 O 94% é real e é carregado por quem não precisa dele.** 72% dos Alunos estão a mais de
duas larguras do corte, e lá o erro é 0,13% — praticamente perfeito, e trivial: são pessoas
muito acima ou muito abaixo da linha. Tirando-os, o erro sobe de 5,36% para **16,66%**.

**4.2 A faixa do meio é onde o modelo brilha de verdade.** Entre 0,5 e 1 largura, o erro é
19,6% contra 47,1% de chutar a maioria — o modelo é **2,4× melhor que o chute**, com Brier
caindo de ~0,25 para 0,145. É a faixa onde há sinal e ele é extraído. Isso é o argumento de
venda honesto, e é melhor do que o número agregado sugere, não pior.

**4.3 Na faixa mais próxima o modelo quase não acrescenta — e é ligeiramente confiante demais.**
Dentro de 0,5 largura (≈7,8 pontos de Argumento Final), o erro é 36,7% contra 46,7% do chute:
há sinal direcional, mas pequeno. O achado que importa está no Brier: **0,2523**, enquanto
responder `50%` para todo mundo dessa faixa daria exatamente **0,2500**. Ou seja, para o Aluno
em cima da linha, a probabilidade que o app mostra é marginalmente **pior calibrada** que
admitir "é cara ou coroa".

Não é defeito de treino nem viés: é a faixa onde o resultado genuinamente é quase moeda, e o
modelo expressa um pouco mais de convicção do que tem. Afeta 7,8% dos Alunos.

**4.4 Consequência de produto, não de modelo.** O Aluno dentro de 0,5 largura não deveria
receber "62%" com a mesma cara de quem recebe "99%". A correção natural é de tela — uma faixa
declarada de indefinição perto do corte — e não mexer no modelo. Fica registrado aqui, sem
ticket.

## 5. Efeito sobre a comunicação

O `94 de cada 100` da landing continua verdadeiro e agora tem procedência repartida. O que
**não** se pode dizer é "94% de acerto para você", porque para 7,8% dos Alunos o número real é
63%. As duas frases defensáveis:

- "acerta o veredito em mais de 94 de cada 100 alunos" — agregado, verdadeiro, medido.
- "quando o aluno está perto da linha, acerta 2 em cada 3, contra metade de um chute" — a
  frase honesta para quem perguntar do caso difícil.

## 6. Glossário

- **Largura de Incerteza** — o quanto o modelo tipicamente erra, usado como o desvio-padrão da
  normal da probabilidade de aprovação. Em Argumento Final é `3 × σ(A3)`, porque
  `Argumento Final = A1 + 2·A2 + 3·Â3` e só o `Â3` é previsto.
- **Erro de decisão** — em que fração dos Alunos o sistema teria dito a coisa errada sobre passar
  (probabilidade acima ou abaixo de 50% contra o que de fato aconteceu).
- **Brier** — erro quadrático médio da própria probabilidade contra o que aconteceu. Menor é
  melhor. Um sistema que responde sempre `50%` tem Brier `0,25` por construção — é a régua
  contra a qual a faixa mais próxima foi lida na §4.3.
- **Saturada** — probabilidade abaixo de 1% ou acima de 99%.
- **Chutar maioria** — sistema burro que responde sempre a classe mais comum da faixa. Piso de
  comparação: erro de decisão só significa alguma coisa lido contra ele.
