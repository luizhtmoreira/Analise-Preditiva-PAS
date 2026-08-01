# 04 — Checksum do Argumento Final + inferência de língua por Etapa

**What to build:** cada registro extraído tem seu Argumento Final recalculado a partir das 9
notas brutas e comparado com o valor impresso no Edital. Isso substitui inspeção humana por
verificação matemática: **um único número verifica 12 campos de uma vez**.

A fórmula é `AF = 1×AP1 + 2×AP2 + 3×AP3`, onde cada `APn = argumento(P1) + argumento(P2) +
argumento(Redação)` e `argumento(x) = ((x − média) / desvio) × peso`, com `PESO_P1=0,72`,
`PESO_P2=8,28`, `PESO_REDACAO=1,00`. É exatamente a fórmula já implementada em
`argument_calculator.py` — o pipeline **reusa essa função** em vez de reimplementá-la. O mesmo
cálculo que hoje prediz passa agora a servir como verificação. A regressão do protótipo sobre
1.261 registros recuperou os pesos `(0,987, 1,972, 2,994)` com R²=0,9984, então a fórmula já está
validada contra dado oficial.

Como subproduto, o pipeline recupera um dado que **não está impresso em lugar nenhum do PDF**: a
língua estrangeira que cada Aluno fez. Ela é inferida testando as 27 combinações (3 línguas × 3
Etapas) e ficando com a que minimiza o delta do checksum.

A língua é inferida **por Etapa, não por Aluno** — 17,4% dos Alunos trocam de língua entre
Etapas, e essa distinção é o que separa um pipeline utilizável de um que descarta dado bom:

```
                          língua fixa por Aluno   língua por Etapa
delta <= 0,005            83,9%                   99,8%  (1258/1261)
falhas (delta > 0,01)     203                     3
```

Tolerância operacional: `delta <= 0,005`. Metade dos registros fecha em ≤0,001; o resto é o
arredondamento oficial de 3 casas.

**Blocked by:** 03 — Extração da tabela de médias e desvios. (O checksum precisa da média e do
desvio oficiais do próprio Edital, e da Parte 1 separada por língua.)

**Status:** ready-for-agent

- [ ] Cada registro tem o Argumento Final recalculado e comparado com o impresso, com o delta gravado na linha
- [ ] O cálculo reusa `argument_calculator.py` em vez de reimplementar a fórmula
- [ ] A língua estrangeira é inferida por qual das três faz o checksum fechar
- [ ] A inferência é por Etapa e não por Aluno, e a língua de cada Etapa é gravada por Aluno
- [ ] A tolerância aplicada é `delta <= 0,005`
- [ ] Nenhum registro é descartado neste ticket — o checksum só marca
