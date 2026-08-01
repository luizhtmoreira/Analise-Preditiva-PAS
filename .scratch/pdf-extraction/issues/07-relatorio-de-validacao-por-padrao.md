# 07 — Relatório de validação agrupado por padrão

**What to build:** um relatório que diz **onde mexer no parser**, agrupando as falhas por padrão
em vez de listar registro por registro. Com 122 mil registros, uma lista linear de falhas é
inútil; o que serve é saber que 200 registros falharam do mesmo jeito.

O relatório precisa distinguir dois tipos de falha, porque exigem ações opostas:

- **Deltas concentrados** — as falhas empilhadas em torno de um mesmo valor indicam **fórmula
  incompleta**. O parser está certo, o cálculo é que está faltando um termo.
- **Deltas espalhados** — indicam **dado corrompido**. O cálculo está certo, o parser é que
  quebrou o número.

Por isso o relatório reporta a **distribuição** dos deltas, não só a taxa de acerto. Este não é
um refinamento estético: durante o protótipo, uma versão do checksum que parecia excelente
(83,9% de acerto) teria descartado 200 de 1.261 registros perfeitamente válidos. O que denunciou
o problema não foi a taxa — foi a forma da distribuição, com as falhas empilhadas em torno de 0,7
em vez de espalhadas. Um gate de qualidade que reportasse só a taxa teria deixado passar.

Daí a regra que o relatório impõe: **nenhum registro é descartado automaticamente por falhar no
checksum sem que o padrão da falha esteja explicado.**

O relatório consolida as camadas de validação já existentes, em ordem de poder de detecção:
checksum do Argumento Final (12 campos de uma vez), classificação como sequência `1..N` (pega o
ponto cego do checksum — o registro que nunca foi extraído), ordem alfabética, formato numérico
exato, e fecho do reticulado de cotas.

Saída em terminal e arquivo. Interface visual está fora de escopo.

**Blocked by:**
- 02 — Validações estruturais por registro
- 04 — Checksum do Argumento Final + inferência de língua por Etapa
- 06 — Dedução das Cotas Declaradas

**Status:** ready-for-agent

- [ ] As falhas são agrupadas por padrão, não listadas registro a registro
- [ ] O relatório reporta a distribuição dos deltas do checksum, não apenas a taxa de acerto
- [ ] Falhas com deltas concentrados são distinguidas de falhas com deltas espalhados
- [ ] Nenhum registro é descartado automaticamente sem que o padrão da falha esteja explicado
- [ ] O relatório cobre as cinco camadas de validação, indicando qual delas pegou cada grupo de falhas
- [ ] Saída em terminal e em arquivo
