# 11 — Relatório de diferenças do `OFFICIAL_STATS`

**What to build:** antes de trocar qualquer valor, o dono do produto vê exatamente quais entradas
do `OFFICIAL_STATS` mudam e em quanto.

O `OFFICIAL_STATS` de `src/pas_intelligence/pas_constants.py` está documentado no próprio arquivo
como *"gerado automaticamente via análise do banco_alunos_pas_final.csv"* — ou seja, média e
desvio-padrão foram **inferidos dos dados** em vez de lidos dos Editais. Para o triênio 2022/2024,
Etapa 1, o projeto usa `m_p2=20.709` enquanto o Edital oficial diz `20.406`. Todo Argumento Final
calculado pelo sistema carrega esse erro de estimativa.

A correção é aplicada em duas etapas, e esta é a primeira: só o relatório. Como a troca altera a
saída de modelos em produção, o diff precisa ser revisado antes de qualquer substituição — daí
ela ser um ticket separado.

O relatório também precisa mostrar a mudança de **forma**, não só de valor: o `ExamStats` atual
tem um `m_p1` único, mas o Edital publica a Parte 1 separada por língua estrangeira. O relatório
deixa visível onde o valor atual é uma agregação indevida de três valores oficiais distintos.

Nenhuma alteração em `pas_constants.py` neste ticket.

**Blocked by:** 03 — Extração da tabela de médias e desvios.

**Status:** ready-for-agent

- [ ] O relatório lista, por `(ano, etapa)`, o valor atual, o valor oficial e a diferença
- [ ] Entradas do `OFFICIAL_STATS` sem cobertura nos Editais extraídos são listadas explicitamente
- [ ] O relatório mostra onde o `m_p1` único agrega três valores oficiais por língua
- [ ] `pas_constants.py` não é alterado neste ticket
