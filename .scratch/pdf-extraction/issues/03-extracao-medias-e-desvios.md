# 03 — Extração da tabela de médias e desvios

**What to build:** o dono do produto obtém, extraída dos próprios Editais, a tabela oficial de
média e desvio-padrão de cada Etapa — o insumo que falta para normalizar notas com valores
oficiais em vez de estimados.

Esta é a terceira Família de Edital, *Médias e Desvios*, e o classificador do ticket 01 já sabe
reconhecê-la pelo schema declarado. Modo de extração: `plain`.

A tabela aparece em **dois lugares diferentes** conforme o triênio: na cauda do Edital de
Resultado Final, ou num Edital avulso só para isso. O pipeline busca nos dois, porque triênios
diferentes publicaram de formas diferentes.

Um detalhe de forma que importa: o Edital publica a média e o desvio-padrão da **Parte 1
separados por língua estrangeira**. O dado atual do projeto agrega isso indevidamente. A extração
preserva a separação — sem ela, o checksum do ticket 04 não fecha.

Sai um CSV próprio, com a mesma proveniência dos demais. É CSV separado porque as três Famílias
têm granularidades diferentes: aqui a linha é uma Etapa, não um Aluno.

**Blocked by:** 01 — Costura `extrair_edital` + classificador de família + CSV de Resultado Final.

**Status:** ready-for-agent

- [ ] A Família *Médias e Desvios* é reconhecida pelo classificador de schema declarado
- [ ] A tabela é encontrada tanto na cauda de um Edital de Resultado Final quanto num Edital avulso
- [ ] Média e desvio-padrão da Parte 1 são gravados separadamente por língua estrangeira
- [ ] Sai um CSV próprio da família, com colunas de proveniência
- [ ] Existe fixture de médias/desvios localmente (gerada pelo utilitário do ticket 01, não commitada), e um teste que verifica os valores extraídos dela, pulando se a fixture não existir
