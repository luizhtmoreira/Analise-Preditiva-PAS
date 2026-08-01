# 08 — Rodada completa sobre os 77 Editais, determinística

**What to build:** um comando só extrai os 77 Editais em `data/pdfs` e produz os CSVs de todas as
Famílias, com o relatório de validação do corpus inteiro. É aqui que o pipeline deixa de ser um
protótipo verificado em fixture e passa a ser a fonte dos ~122 mil registros que vão alimentar o
treino dos modelos.

Duas propriedades que só existem em escala completa:

**Determinismo.** Rodar duas vezes sobre a mesma entrada produz exatamente a mesma saída, byte a
byte. Sem isso não dá para comparar execuções, e o loop de correção de parser perde seu critério
de parada — a única forma de saber se uma mudança melhorou é diffar duas rodadas.

**Reconciliação cruzada entre Editais.** O mesmo número de inscrição aparece em Editais
diferentes; o nome associado tem que bater. É uma verificação independente de todas as outras,
porque não depende de nenhuma fórmula nem de nenhuma suposição sobre o schema — e só é possível
com o corpus inteiro em mãos.

O comando continua aceitando um subconjunto, como no ticket 01. Não há caminho absoluto de
máquina em lugar nenhum: roda em qualquer clone do repositório.

Com o pipeline real existindo, os quatro scripts de protótipo descartáveis
(`scripts/prototype_pdf_census.py`, `prototype_pdf_probe.py`, `prototype_checksum.py`,
`prototype_cotas.py`, e o `prototype_pdf_census.json`) saem. O `scripts/NOTES.md` fica — é o
registro de como cada decisão foi medida. Os extratores antigos (`extrator_master.py` e
companhia) **não** são removidos aqui; isso espera o ticket 09 cobrir a família de convocação.

**Blocked by:**
- 05 — Parse dirigido por seção
- 07 — Relatório de validação agrupado por padrão

**Status:** ready-for-agent

- [ ] Um comando extrai os 77 Editais e produz os CSVs de todas as Famílias já implementadas
- [ ] Duas execuções sobre a mesma entrada produzem saída idêntica byte a byte
- [ ] O mesmo número de inscrição encontrado em Editais diferentes é conferido quanto ao nome
- [ ] O relatório de validação cobre o corpus inteiro, agrupado por padrão
- [ ] Nenhum caminho absoluto de máquina no pipeline
- [ ] Os quatro scripts `prototype_*` foram removidos e o `scripts/NOTES.md` permanece
