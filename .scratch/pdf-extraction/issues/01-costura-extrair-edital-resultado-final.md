# 01 — Costura `extrair_edital` + classificador de família + CSV de Resultado Final

**What to build:** o dono do produto aponta o pipeline para um Edital de Resultado Final e
recebe um CSV com um registro por Aluno, sem ter que informar de que formato o arquivo é nem
manter nenhuma lista de arquivos à mão.

O pipeline descobre sozinho a Família de Edital lendo a frase `"na seguinte ordem: ..."` que o
próprio Edital declara na primeira página. A frase é canonizada antes de comparar — sem acentos,
sem caixa, sem caractere não-alfanumérico. Sem essa canonização aparecem 12 schemas distintos
onde existem 3; a diferença é ruído de extração (`Campus` vs `campus`, `"abaix o"` em vez de
`"abaixo"`, espaço no fim). Depois de canonizar, os grupos restantes colapsam em 3 porque as
diferenças são de redação institucional (`"nome do candidato"` virou `"nome da pessoa candidata"`
a partir de 2023/2025; `"nota final"` virou `"nota provisória"`).

Do Resultado Final saem os 22 campos: inscrição, nome, as 9 notas (P1, P2 e Redação de cada
Etapa), o Argumento Final e as 10 classificações por Sistema de Concorrência. Campus, curso e
turno vêm em cabeçalhos intercalados no fluxo e são carregados como estado durante o parse. `-`
significa "não concorreu naquele sistema". Modo de extração de texto: `plain` — medido, `layout`
produz **mais** números partidos (74 contra 68 hits na amostra) porque injeta espaços para
preservar alinhamento visual.

Cada linha do CSV carrega proveniência — arquivo de origem, edital, triênio, página — para que
qualquer valor possa ser auditado de volta até o PDF.

Este ticket estabelece a costura que todos os outros atravessam:
`extrair_edital(caminho_pdf) -> ResultadoExtracao`. É por essa fronteira que toda a lógica
posterior (parse por seção, checksum, inferência de língua, dedução de cota) vai ser testada, e
não pela estrutura interna do parser — que vai mudar muito durante o loop de correção.

O código vive num pacote novo, `src/pas_extraction/`, separado de `pas_intelligence`: extração é
offline, lê PDF e escreve CSV; `pas_intelligence` é predição dentro do app. Os extratores em
`scripts/` (`extrator_master.py`, `extract.py`, `extract_students.py`, `extrator_teste.py`,
`debug_quota_logic.py`) são referência histórica, não devem ser estendidos, e ficam intactos.

Também entrega o utilitário que fatia fixtures pequenas dos Editais reais. Fatiar em vez de
sintetizar preserva a corrupção real de extração, que é justamente o que os tickets seguintes
precisam testar.

**Fixtures não são commitadas.** São slices de dados reais de Alunos (nome, notas, inscrição) —
caem na mesma regra de "nada que envolva dados sobe pro repo" que vale para os parsers
([[project_parser_privacy]]). Ficam em `tests/fixtures/` (ou equivalente), listado no
`.gitignore`, geradas localmente pelo utilitário de fatiamento a partir de `data/pdfs`. Os testes
que dependem delas devem pular (skip) com uma mensagem clara quando a fixture não existir
localmente, em vez de falhar — assim a suíte roda em qualquer clone sem a pasta `data/pdfs`
completa, e quem tem os PDFs reais localmente consegue gerar as fixtures e rodar a suíte inteira.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Um comando único extrai um Edital de Resultado Final e escreve um CSV, sem lista de arquivos hardcoded e sem nenhum caminho absoluto de máquina
- [ ] O comando aceita um subconjunto de Editais, para iterar rápido durante o desenvolvimento dos parsers
- [ ] A Família de Edital é determinada pelo schema declarado na primeira página, canonizado — um Edital novo com redação institucional diferente é classificado corretamente sem mudança de código
- [ ] Os 22 campos são extraídos, com campus/curso/turno vindos dos cabeçalhos intercalados
- [ ] `-` numa classificação é preservado como "não concorreu", distinto de ausência de dado
- [ ] Cada linha do CSV carrega arquivo de origem, edital, triênio e página
- [ ] Existe o utilitário que fatia uma fixture de 3 a 5 páginas de um Edital real
- [ ] A fixture de Resultado Final de 22 campos existe localmente (gerada pelo utilitário), listada no `.gitignore`, não commitada
- [ ] Um teste exercita a costura `extrair_edital` na fixture e verifica a contagem de registros extraídos, pulando com mensagem clara se a fixture não existir localmente
- [ ] Nenhum extrator antigo em `scripts/` foi modificado ou removido
