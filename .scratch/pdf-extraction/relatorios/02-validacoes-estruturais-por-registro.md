# Relatório — Ticket 02: Validações estruturais por registro

**Ticket:** `.scratch/pdf-extraction/issues/02-validacoes-estruturais-por-registro.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/` (gitignored, mesma política do ticket 01)
**Onde vive o teste:** `tests/test_pas_extraction.py` (versionado normalmente, mesmo arquivo do ticket 01)

---

## 1. O que foi pedido

Cada linha do CSV de Resultado Final passa a carregar o resultado da própria validação
estrutural, para que o consumidor filtre por confiança em vez de confiar cegamente. Três
verificações: (1) todo campo numérico casa `^-?\d+\.\d{3}$` exatamente; (2) a classificação
de cada Aluno é uma sequência `1..N` sem buracos, por curso e por Sistema de Concorrência;
(3) os nomes vêm em ordem alfabética dentro do curso. Mais quatro casos de corrupção real do
protótipo fixados como regressão.

Critérios de aceite (todos atendidos — ver seção 6):

- [x] Cada linha do CSV carrega o resultado da sua própria validação
- [x] Campo numérico que não case `^-?\d+\.\d{3}$` exatamente é sinalizado
- [x] Buraco na sequência `1..N` de classificação é detectado por curso e por Sistema de
      Concorrência, indicando qual posição faltou
- [x] Quebra de ordem alfabética dentro do curso é sinalizada
- [x] Os quatro casos de corrupção do protótipo estão fixados como testes de regressão
      (com uma ressalva de fidelidade — ver seção 3.2)
- [x] Os testes exercitam a costura `extrair_edital`, não a estrutura interna do parser

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/
  models.py       — + ValidacaoRegistro (dataclass) e campo `validacao` em RegistroResultadoFinal
  resultado_final.py — _tentar_float virou reparo tolerante; + _formato_numerico_valido
  validacao.py     — NOVO: validar_sequencia_e_ordem() — buracos de classificação + ordem alfabética
  pipeline.py       — + 1 chamada: validar_sequencia_e_ordem(registros)
  csv_writer.py     — + 3 colunas: campos_formato_invalido, classificacao_buracos, ordem_alfabetica_quebrada
  fixtures.py       — + fatiar_paginas() (fixtures de páginas não contíguas)

tests/test_pas_extraction.py — +10 testes novos (23 no total)
tests/fixtures/resultado_final_curso_completo.pdf — NOVA fixture, 2 páginas
```

Fluxo: `parse_resultado_final()` (parágrafo por registro, com o texto bruto ainda em mãos)
preenche `campos_formato_invalido` no momento do parse; `validar_sequencia_e_ordem()`, logo
depois em `pipeline.py`, preenche `buracos_classificacao` e `fora_de_ordem_alfabetica` — só
possível com a lista inteira de registros do Edital em mãos.

Validado contra os **6 PDFs reais** de Resultado Final em `data/pdfs` (242 a 419 páginas
cada, ~52.700 registros no total): pipeline roda de ponta a ponta sem exceção em nenhum.

---

## 3. Decisões tomadas e o porquê

### 3.1 Número/classificação partidos por espaço deixam de ser descartados

**Decisão:** `_tentar_float` agora remove *todo* espaço interno do campo antes de tentar o
parse (não só a borda) — "1 7.539" vira "17.539", "- 21.683" vira -21.683. O texto bruto
(antes do reparo) é comparado à parte contra `^-?\d+\.\d{3}$` em `_formato_numerico_valido`,
e o nome do campo entra em `campos_formato_invalido` quando não bate. O mesmo reparo (remover
espaço) foi aplicado às 10 classificações, que não passam pelo checador de formato exato
(são inteiros/`-`, não floats de 3 casas) mas sofriam a mesma corrupção.

**Porquê — e por que isso não estava no ticket 01:** medindo o comportamento atual contra os
6 PDFs reais antes de escrever qualquer código, descobri que essa corrupção é muito mais
comum do que "10 registros descartados" (número citado no relatório do ticket 01, que só
contava os 9 campos de nota, não o argumento final): no Ed_38 completo (242 páginas, 8.499
candidatos a registro), **758 (8,9%) eram descartados inteiros** por essa causa antes desta
mudança — 456 números partidos, 302 sinais negativos separados. O ticket 01 tratava isso como
"lixo de extração que nem chegou a virar um registro válido" — decisão correta *para aquele
ticket*, porque a camada que recupera e sinaliza esse tipo de corrupção é exatamente este
ticket 02. Sem essa mudança, a validação de formato pedida no ticket nunca teria o que
validar: o registro já teria sido descartado antes de chegar a ela.

**Consequência que quebra um teste do ticket 01:** `CONTAGEM_ESPERADA` na fixture de 6
páginas subiu de 170 para 189 — os 19 candidatos antes descartados por essa causa agora
aparecem (sinalizados). Só 1 candidato continua descartado de verdade: o último registro da
página 6, cortado pelo limite físico da fixture (span incompleto, não corrupção). Atualizei o
teste existente com um comentário explicando a mudança, em vez de criar um teste novo
duplicado.

### 3.2 Os quatro casos de regressão usam corrupção real, mas não byte-idêntica ao protótipo

**Decisão:** nenhum dos quatro casos do ticket é reproduzido como string literal idêntica ao
que está em `scripts/NOTES.md`. Em vez disso, busquei ocorrências reais da *mesma classe* de
corrupção nos 6 PDFs reais, usando o mesmo pipeline de extração de texto que o código de
produção usa, e usei essas ocorrências.

**Porquê:** medi diretamente, antes de escrever qualquer teste, se os exemplos literais do
protótipo ainda existem no texto extraído hoje. Resultado:

- `"56.29 1"` → não existe hoje nesse texto exato, mas a mesma classe existe abundantemente:
  usei `"1 7.539"` → `17.539` (mesmo tipo de corte, um espaço solto no meio do número).
- `"- 58.570"` → idem; usei `"- 21.683"` → `-21.683`, no mesmo registro real da fixture.
- Cabeçalho `"ENGENHARIA DE REDES DE COMUNICAÇÃO (BACHARELADO)"` engolido: localizei essa
  string exata em 6 dos 7 PDFs de Resultado Final (nem preciso ter sido sintética — é comum),
  mas em **todas** as ocorrências reais que inspecionei, o cabeçalho aparece exatamente na
  fronteira entre dois registros bem formados — o mesmo padrão "cabeçalho colado ao fim do
  último registro" que o ticket 01 já resolve com o parser ancorado por inscrição. A fixture
  de 6 páginas já contém essa mesma classe de corrupção com outro par de cursos
  (ADMINISTRAÇÃO → AGRONOMIA, na transição de página 6) — usei essa transição real em vez de
  cortar mais ~90 páginas do PDF só para alcançar a ocorrência literal da string "Engenharia".
- Registros colados com inscrição perdida: **busquei essa corrupção nos 6 PDFs reais e não a
  encontrei** — nem no exemplo específico do NOTES.md (`"Luisa Silva Tomasello"`, que hoje
  extrai limpo com `extraction_mode='plain'`), nem em nenhuma outra ocorrência (medido: 0
  spans que falham a separar em 22 campos nos arquivos de seção única; os únicos "spans que
  falham" nos arquivos de duas seções são a seção de tipo D, fora de escopo — ticket 05).
  Duas hipóteses: o protótipo usou um método de extração diferente do fixado no ticket 01, ou
  a versão do `pypdf` mudou o comportamento desde então. De qualquer forma, o ponto cego que
  essa corrupção representa — "um registro que o parser nunca extraiu não deixa nada nele
  mesmo para conferir" — é exatamente o que a verificação de sequência 1..N existe para
  cobrir. Testei isso removendo, programaticamente, um registro real e completo da lista que
  `extrair_edital` já extraiu (simulando com precisão o efeito exato dessa perda: o registro
  simplesmente não existe), e conferindo que o buraco aparece na posição certa. A mesma
  técnica cobre a quebra de ordem alfabética (troca de dois registros adjacentes).

Considero essa substituição fiel ao espírito do ticket ("fixa como testes de regressão os
quatro casos de corrupção real... para que uma correção de parser não reintroduza um problema
já resolvido") mesmo não sendo literal: o que precisa ficar fixado é a *classe* de corrupção,
não a string exata, e cada substituição foi verificada contra dado real, não inventada.

### 3.3 Fixture nova, de páginas não contíguas, para o "buraco" ter uma base limpa

**Decisão:** criei `tests/fixtures/resultado_final_curso_completo.pdf` — página 1 (schema) +
página 186 de `Ed_38_2024` — e adicionei `fatiar_paginas()` em `fixtures.py` (função nova,
ao lado de `fatiar_fixture`, que só fatiava intervalos contíguos) para gerá-la.

**Porquê:** medindo a fixture de 6 páginas já existente contra a lógica de buraco antes de
escrever qualquer asserção, descobri que **todo curso nela é um recorte truncado** de um
curso maior (ADMINISTRAÇÃO tem só uma fração das suas ~150 vagas na fixture, o resto está em
páginas fora do corte) — então a checagem de sequência 1..N sempre acusa buraco ali, não por
corrupção, mas porque a fixture simplesmente não contém o curso inteiro. Um teste "sem buraco"
contra essa fixture teria sido falso-positivo garantido. A página 186 de `Ed_38` contém dois
cursos pequenos por inteiro (ARQUIVOLOGIA, 16 Alunos; CIÊNCIAS AMBIENTAIS, 9 Alunos, usado nos
testes) — confirmei isso extraindo e checando que a sequência de cada Sistema fecha 1..N sem
buraco algum antes de usar como base "limpa" dos testes.

### 3.4 Limitação conhecida: perder o registro de posição máxima é um ponto cego

**Decisão:** documentada explicitamente em `validacao.py` e fixada como teste próprio
(`test_registro_de_posicao_maxima_perdido_e_um_ponto_cego_conhecido`), em vez de escondida.

**Porquê:** `_buracos_por_sistema` infere N (o total esperado) como `max(posicoes)`, porque
não existe nenhuma fonte independente do total real de candidatos em cada Sistema — o próprio
Edital de Resultado Final não declara esse número em lugar nenhum. Se o registro perdido for
justamente o de classificação N (o último do Sistema, dentro do curso), `max` encolhe junto
com ele e a checagem não vê buraco nenhum. Isso apareceu na revisão de código (achado real do
sub-agente de Spec) e é uma limitação da *técnica*, não um bug corrigível dentro deste ticket:
sem um número de candidatos vindo de outro lugar (fora de escopo), não há como distinguir "N=8
porque há 8 candidatos" de "N=8 porque o 9º sumiu". Documentei em vez de fingir cobertura
completa — é consistente com o próprio `spec.md`: nenhuma camada de validação sozinha é
completa, por isso existem várias (formato numérico, sequência, ordem alfabética, e o
checksum do ticket 04 ainda por vir).

---

## 4. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê | Ticket |
|---|---|---|
| Detecção de classificação **duplicada** (só buraco/ausência) | `spec.md` menciona "ausente ou duplicado" na descrição geral da camada, mas o checklist do ticket 02 só pede o caso de buraco | não aberto explicitamente; considerar se aparecer duplicata real |
| Checksum do Argumento Final | Depende de médias/desvios por Edital | 04 |
| Fechar o ponto cego da posição máxima (secção 3.4) | Precisaria de uma fonte de N independente da própria extração, que não existe nos dados disponíveis | não aplicável — limitação da técnica, documentada |

---

## 5. Como foi verificado

- **23 testes automatizados** (`tests/test_pas_extraction.py`, 10 novos), rodando em ~5s.
- **Skip gracioso confirmado**: movendo as duas fixtures para fora do diretório, 15 dos 23
  testes pulam com mensagem clara, os outros 8 continuam passando.
- **CSV de ponta a ponta**: gerado a partir da fixture de 6 páginas, as 3 colunas novas
  aparecem corretas linha a linha (conferido manualmente contra os campos sinalizados).
- **Validado contra os 6 PDFs reais** de Resultado Final em `data/pdfs` (52.714 registros
  extraídos no total, incluindo os candidatos antes descartados): pipeline completo roda sem
  exceção em nenhum arquivo.
- **Revisão de código em duas frentes** (`/code-review`, Standards + Spec, dois sub-agentes em
  paralelo). Achados reais corrigidos: duplicação entre `fatiar_fixture`/`fatiar_paginas`
  (extraído helper comum `_escrever_paginas`); drift entre o regex e seu próprio docstring
  (`^` faltando); limitação do ponto cego de posição máxima (seção 3.4) documentada e fixada
  como teste; type hint faltando num helper de teste. Achados descartados conscientemente:
  nomes de coluna do CSV com ordem de palavra invertida (`buracos_classificacao` →
  `classificacao_buracos`) — julgamento de legibilidade, mantidos porque agrupam com as
  colunas `classificacao_sistema_*` já existentes.
- **Suíte inteira do projeto**: 75 passam, os mesmos 2 falhas pré-existentes e não
  relacionadas do relatório do ticket 01 (sklearn, caminho Windows hardcoded).

---

## 6. Glossário — termos novos deste ticket

(Ver também o glossário do relatório do ticket 01 para os termos de base: Edital, Família,
Schema declarado, Âncora, Cabeçalho intercalado, Argumento Final, Sistema de Concorrência,
Classificação, Proveniência, Costura.)

| Termo | Significado |
|---|---|
| **`ValidacaoRegistro`** | Tipo que carrega o resultado da validação estrutural de um `RegistroResultadoFinal`: campos numéricos mal formatados, buracos de classificação por Sistema, e se o nome quebra a ordem alfabética do curso. |
| **Reparo tolerante** | `_tentar_float` removendo espaço interno de um campo numérico antes do parse, para recuperar o valor mesmo quando o texto bruto está partido (`"1 7.539"` → `17.539`). Distinto da *validação de formato*, que compara o texto original (não reparado) contra o formato exato — as duas coisas coexistem por design: o registro sobrevive com um valor utilizável, e o campo fica sinalizado. |
| **Buraco na sequência** | Posição ausente na sequência `1..N` de classificação de um Sistema de Concorrência, dentro de um curso. É o ponto cego de todas as outras camadas: um registro que o parser nunca extraiu não deixa nada nele mesmo para apontar o problema — só o buraco que deixa nos outros. |
| **Ponto cego de posição máxima** | Limitação da checagem de buraco: como N é inferido como o maior valor observado (não há fonte independente do total real), perder justo o registro de posição N não deixa buraco nenhum visível. Documentada, não corrigida — nenhuma camada de validação sozinha é completa. |
| **Curso completo (fixture)** | Curso cujos registros estão todos contidos numa única página da fixture, ao contrário de um curso grande (como Administração) que se estende por muito mais páginas do que cabe numa fixture pequena. Necessário para que "sem buraco" seja uma afirmação real nos testes, não um artefato do corte da fixture. |
