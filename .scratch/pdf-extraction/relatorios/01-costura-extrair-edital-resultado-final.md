# Relatório — Ticket 01: Costura `extrair_edital` + classificador de família + CSV de Resultado Final

**Ticket:** `.scratch/pdf-extraction/issues/01-costura-extrair-edital-resultado-final.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/` (pacote novo, **não versionado** — ver seção
"Por que o código não está no git" abaixo)
**Onde vive o teste:** `tests/test_pas_extraction.py` (versionado normalmente)

---

## 1. O que foi pedido

O dono do produto aponta o pipeline para um Edital de Resultado Final e recebe um CSV com um
registro por Aluno, sem precisar informar de que formato o arquivo é, nem manter lista de
arquivos à mão. O ticket estabelece a *costura* — a fronteira de função única
`extrair_edital(caminho_pdf) -> ResultadoExtracao` — por onde toda a lógica futura do pipeline
(seções, checksum, língua, cotas) vai ser testada. Ele **não** pede validação estrutural
(ticket 02), checksum (ticket 04), parse de duas seções no mesmo PDF (ticket 05) ou dedução de
cotas (ticket 06) — isso é trabalho de tickets seguintes.

Critérios de aceite (todos atendidos — ver seção 6):

- [x] Um comando único extrai um Edital de Resultado Final e escreve um CSV, sem lista de
      arquivos hardcoded e sem caminho absoluto de máquina
- [x] O comando aceita um subconjunto de Editais
- [x] A Família de Edital é determinada pelo schema declarado, canonizado
- [x] Os 22 campos são extraídos, com campus/curso/turno vindos dos cabeçalhos intercalados
- [x] `-` numa classificação é preservado como "não concorreu", distinto de ausência de dado
- [x] Cada linha do CSV carrega arquivo de origem, edital, triênio e página
- [x] Existe o utilitário que fatia uma fixture de um Edital real
- [x] A fixture de Resultado Final existe localmente, gitignored, não commitada
- [x] Um teste exercita `extrair_edital` na fixture e pula com mensagem clara se ela não existir
- [x] Nenhum extrator antigo em `scripts/` foi modificado ou removido

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/
  __init__.py        — reexporta a API pública do pacote
  models.py           — tipos de domínio (dataclasses, enum, exceções)
  schema.py           — canonização + classificador de Família de Edital
  resultado_final.py  — parser da família Resultado Final
  pipeline.py          — a costura: extrair_edital()
  csv_writer.py        — escreve o CSV com proveniência
  fixtures.py          — fatia um trecho pequeno de um Edital real para virar fixture de teste
  cli.py               — comando único: `python -m pas_extraction.cli extract|fixture`

tests/test_pas_extraction.py  — 13 testes de comportamento (não de estrutura interna)
tests/fixtures/resultado_final_22_campos.pdf  — fixture real, 6 páginas, gerada localmente
```

Fluxo de ponta a ponta:

```
PDF do Edital
   │
   ├─ pipeline.extrair_edital()
   │     ├─ lê a página 1 em modo 'layout'
   │     ├─ schema.classificar_familia()  → decide RESULTADO_FINAL / CONVOCACAO / MEDIAS_DESVIOS
   │     ├─ schema.extrair_metadados()    → número do Edital + triênio, lidos do texto
   │     └─ resultado_final.parse_resultado_final()  (só se for RESULTADO_FINAL)
   │           ├─ lê todas as páginas em modo 'plain'
   │           ├─ ancora cada registro pelo próprio número de inscrição
   │           ├─ separa os 22 campos de cada registro
   │           └─ atualiza campus/curso/turno como estado a cada cabeçalho encontrado
   │
   └─ ResultadoExtracao (família, registros, edital, triênio, arquivo)
         │
         └─ csv_writer.escrever_csv()  → um CSV, uma linha por Aluno, com proveniência
```

Validado contra os **77 Editais reais** em `data/pdfs`: o classificador de família acerta
100% (64 convocação, 8 resultado final, 5 médias/desvios, zero desconhecidos/erros). Rodado
também de ponta a ponta num Edital de 242 páginas real (não a fixture): 7.870 registros limpos
extraídos num único comando.

---

## 3. Decisões tomadas e o porquê

### 3.1 Pacote novo e separado de `pas_intelligence`

**Decisão:** todo o código vive em `src/pas_extraction/`, um pacote que não importa nada de
`pas_intelligence` nem é importado por ele.

**Porquê:** são domínios diferentes por natureza — extração lê PDF e escreve CSV, é trabalho
*offline*, rodado manualmente pelo dono do produto; `pas_intelligence` é o motor de predição que
roda *dentro* do app Streamlit, a cada requisição. Misturar os dois acoplaria o ciclo de vida de
um ao do outro sem necessidade. Essa separação já estava decidida no `spec.md` antes deste
ticket começar.

### 3.2 Classificação de Família pelo schema declarado, não por nome de arquivo

**Decisão:** `schema.classificar_familia()` lê a frase que o próprio Edital declara na primeira
página (`"na seguinte ordem: ..."`), **canoniza** essa frase (remove acento, caixa e todo
caractere não-alfanumérico) e procura por âncoras estruturais dentro dela:

- contém `"sistema"` e `"subsistema"` → família **Convocação**
- contém `"numerodeinscricao"` (e não bateu a âncora acima) → família **Resultado Final**
- se a frase `"na seguinte ordem"` nem existe na página 1, procura `"desviopadrao"` no texto
  inteiro da página → família **Médias e Desvios**
- nada disso bate → `FamiliaDesconhecidaError`, erro alto e claro, nunca um "chute" silencioso

**Porquê essas âncoras e não a frase inteira:** o protótipo (`scripts/NOTES.md`, seção 2) já
tinha mostrado que comparar a frase inteira, mesmo canonizada, "descobre" 6 schemas onde
existem 3 — porque a redação institucional muda ao longo dos anos (`"nome do candidato"` virou
`"nome da pessoa candidata"` a partir de 2023/2025; `"nota final"` virou `"nota provisória"`
nos Editais tipo D + redação). Ancorar em termos que **nunca mudam** — `"número de inscrição"`
é estrutural, sempre vai estar lá independente de como o resto da frase for redigido — é o que
permite ao critério de aceite "um Edital novo com redação institucional diferente é classificado
corretamente sem mudança de código" ser verdade de fato, e não só na amostra que eu testei.
Isso foi validado rodando o classificador contra as primeiras páginas dos 77 PDFs reais, não só
contra exemplos sintéticos no teste.

**Detalhe que quase não funcionou:** a regex que captura a frase declarada tinha um limite de
tamanho (`{20,900}` caracteres) copiado do protótipo. A frase real do Resultado Final tem quase
2000 caracteres (lista todos os 22 campos, incluindo cláusulas longas tipo "se houver" para cada
uma das 10 classificações) — com o limite de 900, a regra nunca terminava de capturar e o
classificador falhava com "não foi possível determinar a família" mesmo num Edital óbvio. Corrigi
para `{20,8000}`.

### 3.3 Parser ancorado pela inscrição, não pelo separador `" / "`

**Decisão:** `resultado_final.py` não separa registros pelo `" / "` que aparentemente delimita
cada Aluno no fluxo de texto. Em vez disso, ele varre o texto inteiro procurando **âncoras**: a
sequência de 8 dígitos seguida de vírgula que é sempre o número de inscrição no início de um
registro. Cada registro é o texto entre uma âncora e a próxima.

**Porquê:** medido diretamente no Edital real (`Ed_38_2024`) durante este ticket — o separador
`" / "` **some** exatamente na fronteira entre dois cursos. O último registro de um curso termina
com `"."` (ponto), não com `" / "`, e o cabeçalho do próximo curso vem colado logo depois:

```
..., -, -, -, -, -, -, -, -, -. AGRONOMIA (BACHARELADO)
[inscrição], [nome], ...
```

Um parser que confiasse no `" / "` perderia essa fronteira silenciosamente (é exatamente o tipo
de corrupção "número plausível e errado" que o `spec.md` descreve como o risco real do projeto).
Ancorar pela inscrição resolve isso de graça: qualquer texto entre duas âncoras — inclusive
cabeçalho de curso colado — vira o "ruído" de um registro, tratado à parte (seção 3.4).

### 3.4 Campus/curso/turno como estado, cabeçalho pode estar em qualquer lugar do "ruído"

**Decisão:** depois de separar um registro em seus 22 campos, o texto que sobra até a próxima
âncora (o "ruído") é vasculhado por dois padrões: um regex de cabeçalho de campus
(`CAMPUS ... – DIURNO/NOTURNO/...`) e um regex genérico de trecho em CAIXA ALTA para o nome do
curso. Quando mais de um trecho em caixa alta aparece no mesmo ruído (o que acontece, porque
títulos institucionais como "PROGRAMA DE AVALIAÇÃO SERIADA (PAS)" também são caixa alta), fica
com o **último** candidato — o mais próximo do próximo registro no fluxo do texto.

**Porquê "o último":** foi tentativa e erro guiado por um bug real. A primeira versão pegava
qualquer trecho maiúsculo do início do documento inteiro (todo o texto antes do primeiro
registro) como se fosse um único candidato a curso — resultado: `curso` ficava sempre `None`,
porque o trecho continha majoritariamente a declaração de schema em minúsculas. A correção foi
tratar cada candidato separadamente e escolher o mais próximo do registro seguinte.

**Bug relacionado corrigido durante o code-review:** quando um registro **falha** ao separar em
22 campos (por exemplo, um número partido por espaço — ver 3.5), o texto inteiro daquele span
começa com dígitos de inscrição, não com um cabeçalho de verdade. Antes da correção, esse span
inteiro era vasculhado por cabeçalho mesmo assim, e uma vez um fragmento tipo `"22113758 , G"`
(início do próximo registro, com a inicial do nome do Aluno) foi capturado como se fosse nome de
curso. A correção: span que falhou a separar em 22 campos nunca é vasculhado por cabeçalho —
o risco de confundir nome de Aluno com cabeçalho é maior que o risco de perder um cabeçalho
genuíno nesse caso raro.

### 3.5 Registros corrompidos são descartados nesta camada, sem alarde

**Decisão:** se um dos 9 campos de nota, o argumento final, ou uma classificação não bater no
formato esperado (nota que não é um número puro depois de `strip()`, classificação que não é
`-` nem inteiro, nome com dígito), o registro inteiro é descartado — não aparece no CSV, não
levanta exceção, não é sinalizado.

**Porquê é aceitável aqui e não é "descartar sem explicar" (proibido pelo `spec.md`):** o
`spec.md` proíbe descartar um registro **pelo checksum** sem que o padrão da falha esteja
explicado — isso é sobre a camada de validação (ticket 02), que ainda não existe. Aqui, a
questão é mais simples: um registro cujo *formato* não fecha (18 campos em vez de 22, um número
com espaço no meio) não é um "Aluno com dado suspeito", é lixo de extração que nem chegou a virar
um registro válido — não há o que reportar ainda, porque a camada que reporta padrões de falha é
o próprio ticket 02. Confirmado na prática: dos 119 candidatos a registro nas 6 páginas da
fixture, 10 foram descartados, e **todos os 10** são exatamente a corrupção "número partido por
espaço" já catalogada no protótipo (ex.: `"1 7.539"` em vez de `"17.539"`) — não um bug do meu
parser.

### 3.6 `extraction_mode='plain'` para o corpo, `'layout'` para a página 1

**Decisão:** o corpo do documento (onde estão os registros) é lido em modo `plain`; a página 1
(onde está a frase de schema declarado) é lida em modo `layout`.

**Porquê:** medido no protótipo (`NOTES.md`, seção 6) — `layout` produz **mais** números
partidos que `plain` no corpo (74 contra 68 hits na amostra), porque ele injeta espaços extras
para preservar alinhamento visual de coluna, o que é exatamente o tipo de corrupção que se quer
evitar. Só que a frase de schema na página 1 tende a ficar mais legível em `layout` (é assim que
o protótipo a capturava de forma confiável). São objetivos diferentes na mesma página, por isso
os dois modos coexistem.

### 3.7 Fixture de 6 páginas (não 3–5) — de propósito

**Decisão:** a fixture commitada localmente (`tests/fixtures/resultado_final_22_campos.pdf`) tem
6 páginas do Edital 38 (2022/2024), fatiadas das páginas 1 a 6, um pouco acima do "3 a 5" sugerido
no ticket.

**Porquê:** a primeira versão usava 4 páginas e todos os 109 registros caíam no mesmo curso
(Administração) — o que significa que o critério de aceite "campus/curso/turno vêm dos
cabeçalhos intercalados" nunca era realmente testado, só o campo "não é `None`" era checado. O
revisor de spec (ver seção 5) pegou exatamente essa lacuna. A página 6 é onde acontece a
primeira troca real de curso (Administração → Agronomia) neste Edital — por isso o corte foi
para 6 páginas, o menor intervalo contínuo que inclui uma transição de verdade. Prefiro um pouco
mais de páginas a um teste que finge cobrir o critério de aceite sem cobrir de fato.

### 3.8 Por que o código não está no git

**Decisão:** `src/pas_extraction/` está listado em `.gitignore` (adicionado num commit anterior
a este ticket, `32d1706`) e permanece assim. Só `tests/test_pas_extraction.py` foi commitado.

**Porquê:** política do projeto — parsers de dado (a lógica que sabe como ler os Editais) não
podem ficar públicos no repositório, mesma regra que já valia para `scripts/extrator_master.py`
e companhia. `tests/fixtures/` (as fatias de PDF real, que contêm nome/nota/inscrição de Alunos
de verdade) também é gitignored, pela mesma razão — dado real de Aluno nunca sobe pro repo. O
teste em si pode ficar público porque só afirma comportamento (contagens, formatos), não expõe
lógica de parsing nem dado de Aluno.

**Consequência prática:** rodar `git status` depois deste trabalho mostra só o arquivo de teste
como novidade — isso é esperado, não um erro.

### 3.9 Ajustes feitos depois da revisão de código (Standards + Spec)

Depois da primeira versão funcionar, rodei uma revisão em duas frentes (padrões de código vs.
fidelidade ao ticket). Quatro achados reais, todos corrigidos:

1. **Cobertura de teste da troca de curso** — já descrito na seção 3.7.
2. **`ContextoEdital`** — `arquivo_origem`, `edital` e `trienio` eram passados como três
   parâmetros soltos por duas funções (`parse_resultado_final` e `_montar_registro`), quando na
   prática sempre viajam juntos (são constantes para o Edital inteiro; só a página varia por
   registro). Virou um `NamedTuple` único, reduzindo o acoplamento entre as duas funções.
3. **`classificacoes` virou dicionário** — antes era uma lista posicional
   (`List[Optional[int]]`) em que o índice `i` significava "Sistema `i+1`", uma conta de cabeça
   que o código escondia. Agora é `Dict[int, Optional[int]]`, chave = número do Sistema (1 a
   10) igual aparece em `constants.MAPA_SISTEMAS` — lê-se sem fazer aritmética.
4. **`MAPA_SISTEMAS` estava morto** — a constante existia (documentada como o elo entre as
   famílias Resultado Final e Convocação) mas não era importada em lugar nenhum; os nomes das
   colunas do CSV eram gerados com `range(1, 11)` hardcoded. Agora `csv_writer.py` usa
   `MAPA_SISTEMAS` de verdade para gerar as colunas, então a constante tem um consumidor real.

---

## 4. Escopo deliberadamente fora deste ticket

Para não confundir "não foi feito" com "foi esquecido":

| Não implementado aqui | Por quê | Ticket |
|---|---|---|
| Validação estrutural por registro (formato `^-?\d+\.\d{3}$` exato, sequência 1..N de classificação, ordem alfabética) | O ticket 01 só estabelece a costura; a camada de validação que **reporta** padrões de falha em vez de descartar é outro ticket | 02 |
| Checksum do Argumento Final | Precisa das médias/desvios de cada Edital, que nenhum parser lê ainda | 03, 04 |
| Inferência da língua estrangeira por Etapa | Depende do checksum | 04 |
| Parse dirigido por seção (Editais com duas listas, tipo D + redação) | O parser genérico já tende a ignorar a seção de 4 campos por não fechar no formato de 22, mas a detecção explícita por cabeçalho numerado é outro ticket | 05 |
| Dedução de Cota Declarada a partir do padrão de classificações | Não implementado; `classificacoes` só guarda os números crus | 06 |
| Parser da família Convocação e Médias e Desvios | `extrair_edital` reconhece as duas famílias mas levanta `FamiliaAindaNaoImplementadaError` de propósito se apontado para uma delas | 03, 09 |

---

## 5. Como foi verificado

- **13 testes automatizados** (`tests/test_pas_extraction.py`), rodando em ~2,5s, cobrindo:
  canonização, classificação das 3 famílias (inclusive com redação institucional diferente),
  família desconhecida levanta erro, contagem de registros da fixture, campus/curso/turno como
  estado, **troca real de curso no meio do fluxo**, `-` preservado como "não concorreu",
  proveniência por linha, formato de inscrição/nome.
- **Skip gracioso confirmado de propósito**: renomeei a fixture temporariamente e rodei a suíte —
  os 5 testes que dependem dela pularam com mensagem clara (o comando exato pra regerar), os
  outros 7 continuaram passando.
- **Classificador validado contra os 77 PDFs reais** de `data/pdfs` (não só a fixture): 64
  convocação, 8 resultado final, 5 médias/desvios, **zero** desconhecidos ou erros.
- **Pipeline completo validado num Edital real de 242 páginas** (fora da fixture): 7.870
  registros extraídos num único comando de CLI, com famílias não suportadas puladas com
  mensagem clara em vez de travar a execução inteira.
- **Suíte inteira do projeto** (`pytest tests/`) rodada antes e depois: os únicos 2 testes que
  falham (`test_guaranteed_scenario`, `test_pdf_gen`) já falhavam antes deste ticket por motivos
  não relacionados (incompatibilidade de versão do `sklearn`; caminho absoluto do Windows
  hardcoded no teste) — confirmado por não terem sido tocados nesta sessão.
- **Revisão de código em duas frentes** (padrões de repositório + fidelidade ao ticket), rodada
  como dois sub-agentes em paralelo depois da implementação — achados na seção 3.9.

---

## 6. Glossário — termos necessários para entender este relatório

| Termo | Significado |
|---|---|
| **Edital** | Um PDF publicado pelo Cebraspe. Identificado por número, triênio e data. |
| **Família de Edital** | Um dos três formatos possíveis de Edital: **Resultado Final**, **Convocação**, **Médias e Desvios**. Determinada pelo *schema declarado* (ver abaixo), não pelo nome do arquivo. |
| **Schema declarado** | A frase `"na seguinte ordem: ..."` que o próprio Edital escreve na primeira página, listando os campos de cada registro na ordem exata em que aparecem. É o "gabarito" que o classificador de família lê. |
| **Canonização** | Normalizar um texto removendo acento, diferença de maiúscula/minúscula e todo caractere que não é letra ou número, antes de comparar. Existe porque a mesma frase, extraída de anos diferentes do PDF, aparece com ruído de extração (`Campus` vs `campus`, `"abaix o"` em vez de `"abaixo"`) que não é diferença de conteúdo. |
| **Âncora de registro** | O ponto no texto usado para saber onde um registro de Aluno começa: os 8 dígitos do número de inscrição seguidos de vírgula. Usado em vez do separador `" / "` porque este último desaparece na fronteira entre cursos. |
| **Cabeçalho intercalado** | Texto de campus, curso ou turno que aparece *no meio* do fluxo de registros (não numa tabela separada), e precisa ser lido como **estado**: uma vez encontrado, vale para todos os registros seguintes até o próximo cabeçalho aparecer. |
| **Escore Bruto (EB)** | Nota bruta de uma prova = Parte 1 + Parte 2, numa Etapa. |
| **Etapa** | Uma das 3 fases do PAS (Etapa 1, 2, 3), cursadas ao longo do triênio (ensino médio). |
| **Argumento Final** | Nota final ponderada, calculada a partir das 9 notas brutas + médias/desvios oficiais, usada pra ranquear o Aluno na UnB. Fórmula existe em `pas_intelligence/argument_calculator.py`; este ticket só extrai o valor impresso, não recalcula (isso é o *checksum*, ticket 04). |
| **Sistema de Concorrência** | Um dos 10 sistemas em que um Aluno pode ser classificado/ranqueado: Universal, Cota para Negros, e 8 subsistemas de Escola Pública (combinações de renda, autodeclaração PPI e PcD). Numerados de 1 a 10, nomes em `constants.MAPA_SISTEMAS`. |
| **Classificação** | A posição do Aluno no ranking de um Sistema de Concorrência, dentro do seu curso. `-` significa "não concorreu naquele sistema" (não é ausência de dado, é uma resposta válida). |
| **Cota Declarada** | (Ainda não implementado — ticket 06.) O(s) Sistema(s) de Concorrência que o Aluno escolheu na inscrição, deduzido do padrão de quais classificações estão preenchidas. |
| **Checksum do Argumento Final** | (Ainda não implementado — ticket 04.) Recalcular o Argumento Final a partir das 9 notas brutas e comparar com o valor impresso no Edital, como forma automática de detectar corrupção de extração. |
| **Proveniência** | Os 4 dados que acompanham cada linha do CSV para permitir auditoria: arquivo de origem, número do Edital, triênio, página onde o registro foi encontrado. |
| **Costura (seam)** | O ponto único de entrada — aqui, a função `extrair_edital(caminho_pdf) -> ResultadoExtracao` — por onde toda a lógica interna do pipeline é testada, em vez de testar a estrutura interna do parser (que muda muito durante o ajuste fino). |
| **`ResultadoExtracao`** | O tipo de retorno de `extrair_edital`: guarda a família identificada, a lista de registros extraídos, e os metadados do Edital (número, triênio, arquivo). |
| **`RegistroResultadoFinal`** | Um Aluno extraído: campus/curso/turno, inscrição, nome, as 9 notas, o argumento final, e as 10 classificações. |
| **`ContextoEdital`** | Os 3 dados que são constantes para um Edital inteiro (arquivo de origem, número do Edital, triênio) — agrupados num único tipo para não precisar passá-los como 3 parâmetros soltos por várias funções. |
| **Fixture** | Um PDF pequeno (aqui, 6 páginas), fatiado de um Edital real de verdade — não inventado — para os testes rodarem rápido sem precisar dos PDFs completos (que têm centenas de páginas e não sobem pro repositório). Fatiar em vez de inventar preserva a corrupção real de extração, que é o que os testes de validação (tickets futuros) precisam pegar. |
| **`extraction_mode` (`plain` / `layout`)** | Os dois modos de extração de texto de PDF da biblioteca `pypdf`. `plain` lê o texto como fluxo corrido; `layout` tenta preservar o alinhamento visual de colunas, inserindo espaços extras para isso — o que aqui prejudica mais do que ajuda no corpo do documento. |

---

## 7. Onde continuar

Próximo ticket sem bloqueio: **02 — Validações estruturais por registro** (formato numérico
exato, sequência `1..N` de classificação por curso/sistema, ordem alfabética), que consome
exatamente esta costura sem tocar na estrutura interna do parser.
