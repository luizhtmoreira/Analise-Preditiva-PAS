# Relatório — Ticket 03: Extração da tabela de médias e desvios

**Ticket:** `.scratch/pdf-extraction/issues/03-extracao-medias-e-desvios.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/medias_desvios.py` (módulo novo, autocontido, no
mesmo pacote gitignored de `src/pas_extraction/` — ver ticket 01, seção "Por que o código não
está no git")
**Onde vive o teste:** `tests/test_pas_extraction_medias_desvios.py` (versionado normalmente)

---

## 1. O que foi pedido

O dono do produto aponta o pipeline para a tabela oficial de média e desvio-padrão de cada
Etapa — o insumo que falta para normalizar notas com valores oficiais em vez de estimados
(`OFFICIAL_STATS` hoje é inferido do CSV de Alunos, não lido do Edital). A tabela aparece em
dois lugares diferentes conforme o triênio — na cauda de um Edital de Resultado Final, ou num
Edital avulso dedicado só a isso — e o parser precisa achar os dois. Um detalhe de forma que
importa: a Parte 1 é publicada separada por língua estrangeira (Inglesa/Francesa/Espanhola);
Parte II e Redação não têm essa separação, e agregá-la indevidamente (como o dado atual do
projeto faz) quebra o checksum do ticket 04.

Critérios de aceite (todos atendidos — ver seção 5):

- [x] A Família Médias e Desvios é reconhecida pelo classificador de schema declarado
- [x] A tabela é encontrada tanto na cauda de um Edital de Resultado Final quanto num Edital
      avulso
- [x] Média e desvio-padrão da Parte 1 são gravados separadamente por língua estrangeira
- [x] Sai um CSV próprio da Família, com colunas de proveniência
- [x] Existe fixture de médias/desvios localmente (gerada pelo utilitário do ticket 01, não
      commitada), e um teste que verifica os valores extraídos dela, pulando se a fixture não
      existir

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/medias_desvios.py   — módulo novo, autocontido:
  RegistroMediasDesvios                — dataclass próprio (etapa, prova, língua, média, desvio, proveniência)
  ResultadoExtracaoMediasDesvios       — dataclass próprio (equivalente a ResultadoExtracao)
  FamiliaInesperadaError               — exceção própria (não reusa FamiliaDesconhecidaError)
  parse_medias_desvios(reader, contexto) -> List[RegistroMediasDesvios]
  extrair_edital_medias_desvios(caminho_pdf) -> ResultadoExtracaoMediasDesvios
  escrever_csv_medias_desvios(resultados, destino) -> int

tests/test_pas_extraction_medias_desvios.py   — 8 testes de comportamento
tests/fixtures/medias_desvios_avulso.pdf      — fixture real, 1 página (ED_34, 2019/2021)
tests/fixtures/medias_desvios_cauda.pdf       — fixture real, 1 página (Ed_38_2024, pág. 242)
```

Fluxo, para o caso avulso (equivalente ao `extrair_edital()` do ticket 01, mas só para esta
Família):

```
PDF do Edital avulso
   │
   ├─ extrair_edital_medias_desvios()
   │     ├─ lê a página 1 em modo 'plain'
   │     ├─ schema.classificar_familia()  → confirma MEDIAS_DESVIOS (senão, FamiliaInesperadaError)
   │     ├─ schema.extrair_metadados()    → número do Edital + triênio
   │     └─ parse_medias_desvios()
   │           ├─ lê todas as páginas em modo 'plain', concatenadas com offset por página
   │           ├─ varre o texto inteiro por "PRIMEIRA/SEGUNDA/TERCEIRA ETAPA"
   │           └─ cada Etapa encontrada vira 5 RegistroMediasDesvios (Parte 1 x 3 línguas, Parte II, Redação)
   │
   └─ ResultadoExtracaoMediasDesvios (registros, edital, triênio, arquivo)
         │
         └─ escrever_csv_medias_desvios()  → um CSV, uma linha por (Etapa, prova, língua), com proveniência
```

Para o caso "tabela na cauda de um Resultado Final", o consumidor (ticket futuro de
integração, fora deste escopo) já tem o `ContextoEdital` vindo de `pipeline.extrair_edital()`
para aquele mesmo PDF — só precisa chamar `parse_medias_desvios(reader, contexto)` direto,
sem passar pela função de conveniência (que exige a Família MEDIAS_DESVIOS já na página 1, o
que não é o caso quando a tabela está na cauda de um Resultado Final).

Validado contra os **5 Editais avulsos reais** de `data/pdfs` (`ED_34`, `ED_43`, dois `Ed_32`
de triênios diferentes, `Ed_38_2017-2019`): todos extraem exatamente 15 registros (3 Etapas x
5 linhas). Validado também contra a cauda de **2 Editais de Resultado Final reais completos**
(não só a fixture): `Ed_38_2024` (242 páginas, tabela na página 242) e `Ed_27_2021_2023` (317
páginas, tabela na página 317, o Edital com duas seções do ticket 05) — os dois encontram a
tabela e extraem 15 registros cada, sem precisar saber de antemão em que página ela está.

---

## 3. Decisões tomadas e o porquê

### 3.1 Módulo autocontido, sem reusar helpers de `resultado_final.py`

**Decisão:** `medias_desvios.py` define suas próprias versões privadas de `_construir_blob` e
`_pagina_do_offset` — funções quase idênticas às de `resultado_final.py` — em vez de importar
de lá.

**Porquê:** o ticket pede explicitamente um módulo autocontido, e outro agente estava
trabalhando em paralelo no mesmo pacote (nas regras de isolamento: não editar
`resultado_final.py`, `models.py`, etc.). Importar dessas funções criaria um acoplamento não
pedido pelo ticket, e duplicar ~15 linhas de lógica simples (concatenar texto de página com
offset) é um custo baixo comparado ao risco de um conflito de merge ou de uma mudança
concorrente no outro arquivo quebrar este módulo sem eu perceber.

### 3.2 Ancorar em "PRIMEIRA/SEGUNDA/TERCEIRA ETAPA", não no cabeçalho numerado da seção

**Decisão:** `_ETAPA_RE` procura o texto `"PRIMEIRA ETAPA"` / `"SEGUNDA ETAPA"` /
`"TERCEIRA ETAPA"` seguido da tabela, e ignora completamente o que precede esse texto.

**Porquê:** medido diretamente nos PDFs reais (não só no exemplo do ticket) — o rótulo antes
de "PRIMEIRA ETAPA" muda de forma imprevisível conforme quantas seções numeradas o Edital tem
antes da tabela de médias:

| Origem | Rótulo da seção | Rótulo de cada Etapa |
|---|---|---|
| Edital avulso (ex.: `ED_34`) | `"A Universidade... torna públicos..."` (sem número) | `"1 PRIMEIRA ETAPA"`, `"2 SEGUNDA ETAPA"`, `"3 TERCEIRA ETAPA"` |
| Cauda de `Ed_38_2024` (1 seção antes) | `"2 Média e desvio padrão"` | `"2.1 PRIMEIRA ETAPA"`, `"2.2 SEGUNDA ETAPA"`, `"2.3 TERCEIRA ETAPA"` |
| Cauda de `Ed_27_2021_2023` (2 seções antes, é o Edital tipo D + redação do ticket 05) | `"3 Média e desvio padrão"` | `"3.1 PRIMEIRA ETAPA"`, `"3.2 SEGUNDA ETAPA"`, `"3.3 TERCEIRA ETAPA"` |

O ticket já avisava que o cabeçalho podia ser `"2 Média e desvio padrão"` **ou**
`"3 Média e desvio padrão"` — a inspeção confirmou que o padrão é "o número da seção pai",
não um valor fixo, e que ele também prefixa cada sub-rótulo de Etapa (`"2.1"`, `"3.1"`...).
Um regex que tentasse casar o número da seção precisaria saber de antemão quantas seções o
Edital tem antes da tabela — informação que não está disponível sem já ter feito o parse do
documento inteiro. Ancorar só em "PRIMEIRA/SEGUNDA/TERCEIRA ETAPA" (que é estável nos 7 PDFs
reais testados: 5 avulsos + as 2 caudas) evita essa dependência por completo.

### 3.3 `parse_medias_desvios` nunca chama o classificador de Família

**Decisão:** a função que efetivamente varre o PDF (`parse_medias_desvios`) não chama
`schema.classificar_familia` em nenhum momento — só a função de conveniência
`extrair_edital_medias_desvios` chama, e só sobre a página 1.

**Porquê:** a página 1 de um Edital de Resultado Final declara a Família **Resultado Final**
(a frase "na seguinte ordem: ..."), nunca Médias e Desvios — mesmo quando esse mesmo PDF tem
a tabela de médias na cauda. Se `parse_medias_desvios` exigisse a confirmação de Família antes
de varrer, ela nunca funcionaria no caso "cauda", que é metade do critério de aceite do
ticket. A separação reflete a intenção real: `extrair_edital_medias_desvios` é o equivalente
do `extrair_edital()` do ticket 01 só para o caso avulso (por isso confirma a Família, do
mesmo jeito que `pipeline.extrair_edital` faz para Resultado Final); `parse_medias_desvios` é
a função de baixo nível, reaproveitável para os dois casos, e cabe a quem a chama (o futuro
código de integração, fora deste ticket) decidir se o PDF em mãos é avulso ou se é a cauda de
outro Edital já classificado.

### 3.4 Dataclasses e exceção próprios, não reaproveitados de `models.py`

**Decisão:** `RegistroMediasDesvios`, `ResultadoExtracaoMediasDesvios` e
`FamiliaInesperadaError` são definidos dentro de `medias_desvios.py`, não em `models.py`.

**Porquê:** regra explícita de isolamento do ticket (não editar `models.py`, outro agente
trabalhando em paralelo no pacote). `FamiliaInesperadaError` também é semanticamente diferente
de `FamiliaDesconhecidaError` (já existente em `models.py`): a segunda significa "a página 1
não bateu com nenhuma Família conhecida"; a primeira significa "a Família *foi* determinada,
só que não é a que esta função espera" — situações diferentes que merecem mensagens de erro
diferentes (a de `FamiliaInesperadaError` já indica a solução: usar
`parse_medias_desvios(reader, contexto)` direto).

### 3.5 `prova` como campo explícito, além de `etapa` e `lingua_estrangeira`

**Decisão:** `RegistroMediasDesvios` tem um campo `prova` (`"parte_1"` / `"parte_2"` /
`"redacao"`), não pedido explicitamente pelo ticket, que enumera os campos como "etapa, língua
estrangeira (quando aplicável), média, desvio-padrão, mais a Proveniência".

**Porquê:** sem `prova`, um registro de Parte II e um de Redação da mesma Etapa seriam
indistinguíveis — os dois têm `lingua_estrangeira=None`, e nada mais no registro diz qual é
qual. `prova` é o campo que efetivamente identifica a linha da tabela; `lingua_estrangeira`
só desambigua dentro de `prova == "parte_1"`. Sem esse campo o CSV de saída teria dados
ambíguos, o que contradiz o propósito do próprio ticket (alimentar o checksum do ticket 04
com valores confiáveis).

### 3.6 `extraction_mode='plain'` também para a página 1 (não `'layout'`)

**Decisão:** `extrair_edital_medias_desvios` lê a página 1 em modo `plain`, diferente de
`pipeline.extrair_edital` (ticket 01), que usa `'layout'` para a página 1 do Resultado Final.

**Porquê:** o `'layout'` era necessário no ticket 01 porque a frase "na seguinte ordem: ..."
ficava mais confiável nesse modo (ver relatório do ticket 01, seção 3.6). A classificação de
Médias e Desvios não depende dessa frase — depende só de achar `"desvio-padrão"` em qualquer
lugar da página 1 (ver `schema.classificar_familia`), e a tabela em si (que
`parse_medias_desvios` lê) é fluxo de texto simples, não colunar. O próprio ticket já indicava
`extraction_mode='plain'` para esta Família inteira; testei os dois modos nos 5 PDFs avulsos
reais e `'plain'` produz o texto mais limpo (`'layout'` insere espaços extras em alguns
números, ex. `"15 .492"` em vez de `"15.492"`, o mesmo tipo de corrupção documentado no
protótipo para o corpo do Resultado Final).

### 3.7 Duas fixtures, não uma — avulso e cauda são cenários genuinamente diferentes

**Decisão:** gerei duas fixtures com `fatiar_fixture`, não uma:

- `medias_desvios_avulso.pdf`: página 1 (única página) de `ED_34_..._2019-2021...pdf`.
- `medias_desvios_cauda.pdf`: só a página 242 de
  `Ed_38_2024_..._Res_final_não_eliminados.pdf` — a página onde a tabela de médias começa e
  termina nesse Edital real.

**Porquê:** o critério de aceite do ticket é explícito sobre os dois lugares onde a tabela
aparece, e uma fixture só do caso avulso não prova que `parse_medias_desvios` funciona sem
saber a priori em que página a tabela está — o próprio ponto central da função (ver 3.2). A
fixture "cauda" é literalmente a última página de um Edital real de 242 páginas, fatiada com o
mesmo utilitário do ticket 01; não foi sintetizada.

**Limite consciente desta fixture:** como ela contém só a página 242 (não as 241 anteriores),
o classificador de Família, se rodado sobre a página 1 *desta fixture*, devolveria
`MEDIAS_DESVIOS` (porque a página 1 da fixture *é* a página 242 do PDF original, que já é a
tabela) — o que coincidentemente pareceria "avulso" do ponto de vista do classificador. Isso
não invalida o teste: o conteúdo da página é genuinamente a cauda de um Resultado Final real,
e o teste (`TestParseMediasDesviosCauda`) chama `parse_medias_desvios` diretamente com um
`ContextoEdital` montado à mão (como faria o código de integração futuro, que já tem esse
contexto de ter chamado `pipeline.extrair_edital` antes) — nunca passa pela função de
conveniência `extrair_edital_medias_desvios`, que é só para o caso avulso de verdade (ver
3.3). Documentei esse detalhe no comentário da fixture no arquivo de teste para quem for lê-lo
depois.

### 3.8 Nenhum bug real encontrado em código já existente

Diferente do ticket 01 (que tinha uma seção de bugs corrigidos), não encontrei nenhum defeito
em `schema.py`, `models.py` ou nos outros arquivos que só li como referência — o classificador
de Família e o extrator de metadados já funcionam corretamente para os 5 PDFs avulsos e para
a página 1 dos 2 PDFs de Resultado Final usados nos testes, sem qualquer ajuste necessário.

---

## 4. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê | Ticket |
|---|---|---|
| Integração no dispatch de `pipeline.extrair_edital()` | Combinado explicitamente com quem passou o ticket: "vou integrar isso... manualmente" | fora deste ticket |
| Checksum do Argumento Final usando estas médias/desvios | Depende desta extração existir primeiro | 04 |
| Correção do `OFFICIAL_STATS` em `pas_intelligence/pas_constants.py` | É outra etapa do pipeline (relatório de diff antes da substituição) | fora deste ticket, ver spec.md |
| Deduplicar/reconciliar quando o mesmo (edital, triênio, etapa) aparece em dois PDFs diferentes (avulso E cauda, se algum triênio publicar os dois) | Não observado nos 77 PDFs reais durante este ticket; se ocorrer, é decisão de quem for consumir os dois CSVs | fora de escopo |
| Extração dos PDFs de Médias e Desvios com nomes truncados ou variação de grafia adicional (ex. Editais futuros ainda não publicados) | Coberto pela mesma âncora estrutural ("PRIMEIRA/SEGUNDA/TERCEIRA ETAPA"), mas só testado contra os 7 PDFs reais disponíveis hoje (5 avulsos + 2 caudas completas) | validação contínua, fora de escopo |

---

## 5. Como foi verificado

- **8 testes automatizados** (`tests/test_pas_extraction_medias_desvios.py`), rodando em
  ~0,26s, cobrindo: contagem de 15 registros (3 Etapas x 5 linhas) tanto no avulso quanto na
  cauda, separação da Parte 1 por língua com valores realmente diferentes entre si (não só
  "não é `None`"), Parte II e Redação sem língua, valores numéricos exatos da 1ª Etapa
  conferidos contra o PDF real (avulso e cauda), proveniência por linha, e o erro claro que
  `extrair_edital_medias_desvios` levanta quando apontada para um PDF de outra Família (usando
  a fixture de Resultado Final do ticket 01, pulando se ela não existir).
- **Skip gracioso confirmado**: as duas fixtures deste ticket são geradas localmente e
  gitignored; rodei a suíte com elas ausentes (renomeando o diretório) e todos os testes que
  dependem delas pularam com a mensagem exata do comando `fatiar_fixture` para regerá-las.
- **Validado contra os 5 Editais avulsos reais completos** de `data/pdfs` (`ED_34`, `ED_43`,
  os dois `Ed_32` de triênios diferentes, `Ed_38_2017-2019`): todos os 5 extraem exatamente 15
  registros, sem erro.
- **Validado contra a cauda de 2 Editais de Resultado Final reais completos** (não só a
  fixture, o PDF inteiro): `Ed_38_2024` (242 páginas — a tabela é achada na página 242 sem
  informar a página de antemão) e `Ed_27_2021_2023` (317 páginas, o Edital de duas seções do
  ticket 05 — a tabela é achada na página 317). Os valores da 1ª Etapa de `Ed_38_2024`
  (`m_p2=20.406`, `dp_p2=13.533`, `m_red=5.849`) batem exatamente com os números que
  `scripts/NOTES.md`, seção 7, já tinha registrado manualmente como divergentes do
  `OFFICIAL_STATS` estimado atual (`20.709`/`13.581`/`5.888`).
- **CSV de ponta a ponta**: escrevi um CSV de fumaça combinando os 5 resultados avulsos
  (`escrever_csv_medias_desvios`), confirmei 75 linhas (5 Editais x 15 registros), cabeçalho e
  colunas de proveniência corretos, e removi o arquivo temporário depois (não fica no repo).
- **Suíte inteira do projeto** (`pytest tests/`) rodada depois da implementação: 53 passam
  (incluindo os 8 novos), os únicos 2 que falham (`test_guaranteed_scenario`,
  `test_pdf_gen`) já falhavam antes deste ticket por motivos não relacionados (mesmos dois do
  relatório do ticket 01: incompatibilidade de versão do `sklearn`, caminho absoluto do
  Windows hardcoded no teste) — confirmado por não terem sido tocados nesta sessão.

---

## 6. Glossário — termos necessários para entender este relatório

| Termo | Significado |
|---|---|
| **Média e Desvio-padrão (a tabela)** | A tabela oficial, publicada pelo Cebraspe, com a média e o desvio-padrão de cada prova (Parte 1 por língua, Parte II, Redação) em cada Etapa — o insumo oficial que falta hoje ao `OFFICIAL_STATS` estimado do projeto. |
| **Prova** | Neste módulo, um dos 3 valores `"parte_1"`, `"parte_2"`, `"redacao"` — identifica qual das 3 seções da tabela um `RegistroMediasDesvios` representa. Só existe porque `etapa` + `lingua_estrangeira` sozinhos não bastam para diferenciar Parte II de Redação (as duas têm `lingua_estrangeira=None`). |
| **Edital avulso** | Um Edital dedicado inteiramente à tabela de médias/desvios (ex.: `ED_34_..._Media_e_desvio_padrao.pdf`), publicado separadamente do Resultado Final naquele triênio. |
| **Cauda** | As últimas páginas de um Edital de Resultado Final, onde a mesma tabela de médias/desvios aparece nos triênios que não a publicaram como Edital avulso. |
| **`RegistroMediasDesvios`** | Um registro deste módulo: `etapa` (1/2/3), `prova`, `lingua_estrangeira` (só preenchida quando `prova == "parte_1"`), `media`, `desvio_padrao`, `proveniencia`. Tipo próprio deste módulo, não vive em `models.py` (ver seção 3.4). |
| **`ResultadoExtracaoMediasDesvios`** | O tipo de retorno de `extrair_edital_medias_desvios`: registros + metadados do Edital (edital, triênio, arquivo). Equivalente a `ResultadoExtracao` (ticket 01), mas próprio deste módulo. |
| **`FamiliaInesperadaError`** | Exceção própria deste módulo: levantada quando `extrair_edital_medias_desvios` é apontada para um Edital cuja página 1 declara uma Família diferente de Médias e Desvios. Distinta de `FamiliaDesconhecidaError` (models.py) — ver seção 3.4. |
| **`parse_medias_desvios(reader, contexto)`** | A função de baixo nível que varre qualquer `PdfReader` (avulso ou só a cauda de outro Edital) e devolve os `RegistroMediasDesvios` encontrados, sem checar Família — ver seção 3.3. |

---

## 7. Onde continuar

O ticket 04 (checksum do Argumento Final) consome tanto os registros do ticket 01
(`RegistroResultadoFinal`, as 9 notas + argumento impresso) quanto os deste ticket
(`RegistroMediasDesvios`, as médias/desvios oficiais por Etapa) para recalcular o Argumento
Final e comparar com o valor impresso. A integração combinada — decidir, para um PDF de
Resultado Final, se a tabela de médias vem da própria cauda dele ou de um Edital avulso
correspondente (`(subprograma, triênio)`) — foi deixada explicitamente para quem passou este
ticket integrar manualmente no dispatch de `pipeline.extrair_edital()`, junto com a Família
Convocação (ticket 09).
