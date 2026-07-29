# Relatório — Ticket 02: Extrator de Editais de Etapa vira módulo com teste

**Ticket:** `.scratch/publicar-site/issues/02-extrator-de-editais-de-etapa-vira-modulo.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/etapa.py` (módulo novo, autocontido, no mesmo
pacote gitignored de `src/pas_extraction/`)
**Onde vive o teste:** `tests/test_pas_extraction_etapa.py` (versionado normalmente, fixture
100% sintética via `fixtures.gerar_pdf_texto_sintetico`)

---

## 1. O que foi pedido

Promover `.scratch/publicar-site/medicao-passo-1/extrair_etapa.py` — script descartável que
respondeu a pergunta de medição do Passo 1 — a módulo de verdade em `src/pas_extraction/`, com
teste, porque a extração roda de novo a cada triênio novo (não é operação de uma vez).

Critérios de aceite (todos atendidos):

- [x] Módulo em `src/pas_extraction/` lê um Edital isolado de Etapa e devolve os registros
      validados pelo checksum embutido, com diagnóstico dos descartes por motivo
- [x] Edital parcial é recusado com mensagem nomeando a contagem encontrada; o caso real do
      Edital 8/2023 (827 registros) é a referência do teste
- [x] Números com espaço interno são normalizados, e o checksum prova isso numa fixture
      sintética
- [x] Saída é um `medias_desvios_etapa.csv` com `(ano, etapa)`, `n`, e média/desvio de Parte
      2, Redação e Parte 1 misturada
- [x] Teste sobre fixture sintética — nenhuma linha de Aluno real entra em teste ou fixture
      rastreada
- [x] Rodando sobre os Editais reais de `data/pdfs`, reproduz os números do Passo 1
- [x] `pytest tests/` continua verde (329 → 348 testes, todos passando)

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/etapa.py        — módulo novo, autocontido:
  RegistroEtapa                    — dataclass (inscrição, eb_p1, eb_p2, tipo_d, red)
  DiagnosticoDescartes             — dataclass (blocos, campos_invalidos, nao_numerico, checksum_falhou)
  ResultadoExtracaoEtapa           — dataclass (registros, diagnóstico, ano, etapa, triênio, arquivo)
  LinhaMediasDesviosEtapa          — dataclass de saída, uma linha do CSV agregado
  EditalParcialError               — exceção própria
  normaliza_numero(campo) -> float | None
  parse_registros(texto) -> (List[RegistroEtapa], DiagnosticoDescartes)   — pura, sem PDF
  extrair_edital_etapa(caminho, etapa, minimo_registros=5000) -> ResultadoExtracaoEtapa
  calcular_medias_desvios(resultado) -> LinhaMediasDesviosEtapa
  escrever_csv_medias_desvios_etapa(linhas, destino) -> int
  processar_editais_etapa(mapeamento, destino_csv) -> List[LinhaMediasDesviosEtapa]

src/pas_extraction/fixtures.py      — + gerar_pdf_texto_sintetico(paginas, destino) (reportlab)
tests/test_pas_extraction_etapa.py  — 19 testes, 100% fixture sintética (nenhum PDF real)
```

Fluxo:

```
PDF do Edital isolado de Etapa
   │
   ├─ extrair_edital_etapa(caminho, etapa)
   │     ├─ lê a página 1, acha "TRIÊNIO aaaa/aaaa"  → triênio + início
   │     ├─ ano = início + etapa - 1                  (etapa vem de quem chama, ver §3.2)
   │     ├─ concatena as páginas (_texto_do_edital), cortando ruído de cabeçalho na pág. 1
   │     ├─ parse_registros(texto)  → registros validados + diagnóstico dos descartes
   │     └─ len(registros) < minimo_registros?  → EditalParcialError
   │
   └─ ResultadoExtracaoEtapa
         │
         ├─ calcular_medias_desvios()          → LinhaMediasDesviosEtapa (n, m/dp de P2, Red, P1 misturada)
         └─ escrever_csv_medias_desvios_etapa() → medias_desvios_etapa.csv
```

---

## 3. Decisões tomadas e o porquê

### 3.1 Bug real encontrado e corrigido: o cabeçalho da página 1 comia o 1º candidato

**Achado.** O script original (`extrair_etapa.py`) concatena todas as páginas, inclusive a
página 1 inteira (cabeçalho institucional + declaração de schema "na seguinte ordem: ..."), e
divide o blob inteiro por `"/"` para achar os blocos de candidato. O cabeçalho contém `"/"`
duas vezes por conta própria — `"TRIÊNIO 2023/2025"` e `"PAS/UnB"` — então o primeiro candidato
real da lista (o primeiro em ordem alfabética) fica colado ao texto do cabeçalho no mesmo
"bloco" e é descartado por não bater `_INSCRICAO_RE` (o campo antes da vírgula não é só
dígitos).

**Impacto medido:** rodando o módulo corrigido contra os Editais reais de `data/pdfs`, cada
`n` sai exatamente **+1** acima do que o script original produzia (ex.: Ed_8/2024 sai com
19.128 em vez de 19.127; Ed_15/2024 sai com 16.340 em vez de 16.339, o número citado no mapa).
Invisível em qualquer estatística agregada sobre ~19 mil registros, mas é uma perda sistemática
e silenciosa — exatamente o tipo de coisa que promover o script a módulo com teste deveria
consertar.

**Correção:** `_texto_do_edital` agora corta o texto da página 1 a partir do primeiro
`\d{6,10}\s*,` encontrado (`_PRIMEIRO_REGISTRO_RE`) antes de entrar no blob que
`parse_registros` divide por `"/"`. Páginas seguintes não têm esse problema (só carregam
candidatos, sem cabeçalho institucional).

### 3.2 `etapa` é parâmetro do chamador, não detectado do conteúdo

**Decisão:** `extrair_edital_etapa(caminho, etapa, ...)` exige `etapa` (1 ou 2) como
argumento; não tenta adivinhar "primeira etapa" vs. "segunda etapa" a partir do texto.

**Porquê:** conferi as 6 variantes de Edital isolado em `data/pdfs` diretamente com `pypdf`. Um
Edital **original** declara "referentes à primeira/segunda etapa" explicitamente na página 1 —
dá para detectar. Um Edital de **retificação**, não: ele diz apenas "retificação... divulgados
por meio do Edital nº X", sem repetir a etapa em lugar nenhum da página 1 (conferido nos dois
casos reais que existem: retificação de 2022 e de 2023). Como uma retificação é justamente um
dos dois documentos que o módulo precisa aceitar (a de 2022 é a fonte correta daquele ano — o
Edital original não trazia os escores brutos), detectar a etapa por regex funcionaria só às
vezes, de um jeito que falharia em silêncio exatamente no documento mais armadilhado do ticket.
Pedir `etapa` explicitamente é honesto sobre essa limitação, em vez de inventar uma heurística
frágil. O **ano**, em contraste, é sempre derivado do triênio impresso (nunca hardcoded):
`ano = início_do_triênio + etapa - 1` — conferido contra as 5 combinações reais de
`data/pdfs` (ver §5).

### 3.3 `MINIMO_REGISTROS = 5000`

**Decisão:** limiar fixo de 5.000 registros abaixo do qual `EditalParcialError` é levantado.

**Porquê:** os únicos dois pontos de dado que existem hoje são o menor Edital completo real
(16.990 registros) e o único documento parcial conhecido (827, a retificação de 2023). 5.000
fica no meio dos dois com folga grande dos dois lados — não exige saber a contagem oficial de
candidatos (que o Edital não publica em lugar nenhum) nem arriscar um limiar tão próximo de um
dos dois números que uma variação normal de matrícula entre triênios o cruzasse.

### 3.4 Fixture sintética via PDF gerado (não fatiado de um real)

**Decisão:** adicionei `fixtures.gerar_pdf_texto_sintetico`, que desenha texto arbitrário num
PDF novo via `reportlab`, em vez de reusar `fatiar_fixture`/`fatiar_paginas` (que copiam
páginas de um Edital real).

**Porquê:** critério de aceite explícito do ticket — nenhuma linha de Aluno real em teste ou
fixture rastreada. Um Edital isolado de Etapa lista nota por candidato (inscrição, notas), não
uma tabela agregada como `medias_desvios.py`; fatiar uma página real embutiria dado de Aluno de
verdade no repositório (mesmo em `tests/fixtures/`, que é gitignored, o *padrão* de teste
estabelecido para as outras Famílias hardcoda valores individuais extraídos direto no `.py`
rastreado — ver `ETAPA_1_AVULSO` em `test_pas_extraction_medias_desvios.py` — o que aqui
seriam notas de um Aluno real, não estatística agregada pública). A fixture sintética evita o
problema na raiz: os "candidatos" nos testes são gerados por um laço (`Fulano de Tal`, notas
sintéticas), nunca lidos de um PDF real.

### 3.5 `parse_registros` separado de `extrair_edital_etapa`

**Decisão:** a lógica de parsing de texto (`parse_registros`) é uma função pura que recebe uma
string e não sabe de PDF; `extrair_edital_etapa` é a única que abre arquivo.

**Porquê:** permite testar checksum, número partido por espaço e campo malformado com string
Python simples, sem precisar gerar PDF nenhum para a maioria dos testes (só os testes de
integração ponta-a-ponta — derivação de ano/etapa, rejeição de parcial — precisam do PDF
sintético). Mantém os testes rápidos (a suíte inteira do módulo roda em 0,17s) e a fixture
sintética pequena (poucas páginas), já que o limiar de "documento parcial" é testado contra o
valor de produção (5.000) sem precisar gerar 5.000 candidatos de verdade.

---

## 4. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê |
|---|---|
| Integração com `OFFICIAL_STATS` / `ExamStats` (Parte 1 misturada explicitamente derivada) | Ticket 01 do mapa `publicar-site`, que decide a forma do `ExamStats` |
| Calibração do Deslocamento (+7,87 pontos medido no Passo 1) | Ticket 06 do mapa `publicar-site` |
| Detecção automática de etapa a partir do conteúdo | Decisão consciente — ver §3.2, não é confiável para Editais de retificação |
| CLI (`python -m pas_extraction etapa ...`) | Não pedido pelo checklist; `processar_editais_etapa` já cobre "apontar para uma pasta de PDFs produz o CSV" como função Python |

---

## 5. Como foi verificado

- **19 testes automatizados** (`tests/test_pas_extraction_etapa.py`), 100% fixture sintética,
  rodando em 0,17s: normalização dos 3 exemplos do ticket (`"2. 046"`, `"1 6.005"`,
  `"0 .220"`), checksum que fecha/não fecha, campo malformado, campo não-numérico, ruído sem
  inscrição, derivação de ano/etapa/triênio a partir do conteúdo (Etapa 1 e Etapa 2), etapa
  inválida, documento parcial recusado (com a mensagem citando a contagem — mesmo mecanismo
  que teria recusado o Edital 8/2023 real), documento completo aceito, cálculo de média/desvio,
  escrita do CSV, e a costura `processar_editais_etapa` de ponta a ponta.
- **Validado contra os 6 Editais isolados reais de `data/pdfs`** (as duas retificações de
  2022/2023, o Edital 7 original de 2023, o Edital 8 de 2024, e os dois de Etapa 2 de
  2024/2025):
  - as 5 combinações completas são aceitas, zero falhas de checksum (`checksum_falhou=0` em
    todas), com `campos_invalidos` de 1–2 por documento (ruído residual de extração, não
    checksum);
  - a retificação de 2023 é **recusada** com `EditalParcialError`, contagem 828 (827 do script
    original + 1, ver §3.1);
  - os números agregados batem com a tabela do Passo 1 registrada em
    `.scratch/publicar-site/map.md` (Passo 2, item 1) dentro do arredondamento: `(2024, Etapa
    1)` → `m_p2=23,906 / dp_p2=11,398 / m_red=6,471 / dp_red=2,292`; `(2025, Etapa 2)` →
    `m_p2=27,643 / dp_p2=14,752 / m_red=6,316 / dp_red=2,251` — idênticos aos do mapa.
- **Suíte inteira do projeto** (`pytest tests/`): 348 passam (329 pré-existentes + 19 novos), 2
  pulados (pré-existentes, não relacionados a este ticket).

---

## 6. Glossário — termos necessários para entender este relatório

| Termo | Significado |
|---|---|
| **Edital isolado de Etapa** | O "Resultado final nos itens do tipo D e na prova de redação" de uma Etapa 1 ou 2, publicado no ano da prova — lista nota por candidato (inscrição, EB parte 1, EB parte 2, somatório, tipo D, redação), mas não a língua estrangeira. |
| **Checksum embutido** | `EB parte 1 + EB parte 2 = somatório`, os três impressos no próprio documento — um registro cuja extração saiu corrompida não fecha essa conta e é descartado, sem precisar de conferência humana. |
| **Documento parcial** | Um Edital de retificação (ou outro) que corrige só um trecho da lista de candidatos, não o resultado completo da Etapa — "retificação" no nome não diz, por si, se é parcial ou completo (ex.: retificação de 2023, 827 registros, parcial; retificação de 2022, 18.382, completa — foi o original de 2022 que faltava os escores brutos). |
| **`RegistroEtapa`** | Um candidato: inscrição + as 4 notas que o Edital publica. |
| **`DiagnosticoDescartes`** | Quantos blocos candidatos foram encontrados e por que motivo cada um foi descartado (campos errados / não-numérico / checksum). |
| **Parte 1 misturada** | A média/desvio da Parte 1 sem separar por língua estrangeira — o Edital isolado de Etapa não diz a língua de cada candidato, só o valor agregado (ver ticket 01 do mapa, que decide como o `ExamStats` marca isso como derivado). |

---

## 7. Onde continuar

O ticket 07 do mapa `publicar-site` ("Preditor responde para a Turma viva") é quem consome o
`medias_desvios_etapa.csv` produzido aqui para preencher `(2024, Etapa 1)` e `(2025, Etapa 2)`
no `OFFICIAL_STATS`, depois de o ticket 01 decidir a forma do `ExamStats` (Parte 1 misturada
marcada como derivada) e o ticket 06 calibrar o Deslocamento medido no Passo 1.
