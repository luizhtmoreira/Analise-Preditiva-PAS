# Relatório — Ticket 05: Parse dirigido por seção

**Ticket:** `.scratch/pdf-extraction/issues/05-parse-dirigido-por-secao.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/resultado_final.py` (pacote gigignored — ver
seção "Por que o código não está no git" do relatório do ticket 01)
**Onde vive o teste:** `tests/test_pas_extraction.py`, classe `TestParseDirigidoPorSecao`

---

## 1. O que foi pedido

Editais de *resultado final tipo D + redação* (medido no Ed_27, 2021/2023, 317 páginas) trazem
duas listas com schemas diferentes no mesmo arquivo: páginas 0–98 têm registros de 4 campos
(candidatos eliminados), a partir da página 99 registros de 22 campos (candidatos não
eliminados). O parser existente (ticket 01) assume um schema por arquivo — sem intervenção,
ele varreria o documento inteiro por âncora de inscrição, sem saber que a metade de trás tem um
formato diferente da metade da frente.

A unidade de parse passa a ser a **seção**, não o arquivo. A transição precisa ser detectada
pelo cabeçalho numerado do próprio documento (`"2 DO RESULTADO FINAL DOS CANDIDATOS NÃO
ELIMINADOS"`), não por número de página fixo — para funcionar em qualquer Edital da família, não
só no Ed_27.

Critérios de aceite (todos atendidos — ver seção 4):

- [x] A transição entre seções é detectada pelo cabeçalho numerado, não por número de página
- [x] Apenas a seção de não eliminados é extraída dos Editais de duas seções
- [x] Um Edital de seção única continua sendo extraído normalmente, sem regressão
- [x] Existe fixture com a transição entre as duas seções, gerada localmente (não commitada)
- [x] Um teste verifica que nenhum registro da primeira seção aparece na saída

---

## 2. O que já vinha filtrando a seção 1 "de graça" — e por que isso não bastava

Antes deste ticket, um registro de 4 campos (`inscrição, nome, nota, nota`) nunca produzia um
`RegistroResultadoFinal` válido: `_separar_registro` exige exatamente 22 campos separados por
vírgula entre duas âncoras, e um span da seção 1 nunca tem isso. Então, na prática, os
registros da seção 1 já não apareciam na saída — **por acidente**, não por decisão.

O ticket recusa esse acidente como solução (é literalmente o cenário que o `spec.md` descreve
como o risco do projeto: "produzir lixo sem levantar erro"). O motivo concreto, não hipotético:
nada garante que um span de 4-campos-mais-ruído nunca vai ter, por coincidência de texto vizinho,
21 vírgulas antes da próxima âncora — aí ele passaria pelo `_separar_registro`, e o restante do
pipeline (que já espera 22 campos bem-formados) tentaria montar um registro a partir de texto
que é, na verdade, nota de eliminado + fragmento de ruído do meio da seção 1. Não observei esse
caso acontecer na prática (ver seção 4), mas o ticket pede detecção explícita por cabeçalho
justamente para não depender de "não aconteceu até agora".

---

## 3. Decisão tomada e o porquê

**Decisão:** `resultado_final.py` ganha uma regex que busca o cabeçalho de transição no blob já
concatenado (mesmo blob que `_ANCORA_RE` varre):

```python
_SECAO_NAO_ELIMINADOS_RE = re.compile(
    r"\d\s*DO\s+RESULTADO\s+FINAL\s+DOS\s+CANDIDATOS\s+N[ÃA]O\s+ELIMINADOS",
    re.IGNORECASE,
)
```

Se o cabeçalho é encontrado, `parse_resultado_final` só varre âncoras de inscrição **a partir do
fim do cabeçalho** (`_ANCORA_RE.finditer(blob, inicio_secao)`); o texto antes disso nunca chega a
virar candidato a registro, seja qual for o formato dele. Se o cabeçalho não existe (Edital de
seção única, o caso comum), `inicio_secao = 0` e o comportamento é idêntico ao de antes — sem
bandeira nova, sem parâmetro novo, sem branch visível no resto do pipeline.

**Por que a regex e não comparar a frase inteira canonizada:** `schema.canonizar()` (usado para
classificar Família) remove todo espaço antes de comparar — ótimo para decidir "essa frase é
essa família", ruim aqui porque a posição do casamento no blob **é o dado que eu preciso** (é o
que separa a seção 1 da seção 2). Canonizar apagaria os espaços que sustentam essa posição. Por
isso a regex roda direto no blob normalizado por `_WS.sub(" ", ...)` (que só colapsa espaço
repetido, não remove), com `\s+` cobrindo a variação de espaçamento entre palavras e `N[ÃA]O`
cobrindo a única variação de acento que aparece na frase — mesmo raciocínio do classificador de
família, adaptado para preservar posição.

**Por que `\d\s*DO RESULTADO...` e não `"2 DO RESULTADO..."` fixo:** o número da seção não é
garantidamente "2" em todo Edital da família — é a posição dele na numeração do documento, que
pode mudar se um Edital futuro tiver uma cláusula extra antes. Ancorar no dígito genérico (`\d`)
em vez do literal `2` é o que faz a detecção valer "para qualquer Edital da família, não só o
Ed_27", como o ticket pede — sem essa generalização, a implementação atenderia à letra do
critério de aceite ("cabeçalho numerado, não página fixa") mas continuaria fixa a um número
específico, só que dentro do texto em vez de na posição.

**Por que o cabeçalho da seção 1 não colide:** medido no Ed_27, o cabeçalho da primeira seção é
`"1 DO RESULTADO FINAL NOS ITENS DO TIPO D E DO RESULTADO FINAL NA PROVA DE REDAÇÃO EM LÍNGUA
PORTUGUESA"` — sem `"CANDIDATOS NÃO ELIMINADOS"` em lugar nenhum da frase. A regex é específica o
bastante para não confundir as duas sem precisar de nenhuma lógica de "pega o segundo cabeçalho
numerado que achar".

**Efeito colateral corrigido junto:** `_processar_cabecalhos(blob[:ancoras[0]], estado)`, que lê
o primeiro Campus/Curso/Turno antes do primeiro registro, também passou a receber
`blob[inicio_secao:ancoras[0]]` em vez de `blob[:ancoras[0]]`. Sem essa mudança, o trecho
maiúsculo mais próximo do primeiro registro da seção 2 continuaria vencendo (a lógica de "pega o
último candidato" do ticket 01 já cobre isso), mas o texto da seção 1 inteira seria varrido por
`_CURSO_RE` à toa — desperdício sem efeito prático neste caso, mas incoerente com a decisão de
"a seção 1 não existe para este parser" que é o espírito do ticket.

---

## 4. Como foi verificado

- **Medido diretamente no Ed_27 real** (`data/pdfs/Ed_27_PAS_3_2021_2023_Res_final_tipo_D_
  redação.pdf`, 317 páginas): o cabeçalho de transição aparece na página 100 (1-indexada),
  exatamente como descrito no ticket e no `spec.md` ("a partir da página 99" — a página 99
  ainda é cauda da seção 1; o cabeçalho em si está na 100). Confirma que a medição do ticket
  bate com o dado real, não é uma suposição.
- **Fixture nova, não contígua** (`tests/fixtures/resultado_final_duas_secoes.pdf`, gitignored):
  página 1 (schema, para o classificador de Família) + páginas 99–101 (cauda da seção 1 +
  cabeçalho de transição + início da seção 2), geradas com `fixtures.fatiar_paginas` — o mesmo
  utilitário não contíguo já usado pela fixture `resultado_final_curso_completo.pdf` do ticket
  01, porque página 1 e página 99+ não são um intervalo contínuo.
- **3 testes novos** (`TestParseDirigidoPorSecao`):
  1. `test_apenas_a_secao_de_nao_eliminados_e_extraida` — 55 registros extraídos da fixture,
     todos com as 10 posições de classificação preenchidas (só existe se o registro veio do
     schema de 22 campos).
  2. `test_nenhum_registro_da_secao_1_aparece_na_saida` — verifica por **nome e inscrição reais**
     de um candidato da página 99 (os literais estão nas constantes `NOME_DA_SECAO_1` e
     `INSCRICAO_DA_SECAO_1` do teste, não reproduzidos aqui — são dado pessoal de Aluno, mesma
     razão pela qual as fixtures de PDF não são commitadas; ambos confirmados presentes no texto
     bruto da seção 1 antes de rodar o parser) que não aparecem no resultado. Não é só uma
     contagem: é a garantia de que um registro específico e identificável da seção errada não
     vazou.
  3. `test_edital_de_secao_unica_continua_sem_regressao` — reroda a fixture do ticket 01
     (`resultado_final_22_campos.pdf`, sem o cabeçalho de transição) e confirma a mesma contagem
     de sempre (189), provando que `inicio_secao = 0` quando o cabeçalho não existe não muda
     nada do comportamento anterior.
- **Suíte completa do pacote** (`pytest tests/test_pas_extraction.py`): 26 testes, todos
  passando (23 pré-existentes + 3 novos), ~6s.
- **Suíte inteira do projeto** (`pytest tests/`): 78 passam, 2 falham — os mesmos 2 já
  documentados como pré-existentes e não relacionados no relatório do ticket 01
  (`test_guaranteed_scenario`: incompatibilidade de versão do `sklearn`; `test_pdf_gen`:
  caminho absoluto do Windows hardcoded no teste). Confirmado por não terem sido tocados nesta
  sessão.

---

## 5. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê |
|---|---|
| Parser da seção 1 (eliminados, 4 campos: inscrição, nome, nota tipo D, nota redação) | Fora de escopo por decisão registrada no ticket — só 2 notas, não forma o vetor de 9 usado pelos modelos de treino. Custo aceito: ~1.449 Alunos eliminados por Edital não entram no CSV. |
| Detecção de outras famílias de duas seções | Só Resultado Final tipo D + redação tem esse formato, entre as famílias já implementadas (01/03/09); Convocação e Médias e Desvios são de seção única. |

---

## 6. Glossário — termos necessários para entender este relatório

| Termo | Significado |
|---|---|
| **Edital de resultado final tipo D + redação** | Variante do Edital de Resultado Final que traz duas listas de schema diferente no mesmo PDF: resultado dos itens do tipo D / redação (candidatos eliminados, ainda sem o vetor completo de notas) e resultado final dos candidatos não eliminados (schema completo de 22 campos, igual ao de qualquer outro Edital de Resultado Final). |
| **Seção** | Um trecho do PDF delimitado por um cabeçalho numerado do próprio documento (`"1 DO ..."`, `"2 DO ..."`), cada um podendo declarar um schema de registro diferente. A partir deste ticket, é a seção — não o arquivo inteiro — que o parser trata como unidade de schema único. |
| **Candidato eliminado** | Aluno que não avançou para a etapa seguinte do PAS; aparece só na seção 1 do Edital tipo D + redação, com 2 notas (não 9), por isso não serve para o vetor de treino dos modelos preditivos. |
| **Cabeçalho numerado** | A frase de seção que o próprio Edital declara (ex.: `"2 DO RESULTADO FINAL DOS CANDIDATOS NÃO ELIMINADOS"`), usada aqui como o marcador de transição entre seções — em vez de um número de página fixo, que mudaria de Edital para Edital. |
| **Blob** | O texto de todas as páginas do PDF concatenado numa única string (ver `resultado_final.py`, `_construir_blob`), sobre o qual `_ANCORA_RE` e (a partir deste ticket) `_SECAO_NAO_ELIMINADOS_RE` são varridos. |
| **Âncora de registro** | O ponto no blob usado para saber onde um registro de Aluno começa: os 8 dígitos do número de inscrição seguidos de vírgula (ver relatório do ticket 01). |

---

## 7. Onde continuar

Próximo ticket sem bloqueio direto por este: **06 — Dedução de Cota Declarada**, que consome os
registros de 22 campos já extraídos (agora garantidamente só da seção correta, em Editais de
duas seções) sem depender de nada novo deste ticket além do que o 01 já entregava.
