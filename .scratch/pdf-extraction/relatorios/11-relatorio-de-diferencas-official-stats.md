# Relatório — Ticket 11: Relatório de diferenças do `OFFICIAL_STATS`

**Ticket:** `.scratch/pdf-extraction/issues/11-relatorio-de-diferencas-official-stats.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/relatorio_official_stats.py` (módulo novo) e o
subcomando `stats-diff` em `src/pas_extraction/cli.py` — no mesmo pacote gitignored de
`src/pas_extraction/` (ver ticket 01, seção "Por que o código não está no git")
**Onde vive o teste:** `tests/test_pas_extraction_official_stats.py` (versionado normalmente)
**Onde vive o artefato gerado:** `.scratch/pdf-extraction/official-stats-diff.md` — é este o
documento que o dono do produto revisa antes de destravar o ticket 12

---

## 1. O que foi pedido

Antes de trocar qualquer valor do `OFFICIAL_STATS`, o dono do produto precisa ver exatamente
quais entradas mudam e em quanto. O `OFFICIAL_STATS` de hoje é declaradamente estimado
(*"gerado automaticamente via análise do banco_alunos_pas_final.csv"*), enquanto o Cebraspe
publica média e desvio-padrão oficiais em Edital. Como a substituição altera o Argumento Final
calculado para todo Aluno em produção, ela foi partida em dois tickets: o relatório (este) e a
troca (ticket 12).

O ticket pede também que o relatório mostre a mudança de **forma**, não só de valor: o
`ExamStats` tem um `m_p1` único onde o Edital publica a Parte 1 separada por três línguas
estrangeiras.

Critérios de aceite (todos atendidos — ver seção 6):

- [x] O relatório lista, por `(ano, etapa)`, o valor atual, o valor oficial e a diferença
- [x] Entradas do `OFFICIAL_STATS` sem cobertura nos Editais extraídos são listadas
      explicitamente
- [x] O relatório mostra onde o `m_p1` único agrega três valores oficiais por língua
- [x] `pas_constants.py` não é alterado neste ticket

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/relatorio_official_stats.py   — módulo novo:
  ano_da_prova(trienio, etapa) -> int              — a conversão que liga os dois lados
  ValorOficial / EtapaOficial                      — a tabela oficial de uma (ano, etapa)
  DiferencaCampo / AgregacaoParte1                 — as duas formas de "diferença" (ver 3.3)
  DivergenciaEntreFontes                           — dois Editais discordando da mesma célula
  ColetaOficial / RelatorioOfficialStats           — o que se varreu / o que se concluiu
  coletar_valores_oficiais(pdfs) -> ColetaOficial
  conferir_entre_fontes(a, b) -> List[DivergenciaEntreFontes]
  comparar(coleta, stats_atuais=None) -> RelatorioOfficialStats
  formatar_markdown(relatorio) -> str              — o arquivo revisado por uma pessoa
  formatar_resumo(relatorio) -> str                — a saída de terminal
  gerar_relatorio(pdfs, destino) -> RelatorioOfficialStats

src/pas_extraction/cli.py                        — subcomando novo `stats-diff`
tests/test_pas_extraction_official_stats.py      — 19 testes de comportamento
.scratch/pdf-extraction/official-stats-diff.md   — o relatório gerado sobre data/pdfs (77 Editais)
```

Fluxo:

```
data/pdfs/*.pdf
   │
   ├─ coletar_valores_oficiais()
   │     ├─ página 1 em 'layout' → classificar_familia() + extrair_metadados()
   │     ├─ Convocação            → pulada, com motivo registrado (não publica a tabela)
   │     ├─ Médias e Desvios      → parse_medias_desvios() no PDF inteiro (1 página)
   │     ├─ Resultado Final       → parse_medias_desvios() nas 5 últimas páginas;
   │     │                           se não achar, varredura completa (ver 3.7)
   │     └─ (triênio, etapa) → ano_da_prova() → chave (ano, etapa) do OFFICIAL_STATS
   │                            + conferência quando dois Editais cobrem a mesma Etapa
   │
   ├─ comparar(coleta, OFFICIAL_STATS)
   │     ├─ Parte II e Redação → 4 DiferencaCampo por entrada (m_p2, dp_p2, m_red, dp_red)
   │     ├─ Parte 1            → 1 AgregacaoParte1 por entrada (1 atual x 3 oficiais)
   │     ├─ no OFFICIAL_STATS e não no Edital → sem_cobertura
   │     └─ no Edital e não no OFFICIAL_STATS → ausentes_no_official_stats
   │
   └─ formatar_markdown() → .scratch/pdf-extraction/official-stats-diff.md
      formatar_resumo()   → terminal
```

Comando:

```bash
PYTHONPATH=src python -m pas_extraction.cli stats-diff \
    --out .scratch/pdf-extraction/official-stats-diff.md
```

### O que o relatório encontrou (rodada sobre os 77 Editais de `data/pdfs`)

| | |
|---|---|
| Entradas do `OFFICIAL_STATS` | 21 |
| Entradas comparadas contra Edital oficial | 21 (100%) |
| Entradas sem cobertura nos Editais | **0** |
| Etapas oficiais ausentes do `OFFICIAL_STATS` | 3 (todo o triênio 2023/2025) |
| Campos com comparação 1-para-1 | 84 (21 x `m_p2`, `dp_p2`, `m_red`, `dp_red`) |
| Campos que divergem | **82 de 84** (só dois `dp_red` batem exatamente) |
| Divergências entre Editais da mesma Etapa | 0 |

Magnitude por campo:

| Campo | Δ médio (atual − oficial) | \|Δ\| médio | \|Δ%\| médio | \|Δ%\| máx | Sinal |
|---|---:|---:|---:|---:|---|
| `m_p2` | +0.367 | 0.367 | 1.47% | 3.21% (2017/E2) | **21 de 21 positivos** |
| `dp_p2` | +0.061 | 0.064 | 0.50% | 2.51% (2017/E2) | 19 positivos, 2 negativos |
| `m_red` | +0.039 | 0.039 | 0.61% | 1.52% (2016/E1) | **21 de 21 positivos** |
| `dp_red` | −0.006 | 0.009 | 0.43% | 1.22% (2016/E1) | 4 positivos, 15 negativos, 2 zero |

O caso citado no ticket confere: 2022/2024 Etapa 1, `m_p2` estimado 20.709 contra 20.406
oficial (+1.48%).

**O viés das médias tem uma direção só.** As 42 médias (`m_p2` e `m_red` de 21 entradas) estão
*todas* acima do valor oficial. Isso não parece ruído de amostragem: o
`banco_alunos_pas_final.csv` vem dos Editais de Resultado Final, que listam **apenas os não
eliminados** (decisão registrada em "Out of Scope" da spec), enquanto a média oficial é
calculada sobre todos os candidatos. Estimar a média da prova a partir de quem sobreviveu a ela
puxa o valor para cima — é sobrevivência, não erro numérico, e é por isso que nenhuma
recalibração do CSV resolveria: falta o dado de quem foi eliminado, que o projeto não tem.

E a Parte 1 não é uma diferença de valor, é de forma: o `m_p1` único está a até **3.026** pontos
de distância de uma das três línguas oficiais (2021, Etapa 2, Francesa), e as três línguas de
uma mesma Etapa chegam a diferir **3.672** entre si (2021, Etapa 2 — e 3.864 em 2024, Etapa 2,
já fora do `OFFICIAL_STATS` atual). Não existe um `m_p1` correto a escolher — ver 3.3.

---

## 3. Decisões tomadas e o porquê

### 3.1 O relatório é código, não um Markdown escrito à mão

**Decisão:** o entregável é um módulo + subcomando `stats-diff` que **gera** o relatório; o
Markdown em `.scratch/pdf-extraction/official-stats-diff.md` é saída, não fonte.

**Porquê:** três consumidores diferentes precisam dos mesmos números. O dono do produto lê o
Markdown agora; o ticket 12 precisa dos valores oficiais para escrever no `pas_constants.py`; e
quando um Edital novo entrar em `data/pdfs` (o triênio 2024/2026, por exemplo), a comparação
tem que ser refeita sem ninguém reconferir 84 células na mão. Um Markdown escrito à mão
atenderia só o primeiro, e ficaria desatualizado em silêncio. Como efeito colateral, o ticket
12 recebe uma API (`coletar_valores_oficiais`) em vez de uma tabela para transcrever — a
transcrição manual seria a fonte de erro mais provável naquele ticket.

### 3.2 A junção é `ano = início do triênio + etapa − 1`, e isso foi verificado, não presumido

**Decisão:** `ano_da_prova("2022/2024", 1) == 2022`; a Etapa N de um triênio A/B é a prova do
ano `A + N − 1`.

**Porquê:** os dois lados são indexados de formas diferentes — o `OFFICIAL_STATS` por **ano da
prova**, o Edital por **triênio**. Toda a comparação depende dessa conversão, e um deslocamento
de um ano compararia silenciosamente a Etapa 1 de 2022 com a estimativa de 2021: o relatório
sairia bonito e completamente errado.

Como isso não é demonstrável só por leitura, foi medido. Refazendo a comparação com o ano
deslocado, o erro médio absoluto de `m_p2`:

| Mapeamento | \|Δ\| médio de `m_p2` | \|Δ\| médio de `dp_p2` |
|---|---:|---:|
| `ano − 1` | 10.67% | 6.15% |
| **`ano` (o usado)** | **1.53%** | **0.54%** |
| `ano + 1` | 11.39% | 5.31% |

Uma ordem de grandeza de diferença. Se o mapeamento estivesse errado, ele não ficaria nesse
patamar — a estimativa amostral e o valor oficial da *mesma* prova têm que ficar próximos, e é
exatamente o que acontece.

### 3.3 `m_p1`/`dp_p1` ficam **fora** da tabela de diferença 1-para-1

**Decisão:** a comparação campo a campo cobre só `m_p2`, `dp_p2`, `m_red` e `dp_red`
(`CAMPOS_COMPARAVEIS`). A Parte 1 tem uma seção própria (`AgregacaoParte1`), que mostra o valor
agregado atual ao lado dos três valores oficiais e a distância para cada um, mais a amplitude
entre as línguas.

**Porquê:** para reportar "a diferença" de `m_p1` seria preciso primeiro inventar um `m_p1`
oficial — média das três línguas, ou a de Língua Inglesa por ser a mais comum. Qualquer uma
dessas escolhas produziria um número que **não está em Edital nenhum**, e o relatório existe
justamente para separar o que é oficial do que é estimado. Pior: esconderia o achado principal
desta seção, que é a amplitude entre as línguas (até 3.672 pontos) — a informação de que a
agregação atual é indevida, e não só imprecisa.

A tentação de eleger a Língua Inglesa como proxy também não se sustenta nos dados: das 21
entradas, o `m_p1` estimado fica mais perto da Inglesa em 11, da Francesa em 7 e da Espanhola em
3 — ou seja, em 10 das 21 o agregado atual está mais perto de uma língua que não é a mais
comum, e nenhuma escolha fixa serviria. Qual língua vale para um Aluno depende de qual ele fez, informação que o
Resultado Final não imprime e que o ticket 04 infere por checksum.

### 3.4 "Sem cobertura" e "ausente do `OFFICIAL_STATS`" são duas listas, não uma

**Decisão:** seções 3 e 4 do relatório gerado. A primeira é entrada do `OFFICIAL_STATS` que
nenhum Edital cobre; a segunda é Etapa publicada em Edital que o `OFFICIAL_STATS` não tem.

**Porquê:** são decisões opostas no ticket 12. Uma entrada sem cobertura **continua estimada**
depois da substituição — é dívida que sobra, e o relatório precisa dizer isso na cara. Uma Etapa
oficial ausente é **dado novo** — acrescentar ou não é escolha, não correção, e misturá-la com
as correções faria o ticket 12 parecer maior do que é. Na rodada atual a primeira lista está
vazia e a segunda tem o triênio 2023/2025 inteiro; as duas continuam impressas mesmo vazias,
com a frase explícita "Nenhuma", porque "seção ausente" e "nada a reportar" não podem parecer a
mesma coisa para quem revisa.

### 3.5 Conferência entre Editais, sem desempate automático

**Decisão:** quando dois Editais publicam a mesma `(ano, etapa)`, cada célula é conferida entre
eles. Iguais: o arquivo entra na lista de fontes. Diferentes: vira `DivergenciaEntreFontes` na
seção 5 do relatório, o primeiro lido é mantido, e nada é escolhido automaticamente.

**Porquê:** um valor oficial que aparece em dois documentos oficiais diferentes é a única
chance de detectar erro de extração *no lado oficial* — se o parser quebrasse um número numa das
duas leituras, a conferência acusaria. E se a divergência for real (Edital retificado, por
exemplo), a escolha é do dono do produto, não do código: um desempate silencioso decidiria em
nome dele exatamente o que este ticket existe para evitar.

Na prática a seção saiu vazia, e por um motivo estrutural que só apareceu ao rodar: o Cebraspe
publicou a tabela em Edital avulso até o triênio 2020/2022 e passou a publicá-la na cauda do
Resultado Final a partir de 2021/2023 — **nenhum triênio usa os dois formatos**. Os dois modos
de publicação que o ticket 03 implementou são complementares, não redundantes. O mecanismo de
conferência fica no lugar porque é barato e porque a primeira sobreposição (uma retificação,
uma republicação) é justamente o caso em que ninguém estaria olhando.

### 3.6 `stats_atuais` é injetável e o `OFFICIAL_STATS` é importado tarde

**Decisão:** `comparar(coleta, stats_atuais=None)` importa
`pas_intelligence.pas_constants.OFFICIAL_STATS` **dentro da função**, e só quando o parâmetro
não foi passado.

**Porquê:** dois motivos independentes.

1. `pas_extraction` é extração offline e `pas_intelligence` é predição dentro do app Streamlit —
   a spec separa os dois domínios de propósito. Um `import` no topo do módulo faria todo o
   pacote de extração passar a depender do pacote de predição em tempo de import, inclusive nos
   testes e no `extract`, que não têm nada a ver com isso.
2. Os testes injetam um `OFFICIAL_STATS` falso. Se dependessem do de produção, **quebrariam no
   ticket 12** — quando os valores forem substituídos, que é a única coisa que o ticket 12 vai
   fazer. Um teste que falha quando o trabalho planejado é executado corretamente não é um teste,
   é um alarme falso agendado.

### 3.7 Janela de cauda com varredura completa como rede de segurança

**Decisão:** num Edital de Resultado Final, a tabela é procurada primeiro nas 5 últimas páginas
(`_JanelaDePaginas`, uma vista somente-leitura de `reader.pages`); se não aparecer, o PDF
inteiro é varrido antes de o Edital ser dado como "sem tabela".

**Porquê:** os Resultados Finais têm de 242 a 419 páginas e a extração de texto custa ~0,06 s
por página; varrer os 8 inteiros seria ~2 minutos gastos quase todos em páginas de registros de
Aluno, e a tabela está sempre na última página nos três Editais em que ela existe. Mas uma
janela fixa que não achasse a tabela produziria um "sem cobertura" **falso** — e o relatório
inteiro serve para dizer o que está coberto e o que não está. Otimização que pode mentir sobre
o resultado não vale o tempo economizado, então a janela é só um caminho rápido: quem decide que
não há tabela é a varredura completa.

A janela renumera as páginas a partir de 1, o que corromperia a proveniência; `_com_pagina_corrigida`
soma o deslocamento de volta para a página apontada ser a real do Edital.

**Alternativa descartada:** re-serializar as últimas páginas num PDF em memória (`PdfWriter` +
`BytesIO`) para montar um `PdfReader` de verdade. Funciona, mas paga escrita e releitura do PDF
para resolver um problema que é só de *quais páginas iterar* — `parse_medias_desvios` toca
apenas em `reader.pages`, então uma vista com esse atributo basta.

### 3.8 Convocação é pulada por construção, mas o motivo entra no relatório

**Decisão:** Editais da Família Convocação nem chegam a ser varridos; o arquivo entra em
`ColetaOficial.ignorados` com o motivo, e o relatório publica a contagem por motivo.

**Porquê:** são 64 dos 77 Editais e nenhum publica a tabela (a spec já registra que ela só
aparece em Edital avulso ou na cauda de um Resultado Final). Varrê-los seria minutos de custo
com resultado conhecido. O que **não** dá para fazer é pular em silêncio: quem revisa precisa
poder verificar que 77 Editais entraram e que os 69 que não contribuíram têm cada um um motivo
declarado — senão "sem cobertura: 0" é uma afirmação que ninguém consegue auditar.

Os 5 Resultados Finais que a varredura completa deu como "sem tabela" foram conferidos à parte:
a palavra "desvio" **não aparece nenhuma vez** em nenhum dos 5 (0 ocorrências em 1.566 páginas).
Não é falha de parser — esses triênios publicaram a tabela no Edital avulso, que está em
`data/pdfs` e foi lido.

### 3.9 Formatação separada do cálculo

**Decisão:** `comparar()` devolve um `RelatorioOfficialStats` com os números; `formatar_markdown`
e `formatar_resumo` são funções puras sobre ele.

**Porquê:** o relatório é lido por uma pessoa, e é essa leitura que destrava o ticket 12 — o
texto vai mudar mais do que a regra de comparação. Com a separação, mexer na redação de uma
seção não toca em nada que os testes de comparação verificam, e o ticket 12 pode consumir o
`RelatorioOfficialStats` direto, sem reparsear Markdown.

### 3.10 A estimativa de impacto no Argumento Final ficou **fora** do artefato gerado

**Decisão:** o relatório gerado tem só o diff pedido; a estimativa de impacto está aqui, na
seção 4.

**Porquê:** o ticket pede valor atual, valor oficial e diferença. Impacto no Argumento Final é
uma projeção com premissa embutida (qual Aluno, com que nota) — colocá-la lado a lado com
números lidos de Edital, no mesmo documento, borraria a linha entre "isto está publicado" e
"isto eu calculei", que é a linha que este ticket inteiro existe para reforçar.

### 3.11 Nenhum bug encontrado em código já existente

`medias_desvios.py` (ticket 03) rodou sobre os 8 Editais que têm a tabela sem nenhum ajuste:
extraiu 15 registros em cada um, e os valores da fixture bateram com os do ticket 03. O
classificador de `schema.py` acertou a Família dos 77 Editais. Nada em `pas_intelligence` foi
tocado.

---

## 4. Quanto disso chega no Argumento Final (estimativa)

Não é critério de aceite; é o contexto que dá tamanho à decisão do ticket 12.

O argumento de uma parte é `((nota − média) / desvio) × peso`. Para o Aluno que tirou
exatamente a média oficial da prova, o argumento correto daquela parte é 0, e o que o sistema
calcula hoje é `(m_oficial − m_estimada) / dp_estimado × peso`. Com `PESO_P2 = 8.28` e
`PESO_REDACAO = 1.00`:

| Triênio | Erro no Argumento Final (3 Etapas somadas) |
|---|---:|
| 2016/2018 | −1.280 |
| 2017/2019 | −0.781 |
| 2018/2020 | −0.746 |
| 2019/2021 | −0.563 |
| 2020/2022 | −0.593 |
| 2021/2023 | −0.695 |
| 2022/2024 | −0.664 |

O sinal é sempre o mesmo, porque as médias estimadas são sistematicamente altas (seção 2): hoje
o sistema **subestima** o Argumento do Aluno mediano em ~0,2 ponto por Etapa, ~0,6 a 1,3 por
Argumento Final. A Parte 1 não entra nesta conta — sem saber a língua do Aluno não há um valor
oficial a usar, que é o ponto de 3.3 outra vez. Aluno longe da média sente também a correção do
desvio, que é bem menor (\|Δ%\| médio de 0.50% em `dp_p2`).

---

## 5. Escopo deliberadamente fora deste ticket

- **Trocar qualquer valor do `OFFICIAL_STATS`** — é o ticket 12, e é o motivo de este ticket
  existir separado. `pas_constants.py` está byte a byte como estava.
- **Mudar a forma do `ExamStats`** para acomodar a Parte 1 por língua — também ticket 12. Aqui a
  mudança de forma é só *mostrada*.
- **Decidir o que fazer com o triênio 2023/2025** (as 3 Etapas oficiais sem entrada no
  `OFFICIAL_STATS`) — o relatório lista; incluir é decisão do ticket 12.
- **Reestimar o `banco_alunos_pas_final.csv`** para corrigir o viés de sobrevivência. O dado de
  quem foi eliminado não está em lugar nenhum do projeto; a saída é usar o valor oficial, que é
  o que o ticket 12 faz.
- **Interface visual.** Terminal e arquivo, como o resto do pipeline.

---

## 6. Como foi verificado

**Critérios de aceite:**

1. *Lista por `(ano, etapa)` o valor atual, o oficial e a diferença* — seção 1 do relatório
   gerado, 84 linhas, com `Δ`, `Δ %` e o arquivo de origem de cada valor oficial. Teste:
   `test_diferenca_e_atual_menos_oficial_para_parte_2_e_redacao`,
   `test_markdown_mostra_atual_oficial_e_diferenca_por_ano_e_etapa`.
2. *Entradas sem cobertura listadas explicitamente* — seção 3 do relatório gerado, com a frase
   "Nenhuma" quando vazia. Testes:
   `test_entrada_do_official_stats_sem_edital_e_listada_e_nao_comparada`,
   `test_markdown_lista_as_entradas_sem_cobertura_explicitamente`, e
   `test_official_stats_de_producao_e_o_padrao_quando_nao_injetado`, que coleta um Edital só e
   confirma que a lista de faltantes é calculada (sai não vazia), não presumida vazia.
3. *Mostra onde o `m_p1` único agrega três valores oficiais* — seção 2 do relatório gerado, uma
   linha por Etapa com os três oficiais, os três deltas e a amplitude. Testes:
   `test_parte_1_expoe_os_tres_valores_oficiais_que_o_valor_atual_agrega`,
   `test_m_p1_nao_entra_na_comparacao_1_para_1`,
   `test_markdown_mostra_o_m_p1_unico_contra_as_tres_linguas`.
4. *`pas_constants.py` não alterado* — `git status`/`git diff` em `src/pas_intelligence/`
   voltam vazios depois de toda a rodada.

**Rodada real:** os 77 Editais de `data/pdfs`, ~85 s (quase todo o tempo na varredura completa
dos 5 Resultados Finais que não têm tabela — ver 3.7). 24 Etapas oficiais coletadas de 8
Editais (5 avulsos de médias e 3 caudas de Resultado Final), cobrindo os 8 triênios de 2016/2018
a 2023/2025; as 21 entradas do `OFFICIAL_STATS` comparadas.

**Verificações que não são teste automatizado:**

- *Alinhamento do ano* — comparação refeita com o ano deslocado em ±1: erro médio de `m_p2` sobe
  de 1.53% para 10.67% e 11.39% (tabela em 3.2).
- *Os 5 Resultados Finais "sem tabela"* — busca por "desvio" nas 1.566 páginas dos 5: 0
  ocorrências. Confirma que é ausência do dado, não falha de leitura.
- *Valores conferidos contra o ticket 03* — a Etapa 1 de `ED_34` (2019/2021) sai com
  `parte_2 = 26.738`, `redacao = 6.617`, `parte_1 francesa = 5.064`, iguais aos valores que o
  ticket 03 conferiu direto no PDF; e a Etapa 1 do triênio 2022/2024 sai com `m_p2 = 20.406`,
  o número que o próprio ticket 11 cita como o oficial.

**Testes:** 19 novos em `tests/test_pas_extraction_official_stats.py`, todos passando. Suíte
completa: 71 passando, 2 falhas **pré-existentes e sem relação** com este ticket —
`test_pdf_gen_manual.py` (caminho Windows hardcoded, `c:/Users/user/...`) e
`TestTargetCalculator::test_guaranteed_scenario` (asserção sobre saída de modelo em `models/`).
`tests/test_pas_extraction.py` não pôde ser coletado: no momento desta rodada ele estava
modificado na árvore de trabalho por outro trabalho em andamento e importa
`pas_extraction.cotas`, módulo do ticket 06 que ainda não existe — nada a ver com este ticket, e
nenhum arquivo dele foi tocado aqui.

---

## 7. Glossário — termos necessários para entender este relatório

- **Ano da prova**: o ano em que uma Etapa foi aplicada. É a chave do `OFFICIAL_STATS`, e não
  aparece no Edital: deriva do triênio mais o número da Etapa (`ano_da_prova`).
- **Triênio**: o par de anos que identifica um subprograma do PAS (`2022/2024`), impresso na
  primeira página de todo Edital. Cobre as três Etapas, uma por ano.
- **Entrada sem cobertura**: `(ano, etapa)` que existe no `OFFICIAL_STATS` e que nenhum Edital
  extraído cobre — continua estimada depois do ticket 12.
- **Etapa ausente do `OFFICIAL_STATS`**: o contrário — publicada em Edital, sem entrada no
  `OFFICIAL_STATS`. Dado novo, não correção.
- **Agregação indevida**: o `m_p1`/`dp_p1` único do `ExamStats`, onde o Edital publica três
  valores (Inglesa, Francesa, Espanhola). Não é imprecisão de valor, é perda de dimensão.
- **Divergência entre fontes**: dois Editais publicando valores diferentes para a mesma célula
  da mesma Etapa. Reportada, nunca desempatada automaticamente.
- **Viés de sobrevivência** (no sentido usado aqui): as médias estimadas saírem sistematicamente
  acima das oficiais porque o CSV de origem só contém Alunos não eliminados.

---

## 8. Onde continuar

**Ticket 12 — substituição do `OFFICIAL_STATS`.** Está destravado e recebe deste ticket:

- `coletar_valores_oficiais(pdfs)` como fonte dos valores, em vez de transcrição manual das 84
  células (é a principal fonte de erro que o ticket 12 tem a evitar).
- A decisão de forma já quantificada: o `ExamStats` precisa acomodar Parte 1 por língua, com
  amplitude entre línguas de até 3.672 — não é detalhe cosmético.
- A pergunta em aberto sobre o triênio 2023/2025: 3 Etapas oficiais disponíveis, sem entrada
  atual. `api/services/analytics_service.py` itera `OFFICIAL_STATS.items()` — acrescentar
  entradas muda a série que ele expõe, e isso precisa ser olhado lá.
- O consumidor a não quebrar continua sendo `api/services/analytics_service.py`, que lê `s.m_p1`
  e `s.m_p1 + s.m_p2` — e `s.m_p1` é exatamente o campo que perde o sentido único.

**Depois do ticket 12**, rodar `stats-diff` de novo é o teste de fechamento: as diferenças de
Parte II e Redação devem ir todas a zero, e o que sobrar diferente de zero é entrada que ficou
para trás.
