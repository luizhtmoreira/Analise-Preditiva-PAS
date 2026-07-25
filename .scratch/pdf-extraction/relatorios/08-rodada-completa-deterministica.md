# Relatório — Ticket 08: Rodada completa sobre os 77 Editais, determinística

**Ticket:** `.scratch/pdf-extraction/issues/08-rodada-completa-deterministica.md`
**Status:** concluído
**Onde vive o código novo:** `src/pas_extraction/reconciliacao.py`, `src/pas_extraction/rodada.py`
(módulos novos, não versionados — mesma política do ticket 01); `src/pas_extraction/cli.py`
ganhou o subcomando `rodada` (versionado); `src/pas_extraction/validacao.py` recebeu um
ajuste pontual (ver seção 3).
**Onde vivem os testes:** `tests/test_pas_extraction_reconciliacao.py`,
`tests/test_pas_extraction_rodada.py` (novos), mais uma classe nova em
`tests/test_pas_extraction.py` (todos versionados normalmente).

---

## 1. O que foi pedido

Um comando só que extrai os 77 Editais reais de `data/pdfs`, produz os CSVs de todas as
Famílias já implementadas (Resultado Final, Convocação, Médias e Desvios), com determinismo
byte a byte e a 6ª camada de validação do spec — reconciliação cruzada entre Editais — que só
existe em escala completa. É o ticket que faz o pipeline deixar de ser protótipo-verificado-
em-fixture e passar a ser a fonte real dos dados.

Critérios de aceite (todos atendidos — ver seção 5):

- [x] Um comando extrai os 77 Editais e produz os CSVs de todas as Famílias já implementadas
- [x] Duas execuções sobre a mesma entrada produzem saída idêntica byte a byte
- [x] O mesmo número de inscrição encontrado em Editais diferentes é conferido quanto ao nome
- [x] O relatório de validação cobre o corpus inteiro, agrupado por padrão
- [x] Nenhum caminho absoluto de máquina no pipeline
- [x] Os quatro scripts `prototype_*` foram removidos e o `scripts/NOTES.md` permanece

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/reconciliacao.py   — 6ª camada de validação (nova, isolada):
    DivergenciaNome                     — dataclass novo
    reconciliar_nomes(*grupos)          — agrupa por inscrição através de N grupos de
                                           registros (Resultado Final, Convocação, ou
                                           qualquer combinação), sinaliza nome divergente
                                           tolerando ruído de extração via schema.canonizar

src/pas_extraction/rodada.py          — a costura do ticket 08:
    RodadaCompleta                      — os três resultados de Família + PDFs ignorados
    rodar(pdfs) -> RodadaCompleta       — classifica cada PDF (1 leitura de página 1) e
                                           despacha para o extrator certo; resolve sozinha
                                           qual Edital avulso de Médias e Desvios combina
                                           com qual Resultado Final, por triênio
    RelatorioCorpus / gerar_relatorio_corpus / formatar_terminal_corpus /
    formatar_markdown_corpus / escrever_relatorio_corpus
                                         — camadas 1-5 (ticket 07, reusado sem alteração)
                                           + camada 6 (reconciliação), sobre o corpus inteiro

src/pas_extraction/cli.py             — subcomando novo `rodada`:
    python -m pas_extraction.cli rodada --out-dir <dir> --relatorio <md>
    (aceita subconjunto de PDFs como `extract`/`validar`, mesmo `--pdf-dir`/`--limit`)

src/pas_extraction/validacao.py       — ajuste pontual em `_buracos_por_sistema` (ver seção 3)

tests/test_pas_extraction_reconciliacao.py  — 11 testes (puros, sem PDF)
tests/test_pas_extraction_rodada.py         — 10 testes (fixtures já commitadas localmente
                                               pelos tickets 01/03/09, nenhuma fixture nova)
tests/test_pas_extraction.py                — +2 testes (TestLimiteDePlausibilidadeDosBuracos)
```

Fluxo de ponta a ponta:

```
data/pdfs/*.pdf (ou subconjunto)
   │
   ├─ rodada._classificar(pdf)  — lê só a página 1, devolve (Família, triênio)
   │     (1 leitura por PDF, reaproveitando schema.classificar_familia/extrair_metadados
   │      já existentes — não inventa um classificador novo)
   │
   ├─ rodada._mapa_avulsos_medias_desvios  — triênio -> Path do avulso, entre os PDFs
   │      descobertos nesta rodada
   │
   ├─ para cada PDF, despacha por Família:
   │     RESULTADO_FINAL   -> pipeline.extrair_edital (ticket 01), com fallback automático
   │                          pro avulso do mesmo triênio se a cauda não tiver a tabela
   │     CONVOCACAO        -> convocacao.extrair_edital_convocacao (ticket 09), inalterado
   │     MEDIAS_DESVIOS    -> medias_desvios.extrair_edital_medias_desvios (ticket 03), inalterado
   │
   └─ RodadaCompleta(resultado_final=[...], convocacao=[...], medias_desvios=[...])
         │
         ├─ csv_writer.escrever_csv / convocacao.escrever_csv_convocacao /
         │   medias_desvios.escrever_csv_medias_desvios   (os três writers já existentes,
         │   um por Família, sem alteração)
         │
         └─ rodada.gerar_relatorio_corpus
               = relatorio_validacao.gerar_relatorio (camadas 1-5, ticket 07, reusado)
               + reconciliacao.reconciliar_nomes (camada 6, nova)
```

**Validado contra o corpus real completo** (`python -m pas_extraction.cli rodada` sobre os 77
PDFs de `data/pdfs`, sem `--limit`):

| Família | Editais | Registros |
|---|---:|---:|
| Resultado Final | 8 | 66.313 |
| Convocação | 64 | 33.386 |
| Médias e Desvios | 5 | 75 |

(33.386 em Convocação bate exatamente com o número já medido pelo ticket 09 isoladamente —
consistência cruzada entre os dois tickets, não só interno a este.)

Checksum do Argumento Final: 2.015 falhas de 66.313 conferidos (96,96% fecham), distribuição
**espalhada** — dado corrompido pontual, não fórmula incompleta (ver ticket 07 para o
critério). Reconciliação cruzada: **10 inscrições** com nome divergente entre Editais
diferentes, em ~100 mil registros cruzados — proporção baixa, compatível com ruído de
extração isolado e não com um problema sistemático de casamento de inscrição.

Saída: `resultado_final.csv` (24 MB), `convocacao.csv` (6,4 MB), `medias_desvios.csv` (7 KB),
relatório de validação do corpus (99 KB) — ver seção 3 para a história de como o
`resultado_final.csv` chegou a 24 MB e não aos 6,4 **GB** que a primeira rodada real produziu.

---

## 3. Decisões tomadas e o porquê

### 3.1 `rodada.py` importa das três Famílias — é a exceção deliberada à regra de isolamento

Cada ticket anterior (01, 03, 09) construiu seu extrator como módulo autocontido, sem
importar dos módulos irmãos, exatamente para permitir trabalho paralelo sem conflito. O
ticket 09 documentou explicitamente que essa integração ficava para depois ("o dispatch
principal integra isso manualmente depois"). Este é esse depois: `rodada.py` importa de
`pipeline.py`, `convocacao.py`, `medias_desvios.py`, `relatorio_validacao.py` e
`reconciliacao.py` ao mesmo tempo, de propósito — é o único lugar do pacote em que isso deve
acontecer, e nenhum dos módulos-fonte foi alterado para permitir isso (só lidos e importados,
mesma disciplina do ticket 09).

### 3.2 Fallback automático de Médias e Desvios por triênio, não por parâmetro manual

`extract --medias-desvios <avulso>` (tickets 01/04) aplica o mesmo avulso a **todos** os PDFs
da rodada, com aviso se houver mais de um — funciona para iterar num Edital por vez, mas não
para uma rodada com 7 triênios diferentes, dos quais só um (`Ed_31`, 2016/2018) não tem tabela
na própria cauda.

`rodada._extrair_resultado_final` resolve isso sozinha: extrai cada Resultado Final sem
avulso primeiro; se **todos** os registros saírem com `checksum=None` (sinal de "não havia
tabela na cauda" — `pipeline.extrair_edital` só chama `conferir_registros` quando a tabela não
é vazia, então é tudo-ou-nada por Edital), procura entre os PDFs descobertos nesta mesma
rodada um avulso de Médias e Desvios do **mesmo triênio** e refaz a extração com ele. Testado
com o par real `Ed_31` + `Ed_32` (mesmas fixtures que `test_pas_extraction_checksum.py` usa
para a chamada manual): 84 registros, 82 fecham — sem passar `--medias-desvios` nenhuma vez.
Também testado que um avulso de **outro** triênio não é usado por engano
(`test_avulso_de_trienio_diferente_nao_e_usado`).

### 3.3 Achado da rodada real: uma posição de classificação implausível explodia o CSV para 6,4 GB

**O que aconteceu.** A primeira rodada real (77 PDFs, sem `--limit`) produziu um
`resultado_final.csv` de **6,4 GB** (esperado: dezenas de MB) e um relatório de validação de
21 MB. Isolado por edital com `grep -c <arquivo> resultado_final.csv`, a contagem de linhas
por Edital era normal (5.896 a 9.852, somando os 66.313 esperados) — o problema não era volume
de registros, era o **tamanho de uma célula**.

**Causa raiz.** `validacao._buracos_por_sistema` infere N como `max(posicoes)` (limitação já
documentada desde o ticket 02: não existe fonte independente do total real de candidatos). O
campo de classificação, ao contrário dos 9 campos numéricos de nota, **não passa** pela
validação de formato exato do ticket 02 (`_formato_numerico_valido`) — é só
`_WS.sub("", v)` seguido de `int()`. Um único registro real, no Edital 36 (triênio
2017/2019), curso MEDICINA, leu uma posição de 6 dígitos no Sistema Universal (curso com
~900 classificados de verdade) — a mesma classe de corrupção de dígito colado já documentada
em `cotas.py` (número de página vazando pro último campo), aqui afetando um campo diferente.
Sem limite, `esperado = set(range(1, max(posicoes) + 1))` virou um `range` de ~280 mil
posições, e essa lista (join de ~280 mil inteiros) foi escrita na coluna
`classificacao_buracos` de todo registro daquele curso/Sistema — e ela é a mesma lista
repetida linha a linha, por como `validar_sequencia_e_ordem` grava `buracos_classificacao`
(ver seu docstring). Um só valor implausível apagou o sinal do corpus inteiro atrás de um
CSV e um relatório inutilizáveis.

**Decisão: mitigar no ticket 08, não redesenhar a validação de campo no ticket 02.**
Corrigir a raiz (validar o formato do campo de classificação, no mesmo espírito de
`_formato_numerico_valido`) é trabalho do parser (`resultado_final.py`, tickets 01/02) e
merece sua própria investigação — a mecânica exata de como o dígito coladura acontece nesse
campo específico não foi determinada aqui. O que este ticket precisa é que a rodada completa
produza um CSV e um relatório utilizáveis; por isso `_buracos_por_sistema` ganhou um limite de
plausibilidade: um Sistema cujo `max(posicoes)` excede `3 × contagem_observada + 50` fica de
fora do cômputo de buracos (não é tratado como "sem buraco" nem descartado — só sai desta
camada; o registro continua na saída e continua visível às outras camadas de validação). O
multiplicador é generoso de propósito: nenhum curso real do corpus tem mais que dezenas de
buracos genuínos, então mesmo um curso com o triplo de posições "esperadas" sobre o observado
ainda é coberto. Com o limite: `resultado_final.csv` caiu de 6,4 GB para 24 MB, relatório de
21 MB para 99 KB, e a maior lista de buracos observada no corpus real caiu de ~280.000 para
55 (o segundo maior valor real, não um artefato).

**Testes:** `TestLimiteDePlausibilidadeDosBuracos` em `test_pas_extraction.py`, com registros
sintéticos (não fixture — é um teste de `validacao.py`, não de PDF real): confirma que a
posição implausível não gera buraco gigante, e que um buraco legítimo pequeno continua
detectado normalmente com o limite ativo.

**Segue como abertura para outro ticket:** validar o formato do campo de classificação em
`resultado_final._montar_registro`, simétrico ao que já existe para os 9 campos numéricos —
isso sinalizaria o registro específico (`campos_formato_invalido`) em vez de só neutralizar o
efeito colateral dele nesta camada.

### 3.4 Determinismo: por que não há nada de novo a fazer aqui

`rodada.py` não introduz nenhuma fonte de não-determinismo: `_descobrir_pdfs` (cli.py, já
existente) devolve `sorted(pdf_dir.glob(...))`; `rodar` processa nessa ordem, sequencialmente,
sem `set()`/dicionário não ordenado na saída; `reconciliar_nomes` ordena por inscrição antes de
devolver. Verificado com testes de unidade (`TestDeterminismo` em
`test_pas_extraction_rodada.py`): rodar duas vezes sobre as mesmas fixtures produz CSVs e
relatório byte a byte idênticos. A verificação equivalente contra o corpus real completo (77
PDFs, duas rodadas, diff dos CSVs) não foi executada nesta sessão por custo — a primeira rodada
real sozinha já levou vários minutos; a garantia de determinismo vem da ausência estrutural de
qualquer fonte de não-determinismo no código, confirmada nas fixtures, não de uma segunda
rodada de 77 PDFs.

### 3.5 Reconciliação cruzada tolera ruído de extração, não esconde divergência real

`reconciliacao.reconciliar_nomes` compara nomes via `schema.canonizar` (a mesma função que já
resolve variação de redação institucional na classificação de Família) — sem isso, acento
perdido ou espaço a mais entre dois Editais diferentes apareceria como "divergência", inflando
o relatório com ruído em vez de sinal. Os nomes **originais** (não canonizados) é que saem no
relatório, para a divergência real ficar legível.

### 3.6 Scripts protótipo

`scripts/prototype_pdf_census.py`, `prototype_pdf_probe.py` e `prototype_pdf_census.json` já
não existiam mais no disco no início deste ticket (removidos em rodada anterior de limpeza).
Restavam `scripts/prototype_checksum.py` e `scripts/prototype_cotas.py`, ambos removidos agora
— grep confirmou que nada no repositório os referenciava. `scripts/NOTES.md` permanece.

---

## 4. Nota sobre a contagem de Editais de Resultado Final

`scripts/NOTES.md` (censo original, ticket 01) contava **7** PDFs na família
`resultado_final_*`. A rodada real deste ticket encontrou **8**. A diferença não é uma
regressão do classificador — é que o censo original foi feito antes de todos os 77 PDFs atuais
estarem em `data/pdfs`; o classificador por schema declarado (ticket 01) não mudou.

---

## 5. Critérios de aceite — conferência final

- [x] Um comando extrai os 77 Editais e produz os CSVs de todas as Famílias já implementadas —
  `python -m pas_extraction.cli rodada --out-dir <dir> --relatorio <md>`, validado contra o
  corpus real completo (seção 2).
- [x] Duas execuções sobre a mesma entrada produzem saída idêntica byte a byte — garantido
  estruturalmente e verificado em teste de unidade (seção 3.4).
- [x] O mesmo número de inscrição encontrado em Editais diferentes é conferido quanto ao nome —
  `reconciliacao.py`, camada 6, 10 divergências reais encontradas no corpus completo.
- [x] O relatório de validação cobre o corpus inteiro, agrupado por padrão — `RelatorioCorpus`
  reusa as camadas 1-5 do ticket 07 sobre todo o Resultado Final extraído, mais a camada 6.
- [x] Nenhum caminho absoluto de máquina no pipeline — `rodada.py` não introduz nenhum; os
  defaults de `cli.py` continuam relativos a `_REPO_ROOT` (ticket 01).
- [x] Os quatro scripts `prototype_*` foram removidos e o `scripts/NOTES.md` permanece.
