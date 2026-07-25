# Relatório — Ticket 04: Checksum do Argumento Final + inferência de língua por Etapa

**Ticket:** `.scratch/pdf-extraction/issues/04-checksum-argumento-final-e-lingua-por-etapa.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/checksum.py` (novo), com pontos de costura em
`models.py`, `constants.py`, `pipeline.py`, `csv_writer.py`, `cli.py` e `schema.py` — pacote
gitignored, ver seção "Por que o código não está no git" do relatório do ticket 01
**Onde vive o teste:** `tests/test_pas_extraction_checksum.py` (novo, 22 testes)
**Fórmula reusada, não reimplementada:** `src/pas_intelligence/argument_calculator.py`

---

## 1. O que foi pedido

Recalcular o Argumento Final de cada registro a partir das 9 notas brutas mais a tabela
oficial de médias e desvios do próprio Edital, e comparar com o valor impresso. Isso
substitui inspeção humana por verificação matemática: **um único número verifica 12 campos
de uma vez** (as 9 notas, o Argumento Final impresso, e as médias/desvios usados). Como
subproduto, recuperar a língua estrangeira de cada Etapa — um dado que não está impresso em
lugar nenhum do PDF.

Critérios de aceite (todos atendidos — ver seção 6):

- [x] Cada registro tem o Argumento Final recalculado e comparado com o impresso, com o delta gravado na linha
- [x] O cálculo reusa `argument_calculator.py` em vez de reimplementar a fórmula
- [x] A língua estrangeira é inferida por qual das três faz o checksum fechar
- [x] A inferência é por Etapa e não por Aluno, e a língua de cada Etapa é gravada por Aluno
- [x] A tolerância aplicada é `delta <= 0,005`
- [x] Nenhum registro é descartado neste ticket — o checksum só marca

---

## 2. Visão geral do que foi entregue

```
extrair_edital(pdf, medias_desvios=None) -> ResultadoExtracao
  │
  ├─ parse_resultado_final ......... (ticket 01/05) os registros
  ├─ validar_sequencia_e_ordem ..... (ticket 02) validação estrutural
  └─ NOVO:
     ├─ tabela oficial: da cauda do próprio Edital (parse_medias_desvios, ticket 03)
     │                  ou do Edital avulso passado em `medias_desvios`
     ├─ montar_tabela_oficial(...)  -> {(Etapa, língua): HistoricalStats} ou {}
     └─ conferir_registros(registros, tabela)
            para cada registro, sobre as 27 combinações de língua por Etapa:
              recalcular_argumento_final -> calculate_argument_final (pas_intelligence)
            grava ChecksumArgumentoFinal(recalculado, delta, línguas, lingua_ambigua)
```

Cinco tipos/funções novas em `checksum.py`: `TabelaOficial`, `montar_tabela_oficial`,
`recalcular_argumento_final`, `conferir_argumento_final`, `conferir_registros`. O resultado
por registro é `models.ChecksumArgumentoFinal`, e `RegistroResultadoFinal.checksum` é
`Optional` — `None` significa **"não foi possível conferir"**, nunca "conferido e passou".

Seis colunas novas no CSV: `lingua_e1`, `lingua_e2`, `lingua_e3` (junto das colunas de cota
declarada, porque são a mesma categoria: dado do Aluno que o Edital não imprime) e
`checksum_delta`, `checksum_fecha`, `lingua_ambigua` (no bloco de validação, porque são
sinal de qualidade).

---

## 3. Decisões tomadas e o porquê

### 3.1 Força bruta nas 27 combinações, chamando a função compartilhada 27 vezes

A alternativa era decompor: `AF = Σ peso_etapa × (argP1(língua) + argP2 + argRed)`, com
`argP1` pré-calculado para as 9 combinações (Etapa × língua) e as 27 somas montadas por
aritmética. É ~10× mais rápido, e foi **recusado**: reexpressar `AF = 1×AP1 + 2×AP2 + 3×AP3`
dentro de `pas_extraction` criaria uma segunda cópia da fórmula, capaz de divergir da que
roda em produção. O ticket é explícito ("reusa essa função em vez de reimplementá-la"), e o
custo medido não justifica a troca: das ~29s de uma rodada sobre o Ed_38 inteiro (242
páginas, 8.499 registros), o checksum é ~1s — o gargalo é a extração de texto do PDF.

Consequência prática: nem os pesos das partes (0,72 / 8,28 / 1,00) nem os das Etapas (1, 2,
3) aparecem em lugar nenhum de `pas_extraction`.

### 3.2 `ChecksumArgumentoFinal` mora em `models.py`, a lógica em `checksum.py`

O pacote tem dois precedentes: `ValidacaoRegistro` (resultado de verificação) fica em
`models.py` com a lógica em `validacao.py`; `CotaDeclarada` (atributo do Aluno) fica em
`cotas.py`. O checksum é resultado de verificação, então segue o primeiro. Isso também
resolve o ciclo de import que a outra escolha criaria (`models` → `checksum` →
`medias_desvios` → `models`) sem precisar de truque de `TYPE_CHECKING` no caminho principal.

### 3.3 `RegistroResultadoFinal.notas` — a ordem das 9 notas definida num lugar só

O checksum precisa das 9 notas na ordem do schema. Sem a property, essa ordem apareceria
literal no `pipeline.py`, no `checksum.py` e em cada teste. Foi exatamente o que a primeira
versão dos testes fez (um helper `_notas(registro)` duplicando a property), e a revisão
pegou: a duplicata foi removida.

### 3.4 Tabela incompleta devolve `{}`, e não uma tabela parcial

`montar_tabela_oficial` exige as 3 Etapas × 3 línguas + Parte II + Redação. Faltando
qualquer uma, devolve `{}` e o Edital sai sem checksum. Conferir contra tabela parcial
mediria a falta da tabela e não a qualidade do registro — e produziria justamente o "número
plausível e errado" que o pipeline inteiro existe para não produzir. `conferir_registros`
recusa `{}` com `ValueError` para que a regra não dependa de o chamador lembrar de checar.

### 3.5 A língua é gravada mesmo quando o checksum não fecha — mas marcada

Num registro que não fecha, a combinação escolhida é a menos ruim de 27 todas erradas. Ela
continua gravada (nada é descartado neste ticket, e a combinação é insumo do diagnóstico do
ticket 07), e `checksum_fecha=False` na mesma linha diz para não confiar nela.

### 3.6 `lingua_ambigua`: o limite honesto da inferência, no dado e não só na prosa

Em 25 dos 189 registros da fixture — e 965 dos 8.499 do Ed_38 inteiro (11,4%) — **mais de
uma** das 27 combinações cabe na tolerância. Nesses casos o delta é válido, mas a língua não
fica determinada. O caso extremo é a Etapa com notas zeradas: com P1 = 0, as três línguas
dão resultados próximos.

Isso começou como um parágrafo de docstring e virou coluna depois da revisão de spec, que
apontou a user story 32 ("cada linha carrega o resultado da sua própria validação, para
poder filtrar por confiança em vez de confiar cegamente"): sem a marca, o consumidor não tem
como separar uma língua determinada de um cara-ou-coroa. `lingua_ambigua` qualifica as três
colunas de língua, não o delta.

### 3.7 Determinismo: empate resolvido pela ordem do Edital

`min` sobre a lista na ordem de `_COMBINACOES` (que é a ordem de `LINGUAS_ESTRANGEIRAS`, que
é a do Edital) devolve sempre a primeira em caso de empate. Verificado na prática: duas
rodadas sobre o Ed_38 inteiro produzem CSVs idênticos byte a byte (spec, user story 35).

### 3.8 `medias_desvios=` para os triênios que publicam a tabela avulsa

Triênios diferentes publicam de formas diferentes (user story 8): 2022/2024 traz a tabela na
cauda do Resultado Final; 2016/2018 a publica num Edital avulso (Ed_32). Sem o parâmetro, o
checksum só existiria para o primeiro caso. O avulso passa por
`extrair_edital_medias_desvios` (e não por `parse_medias_desvios` direto) para que a Família
dele seja confirmada pela página 1 e os metadados saiam do próprio arquivo.

### 3.9 Triênio incompatível morre na entrada (`TrienioIncompativelError`)

Conferir um Edital contra a tabela de outro triênio não fecha **nenhum** registro, com
deltas na casa das dezenas (min. 10,3 no experimento). Isso é alto o bastante para não
passar batido — mas só depois de processar o Edital inteiro e de alguém ler o relatório.
Como os dois triênios estão impressos na página 1 dos dois Editais e saem no mesmo formato
de `schema.extrair_metadados` (`"2016/2018"`), o erro é barato de pegar na entrada, e é onde
ele morre.

Não é uma sétima camada de validação além das seis do spec: é a validação do parâmetro
`medias_desvios`, que só existe a partir deste ticket, e recusar entrada errada não é
descartar registro. Quando um dos dois triênios não pôde ser lido do PDF
(`METADADO_DESCONHECIDO`), não há o que comparar e a extração segue — o backstop nesse caso
é a magnitude do delta, que continua fixada por teste.

### 3.10 `LINGUAS_ESTRANGEIRAS` subiu para `constants.py`

Os três nomes de língua eram literais em `medias_desvios.py` (onde são gravados) e de novo
em `checksum.py` (onde são procurados de volta), com um comentário sustentando o acoplamento.
Uma grafia divergente entre os dois quebraria a busca. A constante agora é a única grafia
possível, e `medias_desvios.py` monta as três linhas de Parte 1 iterando sobre ela.

---

## 4. Medições

### 4.1 Fixture do ticket (Ed_38, páginas 1-6 + cauda, 189 registros)

| métrica | valor |
|---|---|
| fecham (`delta <= 0,005`) | **189/189 (100%)** |
| delta mediano | 0,0010 |
| delta máximo | 0,0030 |
| língua mista entre Etapas | 52/189 (27,5%) |
| fecham por Etapa e **não** com língua fixa por Aluno | 47/189 |
| língua ambígua (>1 combinação fecha) | 25/189 |

### 4.2 Ed_38 inteiro (242 páginas, 8.499 registros)

| delta | registros |
|---|---|
| = 0 | 2.309 (27,2%) |
| ≤ 0,001 | 6.004 (70,6%) |
| ≤ 0,002 | 7.882 (92,7%) |
| ≤ 0,003 | 8.441 (99,3%) |
| ≤ 0,005 | **8.499 (100,0%)** |

Língua mista entre Etapas: 1.693/8.499 (19,9%) — o protótipo mediu 17,4%. Distribuição
inferida: Inglesa ~68%, Espanhola ~31%, Francesa ~1%. Língua ambígua: 965 (11,4%).

A forma da distribuição é o que importa aqui, não só a taxa (spec, Further Notes): os deltas
estão **espalhados nas milésimas e colados no zero**, sem pilha em torno de nenhum valor
maior. É o oposto do sintoma que denunciou a fórmula incompleta no protótipo (falhas
empilhadas em torno de 0,7).

### 4.3 O achado: o reparo do ticket 02 recuperou os valores **certos**

Os 758 registros do Ed_38 que tiveram campo numérico partido por espaço reparado
(`"1 7.539"` → 17.539, `"- 21.683"` → −21.683) **fecham todos** o checksum. Isso não era
sabido: o ticket 02 sinalizou esses campos justamente porque um parse tolerante recupera
*algum* número e não há como saber, olhando só para o campo, se é o número certo. O checksum
responde por fora — é a primeira verificação cruzada entre camadas do pipeline. Fixado como
teste (`test_campos_reparados_pelo_ticket_02_fecham_o_checksum`).

### 4.4 Ed_31 (2016/2018) com a tabela avulsa: 82/84 fecham

Os 2 que não fecham (deltas 0,181 e 0,103) têm a **Etapa 1 inteira zerada** — 2 dos 3
registros nessa condição no recorte falham, contra 0 dos 81 restantes. Um deles também tem
campos sinalizados por formato, o que sugere reparo que recuperou valor errado; o outro não
tem sinal nenhum das outras camadas. É padrão para o ticket 07 agrupar e explicar, não para
este ticket descartar. Fixado como teste
(`test_registro_que_nao_fecha_continua_na_saida_com_a_marca`).

---

## 5. Achados da revisão de código incorporados

A revisão de duas frentes (`/code-review`) levantou 15 pontos; os aceitos:

| # | Achado | O que foi feito |
|---|---|---|
| Std 1 | Três nomes de língua duplicados entre `medias_desvios.py` e `checksum.py` | `LINGUAS_ESTRANGEIRAS` em `constants.py`, usada nos dois |
| Std 2 | Teste reimplementava `RegistroResultadoFinal.notas`, a property criada neste ticket para ser o lugar único | helper `_notas` removido, testes usam a property |
| Std 3 | `melhor_delta is None` e `melhores_linguas = LINGUAS` inalcançáveis e enganosos (`LINGUAS` não é atribuição por Etapa) | laço trocado por `min` sobre a lista das 27 |
| Std 4 | "mínimo entre 27" duplicado 4× entre produção e testes | contrafactuais dos testes passam por `conferir_argumento_final` |
| Std 5 | Guarda de tabela vazia em `conferir_registros` sem teste | teste adicionado |
| Std 6 | Três ramos `r.checksum is None` na mesma linha do CSV | um ramo só, as colunas saem vazias juntas |
| Std 8 | `conferir` é verbo pelado; irmãos usam `validar_sequencia_e_ordem` / `deduzir_cota_declarada` | renomeada para `conferir_argumento_final`; "cada candidata" reescrito (candidato = Aluno) |
| Spec 1 | Ambiguidade de língua não filtrável pelo consumidor | `lingua_ambigua` no registro e no CSV, com teste |
| Spec 4 | `_resumo_checksum` reportava taxa sem dispersão, contra o alerta explícito do spec | passou a reportar delta mediano e máximo |
| Spec 5 | `argumento_final_recalculado` gravado e nunca lido | teste fixa que o delta é a distância entre recalculado e impresso |
| Spec 6 | Comentário da tolerância descrevia arredondamento por parcela, que `calculate_argument_final` não faz | comentário corrigido (o arredondamento é dos operandos publicados) |
| Spec 7 | `fecha` lê o delta arredondado | documentado na property |

Recusados, com motivo:

- **Língua como `Enum` em vez de `str`** (Std 7): `medias_desvios.RegistroMediasDesvios.lingua_estrangeira` já é `str` (ticket 03); trocar aqui criaria duas representações da mesma coisa no mesmo pacote. Consistência com o ticket vizinho vence.
- **Não gravar a língua quando o checksum não fecha** (Spec 2): ver 3.5 — nada é descartado neste ticket, e `checksum_fecha` já marca.
- **`TrienioIncompativelError` como escopo excedido** (Spec 3): ver 3.9 — é validação do parâmetro novo, não uma camada sobre registros.

---

## 6. Como foi verificado

```bash
.venv/bin/python -m pytest tests/test_pas_extraction_checksum.py -q   # 22 passed
.venv/bin/python -m pytest tests/ -q                                  # 134 passed, 2 failed
```

As 2 falhas são **anteriores a este ticket e não relacionadas**:
`test_pas_intelligence.py::TestTargetCalculator::test_guaranteed_scenario` (carrega
`.joblib` e quebra em `No module named '_loss'`, incompatibilidade de versão de scikit-learn
neste ambiente) e `test_pdf_gen_manual.py::test_pdf_gen` (caminho absoluto de Windows
hardcoded). Nenhuma toca `pas_extraction`.

Cobertura dos critérios de aceite, um teste por critério:

| Critério | Teste |
|---|---|
| delta gravado por registro | `test_todo_registro_recebe_delta_e_linguas`, `test_o_recalculado_bate_com_o_impresso_dentro_da_tolerancia` |
| reusa `argument_calculator` | `test_o_valor_recalculado_e_gravado_junto_do_delta` + ausência de qualquer peso em `pas_extraction` |
| língua inferida pelo fechamento | `test_a_lingua_gravada_e_a_que_minimiza_o_delta`, `test_a_lingua_de_cada_etapa_e_gravada_por_aluno` |
| por Etapa, não por Aluno | `test_inferir_por_aluno_descartaria_registros_que_fecham_por_etapa`, `test_ha_alunos_que_trocam_de_lingua_entre_etapas` |
| tolerância `<= 0,005` | `test_a_tolerancia_aplicada_e_0_005` (inclusive na borda) |
| nada é descartado | `test_a_contagem_e_a_mesma_com_e_sem_tabela_oficial`, `test_registro_que_nao_fecha_continua_na_saida_com_a_marca` |

Verificações fora da suíte:

- Rodada completa sobre o Ed_38 real: 8.499/8.499 fecham, 29s.
- Determinismo: duas rodadas → CSVs idênticos byte a byte (`cmp`).
- Corrupção simulada: +1,000 em `eb_p2_e1` de um registro que fechava → delta 0,047, não fecha.
- Caminho de erro do CLI: `--medias-desvios` de outro triênio → mensagem clara e `exit 1`, sem escrever CSV.

### Fixtures novas (gitignored — contêm dado real de Aluno)

- `tests/fixtures/resultado_final_com_checksum.pdf` — páginas 1-6 **+ 242** do Ed_38. Nenhuma fixture anterior tinha registros e tabela no mesmo arquivo.
- `tests/fixtures/medias_desvios_avulso_2016_2018.pdf` — página 1 do Ed_32 (2016/2018), o triênio da fixture `resultado_final_cota_suspeita.pdf`. A avulsa que já existia é do ED_34 (2019/2021), triênio que não casa com nenhuma fixture de Resultado Final.

Os comandos de geração estão nos comentários das constantes correspondentes no teste.

---

## 7. Escopo deliberadamente fora deste ticket

- **Relatório de validação agrupado por padrão** — ticket 07. Este ticket produz o delta por registro; agrupar as falhas e explicar cada padrão (inclusive os 2 registros de Etapa 1 zerada da seção 4.4) é lá.
- **Descartar registro que não fecha** — proibido pelo spec sem padrão explicado, e o ticket diz "o checksum só marca".
- **Casar automaticamente cada Edital com o avulso do triênio dele** — o `--medias-desvios` vale para a rodada inteira e avisa quando há mais de um Edital. A rodada completa dos 77 Editais é o ticket 08.
- **Investigar por que Etapa zerada não fecha** — padrão identificado e medido, diagnóstico é do ticket 07.
- **Coluna com `argumento_final_recalculado`** — o ticket pede o delta na linha; o valor recalculado fica no objeto, para quem for consumir pela API.

---

## 8. Glossário — termos necessários para entender este relatório

- **Argumento Final** — nota ponderada acumulada das 3 Etapas que o UnB usa para ranquear. `AF = 1×AP1 + 2×AP2 + 3×AP3`, onde `APn = argumento(P1) + argumento(P2) + argumento(Redação)` e `argumento(x) = ((x − média) / desvio) × peso`, com `PESO_P1=0,72`, `PESO_P2=8,28`, `PESO_REDACAO=1,00`.
- **Checksum do Argumento Final** — recalcular o AF a partir das 9 notas brutas + a tabela oficial, e comparar com o impresso. Verifica 12 campos com um número.
- **Delta** — `|recalculado − impresso|`. Fecha quando `<= 0,005`.
- **Tolerância (0,005)** — não é folga arbitrária: todos os operandos do cálculo são publicados com 3 casas decimais, e esse arredondamento se propaga em milésimos quando o valor é recomposto.
- **Etapa** — uma das 3 provas anuais do PAS (PAS 1, 2, 3). Não confundir com **Parte** (Parte 1 = língua estrangeira; Parte 2 = demais disciplinas).
- **Parte 1** — a única parte com média e desvio publicados **por língua estrangeira**, e por isso a única por onde a língua pode ser inferida.
- **Língua por Etapa** — a língua estrangeira que o Aluno fez em cada Etapa. Não está impressa no Edital; é recuperada pela combinação que minimiza o delta. ~20% dos Alunos trocam de língua entre Etapas.
- **Língua ambígua** — registro em que mais de uma das 27 combinações fecha: o delta vale, a língua não fica determinada.
- **Tabela oficial (Médias e Desvios)** — a tabela de média e desvio-padrão por Etapa × prova publicada pelo Cebraspe, na cauda do Resultado Final ou num Edital avulso (ticket 03).
- **Família de Edital** — um dos três formatos (Resultado Final, Convocação, Médias e Desvios), determinada pelo schema que o próprio Edital declara.
- **Triênio** — o ciclo de 3 anos de uma turma do PAS (ex.: `2022/2024`). A tabela oficial é específica do triênio.

---

## 9. Onde continuar

- **Ticket 07** — relatório de validação por padrão, com distribuição de deltas. Tem agora o insumo por registro (`delta`, `argumento_final_recalculado`, `lingua_ambigua`) e dois padrões já identificados para agrupar: Etapa 1 zerada, e campo reparado pelo ticket 02.
- **Ticket 08** — rodada completa determinística sobre os 77 Editais. Precisa resolver o casamento automático entre cada Resultado Final e a tabela do triênio dele (hoje manual via `--medias-desvios`); `TrienioIncompativelError` já dá o critério de casamento correto.
- **Ticket 10** — Notas de Corte por Sistema de Concorrência. Passa a poder filtrar por `checksum_fecha` antes de derivar a nota, em vez de confiar em toda linha.
