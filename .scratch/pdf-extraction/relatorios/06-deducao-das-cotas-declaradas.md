# Relatório — Ticket 06: Dedução das Cotas Declaradas

**Ticket:** `.scratch/pdf-extraction/issues/06-deducao-das-cotas-declaradas.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/cotas.py` (novo), com pontos de costura em
`models.py`, `resultado_final.py` e `csv_writer.py` — pacote gitignored, ver seção "Por que o
código não está no git" do relatório do ticket 01
**Onde vive o teste:** `tests/test_pas_extraction.py`, classe `TestCotaDeclarada`

---

## 1. O que foi pedido

Deduzir o perfil de cotas de cada Aluno a partir do padrão de preenchimento das 10
classificações — um dado que o Edital não imprime em lugar nenhum, mas que está lá
implicitamente. As 10 classificações **são** os Sistemas de Concorrência; quatro atributos
binários (escola pública, renda ≤1,5 SM per capita, PPI, PcD) geram os 9 sistemas de cota, e os
sistemas são **aninhados, não exclusivos** — a cascata de remanejamento da Lei 12.711. O Aluno é
ranqueado em todos os subsistemas que subsome, e seus atributos são os do subsistema **mais
específico** em que aparece.

Critérios de aceite (todos atendidos — ver seção 5):

- [x] As seis colunas derivadas são gravadas por Aluno, junto das 10 classificações cruas
- [x] Os quatro atributos vêm do subsistema mais específico em que o Aluno aparece
- [x] Todos os Alunos não eliminados recebem perfil de cota, não só os aprovados
- [x] O campo é nomeado *cota declarada* em código, CSV e docs — nunca *cota elegível*
- [x] Padrão que não seja fecho para baixo do reticulado é sinalizado como suspeito, e não descartado
- [x] Um teste verifica o perfil deduzido de um Aluno com padrão conhecido

---

## 2. Decisões tomadas e o porquê

### 2.1 O reticulado foi implementado como especificado, só renumerado

O ticket traz `EP_ATTRS` com chaves 2–9 ("os índices na ordem em que as classificações
aparecem"), que é a indexação 0-based do protótipo. O resto deste pacote numera os Sistemas de
1 a 10 (`constants.MAPA_SISTEMAS`, herdado de `scripts/extrator_master.py`, que é o que liga a
família Resultado Final à família Convocação). Manter as duas numerações vivas no mesmo processo
seria um erro esperando acontecer, então `cotas.EP_ATTRS` usa as chaves 3–10 do
`MAPA_SISTEMAS`, com a correspondência conferida item a item:

| `EP_ATTRS` do ticket | `EP_ATTRS` aqui | Nome em `MAPA_SISTEMAS` |
|---|---|---|
| `2: {R, PPI}` | `3` | EP / Baixa Renda / PPI |
| `3: {R, PPI, PcD}` | `4` | EP / Baixa Renda / PPI / PcD |
| `4: {R}` | `5` | EP / Baixa Renda / Não-PPI |
| `5: {R, PcD}` | `6` | EP / Baixa Renda / Não-PPI / PcD |
| `6: {PPI}` | `7` | EP / Alta Renda / PPI |
| `7: {PPI, PcD}` | `8` | EP / Alta Renda / PPI / PcD |
| `8: {}` | `9` | EP / Alta Renda / Não-PPI |
| `9: {PcD}` | `10` | EP / Alta Renda / Não-PPI / PcD |

O modelo em si (quais atributos cada subsistema exige, e o fecho para baixo como critério de
validade) não foi redesenhado — é literalmente o do ticket. "Escola pública" não aparece em
`EP_ATTRS` porque é exigida pelos oito: é o que os define como subsistemas de Escola Pública, e
o que varia entre eles são os outros três atributos.

### 2.2 A união dos atributos observados *é* o "subsistema mais específico"

O ticket pede os atributos do subsistema mais específico em que o Aluno aparece. A
implementação usa a união dos atributos exigidos por todos os subsistemas observados. Não é uma
troca de modelo: os 8 subsistemas cobrem exatamente os 8 subconjuntos de {Renda, PPI, PcD}, então
num padrão que é fecho a união é, por construção, a exigência do maior elemento — o mais
específico. A união foi escolhida porque continua definida também no padrão **corrompido** (que
não é fecho e por isso pode não ter um elemento máximo), e ali ela é a leitura mais defensável:
não perde nenhum atributo que o Edital afirmou.

### 2.3 `perfil_cota` é o nome do Sistema, tirado do `MAPA_SISTEMAS`

O ticket lista `perfil_cota` entre as colunas sem fixar o formato. Optei pelo nome já canônico
do Sistema mais específico (`"EP / Baixa Renda / PPI"`, `"Cota para Negros"`, `"Universal"`) em
vez de inventar um código novo (`"EP-R-PPI"`) — o vocabulário já existe em `MAPA_SISTEMAS`, é o
mesmo que a família Convocação usa, e um CSV lido por humano não precisa de legenda. As quatro
colunas booleanas continuam sendo a forma de filtrar programaticamente; `perfil_cota` é o rótulo.

O nome sai do reticulado (`EP_ATTRS`), não da interseção com o que foi declarado, porque num
padrão corrompido a união de atributos pode não corresponder a nenhum subsistema observado —
o caso `{1, 3}` do teste: os atributos são {Renda, PPI} e quem os exige é justamente o 3.

### 2.4 A dedução acontece no parse, não numa passada posterior

`deduzir_cota_declarada` é chamada em `resultado_final._montar_registro`, junto com a montagem
do registro — e não numa passada sobre a lista inteira, como `validacao.validar_sequencia_e_ordem`
(ticket 02). O motivo é a dependência de dados: a cota depende só das 10 classificações **deste**
registro, enquanto buracos de sequência e ordem alfabética só existem com o curso inteiro em mãos.
Fazer no parse deixa `cota_declarada` sempre preenchida, sem estado intermediário `None` — o que
é também o que garante o critério "todos os Alunos não eliminados recebem perfil de cota": não há
caminho no código em que um `RegistroResultadoFinal` exista sem cota.

### 2.5 "Cota declarada", nunca "cota elegível" — e onde isso aparece

O nome está no módulo (`cotas.py`), na classe (`CotaDeclarada`), no campo do registro
(`cota_declarada`), na função (`deduzir_cota_declarada`), na coluna de suspeita
(`cota_padrao_suspeito`) e nas docstrings, que explicam o porquê no ponto de uso: para os 67,3%
de Alunos que aparecem só no Universal (medido, ver 4.1 — o ticket estimava 71%) é impossível
distinguir quem não tem direito de quem tem e optou por não usar. O dado registra a opção, não a
elegibilidade. As seis colunas do CSV mantêm os nomes exatos que o ticket pede (`sistema_negros`,
`escola_publica`, `renda_baixa`, `ppi`, `pcd`, `perfil_cota`) e por isso não carregam o prefixo.

### 2.6 Coocorrência Negros × Escola Pública **também** marca padrão suspeito

Esta decisão foi revista durante a revisão de código, e vale registrar as duas versões.

**Primeira decisão (descartada):** deixar `padrao_suspeito` significando exatamente uma coisa —
"o padrão não é fecho para baixo" — porque é o único critério que o ticket 06 enuncia, e ampliá-lo
seria redesenhar o modelo. A coocorrência seria só medida (1 registro em 66.313), não codificada.

**Decisão final:** `padrao_suspeito` marca as **duas** formas de padrão que a estrutura aninhada
torna impossível — não ser fecho, **ou** declarar Cota para Negros junto de subsistema de Escola
Pública. O que mudou minha cabeça foi o buraco concreto que a revisão apontou: um padrão como
`{1, 2, 3, 5, 7, 9}` (Negros + um fecho de EP perfeitamente válido) passaria **limpo** pela
checagem de fecho — o lado de EP está íntegro — e ainda por cima teria a declaração de Negros
apagada do rótulo, porque `perfil_cota` só mostra um Sistema e a precedência é de EP. Ou seja: um
registro corrompido sairia do pipeline sem marca nenhuma e com informação a menos. Isso é
exatamente "produzir lixo sem levantar erro", que o `spec.md` nomeia como *o* risco do projeto.

A ampliação também não é minha invenção: a user story 28 do `spec.md` pede que "padrões de cota
que violem a estrutura aninhada sejam sinalizados como suspeitos", e a não-coocorrência é uma
propriedade dessa estrutura, afirmada tanto no ticket quanto no `spec.md`. A checagem virou a
função `_padrao_impossivel`, com as duas condições documentadas e nomeadas.

**Custo real: zero.** Reprocessei o corpus depois da mudança e o total de suspeitos continua
**8** — o único registro com coocorrência (ver seção 4.3) já era pego pelo fecho por outro motivo.
A segunda condição não marca nenhum registro novo hoje; ela fecha um buraco que hoje ninguém
ocupa, o que é a hora certa de fechá-lo.

A precedência entre os dois no rótulo (`perfil_cota` mostra o subsistema de EP quando ambos
aparecem) só existe para esse caso degenerado, e está documentada em `_perfil`. Os quatro
booleanos + `sistema_negros` continuam mostrando as duas declarações, então nada se perde.

---

## 3. O achado principal: a checagem de fecho pegou um bug de parse real

Esta é a parte do relatório que muda algo para os próximos tickets.

O ticket previa 0 violações de fecho (o protótipo mediu 0 em 1.843 registros). No corpus inteiro
— **66.313 registros, 36× maior que o do protótipo** — apareceram **8 violações**, e todas as 8
são a mesma coisa: **não é dado do Edital, é corrupção de extração**, exatamente o que o ticket
disse que a checagem serviria para pegar.

**Causa raiz, confirmada no texto bruto:** o `pypdf` emite o número da página no **início** do
texto extraído de cada página:

```
--- pág 83 INÍCIO: '83 - / [inscrição], [nome], 5.275, 20.825, ...'
```

Quando um registro é o último da página e seu 22º campo (a 10ª classificação) só começa na página
seguinte, `_separar_registro` lê o número da página no lugar dele. O Aluno (Ed_31, página 82,
identificado no teste automatizado como o único registro suspeito da fixture — ver
`test_padrao_suspeito_real_sobrevive_na_saida_com_a_marca`) tem `-` como 10ª classificação no
Edital e saiu do parser com **83**:

```
[inscrição], [nome], ..., 103.120, 25, -, -, -, -, -, -, -, -,   ← fim da pág 82
83 - / [inscrição do próximo registro], ...                       ← início da pág 83
```

(nome e inscrição substituídos por placeholders nesta versão do relatório — são dado real de
Aluno, mesma razão pela qual as fixtures de PDF não são commitadas; a estrutura numérica que
demonstra o bug está preservada)

**Confirmação quantitativa:** nas 8 violações, o valor lido na 10ª classificação é igual ao
número da página seguinte em **8 de 8** casos. Nenhum dos 8 registros tem
`campos_formato_invalido` — as camadas de validação dos tickets 01/02 não veem nada de errado
neles, porque o valor lido é um inteiro perfeitamente bem formado. **A checagem de fecho é a única
camada que pega essa classe de corrupção.**

**Limite conhecido da detecção:** 10 registros no corpus têm a 10ª classificação igual ao número
da página seguinte; a checagem de fecho pega 8. Os outros 2 caem em padrões que continuam sendo
fecho válido (ex.: `{1, 9, 10}` é o fecho de {PcD}) e ficam invisíveis a esta camada — não dá para
distinguir, só com o fecho, um PcD genuíno de um número de página que caiu num lugar plausível.

**Não corrigi o bug neste ticket.** Ele é da camada de parse (`resultado_final._separar_registro`,
território dos tickets 01/05), a correção exige dar consciência de fronteira de página ao
recorte de campos — que hoje trabalha sobre o blob já concatenado, onde essa fronteira não
existe — e misturá-la aqui esconderia tanto o bug quanto a dedução de cota no mesmo commit.
**Recomendo um ticket de follow-up.** O comportamento atual não é silencioso: os 8 registros saem
no CSV com `cota_padrao_suspeito=True`, que é precisamente o que o ticket pediu (sinalizar, não
descartar).

---

## 4. Medições no corpus real (`data/pdfs`, 8 Editais de Resultado Final, 66.313 registros)

### 4.1 Padrões observados — 14 distintos, de 2⁹ = 512 possíveis

| Padrão (Sistemas com classificação) | Registros | % | Fecho válido? |
|---|---:|---:|---|
| `(1,)` | 44.616 | 67,3% | sim |
| `(1, 9)` | 8.149 | 12,3% | sim |
| `(1, 7, 9)` | 4.844 | 7,3% | sim |
| `(1, 2)` | 3.952 | 6,0% | sim |
| `(1, 3, 5, 7, 9)` | 2.391 | 3,6% | sim |
| `(1, 5, 9)` | 2.264 | 3,4% | sim |
| `(1, 9, 10)` | 55 | 0,1% | sim |
| `(1, 7, 8, 9, 10)` | 19 | 0,0% | sim |
| `(1, 5, 6, 9, 10)` | 9 | 0,0% | sim |
| `(1, 3, 4, 5, 6, 7, 8, 9, 10)` | 6 | 0,0% | sim |
| `(1, 10)` | 5 | 0,0% | **não** |
| `(1, 7, 9, 10)` | 1 | 0,0% | **não** |
| `(1, 2, 10)` | 1 | 0,0% | **não** |
| `(1, 3, 5, 7, 9, 10)` | 1 | 0,0% | **não** |

Os 10 padrões válidos são exatamente os 10 fechos possíveis do reticulado (as 8 combinações de
{Renda, PPI, PcD} sob Escola Pública, mais Universal puro e mais Cota para Negros) — o modelo do
ticket reproduz o dado real sem sobra nem falta. Os 4 padrões inválidos somam 8 registros, todos
explicados na seção 3, e todos envolvendo o Sistema 10 (o último campo do registro, que é o único
que pode cair na página seguinte).

### 4.2 Perfis deduzidos

| `perfil_cota` | Registros | % |
|---|---:|---:|
| Universal | 44.616 | 67,3% |
| EP / Alta Renda / Não-PPI | 8.149 | 12,3% |
| EP / Alta Renda / PPI | 4.844 | 7,3% |
| Cota para Negros | 3.952 | 6,0% |
| EP / Baixa Renda / PPI | 2.391 | 3,6% |
| EP / Baixa Renda / Não-PPI | 2.264 | 3,4% |
| EP / Alta Renda / Não-PPI / PcD | 61 | 0,1% |
| EP / Alta Renda / PPI / PcD | 20 | 0,0% |
| EP / Baixa Renda / Não-PPI / PcD | 9 | 0,0% |
| EP / Baixa Renda / PPI / PcD | 7 | 0,0% |

Os 32,7% de Alunos que declaram alguma cota (Negros ou Escola Pública) são o dado que o ticket 07
(Notas de Corte por Sistema de Concorrência) precisa e que hoje não existe em lugar nenhum do
projeto — é a lacuna nº 3 do `spec.md`.

### 4.3 Coocorrência Negros × Escola Pública

**1 registro em 66.313** (Ed_31, padrão `(1, 2, 10)`) — e é um dos 8 casos de
número de página da seção 3: o padrão real dele é `(1, 2)`, Cota para Negros puro. Descontado
esse artefato, a afirmação do ticket ("nunca coocorrem") vale para **100%** do corpus.

---

## 5. Como foi verificado

- **12 testes novos** em `TestCotaDeclarada` (`tests/test_pas_extraction.py`):
  - 7 unitários sobre `deduzir_cota_declarada`, um por forma do reticulado: só Universal;
    Cota para Negros; Escola Pública sem outro atributo (`{1,9}`); atributos vindos do
    subsistema mais específico (`{1,3,5,7,9}` ⇒ Renda + PPI, e **não** os do 9, onde o Aluno
    também aparece); fecho completo dos quatro atributos (`{1,3,4,5,6,7,8,9,10}`); e padrão que
    não é fecho (`{1,3}`) sendo sinalizado **sem** perder os atributos; e Cota para Negros
    junto de um fecho de EP válido (`{1,2,3,5,7,9}`) sendo sinalizado (seção 2.6).
  - `test_perfil_de_aluno_real_com_padrao_conhecido` — o critério de aceite do ticket: Aluno real
    de ARQUIVOLOGIA (Ed_38), selecionado por curso + tamanho do padrão (não por inscrição),
    padrão `[1,3,5,7,9]` conferido no Edital, perfil `EP / Baixa Renda / PPI` com `pcd=False`
    e `sistema_negros=False`.
  - `test_todo_aluno_nao_eliminado_recebe_cota_declarada` — os 189 registros da fixture do
    ticket 01, todos com cota, não só os aprovados.
  - `test_nenhum_padrao_real_da_fixture_viola_o_fecho` — 0 suspeitos no recorte limpo.
  - `test_padrao_suspeito_real_sobrevive_na_saida_com_a_marca` — o caso da seção 3, com fixture
    nova (ver abaixo): o registro continua na saída, marcado, e é o único marcado no recorte.
  - `test_as_seis_colunas_derivadas_saem_no_csv_junto_das_classificacoes_cruas` — primeiro teste
    de `escrever_csv` do projeto: confere as 6 colunas derivadas **e** as 10 cruas na mesma linha,
    com os valores do mesmo Aluno real de ARQUIVOLOGIA (incluindo `classificacao_sistema_2 ==
    "-"`).
- **Fixture nova** `tests/fixtures/resultado_final_cota_suspeita.pdf` (gitignored): Ed_31,
  páginas 1 + 82 + 83, geradas com `fatiar_paginas` — preserva a virada de página que produz o
  padrão não-fecho. Como todas as outras, o teste faz skip gracioso se ela não existir.
- **Corpus real inteiro**, não só as fixtures: os 8 Editais de Resultado Final de `data/pdfs`,
  66.313 registros — é de onde vêm todos os números das seções 3 e 4. As 8 violações foram
  conferidas uma a uma contra o texto bruto da página de origem, e é assim que a causa raiz foi
  identificada (não por inferência a partir do padrão).
- **Revisão de código em dois eixos** (`/code-review`, sub-agentes de Standards e Spec). O eixo
  Spec confirmou por verificação programática o ponto de maior risco desta entrega — a
  renumeração de `EP_ATTRS` bate atributo a atributo com o ticket, sem off-by-one — e achou o
  buraco da coocorrência, que virou a mudança da seção 2.6. O eixo Standards achou o
  `CotaDeclarada` faltando no `__init__.py`, o termo novo faltando no vocabulário do
  `models.py`, e uma guarda morta (`if subsistemas_ep else frozenset()`, redundante porque
  `frozenset().union()` de nada já é vazio); os três foram corrigidos.
- **Suíte completa** (`pytest tests/`): **109 passam, 2 falham**. As duas falhas são as mesmas
  pré-existentes e não relacionadas, já documentadas nos relatórios dos tickets 01 e 05
  (`test_guaranteed_scenario`, incompatibilidade de versão do `sklearn`; `test_pdf_gen`, caminho
  absoluto do Windows hardcoded no teste). Com a classe nova desselecionada, a mesma árvore dá
  **97 passam / as mesmas 2 falham** — nenhum teste que passava antes falha agora.

---

## 6. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê |
|---|---|
| Correção do número de página lido como 10ª classificação (seção 3) | Bug da camada de parse (tickets 01/05), não da dedução de cota. Exige consciência de fronteira de página em `_separar_registro`, que hoje opera sobre o blob concatenado. Recomendado como ticket de follow-up; enquanto isso os 8 registros saem marcados, não silenciosos. |
| Marcar como suspeito o registro sem classificação nenhuma (nem no Universal) | Padrão degenerado que não ocorre no corpus: todo Aluno não eliminado é ranqueado no Universal. Hoje ele sairia com `perfil_cota = "Universal"` sem marca. Cabe numa terceira condição de `_padrao_impossivel`, mas nem o ticket nem o `spec.md` o mencionam, e eu não tenho medição que justifique. Registrado como lacuna conhecida. |
| Cota nas famílias Convocação e Médias e Desvios | Convocação já traz o sistema/subsistema **impresso** (ticket 09, `convocacao.py`) — não há o que deduzir. Médias e Desvios não tem Aluno. |
| Notas de Corte por Sistema de Concorrência | Ticket 07. Este ticket entrega o insumo (as 6 colunas); o cálculo é lá. |
| Reconciliação da cota deduzida aqui com o subsistema impresso na Convocação | Camada 6 de validação do `spec.md` (reconciliação cruzada entre Editais), ticket próprio. Seria a checagem mais forte possível do modelo — vale registrar como oportunidade. |

---

## 7. Glossário — termos necessários para entender este relatório

| Termo | Significado |
|---|---|
| **Cota Declarada** | O Sistema de Concorrência em que o Aluno **optou** por concorrer, deduzido do padrão das 10 classificações. Deliberadamente distinto de *cota elegível*: para quem aparece só no Universal (67,3%) é impossível saber se não tinha direito ou se tinha e não usou. |
| **Sistema de Concorrência** | Um dos 10 sistemas em que um Aluno pode ser classificado: Universal, Cota para Negros e 8 subsistemas de Escola Pública. São as 10 últimas colunas do registro de Resultado Final. |
| **Subsistema de Escola Pública** | Um dos 8 Sistemas (3 a 10 em `MAPA_SISTEMAS`) que exigem escola pública, diferindo entre si pela combinação de renda ≤1,5 SM, PPI e PcD que exigem além dela. |
| **Sistemas aninhados** | A propriedade, vinda da cascata de remanejamento da Lei 12.711, de que os subsistemas não são exclusivos: quem é ≤1,5 SM concorre também às vagas de >1,5 SM, quem é PPI concorre também às não-PPI, PcD idem. Por isso um Aluno aparece em vários subsistemas ao mesmo tempo. |
| **Fecho para baixo (do reticulado)** | O conjunto de **todos** os subsistemas cuja exigência os atributos do Aluno satisfazem — não só o mais específico. `fecho({Renda, PPI}) = {3, 5, 7, 9}`. Um padrão de classificações válido é sempre exatamente um fecho; é essa propriedade que serve de teste de integridade. |
| **Subsistema mais específico** | O subsistema de maior exigência entre os que o Aluno subsome — de onde saem os quatro atributos dele e o rótulo `perfil_cota`. No fecho `{3, 5, 7, 9}` é o 3. |
| **PPI** | Preto, pardo ou indígena — um dos quatro atributos binários. |
| **PcD** | Pessoa com deficiência — idem. |
| **Padrão suspeito** | Marca (`cota_padrao_suspeito` no CSV) de que o conjunto de Sistemas em que o Aluno aparece é impossível pela estrutura aninhada — ou porque não é fecho para baixo (cascata da Lei 12.711), ou porque declara Cota para Negros junto de Escola Pública (sistemas que se excluem na inscrição). Indica corrupção de extração; o registro é sinalizado, nunca descartado. |
| **Reticulado (de EP_ATTRS)** | A estrutura de ordem parcial dos 8 subsistemas de Escola Pública, ordenados por inclusão dos atributos que exigem — do que não exige nada (Sistema 9) ao que exige os três (Sistema 4). |

---

## 8. Onde continuar

- **Ticket 07 (Notas de Corte por Sistema de Concorrência)** é o consumidor direto das 6 colunas
  desta entrega — é o que fecha a lacuna nº 3 do `spec.md`.
- **Follow-up recomendado (sem ticket ainda):** o número de página lido como 10ª classificação
  (seção 3). Afeta ~10 registros em 66.313 (0,015%), 8 deles já sinalizados. Baixo volume, causa
  raiz conhecida, e a fixture que reproduz o caso já existe
  (`tests/fixtures/resultado_final_cota_suspeita.pdf`).
