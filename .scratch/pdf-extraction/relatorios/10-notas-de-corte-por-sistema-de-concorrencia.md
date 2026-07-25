# Relatório — Ticket 10: Notas de Corte por curso e por Sistema de Concorrência

**Ticket:** `.scratch/pdf-extraction/issues/10-notas-de-corte-por-sistema-de-concorrencia.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/notas_corte.py` (módulo novo, **não versionado** —
mesma política dos tickets 01/09), mais alterações em `convocacao.py`, `rodada.py` e `cli.py`
**Onde vivem os testes:** `tests/test_pas_extraction_notas_corte.py` (41 testes, novo),
`tests/test_pas_extraction_convocacao.py` (+8) e `tests/test_pas_extraction_rodada.py` (+3) —
todos versionados normalmente

---

## 1. O que foi pedido

A Nota de Corte de cada curso deixa de ser um número único e passa a ser um número **por
Sistema de Concorrência** — hoje um Aluno que concorre por Cota para Negros é comparado
contra um corte Universal que não é o dele.

Algoritmo definido pelo dono do produto em 2026-07-24: para cada (curso, Sistema),
identifica-se primeiro a **maior chamada em que houve convocado naquele sistema**; a Nota de
Corte é o **menor** Argumento Final entre os convocados daquele sistema **nessa** chamada —
não a média, não o maior. Um sistema sem convocado na última chamada cai para a anterior.

Critérios de aceite (todos atendidos — ver seção 6):

- [x] Sai uma Nota de Corte por curso e por Sistema de Concorrência, não uma por curso
- [x] Para cada (curso, sistema), a maior chamada com convocado naquele sistema é identificada primeiro
- [x] O corte é o menor Argumento Final entre os convocados daquele sistema nessa maior chamada
- [x] Um curso/sistema sem convocado numa chamada mais recente cai para a anterior mais recente que teve
- [x] A saída é CSV com proveniência, sem carga em banco e sem alteração no app
- [x] Um teste verifica o corte derivado com dado conhecido, incluindo empate/múltiplos Alunos na maior chamada

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/notas_corte.py   — módulo novo:
    ChaveCorte                        (NamedTuple: trienio, semestre, campus, curso, turno, sistema)
    NotaCorte                         (o corte + como ele saiu)
    DerivacaoNotasCorte               (os cortes + o que não deu para derivar, contado)
    derivar_notas_corte(rf, conv)     — o algoritmo
    escrever_csv_notas_corte()        — CSV com proveniência das duas Famílias
    formatar_terminal_notas_corte()   — resumo de cobertura

src/pas_extraction/convocacao.py    — CORRIGIDO (ver 3.1 e 3.2):
    LISTA_CONVOCACAO / LISTA_OUTRA_RELACAO, _classificar_secao(), _rotulo_de_curso()
    RegistroConvocacao.lista (campo novo, com default)

src/pas_extraction/rodada.py        — derivar_notas_corte_da_rodada() +
                                      _registros_das_duas_familias() (extraído)
src/pas_extraction/cli.py           — subcomando `notas-corte`, 4º CSV na `rodada`
```

Fluxo:

```
Resultado Final (ticket 01)        Convocação (ticket 09)
  Argumento Final de cada Aluno      quem foi chamado, em que chamada, em que Sistema
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
            derivar_notas_corte()
              ├─ descarta o que não é convocação desta chamada (3.1)
              ├─ agrupa por (triênio, semestre, campus, curso, turno, sistema)
              ├─ maior chamada COM CONVOCADO NAQUELE SISTEMA
              ├─ busca o Argumento Final por (triênio, inscrição)
              └─ menor Argumento Final entre os daquela chamada
                       ▼
              notas_corte.csv (21 colunas, proveniência das duas Famílias)
```

---

## 3. Decisões tomadas e o porquê

### 3.1 Dois Editais de Convocação publicam **duas** listas de Alunos — e só uma é convocação

**O defeito, encontrado neste ticket:** um Edital de Convocação pode ter duas seções de topo
com lista de Aluno. A seção 1 é a convocação em si (*"1 DA TERCEIRA CONVOCAÇÃO PARA O
PRÉ-REGISTRO ACADÊMICO..."*); a seção 2, quando existe, é outra coisa: *"2 DA RELAÇÃO DOS
CANDIDATOS QUE NÃO COMPARECERAM AO PRÉ-REGISTRO ACADÊMICO"*. O parser do ticket 09 não
distinguia as duas — jogava tudo em `RegistroConvocacao` com a mesma chamada.

**Por que isso é fatal para a Nota de Corte, e não só cosmético:** quem não compareceu foi
convocado numa chamada **anterior** (é a desistência dele que abre a vaga desta chamada).
Contá-lo como convocado desta chamada erra o corte de duas maneiras, e as duas na mesma
direção — para baixo:

1. ele entra na disputa do "menor Argumento Final desta chamada" com uma nota que não é
   desta chamada;
2. pior, ele pode **criar** uma chamada que aquele sistema nunca teve — se o sistema 2 só
   teve convocado até a 1ª chamada, mas aparece na relação de ausentes de um Edital de 3ª
   chamada, o algoritmo passa a achar que a maior chamada daquele sistema é a 3ª.

**Medido no corpus real:** 1.353 registros (4,1% dos 33.386 da Família Convocação) vêm da
segunda lista, em 7 dos 64 Editais — todos do triênio 2016/2018.

**Decisão:** corrigir na origem, em `convocacao.py`, e não filtrar por gambiarra em
`notas_corte.py`. O parser passa a reconhecer o cabeçalho de seção de topo (o único cujo
número **não tem ponto**: `1`, `2`, ... contra `1.1.1 CAMPUS` e `1.1.1.1 MEDICINA`) e grava
em cada registro de qual lista ele veio (`RegistroConvocacao.lista`, coluna nova no CSV de
Convocação). Nada é descartado no parser — as duas listas continuam saindo, agora
distinguíveis; quem descarta é a derivação da Nota de Corte, que é quem tem motivo para isso,
e ela **conta** o que descartou (`DerivacaoNotasCorte.registros_de_outras_listas`).

**Como a seção é classificada — regra negativa, não positiva.** Levantei os títulos de seção
de topo dos 64 Editais. A regra positiva óbvia ("é convocação se o título diz *convocação*")
funciona no corpus atual, mas erra para o lado destrutivo em dois casos plausíveis, e foi
trocada por marcar o que é *outra* relação (`ausente`, `naocomparec`, `naotiveram`
canonizados — as quatro redações observadas):

1. **Título quebrado em duas linhas.** Os títulos são longos e o modo `layout` preserva a
   quebra do PDF. Se "CONVOCAÇÃO" cair na segunda linha, a regra positiva marcaria o Edital
   inteiro como outra relação e **todos os seus convocados sumiriam dos cortes**.
2. **"AUSENTES NA PRIMEIRA CONVOCAÇÃO".** Os Editais de 2016/2018 — justamente os 7 que têm
   duas seções — escrevem "convocação" onde os outros escrevem "chamada" (é o achado 3.3 do
   relatório do ticket 09). Um título assim é redação plausível, e a regra positiva o
   classificaria exatamente ao contrário.

Com a regra negativa, os dois casos caem para o lado conservador, que é o mesmo do default:
**registro sem seção reconhecida conta como convocação**. Se um cabeçalho não for reconhecido,
o parser volta ao comportamento anterior ao ticket 10 (contaminação conhecida) em vez de apagar
convocados. Entre os dois modos de errar, repetir o problema conhecido é preferível a perder o
dado — e o contador da seção 3.7 torna o silêncio visível de qualquer forma. Os dois casos
estão fixados como teste de regressão.

### 3.2 O rótulo de curso é o nome do curso, não a posição dele no sumário

**O defeito, também encontrado neste ticket:** nos Editais até 2020, os cabeçalhos de curso
vêm numerados (`1.1.1.1 ADMINISTRAÇÃO (BACHARELADO)`), e o parser do ticket 09 guardava a
numeração dentro do rótulo. A numeração é posição naquele Edital, não identidade do curso:

- o mesmo curso aparece como `1.1.1.1 ADMINISTRAÇÃO` na seção 1 e `2.1.1.1 ADMINISTRAÇÃO` na
  seção 2 do **mesmo** Edital;
- entre Editais do mesmo triênio ela muda com a lista de cursos daquela chamada — CIÊNCIAS
  BIOLÓGICAS é `1.1.1.10` num Edital e `1.1.1.11` no seguinte.

**Por que quebra este ticket:** o agrupamento da Nota de Corte é por curso. Com a numeração
no rótulo, o mesmo curso vira grupos diferentes, cada um com sua "maior chamada" — que é
exatamente o passo 1 do algoritmo. Medido: 172 cursos com rótulo variando só na numeração
dentro do mesmo (triênio, semestre, campus, turno), e **5.747 chaves de corte que deveriam
ser 4.790** — quase mil cortes eram duplicatas do mesmo curso partido.

**Decisão:** `_rotulo_de_curso()` remove a numeração de sumário no parser (de novo, na
origem). Os Editais de 2021/2023 em diante não numeram esses cabeçalhos, e o rótulo deles
passa intacto — confirmado pelos testes já existentes do ticket 09, que esperam
`"ADMINISTRAÇÃO (BACHARELADO)"` sem número e continuaram verdes sem alteração.

### 3.3 A chave do corte tem seis dimensões, não duas

O ticket fala em "por curso e por Sistema de Concorrência". As outras quatro dimensões de
`ChaveCorte` não são refinamento opcional — são o que faz "curso" ser bem definido:

- **triênio e semestre:** cada processo seletivo tem o seu corte. Juntar o corte de MEDICINA
  de 2016/2018 com o de 2021/2023 num número só não descreve nenhum dos dois. (`semestre`
  pode ser `"desconhecido"` — os 4 Editais do triênio 2018/2020, ver ticket 09 — e continua
  sendo um valor de chave como outro qualquer, porque é constante para aquele triênio.)
- **campus e turno:** a UnB oferta o mesmo nome de curso em campi e turnos diferentes, com
  concorrência diferente, e é assim que o próprio Edital os separa em cabeçalhos.

### 3.4 O Argumento Final é buscado por (triênio, inscrição), não por inscrição

**Decisão:** o índice do Resultado Final é chaveado por `(triênio, inscrição)`.

**Porquê, medido:** **146 inscrições aparecem em mais de um triênio** no corpus (p.ex.
`22108970` em 2022/2024 e 2023/2025). Com chave só de inscrição, o corte de um curso de 2023
poderia sair com a nota que aquele número teve em outro subprograma — erro silencioso, sem
nenhum sinal no CSV. Dentro de um mesmo triênio a inscrição é única: **0 colisões** de
`(triênio, inscrição)` nos 66.313 registros de Resultado Final.

### 3.5 Empate no menor Argumento Final desempata pela inscrição

Se dois Alunos do mesmo sistema têm exatamente o mesmo Argumento Final na maior chamada, o
corte é o mesmo número de qualquer forma — o desempate escolhe apenas **qual linha** vai para
o CSV como representante. Ordenar por `(argumento_final, inscricao)` garante que duas rodadas
sobre a mesma entrada produzam o mesmo CSV byte a byte (spec, user story 35).

### 3.6 Corte parcial sai marcado, não suprimido

Um convocado pode não ter Argumento Final conhecido — o Resultado Final daquele triênio pode
não estar no corpus. Nesse caso o corte sai do que dá para calcular, com
`convocados_na_chamada` / `convocados_com_argumento` / `parcial` na própria linha.

**Porquê marcar em vez de suprimir:** um corte parcial só pode estar **alto demais** — a nota
que faltou pode ser menor que a menor encontrada, nunca maior. O número continua sendo a
melhor resposta disponível; quem consome precisa saber que é um limite superior. Suprimir a
linha trocaria um dado qualificado por um buraco indistinguível de "curso não existe".

Medido: **95 dos 5.747 cortes** são parciais, com 118 convocados sem Argumento Final no total.

### 3.7 O que não vira corte sai contado, nunca em silêncio

`DerivacaoNotasCorte` carrega cinco diagnósticos além das notas, cada um respondendo a uma
pergunta diferente de "o que não virou corte":

| Campo | O que responde |
|---|---|
| `convocados_sem_argumento_final` | quantos convocados da maior chamada não tinham nota, por triênio |
| `grupos_sem_argumento_final` | **quais** (curso, sistema) ficaram sem corte por isso — a chave, não só a contagem |
| `grupos_sem_chamada_conhecida` | quais grupos não têm nenhuma chamada numérica |
| `registros_de_outras_listas` | quantos registros vieram da seção 2 do Edital (3.1) |
| `inscricoes_com_argumento_ambiguo` | (triênio, inscrição) repetida no Resultado Final **com nota diferente** |

É a mesma postura do spec (user story 23) contra descarte silencioso, e aqui ela importa mais
do que nas outras camadas: um curso/sistema que simplesmente não aparece no CSV é
indistinguível de um curso que não existe — por isso `grupos_sem_argumento_final` guarda a
`ChaveCorte` inteira, e não um número.

Os três últimos também servem de alarme de regressão, e são 0 ou constantes hoje: se
`registros_de_outras_listas` cair a zero numa rodada futura, o parser parou de reconhecer a
seção 2 — não os Editais mudaram. `inscricoes_com_argumento_ambiguo` conta só as duplicatas
que **mudam o resultado** (mesma chave, notas diferentes); duplicata com a mesma nota não
altera corte nenhum e é problema de outra camada (ticket 02).

### 3.8 O checksum do registro que define o corte acompanha a linha

Cada corte carrega `checksum_fecha` do registro de Resultado Final que o define: `True`,
`False`, ou vazio quando não houve tabela de médias e desvios para conferir (a mesma distinção
entre "reprovado" e "não verificado" que `RegistroResultadoFinal.checksum` já faz). Medido:
**293 dos 5.747 cortes** são definidos por um registro cujo checksum não fecha — não estão
escondidos nem descartados, estão marcados, e quem quiser um subconjunto de alta confiança
filtra por essa coluna (spec, user story 32).

### 3.9 O comando `rodada` escreve o 4º CSV; `notas-corte` existe para quem só quer esse

`rodada` já tinha as duas Famílias em memória — derivar as Notas de Corte ali não custa
releitura de PDF nenhuma, então ela passa a escrever `notas_corte.csv` junto dos outros três.
O subcomando `notas-corte` existe para quem quer só esse CSV, e avisa em `stderr` quando usado
com `--limit`: um corte derivado de um subconjunto do corpus pode ser o de outra chamada, e o
CSV não tem como sinalizar isso sozinho.

---

## 4. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê |
|---|---|
| Carga em Supabase / consumo no app | Pedido explícito do ticket: "carregar isso em Supabase ou consumir no app está fora de escopo" |
| Substituir as Notas de Corte atuais do projeto | O ticket produz o CSV; trocar a fonte que o app usa é decisão de produto posterior |
| Validação de formato do campo de classificação (follow-up do ticket 08) | Continua pendente, e continua sendo escopo do parser do ticket 01/02 — ver relatório do ticket 08, §3.3 |
| Reconciliar nome entre as duas listas do mesmo Edital | A 6ª camada (ticket 08) já compara nomes por inscrição entre Editais, e agora enxerga as duas listas; separar as métricas por lista não foi pedido |

---

## 5. Alterações em módulos de tickets anteriores

`convocacao.py` (ticket 09) foi alterado — as duas correções da seção 3.1 e 3.2. É fora do
escopo declarado do ticket 10, e foi feito assim de propósito: as duas são defeitos de
**extração**, e corrigi-los na derivação significaria carregar para sempre um filtro que
depende de um artefato de parse (o prefixo numérico) para adivinhar o que o parser já sabia
e jogou fora. Mesma postura do ticket 08 ao encontrar um defeito do ticket 01/02, com uma
diferença: lá o root-cause fix ficou pendente porque o sintoma podia ser contido na camada do
próprio ticket; aqui não podia — sem a marca de lista, `notas_corte.py` não tem como saber
quem foi convocado.

**Consequência para números já publicados:** os 33.386 registros de Convocação do relatório do
ticket 09 continuam sendo 33.386 (nada foi descartado), mas agora 1.353 deles estão marcados
como `outra_relacao`. O CSV de Convocação ganhou a coluna `lista`.

---

## 6. Como foi verificado

- **41 testes** em `tests/test_pas_extraction_notas_corte.py`, escritos antes do módulo
  (vermelhos primeiro), organizados pela regra de negócio que exercitam e não pelo método
  que chamam:
  - maior chamada primeiro, inclusive `"10" > "9"` (ordenação numérica, não textual)
  - menor Argumento Final entre os da maior chamada — explicitamente não a média, não o maior
  - queda para a chamada anterior quando o sistema não tem convocado na última
  - as seis dimensões da chave, uma a uma
  - convocado sem Argumento Final, grupo inteiro sem, chamada desconhecida, outra lista
  - determinismo, não-mutação da entrada, e o CSV
- **+8 testes** em `tests/test_pas_extraction_convocacao.py` para as correções do parser,
  com texto colado de Edital real (Ed_35) através de um reader falso — a fixture real é de um
  Edital de seção única, e o caso sob teste exige um de duas seções.
- **+3 testes** em `tests/test_pas_extraction_rodada.py` para a costura.
- **Suíte inteira**: 235 passam; as mesmas 2 falhas pré-existentes e não relacionadas
  (incompatibilidade de `sklearn` no `TargetCalculator` e caminho absoluto de Windows em
  `test_pdf_gen_manual.py`), já documentadas desde o relatório do ticket 01.
- **Corpus real (77 Editais)** — ver seção 7.

**Limitação conhecida, e por quê:** não existe teste ponta a ponta que produza uma Nota de
Corte a partir de PDFs. Ele exigiria duas fixtures do **mesmo triênio** com inscrições que se
cruzam (uma de Resultado Final, uma de Convocação), e fixtures de Edital real carregam dado de
prova de Aluno identificável — a restrição de privacidade que impede commitar qualquer uma
delas (ver [[project_parser_privacy]] e o relatório do ticket 01, 3.8). Um teste assim pularia
para todo mundo menos quem tem `data/pdfs` local. A regra de derivação está coberta por 41
testes sintéticos; a ponta a ponta está verificada pela rodada sobre os 77 Editais reais
documentada na seção 7, que é reproduzível por quem tem o corpus com um comando só.

**Rodada de `/code-review` (dois eixos, Standards e Spec):** 12 achados, 9 corrigidos —
constante morta (`_DESCONHECIDO`), seis properties que só repassavam (`NotaCorte` achatou as
seis dimensões da chave via `**chave._asdict()`), duplicação entre `_cmd_rodada` e
`_cmd_notas_corte` (extraído `_rodar_descobertos`), `--limit` sem aviso no caminho da `rodada`,
`--out-dir` e o help anunciando três CSVs, `rodada` sem imprimir os contadores, grupo sem corte
só contado e não identificado, colisão de `(triênio, inscrição)` silenciosa, e a fragilidade da
regra positiva de classificação de seção (3.1). Não acatados: trocar `lista: str` por `Enum`
(os campos vizinhos do mesmo dataclass — `semestre`, `chamada` — são strings pela mesma razão,
e o valor vai direto para o CSV) e a falta do teste ponta a ponta (limitação acima).
`CONTEXT.md` foi atualizado: a definição de Nota de Corte não mencionava Sistema de
Concorrência.

---

## 7. Números do corpus real

Rodada completa sobre os 77 Editais em `data/pdfs`, com as correções da seção 3.1 e 3.2
aplicadas — 66.313 registros de Resultado Final e 33.386 de Convocação, **0 Editais
ignorados**.

| Medida | Valor |
|---|---|
| Notas de Corte derivadas | **4.786** (curso × Sistema de Concorrência) |
| Cortes parciais | 77 (1,6%) |
| Convocados sem Argumento Final na maior chamada | 99, em 8 triênios |
| (curso, sistema) sem corte derivável | 17 |
| Grupos sem chamada conhecida | **0** |
| Inscrições com Argumento Final ambíguo | **0** |
| Registros da Família Convocação, por lista | 32.033 `convocacao` + 1.353 `outra_relacao` |
| Cortes por triênio | 2016/2018: 803 · 2017/2019: 785 · 2018/2020: 396 · 2019/2021: 652 · 2020/2022: 581 · 2021/2023: 585 · 2022/2024: 497 · 2023/2025: 487 |
| Checksum do registro definidor | 4.559 fecham · 227 não fecham · 0 não conferidos |
| Determinismo | duas derivações sobre a mesma entrada, idênticas |

**O critério de aceite mais difícil acontece o tempo todo no dado real:** **2.663 dos 4.786
cortes** (56%) têm chamada anterior à maior chamada do próprio curso — ou seja, na maioria dos
casos o sistema *não* teve convocado na última chamada do curso e o algoritmo caiu para uma
anterior. Se a maior chamada fosse calculada por curso em vez de por sistema, mais da metade
dos cortes sairia errado. Exemplo real (ADMINISTRAÇÃO, 2016/2018, Darcy Ribeiro diurno, 1º
semestre): o curso foi até a 3ª chamada, mas o sistema 2 parou na 2ª e os sistemas 5, 7 e 9
pararam na 1ª — quatro cortes, quatro chamadas diferentes, um curso só.

**O caso de empate/múltiplos também é o caso comum:** 2.646 cortes têm mais de um convocado
do mesmo sistema na maior chamada (máximo observado: 169). "Menor, não média nem maior" não é
uma sutileza de borda — decide mais da metade das linhas.

**Efeito medido das duas correções da seção 5**, comparando a mesma rodada antes e depois:

| | Antes | Depois |
|---|---|---|
| Notas de Corte | 5.747 | **4.786** |
| Cursos com rótulo variando só na numeração | 172 | **0** |
| Registros de convocação com curso numerado | 6.357 | **0** |
| Registros de outra lista contados como convocados | 1.353 | **0** |
| Cortes parciais | 95 | 77 |

Os ~960 cortes a menos não são dado perdido: eram o mesmo curso contado duas ou mais vezes,
uma por variante de numeração, cada cópia com sua "maior chamada" calculada sobre uma fatia
dos convocados.

**Um risco levantado no review e medido como não-ocorrente:** `semestre` entra na chave e é
lido da página 1; se um triênio tivesse Editais com semestre declarado e outros com
`"desconhecido"`, os convocados do mesmo curso cairiam em grupos diferentes e a "maior chamada"
sairia de uma fatia. Medido nos 64 Editais: **nenhum triênio mistura** — 2018/2020 é
inteiramente `"desconhecido"` (os 4 Editais que genuinamente não mencionam semestre, achado 3.3
do ticket 09) e os outros sete triênios são inteiramente `['1', '2']`. O risco é real em
princípio, mas hoje não tem nenhuma ocorrência no corpus; se aparecer, o sintoma é uma queda
brusca em `convocados_na_chamada`.

---

## 8. Glossário — termos novos introduzidos neste ticket

(Os termos dos relatórios anteriores — Edital, Família, Sistema de Concorrência, Chamada,
Proveniência, Cota Declarada, Checksum — valem como lá.)

| Termo | Significado |
|---|---|
| **Nota de Corte** | O menor Argumento Final entre os Alunos convocados num curso e num Sistema de Concorrência, na **maior chamada em que aquele sistema teve convocado**. Não é o menor Argumento Final do curso, nem o do sistema em qualquer chamada. |
| **Lista do Edital** (`RegistroConvocacao.lista`) | Qual das seções de topo do Edital de Convocação produziu o registro: `convocacao` (chamado nesta chamada) ou `outra_relacao` (ausente/não compareceu/não homologado em chamada anterior). Só a primeira é convocação. |
| **`ChaveCorte`** | As seis dimensões que identificam um corte: triênio, semestre, campus, curso, turno, sistema. |
| **Corte parcial** | Corte em que nem todos os convocados da maior chamada tinham Argumento Final conhecido. É sempre um **limite superior** do corte verdadeiro. |
| **Corte definidor** | O Aluno cujo Argumento Final é o corte. Gravado por inscrição e nome, para auditar a linha de volta ao PDF. |

---

## 9. Onde continuar

- **Decidir se o CSV vira a fonte das Notas de Corte do app** — hoje o app usa Notas de Corte
  que não distinguem sistema; este CSV distingue. A troca é decisão de produto (o ticket a
  colocou fora de escopo).
- **Follow-up herdado do ticket 08:** validação de formato do campo de classificação no parser
  do Resultado Final, ainda pendente.
- **Cortes com `checksum_fecha=False`:** 293 cortes são definidos por um registro cujo
  Argumento Final não fecha. Vale decidir, como produto, se eles entram no dado publicado ou
  se o corte deve cair para o próximo Aluno cujo checksum fecha.
