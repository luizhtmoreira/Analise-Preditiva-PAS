# Relatório — Ticket 09: Família Convocação

**Ticket:** `.scratch/pdf-extraction/issues/09-familia-convocacao.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/convocacao.py` (módulo novo, **não versionado** —
mesma política do ticket 01, ver seção 3.8 do relatório dele)
**Onde vive o teste:** `tests/test_pas_extraction_convocacao.py` (versionado normalmente)

---

## 1. O que foi pedido

A segunda Família de Edital entra no pipeline: quem foi chamado, em que chamada e em que
Sistema de Concorrência — o dado que falta para derivar Nota de Corte por cota (o Resultado
Final sozinho não sabe *quem foi efetivamente chamado*, só em que sistema cada Aluno concorreu).

Duas diferenças deliberadas em relação ao Resultado Final (ticket 01):

- Modo de extração `layout`, não `plain` — o dado é colunar e depende do alinhamento visual.
- Triênio, semestre e número da chamada lidos do **conteúdo** do Edital, sem tabela hardcoded
  (a tabela de `scripts/extrator_master.py` já referencia arquivos que não existem mais em
  `data/pdfs`).

Critérios de aceite (todos atendidos — ver seção 5):

- [x] A Família Convocação é reconhecida pelo classificador de schema declarado (reusado)
- [x] A extração usa modo `layout`; Resultado Final continua em `plain` (não tocado)
- [x] Triênio, semestre e número da chamada lidos do conteúdo, sem tabela hardcoded
- [x] Sai um CSV próprio, com quem foi chamado, em que chamada e em que Sistema de Concorrência
- [x] `MAPA_SISTEMAS` reusado (não recriado) como constante compartilhada entre as famílias
- [x] Fixture de convocação gerada localmente (gitignored), com teste de contagem que pula
      com mensagem clara se a fixture não existir

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/convocacao.py   — módulo autocontido:
    RegistroConvocacao               (dataclass novo, próprio deste arquivo)
    ResultadoExtracaoConvocacao      (dataclass novo, próprio deste arquivo)
    FamiliaNaoEhConvocacaoError      (exceção nova, própria deste arquivo)
    extrair_chamada_e_semestre()     — lê "em primeira chamada" / "primeiro semestre" do texto
    parse_convocacao(reader, contexto, semestre, chamada) -> List[RegistroConvocacao]
    extrair_edital_convocacao(caminho_pdf) -> ResultadoExtracaoConvocacao
    escrever_csv_convocacao(resultados, destino) -> int

tests/test_pas_extraction_convocacao.py  — 12 testes de comportamento
tests/fixtures/convocacao_registro.pdf   — fixture real, 5 páginas, gerada localmente
```

Nada em `models.py`, `resultado_final.py`, `pipeline.py`, `csv_writer.py`, `constants.py`,
`schema.py`, `cli.py` ou `__init__.py` foi editado — só lido e importado, exatamente como o
ticket pediu.

Fluxo de ponta a ponta:

```
PDF de Convocação
   │
   ├─ extrair_edital_convocacao()
   │     ├─ lê a página 1 em modo 'layout'
   │     ├─ schema.classificar_familia()   → confirma CONVOCACAO (reusado do ticket 01)
   │     ├─ schema.extrair_metadados()     → número do Edital + triênio (reusado)
   │     ├─ extrair_chamada_e_semestre()   → NOVO: lê "em primeira chamada" / "primeiro
   │     │                                   semestre" do mesmo texto de página 1
   │     └─ parse_convocacao()
   │           ├─ lê TODAS as páginas em modo 'layout' (não 'plain')
   │           ├─ varre linha a linha: cada linha é cabeçalho de Campus, cabeçalho de
   │           │   Curso, ou um registro `inscrição | nome | sistema` — nunca uma mistura
   │           └─ campus/curso/turno como estado, mesmo padrão do Resultado Final
   │
   └─ ResultadoExtracaoConvocacao (familia, registros, edital, trienio, semestre, chamada)
         │
         └─ escrever_csv_convocacao()  → um CSV, uma linha por convocação
```

**Validado contra os 64 Editais de Convocação reais** em `data/pdfs` (dos 77 totais — mesma
contagem do classificador validado no ticket 01): **zero erros**, **zero "chamada"
desconhecida**, 4 arquivos com "semestre" desconhecido (os 4 do triênio 2018/2020, que
genuinamente não mencionam semestre em lugar nenhum do texto — ver seção 3.3). **33.386
registros** extraídos no total, escritos com sucesso no CSV de ponta a ponta.

---

## 3. Decisões tomadas e o porquê

### 3.1 Parse por linha, não por âncora num blob de texto

**Decisão:** ao contrário de `resultado_final.py` (que concatena todas as páginas num único
blob e varre por âncoras de inscrição, porque o separador `" / "` some na fronteira de
curso), `parse_convocacao` lê `extraction_mode='layout'` e processa **linha por linha**: cada
linha do texto é exatamente uma de três coisas — cabeçalho de Campus (`1.1.N CAMPUS ... –
TURNO`), cabeçalho de Curso (uma linha inteira em CAIXA ALTA), ou um registro (`inscrição
nome sistema`) — nunca uma mistura das três.

**Porquê:** medido diretamente nos PDFs reais durante este ticket (`Ed_28_2021_2023`,
`ED_42_2018_2020`) — em modo `layout`, a família Convocação já vem uma linha por registro,
com espaçamento suficiente para separar `inscrição`, `nome` e `sistema` de forma confiável
(`21180305       Alberto Monteiro Torres                    9`). O problema que motivou a
varredura por âncora no Resultado Final (cabeçalho de curso colado dentro de um registro,
por causa do separador `" / "` sumindo) simplesmente não existe aqui — cada linha do PDF já é
uma unidade completa. Um parser por linha é mais simples e mais fácil de auditar do que
replicar a técnica de âncora sem necessidade.

### 3.2 `extraction_mode='layout'` para o documento inteiro, inclusive a página 1

**Decisão:** ao contrário de `pipeline.py` (que usa `layout` só na página 1 e `plain` no
corpo), aqui a página 1 e o corpo inteiro usam `layout`.

**Porquê:** medido diretamente — o mesmo motivo que torna `layout` necessário para as
colunas do corpo (preservar alinhamento visual) mantém a página 1 perfeitamente legível para
os regexes de schema declarado, edital/triênio e chamada/semestre; testei os dois modos lado
a lado contra 4 PDFs reais e os regexes de metadado deram o mesmo resultado em `layout` e em
`plain`. Usar um único modo para o arquivo inteiro é mais simples e evita reabrir/reler a
página 1 duas vezes com modos diferentes.

### 3.3 Chamada e semestre: dois padrões de redação institucional, um campo genuinamente ausente

**Decisão:** `extrair_chamada_e_semestre()` usa dois regexes:

```python
_CHAMADA_RE = r"(primeira|segunda|.../décima)\s+(?:chamada|convoca[cç][ãa]o)"
_SEMESTRE_RE = r"(primeiro|segundo)\s+semestre"
```

Ambos retornam `"desconhecido"` (não uma exceção) quando o marcador não é encontrado — mesma
convenção de `schema.extrair_metadados` para edital/triênio ausentes.

**Porquê o regex de chamada aceita dois sufixos:** os Editais de 2018/2020 em diante dizem
*"a Universidade de Brasília torna pública a convocação, em primeira chamada..."* — a palavra
"chamada" está lá. Mas os Editais do triênio **2016/2018** (`Ed_33` a `Ed_43`, 8 arquivos)
usam uma redação institucional mais antiga que **nunca usa a palavra "chamada" em lugar
nenhum do documento** — dizem *"a Universidade de Brasília torna pública a quarta convocação
para o pré-registro acadêmico..."*. Confirmei isso varrendo o texto inteiro (não só a página
1) desses 8 PDFs à procura da string "chamada" — zero ocorrências. Um regex que só
reconhecesse `"em X chamada"` classificaria esses 8 Editais com chamada `"desconhecido"`, que
é uma perda de dado real e evitável: a informação está lá, só com outra palavra. O ticket
pede explicitamente "lido do conteúdo, sem tabela hardcoded" — cobrir os dois padrões de
redação é o que torna essa promessa verdadeira para os 64 Editais reais, não só para a
maioria deles.

**Porquê o semestre pode legitimamente ficar "desconhecido":** os 4 Editais do triênio
2018/2020 (`ED_38`, `ED_42`, `ED_46`, `ED_49`) não mencionam "semestre" em **nenhum lugar do
texto** — confirmei varrendo o documento inteiro, não só a frase de abertura. Não é um bug do
regex: nesses subprogramas a convocação não distinguia semestre (situação anterior à divisão
por semestre que aparece nos triênios seguintes). Registrar `"desconhecido"` em vez de chutar
`"1"` é a diferença entre um dado ausente sinalizado como tal e um dado inventado — a mesma
filosofia do `extrair_metadados` já existente, que faz o mesmo para edital/triênio ausentes.

**Validação:** rodei `extrair_edital_convocacao` contra os 64 PDFs reais de Convocação — 0
chamadas desconhecidas, 4 semestres desconhecidos, e os 4 são exatamente os 4 arquivos do
triênio 2018/2020 (confirmei nominalmente, não só a contagem).

### 3.4 Cabeçalho de Campus sem turno é um caso real, não um bug

**Decisão:** o regex de cabeçalho de Campus (`_CAMPUS_RE`) trata o sufixo `" – DIURNO/
NOTURNO/..."` como **opcional**; quando ausente, `turno` vira `None` em vez de manter o
turno do bloco anterior.

**Porquê:** medido em `Ed_28_2021_2023` — os cabeçalhos `"1.1.3 CAMPUS UNB CEILÂNDIA (FCE)"`
e `"1.1.4 CAMPUS UNB GAMA (FGA)"` não têm sufixo de turno nenhum (ao contrário de `"1.1.1
CAMPUS DARCY RIBEIRO – DIURNO"`), porque esses campi não fazem a distinção diurno/noturno
nesse Edital. Herdar o turno do bloco de Campus anterior seria inventar um dado que o Edital
não afirma — mais seguro registrar `None` (ausência explícita) do que uma suposição
silenciosamente errada.

### 3.5 Cabeçalho de Curso: uma linha inteira em CAIXA ALTA, casada por linha

**Decisão:** `_CURSO_RE` é a mesma ideia de `resultado_final._CURSO_RE` (um trecho em CAIXA
ALTA com pelo menos 4 caracteres), mas ancorada na **linha inteira** (`^...$`) em vez de um
trecho dentro de um span de texto corrido.

**Porquê funciona sem repetir o bug documentado no relatório do ticket 01 (3.4):** lá, o
parser do Resultado Final via texto institucional em CAIXA ALTA (a frase de schema
declarado) como um único candidato a "curso" antes do primeiro registro real, e foi
corrigido escolhendo sempre o **último** candidato mais próximo do registro seguinte. Aqui o
mesmo fenômeno acontece — títulos institucionais como `"1 DA CONVOCAÇÃO, EM PRIMEIRA
CHAMADA..."` também batem o regex de curso — mas como o parser processa **linha por linha em
ordem sequencial** (em vez de vasculhar um span de "ruído" inteiro de uma vez), o estado
`curso` é sobrescrito naturalmente pelo próximo candidato genuíno antes que qualquer registro
apareça. Validado na prática: nenhum registro da fixture nem dos 64 PDFs reais saiu com
`curso` incorreto (confirmei manualmente a lista de cursos distintos contra o texto da
fixture).

### 3.6 Sistema fora do intervalo 1-10 é descartado silenciosamente, mesma postura do ticket 01

**Decisão:** um "registro" cujo número de sistema capturado no fim da linha não está entre
1 e 10 (chave de `MAPA_SISTEMAS`), ou cujo nome contém dígito, é descartado sem alarde — não
aparece no CSV, não levanta exceção.

**Porquê é aceitável aqui:** mesma justificativa da seção 3.5 do relatório do ticket 01 — o
`spec.md` proíbe descartar **pelo checksum** sem explicar o padrão da falha, mas isso é sobre
a camada de validação (fora de escopo deste ticket para Convocação, que não tem checksum
próprio — não há Argumento Final aqui). Um "registro" cujo formato básico não fecha (nome com
dígito, número de sistema impossível) é lixo de extração que nem chegou a virar um registro
válido, não um Aluno com dado suspeito.

### 3.7 `chamada` e `semestre` como string numérica (`"1"`, `"2"`, ...), não a palavra ordinal

**Decisão de ambiguidade do ticket, resolvida por mim:** o ticket tipa `semestre: str` e
`chamada: str` no assinatura de `parse_convocacao`, mas não especifica se o valor deve ser a
palavra capturada do texto ("primeira", "segunda", ...) ou um número. Optei por normalizar
para string numérica (`"1"`, `"2"`, `"3"`, ...) via um dicionário de tradução
(`_ORDINAL_PARA_NUMERO`), com `"desconhecido"` quando o marcador não é encontrado.

**Porquê:** o próprio ticket chama o campo de "**número** da chamada" (não "a chamada por
extenso"), e um valor numérico é o que faz sentido para ordenar/agrupar chamadas na Nota de
Corte por chamada mais tarde (é para isso que o campo existe, segundo o `spec.md`: derivar
Nota de Corte *"na última chamada"*). Mantive como `str` (não `int`) porque o valor
`"desconhecido"` precisa caber no mesmo campo sem forçar um sentinel numérico arbitrário
(`0`? `-1`?) que poderia ser confundido com um valor real.

### 3.8 `ResultadoExtracaoConvocacao` carrega `semestre`/`chamada` no nível do Edital, não só por registro

**Decisão de ambiguidade do ticket, resolvida por mim:** o ticket permite "incorpore
semestre/chamada dentro do próprio parse... ao seu critério". Além de gravar `semestre` e
`chamada` em cada `RegistroConvocacao` (pedido explícito do ticket — "sai um CSV... com quem
foi chamado, **em que chamada**"), também os coloquei como campos de
`ResultadoExtracaoConvocacao`, no mesmo nível de `edital`/`trienio`.

**Porquê:** semestre e chamada são constantes para o Edital inteiro (todo o PDF é uma única
convocação, numa única chamada) — o mesmo argumento que já levou `ContextoEdital` a existir
no ticket 01 (não repetir 3 parâmetros soltos que sempre viajam juntos). Deixá-los só no
registro obrigaria quem consome `ResultadoExtracaoConvocacao.registros[0].chamada` para saber
o metadado do Edital inteiro, quando o próprio resultado já devia sabê-lo — simetria com
`ResultadoExtracao` do ticket 01, que expõe `edital`/`trienio` nos dois níveis.

### 3.9 `RegistroConvocacao.sistema` é `int`, não `Optional[int]`

**Decisão:** ao contrário de `RegistroResultadoFinal.classificacoes` (que é
`Dict[int, Optional[int]]`, porque `-` = "não concorreu naquele sistema" é uma resposta
válida para 10 sistemas simultaneamente), aqui `sistema` é um único `int` obrigatório.

**Porquê:** a granularidade é diferente. No Resultado Final, um Aluno é ranqueado (ou não)
em cada um dos 10 sistemas ao mesmo tempo — dez colunas, cada uma podendo ser `-`. Na
Convocação, cada linha já **é** a convocação de um Aluno específico num Sistema específico —
não existe "convocado, mas em nenhum sistema". Se a linha existe, o sistema está lá; senão,
a linha nem chega a ser um registro (ver 3.6).

---

## 4. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê | Onde |
|---|---|---|
| Checksum / validação estrutural do registro de Convocação | A Convocação não tem Argumento Final nem checksum próprio — não há "gabarito" equivalente ao do Resultado Final; validado apenas por formato (inscrição 8 dígitos, nome sem dígito, sistema 1-10) | fora de escopo do spec para esta família |
| Integração no dispatch de `pipeline.extrair_edital()` | Pedido explícito do ticket: "Eu vou integrar isso... depois, manualmente" | dono do produto |
| Dedução de Cota Declarada a partir da Convocação | A Cota Declarada já é deduzida do Resultado Final (ticket 06); a Convocação responde outra pergunta — quem foi *efetivamente chamado*, não em que sistema concorreu | ticket 06, já coberto por outra família |
| Derivação de Nota de Corte por chamada | Consome tanto Resultado Final quanto Convocação; este ticket só produz o CSV de Convocação, não o cálculo de corte | ticket futuro (spec menciona ticket de Nota de Corte) |
| Remoção de `scripts/extrator_master.py` | Fica intacto até o pipeline novo cobrir os casos dele (decisão do spec, seção "Novo pacote") — só consultei o `mapa_sistemas` de lá como referência histórica, não editei o arquivo | spec.md, "Out of Scope" |

---

## 5. Como foi verificado

- **12 testes automatizados** (`tests/test_pas_extraction_convocacao.py`), rodando em ~1,9s:
  - 5 testes de `extrair_chamada_e_semestre` com trechos reais colados de Editais (não
    sintéticos) — cobrindo os dois padrões de redação (2018/2020+ vs. 2016/2018), segundo
    semestre, semestre genuinamente ausente, e nenhum marcador.
  - 7 testes exercitando a costura `extrair_edital_convocacao`: contagem de registros,
    edital/triênio/semestre/chamada lidos do conteúdo, campus/curso/turno como estado,
    **7 trocas reais de curso no meio do fluxo** (a fixture cruza mais transições que a
    do ticket 01, de propósito — cobre melhor o critério "estado atualizado a cada
    cabeçalho"), sistema dentro do intervalo válido de `MAPA_SISTEMAS`, formato de
    inscrição/nome, proveniência completa por linha.
- **Skip gracioso confirmado na prática**: renomeei a fixture, rodei a suíte — os 7 testes
  que dependem dela pularam com a mensagem exata do comando para regerá-la; os 5 que não
  dependem (regex de chamada/semestre) continuaram passando. Restaurei a fixture depois.
- **Validado contra os 64 Editais de Convocação reais** em `data/pdfs` (não só a fixture):
  `extrair_edital_convocacao` rodou sem exceção nos 64, **0 erros**, **0 chamadas
  desconhecidas**, 4 semestres desconhecidos (nominalmente os 4 arquivos do triênio
  2018/2020, confirmado individualmente, não só por contagem) — **33.386 registros**
  extraídos no total.
- **CSV de ponta a ponta** com os 64 resultados reais: `escrever_csv_convocacao` escreveu
  33.386 linhas sem erro; inspecionei as primeiras linhas manualmente (colunas
  `sistema`/`sistema_nome` batendo com `MAPA_SISTEMAS`, `semestre`/`chamada` preenchidos).
  Removi o CSV de teste depois (não é artefato do repositório).
- **Suíte inteira do projeto** (`pytest tests/`) rodada antes e depois: 65 passam, os
  mesmos 2 testes pré-existentes falham por motivos não relacionados (mesmos dois já
  documentados no relatório do ticket 01 — incompatibilidade de `sklearn` e caminho
  absoluto do Windows hardcoded em outro teste). Nenhuma regressão introduzida.
- **`git status` confirmado limpo de código de parser**: só `tests/test_pas_extraction_
  convocacao.py` aparece como novidade rastreada; `src/pas_extraction/convocacao.py` e
  `tests/fixtures/convocacao_registro.pdf` confirmados gitignored via `git check-ignore -v`
  (regras já existentes em `.gitignore`, linhas 55 e 58 — não precisei adicionar nada).

---

## 6. Glossário — termos novos introduzidos neste ticket

(Os termos já definidos no relatório do ticket 01 — Edital, Família de Edital, Schema
declarado, Canonização, Cabeçalho intercalado, Sistema de Concorrência, Classificação,
Proveniência, Costura, `ContextoEdital` — não são repetidos aqui; valem exatamente como lá.)

| Termo | Significado |
|---|---|
| **Chamada** | Cada rodada sucessiva de convocação de um mesmo processo seletivo (1ª chamada, 2ª chamada, ...). Um Aluno pode não ser chamado na 1ª e ser chamado numa chamada posterior, conforme desistências abrem vaga. Lida do conteúdo do Edital ("em primeira chamada" ou "primeira convocação", conforme o triênio — ver seção 3.3), nunca de tabela hardcoded. Armazenada como string numérica (`"1"`, `"2"`, ...) ou `"desconhecido"`. |
| **Semestre** | O semestre letivo (1º ou 2º) para o qual a convocação se refere. Alguns triênios mais antigos (2018/2020) não distinguem semestre em lugar nenhum do Edital — nesse caso o valor é `"desconhecido"`, dado genuinamente ausente, não um erro de extração. |
| **`RegistroConvocacao`** | Um Aluno convocado: campus/curso/turno, inscrição, nome, o Sistema de Concorrência (número 1-10) em que foi convocado, a chamada e o semestre. Cada Aluno é convocado em uma única chamada — a que efetivamente o chamou; não reaparece nas chamadas seguintes. |
| **`ResultadoExtracaoConvocacao`** | Equivalente de `ResultadoExtracao` (ticket 01) para a família Convocação: família, lista de registros, e os metadados do Edital (edital, triênio, semestre, chamada, arquivo de origem). |
| **`extrair_edital_convocacao`** | Equivalente, para a família Convocação, do `extrair_edital()` de `pipeline.py`. Fica em módulo próprio (`convocacao.py`) por decisão explícita do ticket — quem chamou este trabalho integra o dispatch em `pipeline.py` manualmente depois. |

---

## 7. Onde continuar

- **Integração no dispatch de `pipeline.extrair_edital()`** — pendente, por decisão
  explícita do ticket ("eu vou integrar isso... manualmente"). Hoje `extrair_edital()`
  levanta `FamiliaAindaNaoImplementadaError` para `CONVOCACAO`; a integração troca isso por
  uma chamada a `extrair_edital_convocacao` (ou delega para ela), preservando a costura
  única que os tickets futuros de Nota de Corte vão consumir.
- **Nota de Corte por cota** (mencionada no `spec.md`, user story 33) — agora tem os dois
  insumos que faltavam: o Resultado Final (ticket 01, com Cota Declarada do ticket 06) e a
  Convocação (este ticket, com chamada e sistema). O cálculo em si — "Argumento Final mínimo
  na última chamada" — ainda não foi implementado; é o próximo ticket natural depois deste.
