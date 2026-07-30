# Relatório — Ticket 14: Validação de formato do campo de classificação

**Ticket:** `.scratch/pdf-extraction/issues/14-validacao-formato-classificacao.md`
**Status:** concluído
**Onde vive o código novo:** `src/pas_extraction/resultado_final.py` (não versionado — mesma
política do ticket 01), `src/pas_extraction/models.py` (comentário atualizado);
`src/pas_extraction/notas_corte.py` recebeu um ajuste de escopo estendido (ver seção 3.2 —
decisão tomada com o dono do produto durante a implementação, não estava no ticket original).
**Onde vivem os testes:** classe nova em `tests/test_pas_extraction.py`
(`TestValidacaoFormatoClassificacao`), classe nova em `tests/test_pas_extraction_notas_corte.py`
(`TestChecksumSuspeitoNaEscolhaDoCorte`).

---

## 1. O que foi pedido

O campo de classificação (o ranking do Aluno em cada Sistema de Concorrência) não passava
pela mesma validação de formato exato que os 9 campos numéricos de nota (`_formato_numerico_valido`,
ticket 02) — era só `_WS.sub("", v)` seguido de `int()`. Achado originalmente no ticket 08: um
dígito colado por espaço interno produz uma posição implausível; no corpus real isso já havia
gerado uma posição de 6 dígitos (Edital 36, 2017/2019, MEDICINA) que, sem limite de
plausibilidade, explodia o CSV de saída para 6,4 GB. O ticket 08 mitigou o *sintoma* (limite em
`_buracos_por_sistema`), mas não sinalizava a causa no próprio registro.

Critérios de aceite (todos atendidos — ver seção 5):

- [x] Existe uma validação de formato para o campo de classificação, simétrica a
      `_formato_numerico_valido`, aplicada em `_montar_registro`
- [x] Um valor de classificação com formato inválido marca `campos_formato_invalido` nesse
      registro específico
- [x] O caso real conhecido (Edital 36, 2017/2019, MEDICINA) é reproduzido numa fixture
      sintética de teste e sai marcado como formato inválido
- [x] O limite de plausibilidade em `_buracos_por_sistema` (ticket 08) continua funcionando sem
      regressão, como segunda camada de defesa
- [x] Rodando sobre o corpus real, o outlier de Nota de Corte (MEDICINA, Darcy Ribeiro,
      2020/2022, corte=199.162,872) fica corretamente marcado como suspeito — ver seção 3.2
      para por que "some do CSV" não se aplica a este caso específico

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/resultado_final.py
    _FORMATO_CLASSIFICACAO_RE            — `^-$|^\d+$`, novo
    _formato_classificacao_valido(v)     — novo, simétrico a _formato_numerico_valido
    _montar_registro                     — o loop de classificações agora sinaliza
                                            "classificacao_N" (N = 1..10, ordem de
                                            MAPA_SISTEMAS) em campos_formato_invalido,
                                            junto dos 9 campos numéricos + argumento_final

src/pas_extraction/models.py
    ValidacaoRegistro.campos_formato_invalido — docstring atualizado para cobrir as
                                                 10 classificações, não só as 10 notas

src/pas_extraction/notas_corte.py            — extensão de escopo (seção 3.2):
    DerivacaoNotasCorte.convocados_com_checksum_suspeito_substituidos — novo campo
    derivar_notas_corte                   — na escolha do menor Argumento Final entre os
                                             convocados da maior chamada, um candidato com
                                             checksum.fecha=False só é usado quando NÃO há
                                             alternativa confiável (fecha=True ou não
                                             conferido) na mesma chamada
```

Fluxo de ponta a ponta (rodada completa, 83 PDFs de `data/pdfs`):

```
resultado_final.py (novo)          notas_corte.py (novo, seção 3.2)
   │                                   │
   ├─ 91 registros com classificacao_N ├─ 376 convocados com checksum suspeito
   │  sinalizado (79+8+3+1, 4 padrões) │  substituídos por alternativa confiável
   │  (relatório de validação)         │  (log da rodada)
   │                                   │
   └───────────┬───────────────────────┘
               │
   3 linhas de notas_corte.csv permanecem com |nota_corte| > 200 — todas com
   convocados_na_chamada=1 (sem alternativa possível), todas com checksum_fecha=False
```

---

## 3. Decisões tomadas e o porquê

### 3.1 Formato exato do campo de classificação: `^-$|^\d+$`, não `^-?\d+\.\d{3}$`

A classificação é um ranking (inteiro, sem casa decimal) ou `"-"` ("não concorreu neste
Sistema") — não é o mesmo formato dos 9 campos numéricos de nota, que sempre têm exatamente 3
casas decimais. `_formato_classificacao_valido` compara o texto bruto (só espaço de borda
removido) contra esse formato, do mesmo jeito que `_formato_numerico_valido` faz para os campos
numéricos: pega dígito partido por espaço interno (`"2 80852"` → recuperado para `280852` via
`_WS.sub`, mas o formato bruto não bate e o campo é sinalizado como `classificacao_N`) sem
descartar o registro — o valor recuperado continua indo para a saída, só que marcado.

Rodando sobre o corpus real, a validação pegou **91 registros reais** em 4 padrões distintos
(`classificacao_1`: 79, `classificacao_9`: 8, `classificacao_5`: 3, `classificacao_2`: 1) — não
é um caso hipotético, é sinal real que estava presente e invisível até agora.

**Teste do caso conhecido:** como não há PDF real disponível reproduzindo exatamente o dígito
colado do Edital 36 (a mecânica exata de como ele acontece nunca foi determinada — ver ticket
08, "Segue como abertura"), a fixture é sintética: chama `resultado_final._montar_registro`
diretamente com uma lista de 22 campos fabricados, incluindo `campos[12] = "2 80852"` (o mesmo
valor implausível, ~280 mil, já usado como massa de teste em `TestLimiteDePlausibilidadeDosBuracos`
do ticket 08) — mesmo padrão sintético que aquele ticket já usa para exercitar
`validacao.py` sem depender de PDF real.

### 3.2 Por que o outlier de Nota de Corte ganhou um ajuste em `notas_corte.py`, fora do escopo original do ticket

A evidência do ticket ("Edital 36... nota_corte = 199.162,872") foi investigada rodando o
corpus real. **O campo corrompido não é a classificação — é o `argumento_final`** desse
registro (`RegistroResultadoFinal.argumento_final=199162.872`, já sinalizado em
`campos_formato_invalido=("argumento_final",)` pelo ticket 02, e com `checksum.fecha=False`,
delta ≈ 198999.997). A validação deste ticket (campo de classificação) estruturalmente não
tem como alcançar esse valor — são campos diferentes do registro.

Investigando por que o valor absurdo ainda definia a Nota de Corte apesar de já vir marcado
como suspeito: `derivar_notas_corte` escolhia o **menor** Argumento Final entre os convocados
da maior chamada, sem olhar para o checksum — e neste (curso, Sistema, chamada) havia **um
único convocado** (`convocados_na_chamada=1`), então o candidato corrompido vencia por não
ter concorrente algum, não por ser genuinamente o menor válido.

Combinado com o dono do produto: como a checkbox pede "some do CSV **ou** fica corretamente
marcado como suspeito", e o CSV já expõe `checksum_fecha` para justamente esse filtro, a decisão
foi **implementar a exclusão preferencial de candidatos suspeitos, não excluir a linha em
silêncio quando não sobra alternativa** — descartar o corte inteiro neste caso violaria a
mesma postura anti-descarte-silencioso que já rege o resto do módulo (`DerivacaoNotasCorte`
conta em vez de omitir). Concretamente:

- Ao montar os candidatos de uma chamada, um convocado com `checksum.fecha is False` (não
  `None` — `None` é "não conferido", não "reprovado") vai para uma lista `suspeitos` à parte.
- Se sobrar pelo menos um candidato confiável na mesma chamada, os suspeitos são **excluídos**
  da escolha do menor Argumento Final — mesmo que o suspeito tivesse o menor número.
- Se **não** sobrar nenhum candidato confiável (caso do outlier: 1 convocado, suspeito), a
  escolha cai de volta para os suspeitos — a linha continua saindo (não é descartada), e
  `checksum_fecha=False` permanece visível a quem consumir o CSV.
- Um contador novo (`convocados_com_checksum_suspeito_substituidos`, por triênio) torna a
  substituição auditável — não é um efeito colateral silencioso da escolha do `min()`.

Rodando sobre o corpus real após o ajuste: **376 convocados** tiveram seu Argumento Final
suspeito substituído por uma alternativa confiável na mesma chamada. Sobraram exatamente **3**
linhas com `|nota_corte| > 200` em todo o `notas_corte.csv` — as três com
`convocados_na_chamada=1` (nenhuma alternativa possível), a do ticket entre elas
(MEDICINA/Darcy Ribeiro/2020-2022/Universal), todas com `checksum_fecha=False`. É o caso que a
checkbox previa como alternativa aceitável ("marcado como suspeito"), agora garantido por
código explícito em vez de coincidência de estado anterior ao ticket.

### 3.3 O limite de plausibilidade do ticket 08 continua intacto, como segunda camada

`validacao._buracos_por_sistema` (o limite `3× observado + 50`) não foi tocado por este
ticket — os testes de `TestLimiteDePlausibilidadeDosBuracos` (ticket 08, hoje recuperados em
`tests/test_pas_extraction.py`) continuam passando sem alteração. As duas camadas fazem
trabalhos diferentes e complementares: a validação deste ticket sinaliza o **registro
específico** com formato bruto suspeito (mesmo quando o valor "parece" plausível); o limite do
ticket 08 neutraliza o efeito de qualquer posição implausível — sinalizada ou não — no cômputo
de buracos por curso/Sistema, incluindo casos que passariam pelo formato bruto (ex.: um dígito
duplicado sem espaço, `"280852"` sem corrupção de espaço nenhuma, que a validação deste ticket
não pega porque o texto bruto já bate `^\d+$`).

---

## 4. Recuperação de `tests/test_pas_extraction.py`

Este arquivo (e os demais `test_pas_extraction_*.py`) existia no histórico do projeto mas
sumiu deste worktree — só sobrava o `.pyc` compilado. Confirmado com o dono do produto: os
worktrees paralelos `pas-ticket-01`/`02`/`03` (branches de outra iniciativa, sem relação de
nome com estes tickets) ainda tinham cópias íntegras. `resultado_final.py`, `validacao.py` e
`models.py` eram **byte a byte idênticos** entre este worktree e `pas-ticket-02` antes desta
mudança — confirmado por `diff` antes de copiar — então `tests/test_pas_extraction.py` e
`tests/test_pas_extraction_notas_corte.py` foram trazidos de lá, intactos, como ponto de
partida para as classes novas deste ticket.

---

## 5. Testes

`tests/test_pas_extraction.py` — 5 testes novos em `TestValidacaoFormatoClassificacao`:
partido por espaço e sinalizado; sem corrupção não sinaliza; `"-"` continua válido; duas
classificações partidas na mesma linha são sinalizadas juntas; regressão nomeada do caso real
(Edital 36/MEDICINA, posição de 6 dígitos).

`tests/test_pas_extraction_notas_corte.py` — 4 testes novos em
`TestChecksumSuspeitoNaEscolhaDoCorte`: candidato suspeito perde para alternativa confiável
mesmo tendo o menor número; a substituição é contada por triênio; sem alternativa nada é
contado como substituído (o corte ainda sai, só que suspeito); `checksum=None` não é tratado
como suspeito (não perde para um confiável só por não ter sido conferido).

Suíte completa (`pytest tests/`): 129 testes, 127 passam. As 2 falhas restantes são
pré-existentes e não relacionadas a este ticket (`test_pdf_gen_manual.py`, caminho absoluto do
Windows hardcoded; `test_pas_intelligence.py::TestTargetCalculator`, módulo de calculadora de
metas do app Streamlit, sem relação com o pipeline de extração).
