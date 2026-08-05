# Relatório — Ticket 13: Reparo de nome quebrado por espaço

**Ticket:** `.scratch/pdf-extraction/issues/13-reparo-nome-quebrado-por-espaco.md`
**Status:** concluído
**Onde vive o código novo:** `src/pas_extraction/nome_repair.py` (não versionado — mesma
política do ticket 01), consumido por `src/pas_extraction/resultado_final.py` e
`src/pas_extraction/convocacao.py`; `models.py`, `validacao.py`, `csv_writer.py` e
`relatorio_validacao.py` receberam ajustes para carregar e reportar o sinal.
**Onde vivem os testes:** `tests/test_pas_extraction_nome_repair.py` (novo, testa
`reparar_nome` isolado), classe nova `TestReparoDeNome` em `tests/test_pas_extraction.py`
(via `resultado_final._montar_registro`) e em `tests/test_pas_extraction_convocacao.py` (via
`parse_convocacao`).

---

## 1. O que foi pedido

2,71% dos 66.313 registros de `resultado_final.csv` tinham o nome quebrado por um espaço
espúrio no meio de uma palavra (`"Isabella"` → `"Isabell a"`) ou por espaço duplicado sem
quebra — achado comparando contra a base antiga `data/banco_alunos_pas_final.csv`. Mesma
classe de corrupção já catalogada e reparada para os 9 campos numéricos (tickets 02/04); o
campo `nome` nunca tinha recebido o equivalente.

Critérios de aceite (todos atendidos — ver seção 5):

- [x] Nome com uma palavra partida por um espaço espúrio é reparado no `resultado_final.csv`
- [x] Espaço duplicado entre palavras (sem quebra) também é normalizado
- [x] Partículas curtas legítimas (`de`, `da`, `do`, `dos`, `das`, `e`) nunca são fundidas
- [x] O reparo é sinalizado por uma coluna de proveniência no CSV, não silencioso
- [x] Teste sintético cobre os três casos (quebra real, espaço duplicado, partícula legítima)
- [x] Rodando sobre o corpus real de 77 Editais, a taxa é registrada no relatório de
      validação — **1.865/66.313 = 2,81%**, próxima do baseline conhecido (~2,71%)
- [x] Avaliado e decidido se `convocacao.py` reusa a mesma lógica — decisão: **sim** (seção 3.2)

---

## 2. Visão geral do que foi entregue

```
src/pas_extraction/nome_repair.py                    — novo
    PARTICULAS                                        — {de, da, do, dos, das, e}
    reparar_nome(nome) -> (nome_reparado, foi_reparado)

src/pas_extraction/resultado_final.py
    _montar_registro                                  — chama reparar_nome, propaga o
                                                          sinal para ValidacaoRegistro

src/pas_extraction/convocacao.py
    parse_convocacao                                  — troca _ESPACOS_RE.sub(" ", nome)
                                                          por reparar_nome(nome)
    RegistroConvocacao.nome_reparado                   — campo novo (default False)
    CSV_COLUMNS_CONVOCACAO                             — coluna nome_reparado

src/pas_extraction/models.py
    ValidacaoRegistro.nome_reparado                    — campo novo (default False)
                                                          NÃO entra em .valido (seção 3.3)

src/pas_extraction/validacao.py
    validar_sequencia_e_ordem                          — preserva nome_reparado ao
                                                          reconstruir ValidacaoRegistro

src/pas_extraction/csv_writer.py
    CSV_COLUMNS                                        — coluna nome_reparado

src/pas_extraction/relatorio_validacao.py
    RelatorioValidacao.total_nomes_reparados           — campo novo, taxa impressa em
                                                          formatar_terminal/formatar_markdown
```

Fluxo de ponta a ponta (rodada completa, 77 PDFs de `data/pdfs`):

```
resultado_final.py / convocacao.py (novo)
   │
   ├─ 1.865 nomes reparados em resultado_final.csv (2,81% de 66.313)
   │  (relatório de validação, coluna nome_reparado sinaliza cada linha)
   │
   └─ mesma lógica aplicada em convocacao.csv (33.386 registros)
```

---

## 3. Decisões tomadas e o porquê

### 3.1 A heurística: fundir token minúsculo-não-partícula no token anterior

Um token de nome próprio genuíno sempre começa maiúsculo, ou é uma das seis partículas
curtas do português (`de`, `da`, `do`, `dos`, `das`, `e`). Qualquer outro token inteiramente
minúsculo só existe porque o extrator de PDF injetou um espaço no meio de uma palavra — é o
fragmento que sobrou dela, e pertence ao token anterior (`reparados[-1] += token`). A fusão
é encadeada, não um merge de par isolado: uma palavra partida em 3+ pedaços
(`"Isa bel la"` → `"Isabella"`) fecha numa só passada, da esquerda pra direita — o que também
cobre o caso em que o próprio prefixo é uma letra maiúscula solta (`"F erreira"` →
`"Ferreira"`, achado real de `scripts/NOTES.md`, achado 9).

`" ".join(nome.split())` é o primeiro passo — já resolve "espaço duplicado sem quebra de
palavra" de graça, sem lógica própria: qualquer corrida de espaço colapsa para um antes da
fusão de fragmentos entrar em cena.

**Limitação aceita de propósito:** uma quebra que caia exatamente em cima de uma partícula
(ex.: `"Andra de"` por `"Andrade"`) fica sem reparo — fundir a partícula quebraria a maioria
dos nomes válidos (`"Maria de Souza"`, `"Sousa e Silva"`) para consertar a minoria rara que
coincide com ela. É o "cuidado central" do próprio ticket, implementado como está: nunca
fundir as seis partículas, mesmo ao custo de alguns falsos negativos raros.

### 3.2 `convocacao.py` reusa a mesma lógica — decisão tomada, e por quê

Antes deste ticket, `convocacao.py` só colapsava espaço duplicado (`_ESPACOS_RE.sub(" ", nome)`)
— resolvia a corrida de espaço que o modo `layout` injeta entre colunas, mas não a quebra de
palavra. Como a corrupção é do mesmo extrator de texto (`pypdf`) e a mesma classe de defeito
(`scripts/NOTES.md`, "ARMADILHA B(c)"), a decisão foi reusar `nome_repair.reparar_nome`
diretamente em vez de duplicar a heurística — `nome_repair.py` não depende de nada específico
de nenhuma das duas Famílias, só da string do campo `nome`.

`RegistroConvocacao` não tem um objeto `ValidacaoRegistro` próprio (é um dataclass mais simples
que o Resultado Final), então o sinal `nome_reparado` entra direto no registro, e não aninhado
— documentado no próprio código (`convocacao.py`, comentário do campo).

### 3.3 `nome_reparado` não entra em `ValidacaoRegistro.valido`

Primeira versão desta mudança também tornava `.valido` `False` quando `nome_reparado` era
`True`, por analogia com `campos_formato_invalido` (que reduz `.valido` quando um campo
numérico vem partido). Revisado: o ticket pede uma coluna "no mesmo espírito de
`cota_padrao_suspeito`/`checksum_fecha`" — e nenhum dos dois precedentes citados afeta
`.valido` (`cota_declarada.padrao_suspeito` vive numa dataclass separada; `checksum.fecha` é
uma property num objeto à parte). `nome_reparado` é um sinal de proveniência próprio, não um
motivo a mais para reprovar o registro inteiro — revertido para bater com o precedente que o
próprio ticket citou, não com o padrão de `campos_formato_invalido` (que tem uma razão
diferente para existir: aquele campo tem um formato *exato* conhecido pelo Cebraspe contra o
qual comparar; nome não tem).

### 3.4 Nome real do achado nunca entra em texto versionado

`scripts/NOTES.md` (git-tracked) já cita o nome real de um Aluno (`"Daniela F erreira Miguel
Pereira"`) como achado de produção — pré-existente, fora do escopo deste ticket corrigir. Os
testes novos, porém, usam nomes sintéticos mesmo ao reproduzir esse padrão estrutural (prefixo
de uma letra maiúscula solta) — mesma convenção já em uso em
`test_pas_extraction_convocacao._TEXTO_DUAS_SECOES` ("com os nomes trocados"). Achado e
corrigido durante a revisão deste ticket, antes do commit.

---

## 4. Medição sobre o corpus real

Rodada completa (`python -m pas_extraction.cli rodada`, 77 PDFs de `data/pdfs`):

```
Registros: 66313
Nomes reparados: 1865 (2.81%)
```

2,81% vs. o baseline conhecido de ~2,71% (medido por outro método: diff de string contra
`data/banco_alunos_pas_final.csv`, restrito aos pares casáveis por inscrição+triênio). A
diferença de ~0,1pp é esperada: a medição deste ticket conta todo registro em que
`reparar_nome` mudou o texto, sem depender de existir uma base externa para comparar — um
superconjunto do que o método de diff conseguia enxergar.

A reconciliação cruzada entre Editais (10 inscrições com nome divergente) não muda com este
ticket — `schema.canonizar` já removia espaço na comparação, então um nome quebrado por espaço
sempre canonizava igual à versão intacta; o defeito era invisível a essa camada antes e depois.

---

## 5. Testes

`tests/test_pas_extraction_nome_repair.py` — 15 testes unitários de `reparar_nome`: sufixo de
uma letra, prefixo de uma letra maiúscula, quebra em 3+ pedaços, quebra em qualquer posição do
nome, espaço duplicado (simples e corrida longa), as seis partículas (parametrizado), nome sem
corrupção, espaço de borda não conta como reparo.

`tests/test_pas_extraction.py` — `TestReparoDeNome` (4 testes) via `resultado_final._montar_registro`:
palavra quebrada reparada e sinalizada (com `.valido` permanecendo `True` — ver seção 3.3),
espaço duplicado normalizado e sinalizado, partículas nunca fundidas, nome limpo não sinalizado.

`tests/test_pas_extraction_convocacao.py` — `TestReparoDeNome` (3 testes) via `parse_convocacao`:
mesmos três casos, agora através do parser de linha em modo `layout`.

Suíte completa (`pytest tests/`): 483 testes, todos passam.

Revisão (`/code-review`, eixos Standards + Spec, ambos sem `git diff` — `src/pas_extraction/`
é inteiramente gitignored, revisão por leitura direta dos arquivos): achou e corrigiu (a) nome
real de Aluno hardcoded num teste versionado (seção 3.4) e (b) `.valido` reprovando registros
reparados sem o ticket ter pedido isso (seção 3.3). Heurística de fusão conferida à mão contra
os três exemplos reais citados no ticket e em `scripts/NOTES.md` — todos batem.
