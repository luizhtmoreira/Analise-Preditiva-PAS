# Relatório — Ticket 10: merge do portal para cima do modelo

**Branch:** `feat/pdf-extraction` (o tronco unificado)
**Commit:** `fbdc90a`, merge de `feat/nextjs-frontend` (`5653b05`) sobre `eb87a9f`
**Status:** concluído — `pytest tests/` 406 verdes, `tsc --noEmit` limpo, `eslint` 0 erros

---

## O que mudou

Existiam duas realidades. `feat/pdf-extraction` tinha o modelo promovido, o pipeline de treino e
a extração. `feat/nextjs-frontend` tinha o portal — Preditor com semestre e curso alvo,
Calculadora de Estratégia, header público, recuperação de senha e perfil — e estava 125 commits
atrás. Agora é um tronco só.

A direção do merge importava: `feat/nextjs-frontend` **veio para cima**, e não o contrário. A
verificação depois do merge confirma que os cinco módulos do pipeline sobreviveram
(`model_package.py`, `training_dataset.py`, `training_pipeline.py`, `validation.py`,
`dataset_pas3.py`) e que o `ensemble.py` do ADR-0011 **não voltou** — há teste para isso
(`test_ensemble_aposentado_nao_voltou_no_merge`).

Nove conflitos, não cinco. Os dois de verdade — serviço e página do Preditor — foram resolvidos
mantendo **as duas** evoluções, nunca escolhendo um lado.

---

## Decisões, e por quê

### 1. A probabilidade dos `top_cursos` passa a usar a Largura do manifesto

O código do portal chamava `calculate_approval_probability(arg, nota, rmse=ARG_FINAL_MAE)`.
`ARG_FINAL_MAE` era `13,49` — o resíduo de um modelo aposentado, igual para todo Aluno e errado
por construção a cada retreino. O ADR-0012 tirou essa constante do código; ela vive no manifesto
do pacote e vem por classe de Aluno.

Se o merge tivesse mantido a linha do portal, o Preditor teria **duas** larguras: a do manifesto
no Curso Alvo e a constante morta na lista de cursos. Teste: as duas probabilidades da mesma
nota de corte agora batem exatamente
(`test_probabilidade_dos_top_cursos_usa_a_largura_do_pacote`).

### 2. `get_strategy_prediction` saiu do `target_calculator`

Ela resolvia média e desvio a partir do `TRIENNIUM_STATS` e do `STATS_PAS3_TREND` — os dois
apagados pelo ticket 05. Era o bloqueio declarado no ticket: a Calculadora que chega no merge
consome justamente o dicionário que o 05 apaga.

Não faria sentido remendá-la para receber os dicionários de outro lugar: o `TRIENNIUM_STATS` era
uma **cópia paralela** do `OFFICIAL_STATS`, e o ponto do ticket 05 é que só existe uma porta.
`predict_service._stats_do_ciclo` resolve as três `HistoricalStats` por `stats_da_prova` — a mesma
função que o treino e o Preditor usam — e chama `calculate_required_score` direto. Para a Etapa 3
do triênio vivo entra o Ano-Âncora, exatamente como o `gestao_service` já fazia.

### 3. A Calculadora ganhou `lingua` obrigatória

Consequência da decisão 2, e não uma escolha estética. `stats_da_prova` pede a língua porque o
Cebraspe normaliza a Parte 1 por língua estrangeira — e a Calculadora **reverte** exatamente essa
normalização para chegar ao P2 necessário. Sem a língua certa, o número que a tela pede sai
deslocado, e sempre na mesma direção para o mesmo Aluno.

Sem default, pela razão do ticket 04 §5.3: um default silencioso é o viés. A tela ganhou o
seletor que o Preditor já tinha. O ticket 13 torna isso língua **por Etapa**; aqui é uma só, igual
ao Preditor de hoje.

### 4. Triênio sem Edital de Etapa: `status: "indisponivel"`, não 500

Achado da verificação ponta a ponta, não do código lido. `/api/predict/strategy` **estourava 500**
para `ciclo_aluno=2024-2026`, porque `(2024, Etapa 1)` ainda não está no `OFFICIAL_STATS`. O
Preditor já tratava esse caso (`motivo_indisponivel`); a Calculadora não tinha canal.

`StrategyResponse` já tinha `status` e `mensagem` — o canal existia. A tela mostra o aviso âmbar,
com o mesmo texto do Preditor. O que **não** se fez: chutar um `stats` para a Etapa 1. Isso
destruiria a parte exata da conta, que é a fundação do ADR-0009.

Cuidado extra na tela: o semáforo da Calculadora colore por `prob_hist`, e `prob_hist = 0`
cairia em "Meta de Alto Risco 🔴 — apenas 0% da base histórica". Uma afirmação sobre o Aluno que
ninguém mediu. O semáforo agora é `null` quando o status é `indisponivel`.

### 5. `top_cursos`: 10 a partir de 20%, e a lista é aspiracional

Ficaram os números do portal (10 cursos, piso de 20%) e não os do lado do modelo (8 a partir de
30%) — são a decisão mais nova, e o corte antigo devolvia lista vazia para quem está longe do
curso que quer.

A revisão apontou que a ordenação parecia invertida: o código ordena por **menor** probabilidade,
corta em 10 e inverte. Não é bug — é a intenção do portal (commit `8d7de37`): mostrar os cursos
mais difíceis ainda ao alcance, não os que o Aluno já passaria com folga. O comentário no código
é que estava errado e foi corrigido. Os dois ramos (logado e deslogado) eram o mesmo laço
duplicado; viraram `_selecionar_cursos(candidatos, ordem, limite)`, e o `3` solto virou
`TOP_CURSOS_DESLOGADO`.

### 6. `.gitignore`: lixo binário fora, `/app/` e `/assets/` ancorados

O arquivo estava marcado como binário porque tinha uma linha em UTF-16 (`m\0o\0d\0e\0l\0s\0/`)
no meio. Removida.

Mais importante: o lado do portal carregava a observação de que `app/` **sem barra inicial** casa
com qualquer diretório chamado `app` em qualquer nível — inclusive `landing-page/app/`, que é o
App Router do Next e é rastreado. `/app/` e `/assets/` agora são ancorados. As regras de
`.scratch/` são as do lado do modelo, que rastreia mapas e tickets de propósito e exclui só a
saída de dado.

### 7. A landing afirmava coisas que este merge apagou

A landing do portal (preservada por completo, como o ticket pede) dizia "ensemble de 4 modelos",
"Quatro modelos preveem sua terceira etapa" e mostrava a ficha "±13,49 — erro médio (RMSE) do
modelo". As três estão mortas: ADR-0011 aposentou o ensemble, e a faixa `±` é exatamente o que o
ADR-0012 §7 tirou da tela (acertava 63% e respondia uma pergunta que o Aluno não fez).

Trocadas por afirmações que o tronco sustenta: um modelo prevê a Etapa 3, `A1` e `A2` saem por
aritmética exata, 11 sinais no vetor de features. Não se colocou a Largura no lugar do `13,49` —
ela muda a cada retreino e vive no manifesto.

### 8. `TemporalPage`: estética do portal **e** o fix do laço de resize

Conflito que parecia mecânico e não era. O lado do portal trocou a paleta escura pela clara
(`vp-*`); o lado do modelo tinha envolvido os gráficos num `div` de altura explícita — o commit
`5700fde`, "fix layout breaking on course selection and recharts resize loop". Ficaram os dois: a
paleta clara dentro do wrapper de altura fixa.

---

## Verificação ponta a ponta

Um Aluno de teste (P1/P2/Redação das duas Etapas, espanhol) pelo tronco unificado:

| Caso | Resultado |
|---|---|
| `/api/predict` 2023-2025, deslogado | `arg 75,9` = `12,084 + 2×13,73 + 3×12,122`; largura `14,965` do manifesto; 3 cursos por corte mais próximo |
| `/api/predict` 2023-2025, logado | 10 cursos, curso alvo resolvido por fuzzy match, semestre filtrando |
| `/api/predict` 2024-2026 | `modelo_disponivel: false` + motivo, sem zeros mudos |
| `/api/predict` sem `lingua` | 422 nomeando o campo |
| `/api/predict/strategy` 2023-2025 | 200, `p2_necessario 82,64` |
| `/api/predict/strategy` 2024-2026 | 200, `status: indisponivel` (antes: 500) |
| `/api/courses`, `/chamadas`, `/cutoff` | 115 cursos; chamadas e corte reais para MEDICINA |

---

## Glossário deste ticket

**Merge assimétrico** — quando as duas branches não são pares. Aqui, uma delas é anterior à
outra e "discorda" dela só por ignorância: apagaria arquivos que não existiam quando ela nasceu.
Fazer o merge na direção errada desfaz trabalho **em silêncio**, porque o Git não tem como saber
que a ausência é ignorância e não decisão.

**Conflito mecânico vs. conflito de verdade** — mecânico é quando os dois lados escreveram no
mesmo lugar por acaso (uma linha em branco, um estilo) e dá para ficar com um dos dois. De
verdade é quando os dois lados **resolveram problemas diferentes** no mesmo trecho; escolher um
lado joga fora metade do trabalho. Os dois de verdade aqui foram o serviço e a página do
Preditor.

**Porta única** — uma função por onde *todo* mundo tem que passar para ler um dado. `OFFICIAL_STATS`
é lido só por `stats_da_prova`. Sem isso, treino e runtime podem ler valores diferentes e o `A1`
que a tela mostra deixa de ser o `A1` com que o modelo foi treinado — sem nada quebrar.

**Ano-Âncora** — usar um ano real e já publicado como cenário, no lugar de projetar o ano que
ainda não aconteceu. "E se a sua Etapa 3 for como a de 2025?" em vez de "a Etapa 3 de 2026 vai
ser assim".

**Estado derivado (React)** — em vez de guardar `carregando: true/false` e ligá-lo à mão, guardar
*qual pedido já respondeu* e calcular `carregando` disso. Some o render extra, e a resposta de um
pedido que o Aluno já abandonou não sobrescreve a resposta nova.

---

## Fica para depois

Nada disto é regressão deste merge; é dívida que ele herda.

1. **`p1_ia` / `red_ia` não dizem de onde vieram** — `predict_stable_components` devolve
   `method` e `fallback_reason` de propósito, e `predict_strategy` os descarta. Os `.joblib` de
   P1/Redação não carregam sob o sklearn atual, então hoje o número é **média ponderada**, e a
   tela o chama de "estimativas da IA". É o ticket 11.
2. **`/api/courses/cutoff` responde `0.0`** para curso sem dado, e a Calculadora lê isso como
   `nota_alvo` — um corte ausente vira "você já passou". Tickets 11/12.
3. **`get_course_chamadas` devolve `list[dict]`** enquanto a rota declara `list[ChamadaCorte]`, e
   a regra `min` vs `max` por semestre mora no router, não no serviço.
4. **Glossário do `CONTEXT.md`** ganhou "Simulador de Itens", que veio na branch do portal mas
   está fora desta rodada; e a entrada "Soft Gate" ainda descreve o gate do segundo curso, que a
   própria branch já substituiu por CTA.
5. **8 warnings de eslint** (deps de hooks, imports não usados). Erros estão zerados.
