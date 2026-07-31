# Relatório — Ticket 13: língua estrangeira por Etapa, ponta a ponta

**Ticket:** `.scratch/publicar-site/issues/13-lingua-por-etapa-ponta-a-ponta.md`
**Status:** concluído
**Onde vive o código:** `src/pas_intelligence/model_package.py`, `api/schemas/predict.py`,
`api/schemas/gestao.py`, `api/services/predict_service.py`, `api/services/gestao_service.py`,
`landing-page/lib/api.ts`, `landing-page/components/public/predict/PreditorPage.tsx` — o essencial
do diff em `model_package.py`, `predict.py`, `predict_service.py`, `PreditorPage.tsx` e nos testes
acabou absorvido pelos commits `981048c` (ticket 11) e `2dfb292` (ticket 07), de uma sessão
concorrente que editava os mesmos arquivos ao vivo (mesmo padrão do relatório 05 §4); o que sobrou
fora deles — `gestao.py`, `gestao_service.py`, `landing-page/lib/api.ts` (`fetchPredict`) e este
relatório — ficou neste commit.

---

## 1. O que foi pedido

O Cebraspe registra a língua estrangeira **por Etapa** (`lingua_e1`, `lingua_e2`, `lingua_e3` no
`resultado_final.csv`), e o treino já respeita isso. O runtime não: `EntradaDePrevisao` tinha um
campo `lingua` só, aplicado às duas Etapas — o Aluno que trocou de língua (13,9% da base,
majoritariamente inglês → espanhol) recebia `A1` ou `A2` normalizado com a estatística errada.
Defeito 11 de `defeitos-pendentes.md`.

Critérios de aceite (todos atendidos):

- [x] `EntradaDePrevisao` carrega a língua por Etapa e o cálculo usa cada uma na sua Etapa
- [x] O schema do Preditor exige `lingua_e1` e `lingua_e2`, sem default; faltar qualquer uma
      devolve 422 nomeando o campo
- [x] O formulário tem dois campos, com o segundo pré-preenchido pelo primeiro e editável
- [x] O schema da Gestão tem o default por Etapa
- [x] `test_o_runtime_monta_as_mesmas_features_que_o_treino` ganhou um caso de língua trocada,
      que falha sem a correção
- [x] Um Aluno com línguas diferentes produz o mesmo `A1`/`A2` pelo caminho do treino e pelo
      caminho do runtime
- [x] O defeito 11 de `defeitos-pendentes.md` está marcado como corrigido
- [x] `pytest tests/`, `eslint` e `tsc --noEmit` — ver §6 para o estado exato da suíte

---

## 2. O que foi entregue

```
src/pas_intelligence/model_package.py
  EntradaDePrevisao.lingua  →  lingua_e1, lingua_e2
  _argumentos_exatos: valida as duas contra LINGUAS_OFICIAIS,
    stats_e1 = stats_da_prova(ano_e1, 1, lingua_e1)
    stats_e2 = stats_da_prova(ano_e2, 2, lingua_e2)

api/schemas/predict.py
  PredictInput.lingua  →  lingua_e1, lingua_e2 (Literal, obrigatórias, sem default)

api/services/predict_service.py
  entrada_de_previsao: passa lingua_e1/lingua_e2 para EntradaDePrevisao
  (StrategyInput.lingua — Calculadora reversa — não mudou; fora do escopo deste ticket)

api/schemas/gestao.py
  StudentInput.lingua = "inglesa"  →  lingua_e1 = "inglesa", lingua_e2 = "inglesa"

api/services/gestao_service.py
  _prever: passa lingua_e1/lingua_e2 para EntradaDePrevisao
  Reality Check: stats_p1 usa lingua_e1, stats_p2 e stats_p3 usam lingua_e2

landing-page/lib/api.ts
  fetchPredict: lingua  →  lingua_e1, lingua_e2

landing-page/components/public/predict/PreditorPage.tsx
  um select de língua por card de Etapa (PAS 1 / PAS 2), não mais um só na Configuração
  segundo campo pré-preenchido pelo primeiro até o Aluno editá-lo diretamente

tests/test_model_package.py
  entrada(): lingua_e1/lingua_e2
  + test_lingua_e_por_etapa_nao_por_aluno
  test_lingua_fora_das_tres_oficiais_e_recusada: cobre as duas pontas
  test_o_runtime_monta_as_mesmas_features_que_o_treino: parametrizado com
    ["com Etapa 1", "sem Etapa 1", "língua trocada entre Etapas"]

tests/test_api_predict.py
  entrada_predict(): lingua_e1/lingua_e2
  + test_lingua_e_por_etapa_faltando_qualquer_uma_e_422
  + test_lingua_por_etapa_produz_a1_e_a2_diferentes
  test_lingua_fora_das_tres_e_recusada_pelo_schema: cobre as duas pontas

.scratch/treino-modelos-pas3/relatorios/defeitos-pendentes.md
  defeito 11 riscado e marcado "CORRIGIDO em 2026-07-31 (ticket 13)"
```

---

## 3. Decisões tomadas e o porquê

### 3.1 Dois campos, não uma tupla ou lista

**Decisão:** `EntradaDePrevisao.lingua_e1` e `.lingua_e2` são dois atributos nomeados, não
`linguas: tuple[str, str]`.

**Porquê:** o resto da classe já segue esse padrão (`etapa_1`, `etapa_2` como campos separados,
não uma lista de duas `NotasDeEtapa`), e nomear por Etapa deixa `_argumentos_exatos` legível sem
indexação por posição — `entrada.lingua_e1` não deixa dúvida sobre qual Etapa está sendo lida,
onde `entrada.linguas[0]` exigiria olhar a definição da classe.

### 3.2 Sem alias de compatibilidade no schema do Preditor

**Decisão:** `PredictInput` não aceita mais `lingua`; só `lingua_e1`/`lingua_e2`, e um request
com o campo antigo recebe 422 (campo desconhecido rejeitado — `extra` não configurado como
`allow`, comportamento padrão do Pydantic).

**Porquê:** o próprio ticket decide isso explicitamente — o único cliente é o frontend, que está
sendo reescrito na mesma rodada, e manter um alias treinaria o hábito de ignorar o contrato novo
exatamente no ponto (viés de língua) que o ticket 04 §5.3 se propôs a eliminar.

### 3.3 Pré-preenchimento no frontend é estado de UI, não parte do contrato

**Decisão:** `PreditorPage.tsx` inicializa os dois `useState` com `"inglesa"` e propaga o valor
de `linguaE1` para `linguaE2` só enquanto o Aluno não mexeu em `linguaE2` diretamente (uma flag
`linguaE2Tocada` guardada em `useState`, não em `useRef` — ver §3.5). A API sempre recebe as duas,
nunca uma delas implícita.

**Porquê:** é a decisão de forma explícita do ticket ("o pré-preenchimento é de interface, não de
contrato"). Sem essa distinção, os 86% que não trocaram de língua teriam que preencher o mesmo
campo duas vezes; com ela, só os 13,9% que trocaram precisam tocar no segundo campo.

### 3.4 Seletor de língua migrou para dentro do card de cada Etapa

**Decisão:** o select único "Língua Estrangeira" que vivia no bloco "Configuração do Candidato"
foi removido; cada card (PAS 1, PAS 2) ganhou seu próprio select, posicionado antes dos campos de
nota daquela Etapa.

**Porquê:** a língua deixou de ser um atributo do Aluno (por isso vivia na "Configuração") e
passou a ser um atributo do par (Aluno, Etapa) — colocar o seletor dentro do card da Etapa é o
mapeamento direto dessa mudança de modelo de dados para a tela, e evita a leitura errada de "a
configuração do Aluno tem duas línguas" quando na verdade são duas Etapas com uma língua cada.

### 3.5 Flag de "campo tocado" em `useState`, não em `useRef`

**Decisão:** `linguaE2Tocada` é `useState<boolean>`, não `useRef<boolean>`.

**Porquê:** a primeira tentativa usou `useRef` (não precisa de re-render por si só) mas o eslint
do projeto (`react-hooks/refs`, do `eslint-plugin-react-hooks` novo) acusou "Cannot access ref
value during render" — a leitura de `.current` acontecia dentro de uma função declarada no corpo
do componente e invocada indiretamente via prop `onLingua` passada através do `.map()` do grid de
Etapas, e o linter não consegue provar estaticamente que essa leitura só ocorre dentro de um
handler de evento. Trocar para `useState` remove a ambiguidade — o valor só é lido/escrito em
handlers — ao custo de um re-render a mais por toque no campo, imperceptível numa troca de select.

### 3.6 `StrategyInput` (Calculadora) não mudou

**Decisão:** o campo `lingua` único de `StrategyInput`/`fetchStrategy` (Calculadora de
Estratégia) não foi tocado.

**Porquê:** o checklist do ticket lista explicitamente `EntradaDePrevisao`, o schema do Preditor,
o formulário do Preditor e o schema da Gestão — não a Calculadora. A Calculadora resolve
`A3_necessário` a partir de `stats_p3`, que já é uma aproximação (Ano-Âncora); estender o mesmo
tratamento por Etapa a ela é trabalho de escopo próprio, sinalizado no defeito 11 original como
"item 4: verificar `lingua_e3`" — não fechado aqui, e registrado como aberto no `defeitos-pendentes.md` (§3.7 e §5).

### 3.7 Reality Check da Gestão usa `lingua_e2` como proxy da Etapa 3

**Decisão:** `gestao_service.py` — o bloco de Reality Check que resolve `stats_p3` (para estimar
o Argumento necessário na Etapa 3 via `TargetCalculator`) passou a usar `s.lingua_e2`, não uma
`lingua_e3` (que não existe em `StudentInput`).

**Porquê:** o mesmo raciocínio que `target_calculator.predict_stable_components` já documenta
para o Estimador Auxiliar — o Aluno ainda não sentou a Etapa 3, a troca de língua no PAS é de mão
única (72% das trocas vão de inglesa para espanhola) e a última língua declarada é a melhor
evidência disponível de qual será a da Etapa 3. Deixado como decisão explícita, não descuido — o
item 4 do defeito 11 original ("verificar `lingua_e3`") permanece nomeado como pendência aberta em
`defeitos-pendentes.md`, para revisitar se `A3` exato algum dia entrar em jogo.

---

## 4. Nota sobre sessão concorrente (turma viva / ticket 07 e 11)

Durante a implementação, outra sessão editava ao vivo `src/pas_intelligence/model_package.py`,
`api/schemas/predict.py`, `api/services/predict_service.py`,
`landing-page/components/public/predict/PreditorPage.tsx`, `tests/test_model_package.py`,
`tests/test_api_predict.py`, `.scratch/treino-modelos-pas3/relatorios/defeitos-pendentes.md` e
outros — os tickets 07 ("Preditor responde para a Turma viva com estatística derivada") e 11
("calculadora sem joblib, Estimador Auxiliar e faixa medida da P2"), rodando ao mesmo tempo que
este ticket 13, no mesmo *working tree* (sem *worktree* isolado). Enquanto os dois estavam em
andamento, isso deixava **8 falhas pré-existentes** na suíte — todas assumindo que `(2024, Etapa
1)` e `(2025, Etapa 2)` não têm estatística oficial, premissa que o ticket 07 mudou — confirmadas
como fora do escopo do ticket 13 isolando cada arquivo concorrente via `git stash` (mesmo
conjunto de falhas, com ou sem o diff deste ticket).

Os commits `981048c` (ticket 11) e `2dfb292` (ticket 07) fecharam essa sessão concorrente e
**absorveram, no mesmo commit, o diff inteiro do ticket 13** nos arquivos que os dois tocavam em
comum (`model_package.py`, `predict.py`, `predict_service.py`, `PreditorPage.tsx`, os dois
arquivos de teste, `defeitos-pendentes.md`) — confirmado conferindo que `lingua_e1`/`lingua_e2`,
o caso "língua trocada entre Etapas" do teste de paridade e o defeito 11 riscado já estavam
presentes nesses commits antes deste. Mesmo padrão do relatório 05 §4. O que sobrou fora deles —
`api/schemas/gestao.py`, `api/services/gestao_service.py`, `landing-page/lib/api.ts` e este
relatório — foi o que ficou para este commit.

---

## 5. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê |
|---|---|
| `lingua_e3` em `EntradaDePrevisao`/`StudentInput` | O runtime nunca calcula `A3` exato (é previsto); só o Reality Check e a Calculadora resolvem uma estatística de Etapa 3, hoje via proxy (`lingua_e2`, §3.7). Registrado como pendência do defeito 11 |
| `StrategyInput.lingua` por Etapa (Calculadora) | Fora do checklist do ticket; ver §3.6 |
| Coluna de língua no upload de planilha da Gestão | A Gestão não tem UI de upload que mapeie colunas hoje; o default por Etapa em `StudentInput` já resolve o schema, sem mudar a natureza da dívida do relatório 13 §6.2 |

---

## 6. Como foi verificado

- **`tests/test_model_package.py`**: 20 testes, todos verdes — inclui
  `test_lingua_e_por_etapa_nao_por_aluno` (trocar só `lingua_e1` não mexe em `A2`, e vice-versa) e
  o caso "língua trocada entre Etapas" no teste de paridade treino/runtime.
- **`tests/test_api_predict.py`**: todos verdes.
- **Suíte inteira** (`pytest tests/`): **432 passam, 0 falham** — depois que os tickets 07 e 11
  concorrentes (§4) fecharam e cobriram `(2024, Etapa 1)`/`(2025, Etapa 2)` com estatística
  derivada, as 8 falhas pré-existentes que dependiam dessas duas entradas desapareceram junto.
- **`npx tsc --noEmit`** na `landing-page/`: limpo (exit 0).
- **`npm run lint`** na `landing-page/`: limpo (exit 0) — os 2 erros iniciais de
  `react-hooks/refs` (leitura de `useRef` fora de handler direto) foram corrigidos trocando a
  flag de "campo tocado" para `useState` (§3.5); os warnings remanescentes (`Link`/`BrandMark`
  não usados, dependências do `useEffect`) são pré-existentes, não introduzidos por este ticket.
- Não testado manualmente no navegador (sem servidor local rodando nesta sessão) — a verificação
  visual do formulário com os dois seletores fica pendente para quem revisar.

---

## 7. Glossário — termos necessários para entender este relatório

| Termo | Significado |
|---|---|
| **Argumento de Etapa (`A1`, `A2`, `A3`)** | Nota padronizada (z-score) de uma Etapa do PAS. Para quem já sentou a Etapa, `A1`/`A2` são aritmética exata sobre as seis notas digitadas; só `A3` é previsto pelo modelo. |
| **Parte 1 (P1)** | A nota de Língua Estrangeira do PAS, normalizada pelo Cebraspe com a média/desvio *daquela língua, naquela Etapa* — é o número que sai errado quando a língua da Etapa é a errada. |
| **Train/serve skew** | Quando a forma de montar uma feature no treino diverge da forma de montá-la no runtime; não levanta erro, devolve previsão errada com aparência de certa. |
| **Ano-Âncora** | Um ano real e já publicado usado como estatística de cenário para uma Etapa que ainda não aconteceu, no lugar de projetar por regressão. |
| **Reality Check (cohort)** | Comparação opcional na Gestão de Ativos contra o histórico real de Alunos, para estimar a chance de uma trajetória parecida. |
| **Turma viva** | O triênio em andamento (2024-2026), cujas Etapas mais recentes ainda não têm Edital de média/desvio publicado. |
