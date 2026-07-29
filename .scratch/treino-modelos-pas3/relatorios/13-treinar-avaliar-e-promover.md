# Relatório — Ticket 13: Treinar, avaliar e promover

**Ticket:** `.scratch/treino-modelos-pas3/issues/13-treinar-avaliar-e-promover.md`
**Status:** concluído — **o mapa fecha aqui**
**Tipo:** execução + HITL (dono do produto revisou a comparação antes da promoção)
**Pacote promovido:** `models/pas3/` (`modelo_pas3.txt` + `manifest.json`), commit `355b6e4`
**Lacre aberto:** uma vez, em 2026-07-28, por `scripts/lacre_ticket13.py`
**Privacidade:** só agregados e contagens. A comparação individual usa rótulos anônimos, sem
`id_pseudonimo`, sem curso e sem campus.

---

## 1. O que este ticket fez

Rodou o pipeline do ticket 12 sobre o `resultado_final.csv` real pela primeira vez, abriu o
triênio lacrado, pôs modelo antigo e novo lado a lado, promoveu o pacote e refez a fiação de
`api/` para consumi-lo. **Nenhuma decisão nova de modelagem** — o ticket foi escrito para não
comportar nenhuma, e o único ponto em que uma decisão apareceu está na §6, declarado como tal.

---

## 2. A rodada

`scripts/treinar_pipeline.py` do CSV ao pacote, sem intervenção, na primeira execução:

| | valor |
|---|---:|
| linhas do dataset canônico | 64.298 |
| triênios | 8 (2016/2018 a 2023/2025) |
| previsões fora-da-dobra | 37.844 |
| RMSE geral em `A3` | **5,009** |
| RMSE majoritária (com Etapa 1) | 4,988 |
| RMSE minoritária (sem Etapa 1) | 5,158 |
| viés geral | +0,179 |

**Portão 1 batido nas quatro pernas** (`geral ≤ 5,167`, `majoritária ≤ 5,038`,
`minoritária ≤ 6,028`, `|viés| ≤ 0,500`). O pipeline confere sozinho e recusaria escrever em
disco se não batesse.

Uma discrepância que vale registrar, porque quem cruzar os dois documentos vai tropeçar nela: o
relatório 11 §7.1 previa `sigma_sem_etapa_1 = 5,2174` sobre **3.356** linhas, e o manifesto traz
`5,1584` sobre **2.936**. Não é rodada diferente — é a **trava 1** da régua. O `agrupado_minoritaria`
só soma as dobras em que a classe qualifica (treino da classe pelo menos tão rico quanto o teste);
o número do ticket 11 foi medido sobre todas as linhas da minoria, inclusive as dobras que a régua
recusa. **O do manifesto é o canônico**, porque é o que o código produz e o que viaja com o
artefato. O da §7.1 do relatório 11 era uma previsão escrita à mão.

---

## 3. O lacre — aberto uma vez, e só reporta

Regra escrita no relatório 11 §4.4 **antes de o número existir**: `σ` acima de `5,5` em `A3` = o
app está confiante demais, vira ticket com prioridade; abaixo de `4,5` = nota de relatório; entre
os dois = variação normal entre anos.

Triênio 2023/2025, **n = 8.703** (7.838 com Etapa 1, 865 sem):

| medida | valor | leitura |
|---|---:|---|
| `σ` em `A3` | **4,624** | dentro da banda — **nenhuma ação** |
| viés | +0,172 | mira honesta |
| RMSE/MAE | 1,2614 | normal ≈ 1,25 — a forma sino se sustenta fora da amostra |
| cobertura a 80% | 83,57% | o pacote promete 80% e entrega 83,6% |

O `σ` do lacre é **menor** que o agrupado das dobras (4,624 contra 5,009), o que o relatório 11
§3.4 previa: quanto mais triênios treinam, menor o erro, e o artefato final treina com 7 contra
os 2–6 das dobras. A largura que viaja no manifesto é, portanto, **conservadora de propósito** —
a cobertura a 80% saindo 83,6% é o mesmo fato dito de outro jeito. O modelo promete menos do que
entrega, que é o lado seguro de errar.

**Nada foi reajustado depois de ver este número.** O artefato promovido é bit-a-bit o que o
pipeline produziu antes de o lacre ser aberto.

---

## 4. A comparação lado a lado, revisada antes da promoção

Modelo antigo (`modelo_arg_final.joblib`, Argumento Final direto, `σ = 13,49`) contra o novo
(`A3` previsto, Argumento Final por aritmética, `σ = 3 × σ(A3)`), nos mesmos 8.703 Alunos do
triênio lacrado:

| Argumento Final | antigo | novo |
|---|---:|---:|
| RMSE | 17,942 | **13,871** |
| MAE | 14,270 | **10,996** |
| viés | **+8,658** | **+0,517** |

| decisão (7.449 Alunos com corte casado) | antigo | novo |
|---|---:|---:|
| erro de decisão | 7,81% | **5,41%** |
| Brier | 0,0564 | **0,0391** |
| probabilidades saturadas | 69,2% | 67,9% |

Taxa real de aprovação: 33,20%. Antigo e novo **discordam sobre passar** em 6,32% (471 Alunos).

**O viés de +8,66 do antigo é o achado do ticket 04 §3.1 reaparecendo no lacre** — a rota do
Argumento Final direto era sistematicamente otimista em ~9 pontos. Ele some no novo (+0,52). Um
Aluno em cima da linha recebia, do modelo antigo, quase 9 pontos de esperança que não existiam.

**Ressalva declarada:** não se sabe se o triênio 2023/2025 estava no treino do
`modelo_arg_final.joblib` (artefato de 2026-03-11, sem manifesto — é o problema que o ticket 03
resolveu). Se estava, o `17,942` dele é **otimista**, e a vantagem do novo é maior do que a tabela
mostra. A comparação erra a favor do modelo antigo, nunca contra.

Doze Alunos, estratificados de propósito (sorteio uniforme daria ~10 linhas de `0,0% vs 0,0%`,
porque 63,6% das probabilidades saturam — relatório 11 §8):

```
  #          grupo  AF real   AF ant  AF novo    corte   P ant  P novo  passou
  1  perto, com E1     33.2     36.3     25.0     30.7   66.2%   35.2%     sim
  2  perto, com E1     -8.7    -11.2    -22.2    -25.8   86.0%   59.7%     sim
  3  perto, com E1    -24.5    -31.1    -43.7    -40.8   76.4%   42.4%     sim
  4  perto, sem E1     27.7      9.7      1.2     30.7    6.0%    2.8%     não
  5  perto, sem E1    -30.2    -46.1    -50.7    -18.8    2.1%    2.0%     não
  6  perto, sem E1    -60.1    -73.5    -71.1    -79.4   67.0%   70.5%     sim
  7    longe acima     78.0     94.1     73.8     36.7  100.0%   99.3%     sim
  8    longe acima    -47.8    -40.0    -42.6    -87.0  100.0%   99.8%     sim
  9    longe acima     35.5     54.5     40.2      5.8  100.0%   98.9%     sim
 10   longe abaixo     66.8     58.0     47.7     88.5    1.2%    0.3%     não
 11   longe abaixo     11.0     31.2     29.0    148.5    0.0%    0.0%     não
 12   longe abaixo    -40.1    -47.5    -47.1     67.5    0.0%    0.0%     não
```

O que a revisão procurava e não achou: saída absurda. O que ela achou, e que o número agregado
esconde, são as linhas 1 a 3 — Alunos que **passaram** e a quem o modelo novo dá menos esperança
que o antigo. Isso é o viés otimista sendo retirado, não um erro novo: as três estão dentro de
uma largura do corte, exatamente onde uma probabilidade honesta *deve* ficar perto de 50%. O
antigo dizia 66–86% com o mesmo dado. O dono do produto revisou e autorizou a promoção.

---

## 5. A promoção

- **`models/pas3/`** ← `modelo_pas3.txt` + `manifest.json` do pacote.
- **`models/aposentados-2026-07-28/`** ← os dez `.joblib` do ensemble, preservados para reverter.
  `p1_pas3_model.joblib` e `red_pas3_model.joblib` **ficaram onde estavam**, porque
  `target_calculator.py` ainda os procura (e ainda falha em carregá-los — defeito 3 de
  `defeitos-pendentes.md`, intocado aqui).

### ⚠ O que a promoção **não** fez: o domicílio

A checklist pedia `models/` atualizado *"no formato **e no domicílio** decididos pelo ticket 03"*.
**Formato sim, domicílio não.** A Decisão 3 do ticket 03 põe o pacote num repositório privado no
Hugging Face, com a Decisão 4 assando-o na imagem no build; nada disso existe — não há upload, não
há Dockerfile, e `grep -ri huggingface` no repositório só encontra o próprio relatório 03.

Consequências práticas, que valem mais que a pendência em si:

- **Reverter não é `git revert`.** `models/` é gitignored: reverter o commit devolve o código
  antigo e deixa o pacote novo no disco. Reverter de verdade é copiar
  `models/aposentados-2026-07-28/` de volta **à mão**, e só depois reverter o commit.
- **Máquina nova sobe sem pacote.** Um clone limpo não tem `models/pas3/`, e a API responde
  `modelo_disponivel: False` até alguém copiar o diretório. Hoje isso é aceitável porque o
  deploy ainda não existe; deixa de ser no dia em que existir.

Item para o mapa de deploy, não para este. Registrado também na §8.

### O que saiu do código

| removido | por quê |
|---|---|
| `src/pas_intelligence/ensemble.py` | ADR-0011 aposentou o ensemble; o mecanismo sobrevive em `scripts/baseline_honesto.py::peso_sigmoide_da_volatilidade`, que é a medição que o aposentou e precisa continuar reproduzível |
| `calculate.py` | avaliava o ensemble com os artefatos aposentados, pelo método que o ADR-0007 invalidou |
| `scripts/baseline_avaliacao.py` | já declarado superado no próprio topo; carregava os dez `.joblib` |
| `ARG_FINAL_MAE = 13.49` (`gestao_service.py:34`) | a largura vem do manifesto |
| `ARG_MARGEM = 13.49` (`RelatoriosPage.tsx:6`) | mesma constante, no frontend |
| `arg_min`/`arg_max` da resposta | ADR-0012 §7 |

---

## 6. As decisões que este ticket precisou tomar, e por quê

O ticket abre com *"Nada se decide aqui"*. Duas coisas foram decididas assim mesmo. As duas estão
aqui porque decisão não declarada é decisão escondida.

### 6.1 O `eb_pas3_previsto` saiu da resposta da API e da tela

Ele era produzido por `modelo_lgbm.joblib`, que o ADR-0009 tirou do caminho de produção. As
opções eram três: ressuscitar o modelo aposentado (traz de volta a contradição de 11% entre as
duas rotas, medida no ticket 04 §3.2), derivá-lo pelo Estimador Auxiliar + Ano-Âncora (é a rota
certa, mas o próprio ticket 13 chama a forma disso de *"névoa no mapa"* — decidir aqui seria
exatamente o que o ticket proíbe), ou tirá-lo até que a rota certa exista.

Tirei. É a única das três que não decide nada: não afirma como o EB deve ser derivado, não
ressuscita nada, e não deixa na tela um segundo número previsto capaz de discordar do primeiro.
No lugar dele, o card passa a mostrar a **decomposição** `A1 ×1 · A2 ×2 · Â3 ×3`, que é
informação que o Aluno não tinha e que deixa visível o que é conta fechada e o que é previsão.

**Isto é dívida declarada, não escopo cumprido.** Ver §8.

### 6.2 A língua estrangeira tem default na Gestão de Ativos, e não no Preditor

O ticket 04 §5.3 decidiu **perguntar** a língua, porque agrupar as três embute viés contra a
minoria. No Preditor público isso é literal: o campo é obrigatório, sem default, e cliente que
não enviar recebe `422` nomeando o campo — mesmo tratamento das seis notas.

Na Gestão de Ativos o campo tem default `"inglesa"`. A planilha que a escola envia não tem a
coluna, e exigi-la deixaria o lote **inteiro** sem resposta. O custo é conhecido e não é zero: o
Aluno de espanhol ou francês daquela planilha tem a Parte 1 calculada com a estatística de inglês
— 7,2% do peso do Argumento de Etapa, sempre na mesma direção, sempre sobre a minoria.

Escolhi o lado que responde. Mas é o viés que o ticket 04 mediu, vivo num canto do produto, e a
saída é uma coluna na planilha — não uma decisão de modelagem. Está no código, com o porquê.

---

## 7. A fiação

### 7.1 `src/pas_intelligence/model_package.py` — o módulo novo

A única porta entre `api/` e o artefato. Recebe as seis notas, a língua declarada e o triênio;
devolve `A1`, `A2`, `Â3`, Argumento Final e a Largura de Incerteza da classe do Aluno.

A decisão de projeto que mais importa nele: **ele monta as features chamando as funções do
treino** (`adicionar_features_legadas` → `adicionar_derivadas_trajetoria` →
`com_faltante_nativo_etapa1`), sobre um `DataFrame` de uma linha, em vez de reescrever a
aritmética para o caso individual. Reescrever seria mais direto e é o caminho que quase todo
serviço toma; o custo é o *train/serve skew* — o desencontro entre como uma feature é montada no
treino e no request. Ele **não levanta exceção nenhuma**: devolve previsão errada com cara de
certa, para sempre. `test_o_runtime_monta_as_mesmas_features_que_o_treino` passa o mesmo Aluno
pelas duas portas e compara os dois vetores.

Duas condições fazem o módulo **recusar** em vez de responder:

1. **Pacote ausente** — sem pacote não há previsão *nem* largura. O estado "previsão sim, largura
   não" não é representável (ADR-0012 §6).
2. **Edital de Etapa não extraído** — `A1` e `A2` são a parte *exata* da conta; aproximá-los
   destruiria a fundação do ADR-0009. É a §8.

### 7.2 O Aluno declara a língua e declara a ausência da Etapa 1

Dois campos novos no formulário, os dois vindos de decisões anteriores:

- **Língua estrangeira** (ticket 04 §5.3). O Cebraspe normaliza a Parte 1 por língua e o Edital
  não diz qual cada candidato fez; agrupar embute viés contra espanhol e francês — em
  `(2024, Etapa 2)` a diferença entre a maior e a menor média foi de 3,86 pontos, mais de um
  desvio inteiro.
- **Botão "Não fiz o PAS 1"** (ADR-0012 §8), que preenche os três campos da Etapa 1 com zero. Sem
  ele o Aluno teria que **adivinhar** que "tudo zero" é a codificação da ausência — e quem
  adivinhasse errado receberia a previsão de alguém que fez a prova e zerou. A detecção em runtime
  não é sobre incerteza: sem ela o modelo lê o zero como nota e erra a **previsão**.

### 7.3 Gestão de Ativos ganhou um quarto estado

`grey` / *Sem previsão*, fora do semáforo. Um Aluno sem previsão que aparecesse com Argumento
`0,0` e probabilidade `0,0%` viraria **"Alto Risco"** na tela da coordenação — uma afirmação sobre
ele que ninguém mediu. `grey` diz "não sei", ordena primeiro na lista (é o único estado que pede
ação de quem opera o sistema, não do Aluno) e é contado em `kpis.n_sem_previsao`.

### 7.4 Consumidores conferidos

| consumidor | situação |
|---|---|
| `api/services/predict_service.py` | reescrito sobre o pacote |
| `api/services/gestao_service.py` | reescrito; `ARG_FINAL_MAE` removido |
| `api/services/analytics_service.py` | não toca em modelo — intocado |
| `src/pas_intelligence/statistics.py` | parâmetro renomeado para `largura_incerteza`, sem padrão |
| `src/pas_intelligence/target_calculator.py` | **intocado — e não é "funcionando"**: ele continua com o defeito 3 de `defeitos-pendentes.md` (os dois `.joblib` não carregam sob o sklearn atual e ele responde por média ponderada sem avisar). Este ticket não melhorou nem piorou isso. A cura é a simplificação do ticket 04 §7.1, que depende do Estimador Auxiliar (§8) |
| `app/streamlit_app.py` (gitignored) | **quebrado por este ticket**, e sem verificação prévia com o dono do produto — a checklist pedia *"verificar se ainda é usado localmente antes de quebrá-lo em silêncio"*, e o que houve foi presunção de que está descontinuado, não uma pergunta. Ele importava `ensemble.py` (removido) e chama `calculate_approval_probability(rmse=...)` (renomeado). Consertá-lo é meia hora; a decisão de consertar ou aposentar de vez é do dono do produto |
| `tests/` | 327 passando (eram 336; 9 eram do `ensemble.py` removido, 19 são novos) |

---

## 8. O que ficou de fora, e o que isso bloqueia

**As chaves `(2024, Etapa 1)` e `(2025, Etapa 2)` não existem no `OFFICIAL_STATS`**, e os Editais
dessas Etapas não estão em `data/pdfs/` — só há Editais de PAS 3, que publicam as três Etapas de
um triênio já encerrado. Consequência direta: **a turma viva (2024-2026) não recebe previsão** até
que esses Editais sejam extraídos. O Preditor a tem como triênio padrão.

A alternativa — aproximar `A1` e `A2` com a estatística do ano mais próximo — foi considerada e
descartada com o dono do produto na sala: ela troca a única parte **exata** da conta por
estimativa, que é a fundação inteira do ADR-0009. Recusar dizendo qual chave falta é a resposta
honesta, e o código já está pronto para o dia em que ela existir: `OFFICIAL_STATS` é lido em tempo
de requisição, então a turma viva volta a responder **sem nenhuma mudança de código**.

O dono do produto tem os PDFs e vai adicioná-los a `data/pdfs/`. Duas coisas a conferir quando
isso acontecer: o parser do mapa `pdf-extraction` nunca viu Edital de Etapa 1 ou 2 (família de
documento diferente), e `OFFICIAL_STATS` precisa da substituição do ticket 12 daquele mapa.

**Aberto por este ticket, e declarado:**

- **O domicílio do ticket 03** (§5) — o pacote mora só em disco local. Reverter é manual e uma
  máquina nova sobe sem modelo. Item do mapa de deploy.
- **`app/streamlit_app.py` quebrado sem verificação prévia** (§7.4) — a checklist pedia perguntar
  antes; não perguntei. Conserto de meia hora, decisão do dono do produto.
- **O default de língua na Gestão de Ativos** (§6.2) — some quando a planilha ganhar a coluna.

**Dívidas herdadas, nenhuma nova:**

1. **O EB PAS 3 derivado** (§6) — Estimador Auxiliar + Ano-Âncora, ticket 04 §7.1. Sem ticket.
2. **`target_calculator.py`** — o ticket 04 §7.1 o simplifica por remoção (some o carregamento de
   `.joblib`, some o `ModelLoadError`), o que resolveria de quebra o defeito 3 de
   `defeitos-pendentes.md`. Depende de (1).
3. **`TRIENNIUM_STATS`** (`gestao_service.py`) — números que não são os do Cebraspe (defeito 1 do
   ticket 04 §9). Hoje alimentam só o Reality Check, que é opcional; o Argumento previsto não passa
   mais por lá. `TODO` deixado no código.
4. **A saturação** (relatório 11 §8) — 2 em cada 3 Alunos recebem `<1%` ou `>99%`. É tela de
   produto, não de modelo.

---

## 9. Glossário

- **Portão 1** — o critério de aceite congelado no ticket 07 §8: os quatro números que qualquer
  candidato precisa bater para não regredir contra o baseline honesto. O pipeline o verifica
  sozinho e **recusa escrever em disco** se falhar.
- **Lacre** — o triênio 2023/2025, mantido fora de todo treino e de toda dobra desde o ticket 06,
  para existir um ano que nenhuma decisão do mapa tocou. Aberto uma vez, aqui.
- **Brier** — erro quadrático médio da própria probabilidade contra o que aconteceu
  (`média((p − aconteceu)²)`, com `aconteceu` valendo 0 ou 1). Menor é melhor. Pune tanto errar
  quanto acertar sem convicção, e por isso enxerga o que o "erro de decisão" (que só olha se a
  probabilidade passou de 50%) joga fora.
- **Erro de decisão** — em que fração dos Alunos o sistema teria dito a coisa errada sobre passar.
- **Saturada** — probabilidade abaixo de 1% ou acima de 99%.
- **Train/serve skew** — o desencontro entre como uma feature é montada no treino e como é montada
  no request. Não quebra nada; produz número errado com cara de certo. §7.1.
- **Estimador Auxiliar** — a regra que estimaria P1 e Redação da Etapa 3 só para **repartir** o
  `A3` entre as três partes e poder falar em escore em vez de desvio-padrão. Não é fonte de
  verdade. É a peça que falta para o EB voltar à tela (§6).
- **Ano-Âncora** — ano real e já publicado usado como cenário (*"e se a minha Etapa 3 for como
  2025?"*), amarrando junto a Nota de Corte e as estatísticas da prova daquele ano.

---

## 10. Onde continuar

- **O mapa `treino-modelos-pas3` fecha aqui.** Os 13 tickets estão resolvidos.
- **Régua de parada do ticket 06:** este ticket não move acurácia e não entra na contagem. Ela
  segue onde o ticket 10 a deixou — um ticket de distância do gatilho.
- **O próximo retreino** é `scripts/treinar_pipeline.py` mais a promoção da §5. O lacre, porém,
  **já foi gasto**: 2023/2025 é agora um triênio como qualquer outro, e não existe outro ano limpo
  até o Edital de 2027. Uma rodada futura que queira a mesma frase — *"avaliado num ano que nenhuma
  decisão tocou"* — precisa lacrar 2024/2026 antes de olhar para ele.
- **O bloqueio da turma viva (§8) é o que importa destravar primeiro**, e ele mora no mapa
  `pdf-extraction`, não neste.
