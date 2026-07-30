# Mapa — Publicar o site (lado público)

**Label:** `wayfinder:map`
**Criado:** 2026-07-29
**Atualizado:** 2026-07-30 — ticket 06 rodou e **reprovou**: o deslocamento não é estável entre
triênios, o Preditor continua recusando a Turma viva, e os tickets 07/14 seguem bloqueados
(ver "O que pode reordenar tudo" e o relatório do ticket 06).

---

> **O mapa cumpriu o papel dele.** A rodada está escrita em `spec.md` e quebrada em 14 tickets em
> `issues/`, em ordem de dependência. Comece por lá; este arquivo fica como o registro de *por que*
> a ordem é essa.
>
> Três coisas que a spec acrescentou e que o mapa não sabia:
> 1. **`feat/proof-section` tem PII na árvore**, não só no histórico — mergear como está publica
>    nomes de Alunos na `main` (ticket 04).
> 2. **`TRIENNIUM_STATS` virou bloqueador**, não faxina: é ele que alimenta a Calculadora que entra
>    nesta rodada (ticket 05).
> 3. **"4.786 cortes" não bate com o arquivo** — são 5.225 linhas, 4.986 limpas (ticket 09).

---

## Comece aqui

**O Passo 1 respondeu: o Preditor público pode atender a turma viva.** A pergunta que decidia a
ordem de tudo está fechada, e a resposta veio com uma correção obrigatória junto (§ Passo 1).

O trabalho agora começa no **Passo 2**, que cresceu: não é mais "extrair dois Editais", é extrair
os dois, calibrar o deslocamento que o Passo 1 mediu, e resolver uma questão de forma no
`ExamStats`. Os Passos 3 e 4 continuam podendo andar em paralelo desde já.

---

## Destino

O lado **público** do produto no ar, em `vetorpas.com.br`, servido pelo modelo novo:

- **Preditor PAS 3** respondendo para um aluno da turma viva (2024-2026) — hoje ele recusa;
- **Análise Temporal** pública, com a série oficial e a evolução de nota de corte por curso;
- a **landing** atual continuando de pé, sem regressão;
- a **API** hospedada de verdade, não em `localhost`, com o pacote de modelo dentro da imagem.

O B2B (Gestão de Ativos, Escola, Comparação, Relatórios) fica **fora desta rodada** por decisão sua
— entra depois, e o inventário do que ele precisa está em `app/INVENTARIO-STREAMLIT.md`.

---

## O terreno, em uma tela

| Frente | Onde está | O que falta |
|---|---|---|
| Modelo PAS 3 | ✅ pronto e promovido (`models/pas3/`, `RMSE 5,009` em `A3`) | mora só no seu disco |
| Método de inferir média e desvio | ✅ medido e aprovado com correção (Passo 1) | calibrar o deslocamento em mais triênios |
| Editais de Etapa 1/2 | ✅ 6 baixados e extraídos sem falha de checksum | os de validação dos triênios antigos |
| Extração de Editais | ✅ 12 tickets fechados, 77 PDFs processados | 6 tickets abertos; parser não conhece Etapa 1/2 |
| Landing (`main`) | ✅ em produção na Vercel | — |
| Portal Next.js | ⏳ `feat/nextjs-frontend`, 22 commits, **52 atrás da `main`** | não pronto; não conhece o modelo novo |
| Visual da landing | ✅ `feat/proof-section`, 14 commits, contém a `main` | só falta mergear |
| API FastAPI | ⚠️ funciona em `localhost:8000` | sem Dockerfile, sem Space, CORS quebrado |

---

## A ordem

Cada passo diz **o que é**, **por que nesta posição**, e **como você sabe que acabou**. A ordem
entre passos é o que importa; dentro de cada um, mude à vontade.

### Passo 1 — Medir se a média inferida reproduz a oficial ✅ **RESOLVIDO**

**A resposta: serve com correção.** Usando só os Editais isolados de Etapa 1 e 2, o Argumento Final
sai **+7,87 pontos acima** do verdadeiro. O limiar era `3 × RMSE = 15,03`, então cabe — mas com
pouca folga e no sentido perigoso: o Preditor ficaria **otimista**, dizendo ao aluno que ele está
melhor do que está.

O que salva é que **não é ruído, é um degrau**. A média do erro e a média do valor absoluto são o
mesmo número (7,867 e 7,867): todo aluno erra na mesma direção e quase na mesma quantidade.
Tirando o degrau:

| | \|erro\| médio | p95 | máx |
|---|---:|---:|---:|
| Bruto | 7,87 | 9,85 | 11,46 |
| **Corrigido pelo deslocamento** | **1,14** | 2,62 | **4,19** |

**A causa, identificada.** O erro não está distribuído entre as etapas — está quase todo na Etapa 2:

| Validação | Erro em `m_p2` |
|---|---:|
| (2022, Etapa 1) | −1,35 |
| (2023, Etapa 1) | −2,18 |
| **(2024, Etapa 2)** | **−4,61** |

O Edital isolado de Etapa 2 de 2024 tem 16.339 candidatos; os concluintes daquele triênio são
8.703. **O Cebraspe calcula a média da Etapa 2 sobre os concluintes**, e faz sentido: ele só
publica o Edital de média e desvio depois do PAS 3, quando já sabe quem chegou ao fim. Estimando
sobre os 16.339, pegamos metade a mais de gente, e essa metade é mais fraca — 0,31 desvio-padrão
de diferença.

**Não dá para resolver por filtro.** Foram testados sete recortes da lista do Edital (tirando
faltoso, nota zero, redação zero, tipo D zero) e nenhum reproduz o oficial. O desvio erra por
−1,5 em todos eles: para o desvio subir seria preciso gente com mais dispersão, para a média subir
seria preciso gente com nota mais alta, e as duas direções se contradizem. Existe uma população que
o Cebraspe usa e que não temos.

**O que ficou provado de bônus:** o método de extração. Seis Editais, 19,5 mil registros cada,
**zero falhas** no checksum embutido (`EB parte 1 + EB parte 2 = somatório`), e as notas batem em
99,63% com o CSV para os alunos que aparecem nos dois lados.

**Armadilha de documento, registrada:** "Retificação" não quer dizer parcial nem completo — tem que
conferir. O Edital 8 de 2023 (retificação) tem **827** registros e não serve; o Edital 7 do mesmo ano
tem 19.505. Já em 2022 foi o contrário: o Edital original **não trazia os escores brutos das partes
1 e 2**, que só apareceram na retificação. Sempre conferir a contagem antes de usar.

Medições em `.scratch/publicar-site/medicao-passo-1/`.

---

### Passo 2 — Extrair os Editais de Etapa e calibrar o deslocamento

**O que é.** Quatro coisas, nesta ordem:

1. **Produzir as duas entradas que faltam.** Já extraídas dos Editais que estão em `data/pdfs`:

   | | `(2024, Etapa 1)` | `(2025, Etapa 2)` |
   |---|---:|---:|
   | candidatos | 19.127 | 16.990 |
   | `m_p2` / `dp_p2` | 23,906 / 11,398 | 27,644 / 14,752 |
   | `m_red` / `dp_red` | 6,471 / 2,292 | 6,316 / 2,251 |
   | Parte 1 misturada | 2,787 / 2,466 | 3,066 / 3,100 |

2. **Calibrar o deslocamento.** Os +7,87 estão medidos em **um** triênio para a Etapa 2 e dois para
   a Etapa 1 — e os dois da Etapa 1 já divergem entre si (+1,23 em 2022, +1,81 em 2023). Sem mais
   pontos, a correção é um número solto. Com os Editais isolados de Etapa 1 e 2 de mais três ou
   quatro triênios fechados, ela ganha média e dispersão próprias. É só download: o extrator já
   roda nessa família de documento sem alteração.

3. **Resolver a forma do `ExamStats`.** O ticket 12 tornou `parte_1` um campo obrigatório com as
   três línguas, e o Edital isolado **não diz a língua de cada candidato** — só dá a Parte 1
   misturada. Preencher as três exigiria inventar valores. A saída natural é o `ExamStats` aceitar
   uma Parte 1 misturada explicitamente marcada como derivada, em vez de fingir três línguas.

4. **Marcar as entradas como derivadas.** Quando o Edital de verdade sair em 2026, esses números
   serão substituídos e as previsões vão mexer. Isso precisa estar registrado, não descoberto depois.

**Por que aqui.** É o único bloqueador de produto que sobra no Preditor. Os itens 1 e 3 destravam a
resposta; o item 2 decide se ela sai calibrada ou com um viés de +7,87 embutido.

**O que já não é mais problema.** A língua estrangeira, que parecia o bloqueador, custa **0,46 ponto
de Argumento Final em média** (máx 3,21) e tem viés zero — é ruído, não erro sistemático. A Parte 1
pesa 0,72 numa conta que soma 10, e a média misturada cai praticamente em cima da média da inglesa,
que é 66% a 73% da população.

**Pronto quando** o `OFFICIAL_STATS` tiver 26 entradas com as duas novas marcadas como derivadas, o
deslocamento estiver calibrado sobre pelo menos quatro triênios, e o Preditor responder para um
aluno 2024-2026 sem levantar `EstatisticasIndisponiveisError`.

---

### Passo 3 — Hospedar a API

**O que é.** Dockerfile + Space no Hugging Face + as três coisas que a imagem precisa e o git não
carrega: o pacote em `models/pas3/`, os CSVs de `data/`, e (mais tarde, no B2B) os templates de
`assets/`. Mais o CORS, que hoje libera `"https://*.vercel.app"` como texto literal — o Starlette não
trata isso como curinga, e `vetorpas.com.br` nem está na lista. O Preditor chama a API do navegador,
então isso falha na primeira requisição em produção.

**Por que aqui, e não antes.** Não depende de 1 nem de 2 — pode andar em paralelo desde já se você
tiver braço. Está em terceiro porque é trabalho de infraestrutura que não muda de forma conforme a
resposta do Passo 1, então adiantá-lo nunca é errado, só não é o que decide.

**Esta é a dívida (a) do ticket 13:** o domicílio decidido no ticket 03 — repositório privado no
Hugging Face, artefato assado na imagem no build, promoção por commit de ponteiro — **nunca foi
feito**. Hoje o pacote existe só na sua máquina; máquina nova sobe sem modelo, e reverter é manual.

**Pronto quando** o `/health` responder numa URL pública, o Preditor funcionar num navegador contra
essa URL, e uma máquina limpa reproduzir o deploy sem cópia manual de arquivo.

---

### Passo 4 — Trocar os CSVs velhos pelos novos

**O que é.** A API lê `data/notas_corte_pas.csv` (2.307 linhas, base ad-hoc antiga) e
`data/banco_alunos_pas_final.csv`. A frente de extração já produziu `notas_corte.csv` (**4.786**
cortes, incluindo 2023-2025) e `resultado_final.csv` (66.313 registros, 8 triênios). A troca nunca
foi feita — o modelo novo está sendo servido contra notas de corte velhas.

**Ordem interna que não dá para inverter:** antes de promover o `notas_corte.csv`, feche o **ticket
14** da frente de extração (validação de formato do campo de classificação). Sem ele, cortes
implausíveis passam — o caso conhecido é MEDICINA/Darcy/Universal em 2020-2022 saindo com
`199.162,872`. Um corte desses no Preditor público vira uma probabilidade absurda na tela de um
aluno.

**O `resultado_final.csv` tem o mesmo tipo de sujeira, e o checksum pega tudo.** 510 linhas (0,77%)
têm nota com escala corrompida — `eb_p2` chegando a 39.617. **Todas as 510** falham o
`checksum_fecha` do ticket 04. Filtrar por `checksum_fecha == True` deixa 64.298 de 66.313, e a
contaminação está só nos cinco triênios mais antigos; os três recentes estão limpos.

**Pronto quando** as duas fontes novas estiverem em produção, e uma varredura de plausibilidade nos
4.786 cortes não achar nenhum valor fora de faixa.

---

### Passo 5 — Unificar as branches e mergear o público na `main`

**O que é.** Trazer `feat/nextjs-frontend` (o portal) para cima de `feat/pdf-extraction` (o modelo).
Simulado: **5 conflitos**, dois de verdade — `api/services/predict_service.py` e `PreditorPage.tsx`,
porque os dois lados reescreveram o Preditor por motivos diferentes (um trocou o miolo para `A3` +
incerteza do manifesto; o outro acrescentou semestre, curso alvo e persistência do aluno logado). Os
outros três são mecânicos. A branch do portal ainda importa o `ensemble.py`, que foi aposentado.

**A Calculadora de Estratégia: a decisão mudou de forma.** O mapa dizia que havia duas saídas —
deixá-la fora, ou reconstruir o estimador de P1/Redação (trabalho de modelo). **Existe uma terceira,
e ela já está desenhada:** o **Estimador Auxiliar** do relatório 04 (Alvo Canônico, §2.1 e §7.1).
Em vez de prever P1 e Redação com `.joblib`, prevê-los por média ponderada de z-scores — aritmética
sobre notas que já temos:

```
Â3                ← única previsão do modelo
P1̂, R̂ed           ← Estimador Auxiliar (média dos z das Etapas 1 e 2) + override do Aluno
P2                = resolvido:  z_p2 = (A3 − 0,72·z_p1 − 1,00·z_red) / 8,28
```

Isso **remove** o carregamento de `.joblib` do `target_calculator.py` em vez de consertá-lo, e mata
por remoção o defeito 3 de `defeitos-pendentes.md` — o `ModuleNotFoundError: _loss` que hoje é
engolido na linha 66, fazendo a Calculadora responder por média ponderada sem avisar ninguém.

**Medido nos três triênios recentes:** o Estimador Auxiliar erra 1,47 ponto em P1 e 1,36 na
Redação. Com `A3` fixo, o erro de P1 é amortecido em 60% (move o P2 necessário em ~0,59); o da
Redação passa quase inteiro (~1,29) — e é exatamente ali que a caixa de override do aluno vale mais.

**A faixa da P2 também deixou de ser chute.** Ela era o segundo bloqueador da Calculadora: as
constantes `P2 ∈ [−100, 100]` decidem sozinhas quando o produto diz "impossível" e quando diz
"garantido", e ninguém sabia os valores certos. Agora estão medidos — Etapa 3, 8 triênios,
~64 mil alunos:

| | Chute atual | Medido |
|---|---:|---:|
| Piso de P2 | −100 | **0,24** (0% negativo em 8 triênios) |
| Teto de P2 | +100 | **85,6** (o maior de 64 mil alunos) |
| P2 no percentil 99,9 | — | ~78 |
| Teto de `EB = P1 + P2` | — | 92,3 |

O teto teórico continua 100, porque o fator de normalização existe para que acertar tudo dê 100 —
mas ele é de `P1 + P2` **juntos**, e a P1 sozinha já come até 8,5. O recorde absoluto de EB em oito
triênios é 92,3. Com a faixa antiga, uma nota necessária de 95 na P2 era classificada como
"possível"; com a medida, não é.

**A faixa é por etapa, não global:** nos Editais de Etapa 2, 2,3% dos candidatos ficaram abaixo de
zero (o pior em −19,6); na Etapa 3, zero em 64 mil.

Os quatro status que o código já tem passam a ter significado medido:

- **impossível** — a nota necessária passa de `100 − P1 estimado`. Aritmética, não opinião.
- **improvável** — passa de ~85,6, o recorde histórico. Existe no papel, nunca aconteceu.
- **possível** / **garantido** — o resto, como já é.

**O que a Calculadora *não* precisa:** o modelo de correção item a item (110 itens na Parte 2,
tipos A/B/C/D com pesos 1/2/2/3, desconto por erro, fator de normalização). Isso alimenta o
**Simulador de Itens**, que é outra tela e depende de saber quantos itens de cada tipo tinha cada
prova — dado que **não sai em Edital**, só no caderno de questões, e exigiria um parser que não
existe. Confundir as duas coisas foi o que manteve a Calculadora bloqueada por engano.

Ou seja: a Calculadora **pode entrar na primeira publicação**, e entrar deixando o código mais
simples do que está. Os dois bloqueadores que ela tinha caíram — o estimador de P1/Redação vira
aritmética, e a faixa da P2 vira número medido. **Recomendação: incluir nesta rodada.** O custo é
reescrever o `target_calculator.py` (remoção de código) e trocar duas constantes.

**Ganho barato e independente:** `feat/proof-section` tem 14 commits de visual da landing, já contém
a `main` inteira e não conflita com nada. Pode ir para produção sozinha, a qualquer momento, sem
esperar o resto do mapa.

**Pronto quando** a `main` tiver Preditor + Temporal + landing funcionando contra a API hospedada, e
o deploy da Vercel estiver verde.

---

## Rota até produção

Uma sessão por ticket, limpando o contexto entre elas. **O número da rota é o número do arquivo**
nesta frente — os tickets já nasceram em ordem de dependência. O status de cada um vive no próprio
arquivo, nunca aqui: esta seção só ordena e diz quem precisa de você.

| Rota | Ticket | Você na sala? | Modelo / esforço |
|---|---|---|---|
| 1 | [`ExamStats` com Parte 1 misturada e procedência](issues/01-examstats-parte-1-misturada-e-procedencia.md) | em parte — o ADR quer sua assinatura | Opus, médio |
| 2 | [Extrator de Editais de Etapa vira módulo](issues/02-extrator-de-editais-de-etapa-vira-modulo.md) | não — delegável | Sonnet, médio |
| 3 | [CORS vindo do ambiente](issues/03-cors-vindo-do-ambiente.md) | não — delegável | Sonnet, baixo |
| 4 | [PII sai da `proof-section`, visual vai a produção](issues/04-pii-sai-da-proof-section-e-o-visual-vai-a-producao.md) | **sim** — force-push e merge na `main` | Opus, médio |
| 5 | [`TRIENNIUM_STATS` sai, tudo lê `OFFICIAL_STATS`](issues/05-trienniumstats-sai-tudo-le-official-stats.md) | não — delegável | Sonnet, médio |
| 6 | [Calibração do Deslocamento e o portão](issues/06-calibracao-do-deslocamento-e-o-portao.md) | **sim, agora** — **reprovou** (resíduo 5,751 ≥ 5,009 em 2021/2023); 7 e 14 bloqueados até decisão | Opus, alto |
| 7 | [Preditor responde para a Turma viva](issues/07-preditor-responde-para-a-turma-viva.md) | **sim** — põe número derivado na frente do Aluno | Opus, alto |
| 8 | [API hospedada: Dockerfile, Space, pacote na imagem](issues/08-api-hospedada-dockerfile-space-pacote-na-imagem.md) | em parte — credenciais do HF e o domínio são suas | Sonnet, alto |
| 9 | [Troca dos CSVs de Nota de Corte e população](issues/09-troca-dos-csvs-de-nota-de-corte-e-populacao.md) | não — delegável | Sonnet, médio |
| 10 | [Merge do portal para cima do modelo](issues/10-merge-do-portal-para-cima-do-modelo.md) | **sim** — dois conflitos onde escolher um lado perde metade | Opus, alto |
| 11 | [Calculadora sem `.joblib`: Estimador Auxiliar e faixa medida](issues/11-calculadora-sem-joblib-estimador-auxiliar-e-faixa-medida.md) | em parte — as decisões estão tomadas, a cirurgia é grande | Opus, alto |
| 12 | [Ano-Âncora: cinco anos reais na tela](issues/12-ano-ancora-cinco-anos-reais-na-tela.md) | em parte — é tela nova | Opus, médio |
| 13 | [Língua por Etapa, ponta a ponta](issues/13-lingua-por-etapa-ponta-a-ponta.md) | não — o contrato já está escrito | Sonnet, médio |
| 14 | [Publicação: a `main` no ar](issues/14-publicacao-main-no-ar-contra-a-api-hospedada.md) | **sim** — vai ao ar | Opus, médio |

**Bloqueador externo:** o ticket 9 depende do
[ticket 14 da frente de extração](../pdf-extraction/issues/14-validacao-formato-classificacao.md)
(validação de formato do campo de classificação). Sem ele, cortes implausíveis passam.

### Quando dá para rodar em paralelo

```
onda 0            onda 1           onda 2      onda 3     onda 4    onda 5

01 ExamStats ───→ 05 TRIENNIUM ──→ 10 merge ─┬→ 11 Calc ─→ 12 Âncora ─┐
      │                                      └→ 13 língua ────────────┤
      └────────────────────────┐                                      │
02 extrator ────→ 06 portão ───┴→ 07 Preditor ────────────────────────┤
                                                                      │
03 CORS ────────→ 08 hospedar ────────────────────────────────────────┤
                                                                      │
04 PII + landing ─────────────────────────────────────────────────────┤
                                                                      │
(extração #14) ─→ 09 CSVs ────────────────────────────────────────────┤
                                                                      ▼
                                                                14 no ar
```

**Cinco frentes independentes na onda 0** — 01, 02, 03, 04 e o ticket 14 da extração. Nenhuma
espera nenhuma. Se você quiser paralelizar em máquinas ou sessões diferentes, é aqui que rende.

**O caminho crítico não é o do modelo, é o da Calculadora:**
`01 → 05 → 10 → 11 → 12 → 14`, seis tickets de profundidade. A cadeia do Preditor
(`02 → 06 → 07 → 14`) tem quatro e a da infraestrutura (`03 → 08 → 14`) tem três — as duas cabem
inteiras dentro do tempo da primeira. **Comece pelo 01**, não pelo 02: adiar o 01 empurra a data
de publicação um a um, e adiar o 02 não.

**Onde o paralelismo morre:** o ticket 10 (merge) é o gargalo do lado do produto — 11, 12 e 13
todos passam por ele. Vale chegar nele cedo, e é por isso que o 05 (que o bloqueia) é delegável de
propósito.

**Três pontos onde a rota pode travar esperando você:**

1. **Antes do 6** — alguém precisa baixar os Editais isolados de Etapa 1 e 2 de mais três ou quatro
   triênios para `data/pdfs`. Não há automação de download. Faça isso enquanto o 2 roda.
2. **Antes do 8** — credenciais do Hugging Face e o domínio apontado. O agente não cria conta.
3. **No 6** — se o portão de calibração reprovar, 7 e 14 param e o mapa é reordenado. É o único
   risco vivo, e é por isso que o 6 vem cedo.

---

## O que fica de fora desta rodada (e onde está registrado)

Nada disto some — só não bloqueia publicar o lado público:

- **Todo o B2B.** Upload da base da escola e gravação no Supabase (era a tela "Análise Temporal" do
  Streamlit e é o onboarding inteiro do cliente), geração de PDF de verdade (arquivo, lote em ZIP,
  templates whitelabel, PDF de cursos, PDF de comparação). Inventário completo em
  `app/INVENTARIO-STREAMLIT.md`.
- **Defeito do nome quebrado** — 2,71% dos nomes saem com espaço no meio da palavra. O mesmo defeito
  aparece nos números dos Editais de Etapa (`2. 046`, `1 6.005`), e ali o checksum embutido o
  neutraliza. Não toca nenhum número; aparece quando o nome do aluno for impresso em relatório.
  Ticket 13 da extração.
- **O Simulador de Itens** — a tela que traduz "acertei tantos do tipo C" em nota. Depende do
  modelo de correção item a item (`[[project_regras_correcao_item_pas]]`) e da contagem de itens
  por tipo de cada prova, que **não sai em Edital**: só no caderno de questões, com um parser que
  ainda não existe. Vive só em `feat/nextjs-frontend` (`simulador_itens.py`, commit `d5d97ed`), com
  máximos hardcoded e sem o desconto por erro nem o fator de normalização. Não bloqueia nada nesta
  rodada — e, ao contrário do que se supunha, **não bloqueia a Calculadora** (ver Passo 5).
- **A população que o Cebraspe usa na Etapa 1.** Fica entre a lista do Edital e a dos concluintes, e
  nenhum recorte testado a reproduz. Não bloqueia: o deslocamento corrige o efeito sem que a causa
  esteja explicada.
- **O Streamlit quebrado** pelo ticket 13. Com o inventário em disco, consertá-lo virou opcional.

---

## O que pode reordenar tudo

1. ~~O Passo 1 dar negativo.~~ **Resolvido:** serve com correção.
2. **O deslocamento não é estável entre triênios — aconteceu.** Ticket 06 rodou com 6 triênios
   fechados (2018/2020 a 2023/2025, todos com Edital isolado de Etapa 1 **e** 2 em disco — 4 a
   mais do que os 3 pontos do Passo 1). O Deslocamento por Etapa saiu com dispersão real entre
   anos (Etapa 1: média 1,81, desvio 0,77 entre 6 anos; Etapa 2: média 3,21, desvio 0,35 entre 5
   anos), e aplicar a correção **média** a cada triênio deixa um resíduo que passa do limiar em
   pelo menos um deles: **2021/2023 fecha em 5,751**, acima do portão de 5,009. **O portão
   reprova.** Nenhuma entrada foi escrita no `OFFICIAL_STATS`; o Preditor continua recusando a
   Turma viva, e os tickets 07 e 14 seguem bloqueados até uma decisão do dono do produto — seja
   um deslocamento por (triênio próximo, Etapa) em vez de global, seja aceitar o risco residual,
   seja outra saída. Medição e decisões em
   `.scratch/publicar-site/relatorios/06-calibracao-do-deslocamento-e-o-portao.md`.
3. ~~Você decidir publicar a Calculadora.~~ **Resolvido:** ela entra nesta rodada. Não é passo de
   modelo nem de pesquisa — é troca do estimador por aritmética e de duas constantes por valores
   medidos, tudo dentro do Passo 5.

---

## Glossário deste mapa

- **`A1`, `A2`, `A3` (Argumento de Etapa):** a nota de uma etapa já padronizada pela média e pelo
  desvio daquele ano. `Argumento Final = A1 + 2·A2 + 3·A3`. `A1` e `A2` são conta exata; só `A3` é
  previsto. Como `A2` entra com peso 2, um erro na Etapa 2 vale o dobro de um erro na Etapa 1.
- **Média e desvio oficiais:** os números que o Cebraspe publica por Edital e que entram na conta do
  Argumento. Um Edital só por triênio, com as três etapas, e só depois do PAS 3.
- **População do Cebraspe:** o conjunto de candidatos sobre o qual ele calcula essa média. Na Etapa 2
  são os **concluintes do triênio**, não quem fez a prova — daí o erro de −4,61 e a necessidade de
  correção.
- **Deslocamento:** o degrau de +7,87 pontos de Argumento Final entre o estimado e o oficial. É
  sistemático, não ruído, e é por isso que subtrai-lo funciona.
- **Edital isolado de Etapa:** o "Resultado final nos itens do tipo D e na prova de redação" de uma
  Etapa 1 ou 2, publicado no ano da prova. Lista nota por candidato — inscrição, nome, EB parte 1,
  EB parte 2, somatório, nota tipo D, nota de redação — mas **não a língua estrangeira**.
- **Estimador Auxiliar:** prever P1 e Redação do PAS 3 pela média ponderada dos z-scores das etapas
  anteriores, em vez de por modelo. Aritmética, não ML. Relatório 04, §2.1.
- **Escore Bruto (EB):** `P1 + P2`, já normalizado para que acertar 100% das duas partes juntas dê
  100. Por isso o teto da P2 sozinha não é 100: é `100 − P1`.
- **Faixa de P2:** os limites que decidem quando a Calculadora diz "impossível" e quando diz
  "garantido". Eram o chute `[−100, 100]`; agora são medidos, e são **por etapa** — a Etapa 2
  admite nota negativa, a Etapa 3 não.
- **Simulador de Itens:** tela que traduz contagem de acertos por tipo de item (A/B/C/D) em nota.
  Outra coisa que a Calculadora — e fora desta rodada, porque depende de dado que só existe no
  caderno de questões.
- **Turma viva:** o triênio 2024-2026, que faz o PAS 3 em 2026. É quem procura o Preditor hoje.
- **Nota de corte:** o menor Argumento Final entre os aprovados de um curso, num Sistema de
  Concorrência, na última chamada.
- **Space (Hugging Face):** onde a API Python vai rodar. A Vercel só hospeda o Next.js; modelo Python
  não roda lá.
- **CORS:** a regra que diz de quais endereços o navegador pode chamar a API. Se
  `vetorpas.com.br` não estiver na lista da API, o navegador recusa a chamada.
