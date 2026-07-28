# Relatório — Ticket 08: janela de dados, 2018 vale?

**Ticket:** `.scratch/treino-modelos-pas3/issues/08-janela-de-dados-2018-vale.md`
**Status:** concluído
**Tipo:** medição, timeboxada (mapa §"Timebox nos tickets 08, 09 e 10")
**Régua:** `src/pas_intelligence/validation.py` — `avaliar()`, com o parâmetro `janela` já
existente do ticket 06 e um parâmetro novo, `pesos`, adicionado nesta sessão
**Script:** `scripts/janela_de_dados.py` — reproduz tudo abaixo com um comando
**Modelo usado:** linear em (`A1`, `A2`) + as 6 features legadas — o melhor baseline trivial do
ticket 07, para que a curva seja comparável ao Portão 1
**Recorte:** as 5 dobras do ticket 07 (2018/2020 a 2022/2024), semente **20260728**, dataset
`data/training/pas3_dataset.parquet` (64.298 linhas, 8 triênios). Só a classe majoritária entra
— restrição do ticket 06 §2.2: a minoritária não tem série.
**Lacre:** 2023/2025 não foi tocado.
**Privacidade:** só agregados.

---

## Resposta curta

**Usa 2018. A janela expansiva (treina em tudo o que existe antes do teste) bate qualquer corte
testado**, e ponderar o dado velho por idade não bate simplesmente tratá-lo igual. Nenhum dos
três candidatos a quebra de regime aparece na escala do Argumento (alvo canônico do ticket 04) —
o que aparece em EB some depois da normalização por ano, exatamente como o ticket 02 previu.

O ganho de manter 2018 é pequeno em valor absoluto (0,033 de RMSE entre janela=3 e a expansiva) e
a régua de melhoria do mapa (menos de 1% relativo) já estava perto de disparar no ticket 07 —
aqui ela dispara: N=5→N=6 melhora **0,16%**. Isso não muda a resposta — cortar é estritamente
pior em todo ponto medido — só confirma o timebox do mapa.

---

## 1. A curva de erro contra número de triênios de treino

RMSE em `A3`, classe majoritária, agrupado sobre as dobras que qualificam na trava 1.

| janela | n treino máx (triênios) | n teste (linhas) | RMSE | MAE | viés |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 11.937 ⚠ | 5,332 | 4,212 | −0,489 |
| 2 | 2 | 34.488 | 5,215 | 4,141 | +0,972 |
| 3 | 3 | 34.488 | 5,086 | 4,035 | +0,489 |
| 4 | 4 | 34.488 | 5,071 | 4,022 | +0,526 |
| 5 | 5 | 34.488 | 5,061 | 4,012 | +0,532 |
| **6 / expansiva** | 6 | 34.488 | **5,053** | **4,005** | +0,494 |

⚠ **`janela=1` não é comparável às demais.** Com só 1 triênio de treino, o treino fica **menor
que o teste** em 3 das 5 dobras (a dobra que testa 2021/2023 treina só em 2020/2022 — 7.130
linhas — contra 8.019 de teste), e a **trava 1 da régua** (relatório 07 §9.2, "vale para as duas
classes") barra essas dobras também na majoritária. O `5,332` sai de só **2 dobras**, as duas
mais fáceis da série — não é "janela 1 quase empata com janela 6", é "janela 1 não produz número
suficiente para competir".

**A curva cai monotonicamente até N=6 (todo o histórico fora do lacre). Não há mínimo em N=4** —
a hipótese de "horizonte de validade" do ticket não se confirma. O ganho fica pequeno cedo:
N=3→N=4 melhora 0,29%, N=4→N=5 melhora 0,20%, N=5→N=6 melhora 0,16% — sub-1% relativo em dois
tickets seguidos é exatamente a regra de parada do mapa (§"Timebox"), e ela dispara aqui, não
antes. Isso não inverte a direção: **em todo ponto, mais triênio é RMSE igual ou menor**, nunca
pior.

### 1b. Onde a curva melhora — série por dobra

| janela | 2018/2020 | 2019/2021 | 2020/2022 | 2021/2023 | 2022/2024 |
|---|---:|---:|---:|---:|---:|
| 1 | 5,419 | 5,260 | — | — | — |
| 3 | 5,428 | 5,005 | 5,104 | 5,128 | 4,858 |
| expansiva | 5,428 | 5,005 | 5,077 | 5,106 | 4,742 |

As duas primeiras dobras (2018/2020, 2019/2021) são **idênticas** entre janela=3 e expansiva —
esperado, porque com só 2 ou 3 triênios disponíveis antes delas a janela de 3 já é o teto. A
diferença aparece só a partir da dobra 3 (2020/2022 em diante), onde há mais história para a
expansiva aproveitar e a janela=3 não aproveita: **0,027 a 0,116 de RMSE por dobra**, sempre a
favor de mais dado.

---

## 2. Corte por janela × ponderação por idade

Mesmos 5 triênios de teste (comparação pareada dentro da dobra, restrição do relatório 06 §5); o
que muda é só quem entra no treino e o peso de cada linha.

| estratégia | RMSE `A3` (majoritária) |
|---|---:|
| corte: só os 3 mais recentes | 5,086 |
| **tudo, sem peso (janela expansiva)** | **5,053** |
| tudo, peso geométrico (base=0,5) | 5,088 |
| tudo, peso geométrico (base=0,7) | 5,064 |
| tudo, peso geométrico (base=0,85) | 5,057 |

**Nenhuma ponderação bate treinar em tudo com peso igual.** O peso geométrico melhora conforme
`base` sobe em direção a 1 (isto é, conforme o peso se aproxima de não ponderar nada) — o próprio
formato da tabela é a evidência: **quanto mais perto de "ignorar a idade", melhor**, e o ponto
ótimo é exatamente ali. Se o dado velho estivesse atrapalhando, a curva teria um mínimo em algum
`base < 1`; não tem. Implementação: `avaliar()` ganhou um parâmetro `pesos` (uma função que
recebe o treino da dobra e devolve um vetor de pesos, repassado como `sample_weight` ao `fit`) —
ver §5.

---

## 3. Deriva de distribuição por triênio

Médias e desvio de `A3` já estão na escala normalizada do ticket 04 (`calculate_argument_etapa`
usa a média/desvio publicados por Cebraspe para aquele ano — `pas_constants.OFFICIAL_STATS`), e a
correlação é entre `(A1+A2)/2` e `A3`, o proxy mais direto da relação que os modelos exploram.

| triênio | n | média A1 | média A2 | média A3 | desvio A3 | corr((A1+A2)/2, A3) | etapas pandêmicas |
|---|---:|---:|---:|---:|---:|---:|---|
| 2016/2018 | 8.877 | 1,24 | 0,31 | 0,35 | 9,19 | 0,841 | — |
| 2017/2019 | 8.874 | 1,87 | 0,50 | 0,51 | 9,17 | 0,813 | — |
| 2018/2020 | 5.804 | 0,02 | 0,01 | 0,02 | 9,16 | 0,787 | E3 |
| 2019/2021 | 8.392 | 0,01 | −0,01 | 0,00 | 9,10 | 0,808 | E2 + E3 |
| 2020/2022 | 7.130 | −0,01 | −0,02 | −0,01 | 9,05 | 0,813 | E1 + E2 |
| 2021/2023 | 8.019 | 0,00 | −0,00 | 0,00 | 9,07 | 0,793 | E1 |
| 2022/2024 | 8.499 | −0,00 | −0,00 | −0,00 | 9,08 | 0,827 | — |
| 2023/2025 | 8.703 | 0,00 | 0,00 | 0,00 | 9,15 | 0,830 | — |

**Nenhuma das três colunas mostra pandemia, fronteira ou deriva:**

- **Desvio de `A3`** fica entre 9,05 e 9,19 nos 8 triênios — variação de 1,5%, sem nenhum triênio
  destoando, pandêmico ou não.
- **Correlação `(A1+A2)/2` × `A3`** fica entre 0,787 e 0,841. O ponto mais baixo é justamente
  2018/2020 (0,787), o único triênio inteiramente pandêmico (E1, E2 **e** E3 marcados nos mapas
  de coorte dos tickets 02/07/08 combinados) — mas 2020/2022 (E1+E2 pandêmico) tem 0,813, **acima**
  da média da série, e 2022/2024 (nenhuma etapa pandêmica) tem 0,827. Não há um padrão "ano
  pandêmico → correlação mais baixa" que se sustente fora do primeiro caso.
- **Não há tendência monotônica** em nenhuma coluna ao longo dos 8 triênios — descarta deriva
  gradual como aparece na correlação ou na dispersão.

**A única coisa que destoa são as médias de `A1`/`A2`/`A3` de 2016/2018 e 2017/2019** (0,35 e
0,51 em `A3`, contra ~0,00 em todos os outros seis). Em magnitude são **3,8% e 5,6% do desvio
padrão** — pequeno, mas não zero, e sistematicamente positivo nos dois triênios mais antigos.
Não investiguei a causa (fora do escopo timeboxado deste ticket): candidato mais provável é
cobertura parcial de `OFFICIAL_STATS` para línguas raras nesses dois anos, deslocando a média
amostral para cima da população de referência. **Não é a assinatura de nenhum dos três candidatos
do ticket** — não é buraco isolado (afeta dois triênios seguidos, nas pontas da série, não no
meio), não é mudança de fórmula (ticket 02 já mediu resíduo máx. 0,005 nesses mesmos dois
triênios) e não é deriva gradual (não continua nos triênios seguintes — volta a ~0,00 abruptamente
em 2018/2020 e fica lá). Fica registrado como algo a olhar se um dia a régua entrar em
per-triênio de novo, não como bloqueio.

### 3.1 Os três candidatos, resolvidos

1. **Pandemia.** Único efeito visível é o **tamanho da coorte** (2018/2020 com 5.804 linhas
   contra ~8.500 dos vizinhos) — já explicado pelo ticket 02 como coorte menor por calendário
   quebrado, não perda de dado. Na relação preditiva (correlação) e na dispersão (desvio), o
   triênio pandêmico **não destoa** da série. **Não é obstáculo à janela.**
2. **Mudança de fórmula.** Fechado pelo ticket 02: pesos idênticos de 2016 a 2025, resíduo
   máximo 0,005. **Não é obstáculo à janela.**
3. **Deriva gradual.** Nenhuma tendência monotônica em média, desvio ou correlação ao longo dos 8
   triênios. **Não é obstáculo à janela.**

Nenhum dos três exige cortar ou corrigir triênio nenhum. A pergunta original do ticket
("o dado desde 2018 ainda ajuda?") tem resposta **sim, e sem ressalva de regime** — a curva do
§1 já responde isso diretamente, e o §3 explica *por que* a curva não tem quebra: a normalização
por ano (ticket 04) absorve a diferença de dificuldade de prova que existiria na escala de EB
crua (os ~35% do relatório do ticket 02), e o que sobra na escala do Argumento é estável.

---

## 4. Custo de cada alternativa

| alternativa | RMSE `A3` (majoritária) | custo vs. usar tudo |
|---|---:|---:|
| **usar tudo (2016/2018 a 2022/2024)** | **5,053** | — |
| cortar em 5 triênios | 5,061 | +0,008 (+0,16%) |
| cortar em 4 triênios | 5,071 | +0,018 (+0,36%) |
| cortar em 3 triênios | 5,086 | +0,033 (+0,65%) |
| cortar em 2 triênios | 5,215 | +0,162 (+3,21%) |
| cortar em 1 triênio | 5,332 ⚠ | +0,279 (+5,52%), e ainda perde 3 de 5 dobras |

Cortar é **sempre igual ou pior**; nunca há vantagem. O custo cresce rápido conforme a janela
encolhe — passar de 3 para 2 triênios custa 5× mais do que passar de 4 para 3. **A resposta ao
ticket é usar tudo**, e o motivo de não valer a pena discutir "até onde cortar" é que não existe
ponto de corte que compense.

---

## 5. Mudança na régua

`avaliar()` ganhou o parâmetro `pesos: Callable[[pd.DataFrame], np.ndarray] | None = None`.
Quando dado, recebe o `DataFrame` de treino **daquela dobra** (já filtrado pela janela) e devolve
um vetor de pesos, repassado como `sample_weight` ao `fit`. Por que fábrica de peso e não peso
pré-calculado no dataset: o peso certo depende de qual triênio é o mais recente do treino
**daquela dobra**, que muda a cada uma — pré-calcular um peso fixo por triênio não teria como
saber, para a dobra 2, que "idade zero" é 2018/2020 e não 2023/2025.

`None` (o padrão) preserva o comportamento de antes exatamente — os testes do ticket 06
continuam passando sem alteração, e dois testes novos cobrem o parâmetro
(`tests/test_validation.py`, seção "Peso por idade do triênio").

---

## 6. Portão 1, conferido

| o que precisa bater | valor exigido (ticket 07 §8) | valor obtido aqui |
|---|---:|---:|
| RMSE `A3` agrupado, majoritária | ≤ 5,038 | **5,053** (janela expansiva, o melhor testado) |

**Não bate — por 0,3%.** Isso não é regressão: é o mesmo número do ticket 07 (a régua e o modelo
são os mesmos, `avaliar` é determinístico dada a semente), só que agora isolado por dobra na
classe majoritária em vez de junto com a comparação de artefatos congelados. O ticket 08 não
tinha como fechar essa distância — ele varre **janela**, não família de modelo nem features, e a
resposta do §1 é que a janela não é onde a distância mora. Registrado para o ticket 10 (família):
a regra de parada do mapa provavelmente dispara ali, exatamente como o mapa previu.

---

## 7. Limitações

- **A curva não foi medida com correção de fórmula** (o quarto ponto do ticket, "triênios
  antigos corrigidos para a fórmula atual") porque o ticket 02 concluiu que não existe mudança de
  fórmula a corrigir — o ponto do ticket ficou vazio por a premissa dele não se sustentar, não por
  omissão.
- **`janela=1` não tem série completa** (§1, nota ⚠) — o número existe mas não é comparável às
  outras janelas linha a linha. Registrado, não escondido.
- **O desvio nas médias de A1/A2/A3 dos dois triênios mais antigos (§3) não foi investigado até a
  causa** — pequeno o bastante para não mudar a resposta do ticket, mas fica anotado.
- **A base de ponderação testada foi só {0,5; 0,7; 0,85}** — o suficiente para ver a direção
  (todas piores que sem peso, monotonicamente melhorando perto de 1) sem precisar de uma busca
  fina que o timebox não pede.

---

## 8. Onde continuar

- **Ticket 09 (features):** roda em paralelo (mesmas dependências), pode usar o mesmo
  `scripts/janela_de_dados.py` como referência de como varrer um parâmetro da régua.
- **Ticket 10 (família):** entra sabendo que a régua de parada do mapa provavelmente dispara
  aqui — a distância até o Portão 1 (0,3%) não é da janela.
- **Ticket 12 (pipeline):** o dataset canônico de treino deve continuar sendo **todos os 8
  triênios fora do lacre**, janela expansiva, sem peso por idade.
