# Relatório — Ticket 09: conjunto de features

**Ticket:** `.scratch/treino-modelos-pas3/issues/09-conjunto-de-features.md`
**Status:** concluído
**Tipo:** medição
**Régua:** `src/pas_intelligence/validation.py` (ticket 06), mesma semente e mesmo recorte de
teste do ticket 07 (**37.844** linhas, 5 dobras, semente **20260728**)
**Script:** `scripts/features_ticket09.py`
**Linha de base:** a do ticket 07 — **linear em (`A1`, `A2`) + as 6 features legadas**, RMSE
**5,167** em `A3` (agrupado, geral)
**Privacidade:** `resultado_final.csv` foi lido só para casar `perfil_cota` e as 5 booleanas por
`id_pseudonimo`+`trienio`; nada além de agregado sai daqui.

---

## 1. Resultado — cada bloco, isolado, contra a base

| bloco | RMSE `A3` | MAE | viés | ganho |
|---|---:|---:|---:|---:|
| **BASE** (ticket 07) | 5,167 | 4,088 | +0,177 | — |
| `+ curso` | 5,145 | 4,071 | +0,154 | +0,43% |
| `+ campus + turno` | 5,165 | 4,086 | +0,140 | +0,03% |
| `+ lingua_e1/e2/e3` | 5,149 | 4,071 | −0,044 | +0,35% |
| `+ perfil_cota + 5 booleanas` | 5,170 | 4,091 | +0,188 | **−0,07%** |
| `+ ano_inicio` (trienio como numérica) | 5,240 | 4,140 | −0,486 | **−1,43%** |
| **`+ derivadas de trajetória`** | **5,057** | **4,008** | +0,215 | **+2,13%** |
| todos os blocos juntos (kitchen sink) | 5,040 | 3,993 | +0,021 | +2,45% |

Por classe (ticket 14 — dois números sempre):

| bloco | RMSE majoritária | RMSE minoritária |
|---|---:|---:|
| BASE | 5,053 | 6,028 |
| `+ curso` | 5,037 | 5,963 |
| `+ campus + turno` | 5,050 | 6,038 |
| `+ lingua_e1/e2/e3` | 5,034 | 5,984 |
| `+ perfil_cota + 5 booleanas` | 5,058 | 6,020 |
| `+ ano_inicio` | 5,092 | 6,224 |
| **`+ derivadas de trajetória`** | **5,014** | **5,357** |

**Só um bloco vale a pena: as derivadas de trajetória.** Elas são o único ganho que não é
marginal (2,13% contra ≤0,43% de tudo o resto), o único que também melhora a classe minoritária
de forma relevante (6,028 → 5,357, **11%**), e são **grátis** — derivadas das 6 features legadas
que já entram na base, sem exigir nada novo do Aluno.

O "tudo junto" ganha só **0,32 ponto percentual a mais** que as derivadas sozinhas (2,45% contra
2,13%), ao custo de somar `curso` (≈100 categorias), `campus`, `turno`, 3 colunas de língua e 6
de cota. **Não paga.**

---

## 2. Conjunto final declarado

```
[a1, a2, EB_PAS1, Red_PAS1, EB_PAS2, Red_PAS2, Cresc_EB, Cresc_Red,
 cresc_eb_pct, cresc_red_pct, sinal_cresc_eb]
```

As 6 legadas continuam por decisão do ticket 07 (§3: somar as 6 cruas ao `(A1,A2)` move menos de
meio ponto percentual sozinhas, mas o ticket 10 é quem decide a forma final do modelo — aqui elas
ficam como parte do que já bateu o Portão 1). As três novas são as razões que o `meta_scaler.joblib`
atual já usa (`|c_eb|/|eb_p1|`, `|c_red|/|red_p1|`, `sign(c_eb)`) — não são invenção deste ticket,
são a mesma ideia, só que testada como feature de regressão em vez de insumo de roteador.

RMSE do conjunto final: **5,057** em `A3` (contra o piso do Portão 1 do ticket 07: **≤5,167**
geral, **≤5,038** majoritária, **≤6,028** minoritária — bate as três, e a minoritária por uma
margem grande).

---

## 3. Por que cada candidato descartado foi descartado

### 3.1 `curso` — pequeno, real, e não paga o custo de produto

Isolado, `curso` ganha 0,43% e **é consistente**: melhora em todas as 5 dobras, nunca piora
(5,428→5,397, 5,005→4,993, 5,077→5,064, 5,106→5,093, 4,742→4,732 — série completa medida). Não é
ruído de dobra — é sinal real, só que pequeno.

**Não está disponível hoje.** `api/schemas/predict.py` — o schema do endpoint que faz a previsão
que este ticket abastece — não tem um campo `curso` de entrada. O único campo relacionado é
`curso_alvo: Optional[str]`, e é outra coisa: o curso que o Aluno **quer saber se passa**, usado
pelo calculador reverso (`target_calculator.py`), não o curso em que o Aluno **está matriculado**.
Adicionar `curso` como feature do modelo principal exigiria um campo novo e obrigatório na tela —
custo de produto por 0,43% de RMSE, que nesta base é **0,022 pontos absolutos**, bem dentro do
ruído entre dobras que o ticket 07 já mediu (**±0,37**).

**Verificação da armadilha do ticket:** `curso` não está codificando a Nota de Corte por vias
tortas. Regredi `A3` em (`A1`,`A2`), tirei a média do resíduo por curso (o que sobra depois de
descontar a trajetória do Aluno) e correlacionei com a Nota de Corte média do curso, em 134 cursos
casados: **correlação 0,126** — fraca. Se `curso` fosse majoritariamente um proxy do corte, essa
correlação sairia alta; o pouco que `curso` acrescenta é outra coisa (provavelmente perfil médio
de dificuldade da prova por área, não o corte em si). Mas isso é uma nota de rigor, não uma
mudança de decisão: **mesmo limpo da armadilha, o ganho continua pequeno demais para o custo.**

### 3.2 `campus` + `turno` — ganho nulo

+0,03%, dentro de qualquer margem de ruído razoável. Descartado sem debate — nem vale medir o
custo de produto.

### 3.3 `lingua_e1/e2/e3` — o teste específico contra a normalização de `pas_constants.py`

Item do checklist: `lingua_e*` avaliada contra a normalização de P1 por língua. **O resultado é
que a normalização já fez o trabalho pesado.** `build_training_dataset` (ticket 05) calcula `a1`,
`a2` e `a3` via `calculate_argument_etapa`, que consome `OFFICIAL_STATS[(ano, etapa,
lingua)].parte_1[lingua]` — ou seja, **a língua já normaliza a Parte 1 dentro de `A1`/`A2`**, que
já estão na base. As 6 legadas (`EB_PAS1`, `Red_PAS1`, ...), essas sim, usam `eb_p1_e1` cru, sem
normalização por língua — é aí que sobraria sinal de língua para capturar.

Medido: acrescentar `lingua_e1/e2/e3` cru ganha **0,35%** — pequeno, porque a maior parte do
efeito de língua já está embutida em `A1`/`A2`. O que sobra é o resíduo específico das 6 features
cruas, e é pouco. **Decisão:** não incluir. O ganho é do mesmo porte (e mesma conclusão) que
`curso` — real mas marginal — e, ainda que `lingua_e1`/`lingua_e2` sejam baratas de coletar
(o Aluno sabe informar, ticket 09 nota isso), não há ganho suficiente para justificar 3 colunas
categóricas a mais na receita.

### 3.4 `perfil_cota` + 5 booleanas — decisão fechada pelo próprio número, sem precisar pesar a ética

Isolado, o bloco **piora** o RMSE agrupado em 0,07% (5,167 → 5,170). A dimensão ética que o
ticket pede para decidir "explicitamente" — o custo de pedir dado sensível ao Aluno — não chega a
entrar em jogo: **não há ganho estatístico para pesar contra o custo ético.** A decisão é
rejeitar, e ela não depende de opinião sobre a ética, só do número.

Nota à parte: `perfil_cota` (como `cota`, com default `"Sistema Universal"`) **já é coletado**
pelo endpoint de previsão hoje (`api/schemas/predict.py:12`) — então a rejeição aqui não é por
indisponibilidade, é porque adicioná-lo mede pior.

### 3.5 `trienio` — a armadilha confirmada por medição, não só por argumento

O ticket avisa que `trienio` como categoria não generaliza para o ano seguinte. Testei a versão
"seguro" que o próprio ticket sugere — variável temporal contínua (`ano_inicio`, o primeiro ano do
triênio, nunca categoria) — e ela **piora 1,43%**, a pior de todas as opções testadas, com o viés
saltando para **−0,486** (quase o teto de ±0,5 do Portão 1). A razão é a mesma que o ticket previu
por argumento: a régua treina no passado e testa no futuro, e uma tendência linear em ano
aprendida sobre triênios passados **extrapola** para o triênio de teste em vez de generalizar — e
o ticket 08 já tinha mostrado que a relação entre Etapas 1/2 e a Etapa 3 é estável ano a ano na
escala do Argumento, então não existe tendência real para a extrapolação capturar; só ruído
sistemático. **`trienio` (em qualquer forma) fica fora.**

---

## 4. Checklist do ticket

- [x] Ganho de cada bloco de feature medido isoladamente sobre o holdout do ticket 06, contra a
      linha de base do ticket 07 — §1
- [x] Disponibilidade de cada feature vencedora confirmada com o produto — só as derivadas de
      trajetória venceram, e são derivadas de dado já coletado; nenhuma feature que exigiria
      mudar a tela passou do próprio filtro estatístico, então não há confirmação de produto
      pendente
- [x] `lingua_e*` avaliada especificamente contra a normalização de P1 por língua do
      `pas_constants.py` — §3.3: a normalização já está embutida em `A1`/`A2`, por isso o ganho
      do bruto é pequeno
- [x] `trienio` tratado como efeito temporal ou ponderação, nunca como categoria — testado como
      numérica contínua (nunca categórica) e descartado por piorar — §3.5
- [x] Verificado se `curso` está atuando como proxy da Nota de Corte — correlação 0,126 do
      resíduo por curso com o corte médio do curso: fraca — §3.1
- [x] Decisão sobre features de cota tomada explicitamente, incluindo a dimensão ética — §3.4:
      rejeitada por não ter ganho, a ética não precisou decidir
- [x] Conjunto final de features declarado, com o ganho de cada uma — §1, §2
- [x] Relatório em `relatorios/09-conjunto-de-features.md`

---

## 5. Limitações

- **O teste de `curso` usou one-hot + regressão linear**, não a família de modelo que o ticket 10
  vai escolher. Um modelo em árvore poderia extrair mais de `curso` (interações, splits por
  categoria) do que uma regressão linear consegue — o número de 0,43% é o ganho **sob um modelo
  linear**, não um teto absoluto do que `curso` pode valer. Se o ticket 10 escolher uma família de
  árvore, vale re-testar `curso` sob ela antes de fechar de vez a exclusão.
- **A correlação curso×corte (§3.1) é sobre 134 cursos**, cada um com um único ponto (a Nota de
  Corte média do Sistema Universal). Não é uma medição por Aluno, é uma medição por curso —serve
  para descartar "curso é essencialmente o corte", não para caracterizar toda a relação.
- **`perfil_cota` foi testado como bloco único** (a categoria + as 5 booleanas juntas). Não testei
  cada booleana isolada — dado que o bloco inteiro já piora, não há razão para procurar dentro
  dele um componente que ganhe.

---

## 6. Onde continuar

- **Ticket 10 (família de modelo):** recebe o conjunto de §2 como a receita de features a testar
  contra a linha de base do ticket 07. Vale a nota da limitação de `curso` acima se o ticket 10
  testar árvores.
- O timebox do mapa (ticket 06, §3 do relatório 07) permanece: o teto está a 0,2% do melhor
  baseline, e este ticket também não o quebrou — moveu 2,13%, dentro da faixa "afinação", não
  "salto".
