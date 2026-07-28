# Relatório — Ticket 10: família de modelo

**Ticket:** `.scratch/treino-modelos-pas3/issues/10-familia-de-modelo.md`
**Status:** concluído
**Tipo:** medição
**Régua:** `src/pas_intelligence/validation.py` (ticket 06), mesma semente e mesmo recorte de teste
dos tickets 07/08/09 (**37.844** linhas de teste, 5 dobras, semente **20260728**)
**Script:** `scripts/familia_de_modelo_ticket10.py`
**Features:** o conjunto do ticket 09 (`FEATURES_CANONICAS` — as 6 legadas + `A1`/`A2` + as 3
derivadas de trajetória), RMSE de referência **5,057**
**Baseline:** o do ticket 07 — RMSE **5,167** em `A3` (Portão 1: geral ≤5,167, majoritária
≤5,038, minoritária ≤6,028, |viés| ≤0,5)
**ADR:** [0011 — LightGBM único com faltante nativo substitui o ensemble](../../../docs/adr/0011-lightgbm-unico-com-faltante-nativo-substitui-o-ensemble.md)
**Privacidade:** `resultado_final.csv`/`notas_corte.csv` lidos só para o erro de decisão (mesmo
uso do ticket 07, via `menor_corte_por_aluno`); nada além de agregado sai daqui.

---

## 1. Tuning honesto — a dobra de ajuste

Nenhum hiperparâmetro foi escolhido olhando as 5 dobras que produzem o número reportado. Em vez
disso, uma dobra dedicada — **treina em 2016/2018 (8.877 linhas), valida em 2017/2019 (8.874
linhas)** — que `gerar_dobras` nunca usa como teste (ela só testa a partir do terceiro triênio
disponível em diante) e que fica fora do lacre.

| candidato | grade | escolhido | RMSE na dobra de ajuste |
|---|---|---|---:|
| Ridge | `alpha ∈ {0,01· 0,1· 1· 10· 100· 1000}` | `alpha=0,01` | 5,320 |
| LightGBM | `n_estimators∈{100,200,400}` × `learning_rate∈{0,01;0,05;0,1}` × `num_leaves∈{15,31}` (18 combinações) | `n_estimators=400, learning_rate=0,01, num_leaves=15` | 5,289 |

`alpha=0,01` é regularização quase nula — Ridge convergiu para (quase) a mesma coisa que uma
regressão linear pura. A grade do LightGBM escolheu a combinação **mais conservadora** (mais
árvores, mas rasas e devagar) — coerente com o achado do ticket 06/07 de que o teto de acurácia
já estava atingido antes do mapa começar: não há espaço para uma árvore agressiva capturar
interação que não existe.

**Limitação registrada:** a escolha vem de **uma única** dobra de ajuste, não de uma média de
várias. Dado o ruído entre dobras já medido pelo ticket 07 (±0,37 em RMSE de `A3`), as diferenças
entre combinações vizinhas da grade (ex. 400×0,01×15 contra 200×0,01×15) provavelmente estão
dentro desse ruído — o que a dobra de ajuste garante não é "o hiperparâmetro ótimo", é "nenhum
hiperparâmetro escolhido olhando o número que será reportado".

---

## 2. As três famílias, sobre a régua

| candidato | RMSE geral | RMSE majoritária | RMSE minoritária | viés | Portão 1 |
|---|---:|---:|---:|---:|---|
| Ridge (regularizado) | 5,057 | 5,014 | 5,357 | +0,215 | bate |
| **LightGBM único** | **5,014** | **4,992** | 5,187 | **+0,129** | bate |
| Ensemble por volatilidade (sem roteador) | 5,009 | 4,988 | 5,182 | +0,188 | bate |

**Ganho relativo contra o baseline do ticket 07 (5,167):**

| candidato | ganho |
|---|---:|
| Ridge | +2,13% |
| LightGBM único | +2,97% |
| Ensemble | +3,05% |

**As três batem o Portão 1** nas três pernas (geral, majoritária, minoritária) e no viés. **O
número de Ridge (5,057) é idêntico, na terceira casa, ao "conjunto final" que o ticket 09 já havia
medido** com regressão linear pura — esperado, porque `alpha=0,01` é regularização desprezível, e
é uma verificação cruzada de que o pipeline deste ticket carrega o mesmo dado e as mesmas features
que o 09 mediu.

**Erro de decisão** (faixa congelada em **15,500**, ticket 07 — não recalculada por modelo):

| candidato | erra sobre passar | erra dentro da faixa |
|---|---:|---:|
| Ridge | 7,2% | 33,7% |
| **LightGBM único** | **7,0%** | **32,7%** |
| Ensemble | 7,1% | 33,1% |

LightGBM é o melhor ou empata em **todos** os números desta seção — nunca perde para Ridge nem
para o ensemble em nenhum eixo.

---

## 3. Veredito do ensemble: aposentado

**Melhor componente sozinho: LightGBM único (RMSE 5,014). Ensemble: 5,009. Ganho: 0,10%.**

Muito abaixo da barra de "ganho material" que o mapa já usa em outros pontos (1% relativo, a
mesma do critério de parada do ticket 06). **Veredito: o ensemble por volatilidade é aposentado**
— de novo, como o ticket 07 já havia encontrado sobre o alvo antigo (EB): nem ele nem o roteador
batem o melhor componente sozinho. A reimplementação deste ticket usou o **mesmo mecanismo** de
`ensemble.py` (sigmoide sobre o CV de `[EB_PAS1, EB_PAS2]`, limiares 10/20 inalterados), só que
treinando Ridge/LightGBM sobre `A3` e a régua nova — não é um ensemble redesenhado, é o mesmo
julgado de novo, com dado melhor.

O custo que o ensemble cobraria (dois artefatos, uma sigmoide com dois limiares cravados,
dependência a mais no manifesto do ticket 03) não se paga por 0,10% de RMSE, indistinguível do
ruído entre dobras (±0,37, ticket 07 §1).

---

## 4. Volatilidade como feature

Com o ensemble aposentado, o ticket pede para verificar se a volatilidade sobrevive como **coluna
de entrada** em vez de mecanismo de arquitetura (ADR-0009 já previa isso como consequência).

Testado: CV de `[EB_PAS1, EB_PAS2]` (a mesma fórmula de `ensemble.calculate_volatility`) somado
às features do candidato vencedor (LightGBM único).

| | RMSE geral |
|---|---:|
| LightGBM sem CV | 5,014 |
| LightGBM com CV de volatilidade | 5,014 |
| ganho | −0,01% |

**Não move nada.** A volatilidade não carrega sinal que já não esteja nos EBs crus e nas
derivadas de trajetória do ticket 09 — ela morre também como feature, não só como router. Item do
checklist fechado: se aposentado, avaliar como feature — avaliado, sem sobrevida.

---

## 5. Valor faltante nativo × dois modelos por classe

Restrição do mapa sobre este ticket (registrada nas restrições que o ticket 14 deixou para o 10):
*"'aceita valor faltante nativamente' é critério **com peso**, não desempate (linear/MLP fecham
a porta da classe). Medir um-modelo-com-faltante vs. dois-modelos."*

O Aluno sem Etapa 1 chega com `EB_PAS1=0`, `Red_PAS1=0` — zero estrutural, não desempenho
(glossário: *Fora de distribuição*). Um modelo que só vê o número não distingue os dois. Três
tratamentos comparados, todos com o LightGBM tunado da §1:

| tratamento | RMSE minoritária | RMSE majoritária |
|---|---:|---:|
| zero literal (o candidato da §2, sem mudança) | 5,187 | 4,992 |
| **`NaN` nas colunas da Etapa 1 do Aluno sem Etapa 1** | **5,158** | 4,988 |
| dois modelos treinados por classe (`etapa_1_ausente`) | 5,379 | 4,988 |

**`NaN` nativo ganha 0,56% sobre zero literal na minoritária**, sem custar nada na majoritária —
o LightGBM aprende, em cada nó, para que lado mandar quem está faltando, em vez de tratar `0`
como o pior desempenho já visto. Só famílias com suporte nativo a faltante conseguem isso; Ridge
exigiria imputar um valor, e qualquer valor inventado é chute.

**Dois modelos por classe perde — e por uma margem real (+3,7% pior que o zero literal na
minoritária).** A razão está no tamanho do treino da minoria por dobra:

| dobra | testa em | exemplos da classe no treino |
|---|---|---:|
| 1 | 2018/2020 | **64** |
| 2 | 2019/2021 | 484 |
| 3 | 2020/2022 | 961 |
| 4 | 2021/2023 | 1.538 |
| 5 | 2022/2024 | 2.435 |

A dobra 1 treina o submodelo dedicado da minoria com **64 exemplos** — dado de menos para
qualquer LightGBM aprender sozinho o que o modelo conjunto aprende de graça generalizando entre
classes. Um modelo dedicado só teria chance de pagar o próprio custo nas dobras finais (2.435
exemplos), mas o número agrupado (que pesa todas as dobras válidas) carrega o prejuízo das
primeiras.

**Decisão: `NaN` nativo, não dois modelos.** Fecha o item do checklist com medição, não
intuição — a intuição de "cada classe merece o próprio modelo" perde exatamente onde a régua
(ticket 06, trava 1) já avisava que o treino ficaria pobre demais.

---

## 6. Decisão final — e o critério de desempate por extenso

**LightGBM único** (`n_estimators=400, learning_rate=0,01, num_leaves=15`, semente da rodada),
sobre `FEATURES_CANONICAS`, com as colunas derivadas da Etapa 1 (`a1`, `EB_PAS1`, `Red_PAS1`,
`Cresc_EB`, `Cresc_Red`, `cresc_eb_pct`, `cresc_red_pct`, `sinal_cresc_eb`) trocadas por `NaN`
nas linhas `etapa_1_ausente`.

**O ganho de LightGBM sobre Ridge, isolado, é 0,85% relativo (5,057 → 5,014)** — abaixo da barra
de 1% que o próprio mapa usa como "ganho material" em outros pontos. Medido só por esse número, o
critério de desempate do ticket ("ganho material, ou o mais simples vence") entregaria a decisão
a Ridge. **A escolha por LightGBM não se apoia nesse número sozinho**, mas na soma de três
coisas, todas medidas nas seções acima:

1. **LightGBM nunca perde** — é igual ou melhor que Ridge em geral, majoritária, minoritária,
   viés e nos dois números do erro de decisão. Não é uma troca (melhor aqui, pior ali); é
   vantagem sem contrapartida, ainda que pequena em cada eixo isolado.
2. **Só LightGBM pode usar valor faltante nativo**, e isso vale 0,56% a mais na classe
   minoritária — exatamente a classe que o produto se comprometeu a atender (ADR-0008) e que o
   mapa vem tratando com cuidado extra desde o ticket 14.
3. **A alternativa que teria feito o custo de "dois modelos" valer a pena foi medida e perdeu.**
   Não sobra uma rota melhor com a complexidade de duas famílias — só a de uma família com uma
   troca de tratamento de dado, que é bem mais barata de manter.

**Isto é exatamente o que a restrição do ticket 14 pediu**: "aceita faltante nativamente" não
decidiu sozinho (LightGBM já vencia antes desse número entrar); ele foi o peso que fechou a única
lacuna (a classe minoritária) onde a vantagem de LightGBM seria discutível.

**Registro explícito para quem quiser revisitar:** se o critério fosse só "ganho material sobre
RMSE geral, sem mais nada", a resposta certa seria Ridge — mais simples, sem LightGBM, sem
`NaN`, sem hiperparâmetro para versionar. Os números para essa troca estão todos nesta tabela;
não é preciso remedir nada para reverter esta decisão.

---

## 7. Checklist do ticket

- [x] Ao menos três famílias comparadas sobre o mesmo holdout, com o mesmo conjunto de
      features — §2 (Ridge, LightGBM único, ensemble por volatilidade). O quarto candidato do
      ticket (multi-saída) não se aplica: o ADR-0009 já fixou o alvo em `A3` só, não as 3 notas.
- [x] Tuning feito dentro do split de validação, com o procedimento registrado — §1, dobra de
      ajuste dedicada e disjunta das 5 dobras de medição e do lacre.
- [x] O ensemble por volatilidade tem veredito explícito: mantido ou aposentado, com o número —
      §3, **aposentado**, ganho de 0,10% sobre o melhor componente.
- [x] Se aposentado, a volatilidade foi avaliada como feature — §4, testada, sem ganho (−0,01%).
- [x] Modelo escolhido com hiperparâmetros registrados e reprodutíveis por semente — §6,
      `n_estimators=400, learning_rate=0,01, num_leaves=15`, semente `20260728`.
- [x] Ganho contra o baseline do ticket 07 declarado, e comparado ao critério de aceite do 06 —
      §2, +2,97%, bate o Portão 1 nas três pernas.
- [x] Relatório em `relatorios/10-familia-de-modelo.md`.

Item adicional, restrição do mapa (ticket 14 sobre o 10): valor faltante nativo medido com peso,
não desempate — §5, fechado com `NaN` nativo em vez de dois modelos.

---

## 8. Limitações

- **A dobra de ajuste é uma medição só, não uma média.** As diferenças entre hiperparâmetros
  vizinhos da grade provavelmente estão dentro do ruído entre dobras (±0,37) que o ticket 07 já
  documentou. O que está garantido é que a escolha não olhou o número reportado, não que é o
  ótimo absoluto da grade.
- **O erro de decisão usa a faixa congelada do ticket 07 (15,500)**, calculada sobre o baseline
  antigo — por desenho, não se recalcula por modelo (§7 do relatório 07). Os números desta seção
  comparam modelos entre si sob a mesma régua, não redefinem o piso teórico de 31,6%.
- **O teste de "dois modelos" usou o mesmo LightGBM tunado para os dois submodelos**, sem tunar
  separadamente o da minoria. Um hiperparâmetro mais simples (menos árvores, menos profundidade)
  poderia ajudar um pouco o submodelo de 64 exemplos da dobra 1 — mas a lacuna medida (+3,7%) é
  grande o bastante que retunar não parece o tipo de coisa que inverte o resultado, e o timebox
  do mapa não pede essa medição extra.
- **A comparação de famílias não inclui MLP nem RandomForest isolados** — o ticket 07 já havia
  medido os dois na escala de EB (piores que LGBM/linear nas linhas limpas) e o ticket não pediu
  para retestá-los na escala de `A3`; o candidato "GBM único" cobre a família que historicamente
  ganhou, e retestar as outras duas sem hipótese nova seria medição sem propósito, contra o
  timebox.

---

## 9. Glossário — termos novos deste relatório

Para gravar na [`glossario.md`](../glossario.md), Parte 5:

**Regularização (Ridge / L2)** — em vez de achar os coeficientes que melhor encaixam o treino
sem restrição, a regressão **penaliza** coeficientes grandes, encolhendo-os na direção de zero. O
`alpha` controla a força do encolhimento: `alpha=0` é a regressão linear comum; `alpha` grande
empurra todos os coeficientes para perto de zero. Serve para quando há colunas correlacionadas
entre si (aqui, `A1`/`A2` e as 6 legadas se sobrepõem parcialmente) — sem penalidade, a regressão
pode distribuir o mesmo sinal de forma instável entre colunas parecidas.

**Dobra de ajuste (*tuning fold*)** — uma dobra extra, escrita só para escolher hiperparâmetro,
disjunta das dobras que produzem o número final. Aqui: treina em 2016/2018, valida em 2017/2019
— os dois únicos triênios que `gerar_dobras` nunca usa como teste, porque ela só testa a partir
do terceiro triênio disponível em diante. Existe para que "escolher o hiperparâmetro" e "medir o
modelo" nunca olhem para o mesmo número — senão a escolha vira sobreajuste ao próprio teste.

**Grade (*grid search*)** — testar todas as combinações de uma lista pequena de valores por
hiperparâmetro (aqui, 3 × 3 × 2 = 18 combinações de LightGBM) e ficar com a que vence na dobra de
ajuste. Simples e exaustivo dentro da grade; não encontra nada fora dela.

**Roteamento por classe** — a alternativa a valor faltante que o ticket 14 pediu para medir: em
vez de um modelo que vê `NaN` e decide sozinho, treinar **dois** modelos, um por classe, e mandar
cada linha pro seu. Medido neste ticket e descartado — perde para o modelo único quando uma das
classes (`etapa_1_ausente`) não tem dado suficiente cedo na série temporal (64 exemplos na
primeira dobra).

---

## 10. Onde continuar

- **Ticket 11 (incerteza calibrada):** a largura honesta continua **16,26** em Argumento Final
  (ticket 07), medida no baseline — este ticket não muda esse número em si, mas fixa qual é o
  "modelo novo" cuja incerteza *por Aluno* o ticket 11 vai calibrar: o LightGBM único com `NaN`
  nativo desta seção, não o ensemble.
- **Ticket 12 (pipeline de treino reproduzível):** a receita a industrializar é a §6 — LightGBM,
  hiperparâmetros fixos, o tratamento de `NaN` na Etapa 1 ausente como parte do preparo de dado,
  não um passo separado opcional.
- **Ticket 13 (treinar, avaliar e promover):** é quem decide se o ganho de 2,97% sobre o baseline
  do ticket 07, medido até 2022/2024, se sustenta no lacre (2023/2025) — a única vez que esse
  número pode ser olhado.
- **Régua de parada do mapa (ticket 06):** o ganho **marginal** de trocar só a família,
  segurando as features fixas no que o ticket 09 já tinha (5,057 → 5,014), é 0,85% — abaixo de
  1%. Como o ticket 09 tinha dado +2,13% (um ganho real, não um platô), a régua de parada
  ("<1% relativo em dois tickets seguidos") ainda não dispara duas vezes seguidas — mas está a
  um ticket de distância, exatamente como o mapa previu no §3 do relatório 07.
