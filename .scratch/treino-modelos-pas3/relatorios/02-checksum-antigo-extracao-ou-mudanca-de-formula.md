# Relatório — Ticket 02: o checksum falha nos triênios antigos por extração ruim ou por mudança de fórmula?

**Ticket:** `.scratch/treino-modelos-pas3/issues/02-checksum-antigo-extracao-ou-mudanca-de-formula.md`
**Status:** concluído
**Tipo:** investigação — nenhum código de produção foi alterado
**Dado analisado:** `.scratch/pdf-extraction/saida-nova/resultado_final.csv` (66.313 registros,
8 triênios), `medias_desvios.csv` (75 linhas), e os PDFs de origem em `data/pdfs/`
**Privacidade:** só agregados e contagens. Nenhum nome, inscrição ou linha individual aparece
aqui nem em arquivo intermediário commitado.

---

## 1. Veredito

> **Nenhuma das três hipóteses puras. O degrau é (d): uma quarta explicação, que a investigação
> isolou — a regra de pontuação da Etapa ausente (`0.000/0.000/0.000`) mudou entre triênios.**

Decompondo as 2.015 falhas de checksum, **todas elas** caem em exatamente duas populações
disjuntas, sem resíduo:

| Triênio | n | falhas | **A** — Etapa 1 zerada | **B** — corrupção grossa | outras |
|---|---:|---:|---:|---:|---:|
| 2016/2018 | 9.611 | 734 | **600** | 134 | 0 |
| 2017/2019 | 9.852 | 978 | **846** | 132 | 0 |
| 2018/2020 | 5.896 | 92 | 0 | 92 | 0 |
| 2019/2021 | 8.505 | 113 | 0 | 113 | 0 |
| 2020/2022 | 7.228 | 98 | 0 | 98 | 0 |
| 2021/2023 | 8.019 | 0 | 0 | 0 | 0 |
| 2022/2024 | 8.499 | 0 | 0 | 0 | 0 |
| 2023/2025 | 8.703 | 0 | 0 | 0 | 0 |

- **(a) Extração — verdadeira, mas explica só a população B** (~100 registros por Edital antigo,
  1,0 % a 1,4 %), e ela é *constante* entre os cinco triênios antigos, não crescente com a idade.
  Não produz degrau nenhum.
- **(b) Mudança de fórmula — REJEITADA.** Os pesos oficiais `0,72 / 8,28 / 1,00` e os pesos de
  Etapa `1 / 2 / 3` são recuperados **exatamente** dos Editais de 2016/2018 e 2017/2019 (seção 3).
- **(c) Falta de cobertura de `OFFICIAL_STATS` — REJEITADA.** Todas as 24 chaves `(ano, etapa)`
  necessárias para os 8 triênios existem, e o `medias_desvios.csv` cobre os antigos (seção 4).
- **(d) O que realmente causa o degrau:** nos dois triênios mais antigos, o Argumento Final
  impresso para o candidato **ausente da Etapa 1** não é o que a fórmula produz colocando
  `0,000` nas três notas. É sistematicamente **maior** (menos negativo) — mediana **+2,704** em
  2016/2018 e **+3,549** em 2017/2019, em unidades de Argumento de Etapa. A partir de 2018/2020
  esse desvio desaparece e o `z` de zero passa a valer literalmente.

**Consequência para o ticket 08:** o degrau **não** desqualifica 2016/2018 e 2017/2019. Ele
desqualifica uma *sub-população* desses triênios — a de quem não tem Etapa 1 —, que é
exatamente a sub-população que um modelo de previsão do PAS 3 a partir de PAS 1 e PAS 2 já teria
de excluir por não ter feature. Depois de removê-la, **98,6 %** de 2016/2018 (8.843 de 8.966) e **98,6 %** de
2017/2019 (8.844 de 8.968) fecham o checksum na tolerância de 0,005 — e o 1,4 % restante é
corrupção de extração auto-sinalizada, na mesma taxa dos triênios de 2018 a 2022. Detalhe na
seção 9.

---

## 2. Como foi medido

Todo número deste relatório vem de um recorte declarado, reproduzível a partir do CSV de saída
da rodada de 2026-07-26 (`.scratch/pdf-extraction/saida-nova/`):

1. **Recálculo independente.** O Argumento Final foi recalculado fora do pipeline, direto do CSV,
   com a fórmula de `pas_intelligence.argument_calculator` reimplementada em NumPy e a tabela
   oficial montada de `medias_desvios.csv`. O resultado bate com o `checksum_delta` gravado em
   **41.092 de 41.092** registros dos cinco triênios com tabela avulsa (divergência > 0,01: zero).
   Isso valida o instrumento antes de qualquer conclusão.
2. **Delta com sinal.** O CSV grava `checksum_delta` em valor absoluto. O recálculo devolve
   `delta_com_sinal = recalculado − impresso`, que é o que distingue erro sistemático (viés numa
   direção) de erro disperso.
3. **Regressão de recuperação de pesos.** `AF_impresso ~ 9 z-scores`, sem intercepto, por triênio,
   só nas linhas que fecham. Se os pesos tivessem mudado, os coeficientes sairiam diferentes.
4. **Varredura de constante.** Para as linhas com Etapa 1 zerada, varredura de uma constante `C`
   em passo de 0,001 no intervalo `[−22, −10]`, contando quantas linhas fechariam se o Argumento
   da Etapa 1 fosse `C` para todo mundo.
5. **Controle do estimador.** Toda estimativa que fixa a língua (necessário porque a língua não é
   impressa) foi repetida nas linhas que *fecham*, onde a resposta certa é conhecida. Isso mede o
   borrão que a fixação de língua introduz, e é o que permite dizer que um desvio observado é real
   e não artefato do método.
6. **Conferência contra o PDF.** Inspeção do texto bruto do `Ed_31_2016-2018` para confirmar o que
   o Edital de fato imprime nos registros com Etapa 1 zerada.

Scripts de análise ficaram no scratchpad da sessão (`an1.py`…`an13.py`), fora do repositório —
eles carregam o CSV com nome de Aluno e por isso não vão para o git.

---

## 3. A fórmula não mudou — evidência decisiva

### 3.1 Os pesos são recuperados exatamente, inclusive em 2016/2018

Regressão de mínimos quadrados de `AF_impresso` sobre os 9 z-scores `(nota − média)/desvio`, sem
intercepto, **só nas linhas que fecham o checksum**, por triênio:

| Triênio | n | P1·E1 | P2·E1 | Red·E1 | P1·E2 | P2·E2 | Red·E2 | P1·E3 | P2·E3 | Red·E3 | max\|resíduo\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| esperado | | 0,72 | 8,28 | 1,00 | 1,44 | 16,56 | 2,00 | 2,16 | 24,84 | 3,00 | — |
| 2016/2018 | 8.877 | 0,720025 | 8,279995 | 1,000013 | 1,440020 | 16,560039 | 1,999993 | 2,159970 | 24,839965 | 2,999989 | 0,0053 |
| 2017/2019 | 8.874 | 0,719979 | 8,279988 | 1,000004 | 1,439995 | 16,559991 | 1,999969 | 2,160028 | 24,840009 | 3,000015 | 0,0051 |
| 2018/2020 | 5.804 | 0,720013 | 8,280009 | 1,000006 | 1,440020 | 16,559981 | 1,999962 | 2,159970 | 24,840070 | 2,999997 | 0,0046 |
| 2019/2021 | 8.392 | 0,720001 | 8,280052 | 1,000037 | 1,439990 | 16,559982 | 1,999983 | 2,159972 | 24,840021 | 2,999999 | 0,0048 |
| 2020/2022 | 7.130 | 0,720002 | 8,279896 | 0,999962 | 1,439981 | 16,560061 | 1,999996 | 2,159999 | 24,839977 | 3,000030 | 0,0046 |

O resíduo máximo em todos os triênios é ≤ 0,0053 — a **tolerância de arredondamento de 3 casas**,
não erro de modelo. O RMS é 0,0014 em todos.

**Por que isso é decisivo:** se o Cebraspe usasse outros pesos em 2016/2018, a regressão devolveria
outros números *nesse* triênio. Ela devolve os mesmos oito dígitos significativos em todos os
cinco. A fórmula, os pesos das partes, os pesos das Etapas e a padronização por média/desvio são
**idênticos** de 2016 a 2025.

### 3.2 O teste de "resolver os pesos antigos" nas linhas que falham não converge

A mesma regressão rodada só nas 600/846 falhas finas devolve coeficientes incoerentes para a
Etapa 1 (`P2·E1 ≈ 3,4–3,7` onde deveria dar 8,28; `Red·E1 ≈ 4,0–4,6` onde deveria dar 1,00) e
coeficientes **corretos** para as Etapas 2 e 3. Isso não é um conjunto de pesos antigos: nessas
linhas as três notas da Etapa 1 são todas `0,000`, então os três z-scores da Etapa 1 são
*constantes por linha* e a regressão fica degenerada — os coeficientes de E1 são colineares e não
significam nada. O sinal útil é o outro: **as Etapas 2 e 3 estão perfeitas e o problema é
inteiramente da Etapa 1.**

### 3.3 A hipótese de reescala do Argumento Final foi testada e rejeitada

Se o Edital antigo tivesse computado `AF = (2·A2 + 3·A3) × k` para o candidato sem Etapa 1, o
"Argumento de Etapa 1 implícito" seria proporcional a `(2·A2 + 3·A3)` com inclinação `k − 1`.
Medido: inclinação **−0,9991** (2016/2018) e **−1,0001** (2017/2019), `r = −1,000`. Inclinação
−1 é a assinatura algébrica de um *termo aditivo* (o A1 implícito não depende de A2/A3), não de
reescala. Rejeitada.

---

## 4. Cobertura de `OFFICIAL_STATS` e de `medias_desvios.csv` — hipótese (c) rejeitada

- `pas_constants.OFFICIAL_STATS` (reescrito no ticket 12 do mapa `pdf-extraction`, com Parte 1 por
  língua) tem **24 chaves** `(ano, etapa)`. Todas as 24 combinações exigidas pelos 8 triênios
  existem — inclusive `(2016,1)`, `(2017,2)`, `(2018,3)` para 2016/2018 e `(2017,1)`, `(2018,2)`,
  `(2019,3)` para 2017/2019. Verificado programaticamente, 24/24.
- `medias_desvios.csv` tem 75 linhas = **5 triênios × 15** (3 Etapas × [3 línguas de Parte 1 +
  Parte 2 + Redação]), cobrindo 2016/2018 a 2020/2022 — exatamente os triênios cujo Edital publica
  a tabela **avulsa**. Os três recentes não aparecem no arquivo porque publicam a tabela na
  **cauda do próprio Resultado Final**, e o `rodada.py` só grava esse CSV a partir dos Editais
  avulsos. Não é falta de cobertura: o checksum foi computado para os 8 triênios (`checksum_fecha`
  nulo em **0** dos 66.313 registros).
- A prova final de que a tabela antiga está certa é a seção 3.1: com a tabela do `Ed_32/2016-2018`
  e do `Ed_38/2017-2019`, 8.877 e 8.874 registros fecham com resíduo ≤ 0,005. Uma tabela errada
  não produziria isso.

---

## 5. População A — Etapa 1 zerada: onde mora o degrau

### 5.1 A correlação é total

| Triênio | n | linhas com Etapa 1 = `0,000/0,000/0,000` | delas, falham | % | falhas fora dessa população |
|---|---:|---:|---:|---:|---:|
| 2016/2018 | 9.611 | 645 | 611 | **94,7 %** | 123 |
| 2017/2019 | 9.852 | 884 | 854 | **96,6 %** | 124 |
| 2018/2020 | 5.896 | 426 | 6 | 1,4 % | 86 |
| 2019/2021 | 8.505 | 482 | 5 | 1,0 % | 108 |
| 2020/2022 | 7.228 | 584 | 7 | 1,2 % | 91 |
| 2021/2023 | 8.019 | 897 | 0 | 0,0 % | 0 |
| 2022/2024 | 8.499 | 985 | 0 | 0,0 % | 0 |
| 2023/2025 | 8.703 | 865 | 0 | 0,0 % | 0 |

Na direção inversa: **100 %** das falhas finas (`|delta| ≤ 10`) dos dois triênios antigos —
600 de 600 e 846 de 846 — são linhas com a Etapa 1 inteiramente zerada. Não há uma única exceção.

Nenhum triênio tem Etapa 2 ou Etapa 3 inteiramente zerada (0 linhas em todos os 8). A ausência só
acontece na Etapa 1, o que é coerente com o domínio: quem falta à Etapa 3 não aparece no Resultado
Final do PAS 3.

### 5.2 O `0,000` é real — não é falha de extração

Inspeção do texto bruto do `Ed_31_2016-2018_PAS_3_Res_final_nao_eliminados.pdf`: o Edital imprime
literalmente `0.000, 0.000, 0.000` nas três primeiras notas desses registros, com as Etapas 2 e 3
preenchidas normalmente. O parser leu certo. A hipótese "o parser antigo perdeu as notas da Etapa
1" está descartada na fonte.

### 5.3 A distribuição do delta: erro pequeno, assimétrico e limitado — não é dígito corrompido

Histograma do `delta_com_sinal` das falhas finas (recalculado − impresso):

| faixa | 2016/2018 | 2017/2019 |
|---|---:|---:|
| (−5, −3] | 0 | 18 |
| (−3, −2] | 1 | 195 |
| (−2, −1] | 22 | 227 |
| (−1, −0,5] | 27 | 72 |
| (−0,5, −0,2] | 87 | 52 |
| (−0,2, −0,05] | 148 | 85 |
| (−0,05, −0,005] | 83 | 62 |
| (0,005, 0,05] | 73 | 68 |
| (0,05, 0,2] | 114 | 65 |
| (0,2, 0,5] | 45 | 2 |
| **> 0,5** | **0** | **0** |

Duas coisas: (i) a cauda positiva **é cortada em +0,5 e +0,2** — nenhuma linha tem o recalculado
mais de meio ponto acima do impresso; (ii) a cauda negativa vai até −3. Um dígito corrompido numa
nota produziria erro **simétrico** e de magnitude discreta (um dígito na casa das unidades de P2
vale ~0,6 a 1,9 pontos de Argumento Final, dependendo da Etapa). O que se observa é um viés
unidirecional e contínuo. **Isso é assinatura de regra, não de ruído de leitura.**

O sinal é o esperado por (d): `delta = recalculado − impresso < 0` significa que o impresso é
**maior** que o recalculado, ou seja, o Edital foi mais generoso com a Etapa ausente do que o `z`
de zero.

### 5.4 Quanto mais generoso: o desvio medido

Estimador: fixando `língua(E2) = língua(E3) = inglesa` (a modal, 56 % a 67 % da população que
fecha, conforme a Etapa e o triênio), resolve-se
o Argumento de Etapa 1 implícito `A1_imp = AF_impresso − 2·A2 − 3·A3` e compara-se com o `A1` que
a fórmula produz com as três notas em zero. Quantis do desvio `A1_imp − A1_zero`:

| Triênio | q05 | q25 | **mediana** | q75 | q95 | **controle** (mesmo estimador nas linhas que fecham) |
|---|---:|---:|---:|---:|---:|---|
| 2016/2018 | 1,103 | 2,085 | **2,704** | 3,406 | 4,339 | q25 −0,001 / **med 0,000** / q75 0,243 |
| 2017/2019 | 1,697 | 2,612 | **3,549** | 4,528 | 5,614 | q25 −0,000 / **med 0,002** / q75 1,898 |
| 2018/2020 | −0,002 | 0,000 | 0,720 | 1,484 | 2,174 | q25 −0,000 / med 0,002 / q75 1,187 |
| 2019/2021 | −0,905 | −0,271 | **−0,002** | 0,000 | 0,005 | q25 −0,296 / med −0,001 / q75 0,000 |

O controle é o que dá confiança: nas linhas que fecham, o mesmo estimador acerta a mediana em
0,000–0,002, e todo o espalhamento visível ali é o borrão de fixar a língua. Em 2019/2021 o desvio
das linhas zeradas é indistinguível do controle → **regra do `z` de zero, literal**. Em 2016/2018 e
2017/2019 a mediana está 2,7 e 3,5 pontos deslocada — muito acima do borrão. **O desvio é real.**

### 5.5 Não é uma constante — a regra antiga varia por candidato

Varredura de uma constante `C` que faria as linhas zeradas fecharem (passo 0,001, tolerância 0,005,
melhor de 9 combinações de língua para E2/E3):

| Triênio | n zeradas | melhor `C` | fecha | % |
|---|---:|---:|---:|---:|
| 2018/2020 | 425 | **−18,228** | **420** | **98,8 %** |
| 2016/2018 | 644 | −17,099 | 23 | 3,6 % |
| 2017/2019 | 883 | −17,110 | 34 | 3,9 % |

Em 2018/2020 a constante vencedora é `−18,227`, que é **exatamente** o `A1` do `z` de zero com a
Parte 1 na tabela de língua inglesa — 98,8 % das zeradas obedecem a essa única regra. Nos dois
triênios antigos **nenhuma constante funciona**: a maior cobertura é 3,9 %. O Argumento de Etapa 1
do candidato ausente naqueles anos era um número que **variava de candidato para candidato**, com
IQR ≈ 1,3 (2016/2018) e ≈ 1,9 (2017/2019) além do ruído do estimador.

### 5.6 A mecânica exata da regra antiga não foi identificada (limitação declarada)

Testes de reconstrução por campo único (supondo que P1 e P2 da Etapa 1 valeram 0 e que um terceiro
campo entrou com valor real):

| hipótese | valor implícito 2016/2018 (q25 / med / q75) | população que fecha (referência) | veredito |
|---|---|---|---|
| Redação de E1 contou com valor real | 5,63 / **7,31** / 9,20 | 5,27 / 6,54 / 8,20 | **compatível** em 2016/2018 |
| Parte 2 de E1 contou com valor real | 3,30 / 4,28 / 5,39 | 17,20 / 24,42 / 32,63 | incompatível |
| Parte 1 de E1 contou com valor real | 7,78 / 10,09 / 12,71 | máximo possível = 10 | impossível |

A hipótese "a Redação da Etapa 1 foi contada" reproduz bem 2016/2018, mas em 2017/2019 o mesmo
cálculo exige uma Redação mediana de **9,46** (contra 7,08 na população) e um q75 de 12,06, acima
do máximo de 10 — logo **não é confirmada**. Fica registrado como pista, não como conclusão.

Isso é honesto e suficiente para a decisão do ticket 08: **o que precisa ser certo é que essas
linhas são um regime diferente e devem ser tratadas à parte, e isso está provado.** Descobrir a
regra exata só valeria a pena se alguém quisesse *recuperar* esses registros, o que a seção 9
argumenta que não vale.

---

## 6. População B — corrupção grossa de extração

As demais 569 falhas (134 + 132 + 92 + 113 + 98) têm `|delta|` entre 10 e 311.000. Caracterização:

| Triênio | grossas | com valor fora de faixa física | com flag `campos_formato_invalido` | sem nenhum sinal |
|---|---:|---:|---:|---:|
| 2016/2018 | 134 | 134 | 134 | 0 |
| 2017/2019 | 132 | 132 | 132 | 0 |
| 2018/2020 | 92 | 91 | 91 | **1** |
| 2019/2021 | 113 | 113 | 113 | 0 |
| 2020/2022 | 98 | 98 | 98 | 0 |
| 2021/2023 a 2023/2025 | 0 | 0 | 0 | 0 |

"Fora de faixa física" = alguma Parte 1 ou Redação com `|nota| > 10`, alguma Parte 2 com
`|nota| > 100`, ou `|Argumento Final| > 200`. Exemplos de magnitude: Argumento Final impresso
chegando a 311.102 e a −40.031 — números que não existem na escala do PAS. **568 das 569 já vêm
auto-sinalizadas** pela flag `campos_formato_invalido` do ticket 02 do mapa `pdf-extraction`;
uma única linha (2018/2020) falha grosso sem sinal algum.

Dois fatos que fecham a leitura:

1. **A taxa é praticamente constante** entre os cinco Editais antigos (0,9 % a 1,4 %), sem
   gradiente por idade. Não é isso que produz um degrau de 7,6 % para 0 %.
2. Os três Editais recentes **também** têm registros com `campos_formato_invalido` (709, 758, 828)
   — mas **nenhum** deles falha o checksum. Ou seja: nos PDFs recentes o reparo do número partido
   por espaço recupera o valor certo em 100 % dos casos; nos antigos, `P(falha | flag) = 13 % a
   22 %`. Existe sim um gradiente de qualidade de extração por idade do PDF, e ele é medido aqui —
   mas custa ~100 linhas por Edital, não 800.

---

## 7. Segunda pergunta: o déficit de ~2.600 registros em 2018/2020

**Resposta: menos candidatos, ponto. Não é Edital parcial, não é perda de extração, não é
eliminação em massa.**

### 7.1 Não é perda de extração

| Triênio | arquivo | páginas do PDF | seção de não eliminados | registros | buracos de página | registros/página |
|---|---|---:|---|---:|---:|---:|
| 2016/2018 | Ed_31 | 257 | 1–257 | 9.611 | **0** | 37,4 |
| 2017/2019 | Ed_36 | 419 | 143–418 | 9.852 | **0** | 35,7 |
| 2018/2020 | ED_37 | 243 | **76–242** | 5.896 | **0** | 35,3 |
| 2019/2021 | Ed_30 | 350 | 113–350 | 8.505 | **0** | 35,7 |
| 2020/2022 | Ed_30 | 297 | 99–297 | 7.228 | **0** | 36,3 |
| 2021/2023 | Ed_27 | 317 | 100–317 | 8.019 | **0** | 36,8 |
| 2022/2024 | Ed_38 | 242 | 1–241 | 8.499 | **0** | 35,3 |
| 2023/2025 | (hash) | 247 | 1–247 | 8.703 | **0** | 35,2 |

A densidade é uniforme (35,2 a 37,4 registros por página) e não há uma única página vazia dentro do
intervalo extraído em nenhum triênio. **O Edital de 2018/2020 simplesmente tem 167 páginas de não
eliminados, contra 238 e 276 dos vizinhos.** O documento é menor, não a leitura.

### 7.2 Não é mais eliminação

Contando âncoras de inscrição (`\d{8},`) na seção que antecede a de não eliminados — que é a lista
de eliminados — dá para reconstruir a coorte inteira:

| Triênio | eliminados (âncoras) | não eliminados | **coorte total** | % eliminados |
|---|---:|---:|---:|---:|
| 2017/2019 | 12.926 | 9.852 | **22.778** | 56,7 % |
| **2018/2020** | **6.844** | **5.896** | **12.740** | **53,7 %** |
| 2019/2021 | 10.221 | 8.505 | **18.726** | 54,6 % |
| 2020/2022 | 8.927 | 7.228 | **16.155** | 55,3 % |
| 2021/2023 | 9.042 | 8.019 | **17.061** | 53,0 % |

A taxa de eliminação de 2018/2020 (53,7 %) é a **segunda mais baixa** da série. O que encolheu foi
a coorte inteira: 12.740 contra 18.726 do triênio seguinte — **−32 %**, praticamente o mesmo −31 %
observado só nos não eliminados. O funil manteve a proporção; entrou menos gente nele.

### 7.3 Por que entrou menos gente: o calendário quebrou

Data de publicação do Resultado Final, lida da página 1 de cada PDF:

| Triênio | publicação | intervalo desde o anterior |
|---|---|---:|
| 2016/2018 | 31/01/2019 | — |
| 2017/2019 | 22/01/2020 | 356 dias |
| **2018/2020** | **27/10/2021** | **644 dias** |
| 2019/2021 | 30/03/2022 | **154 dias** |
| 2020/2022 | 24/01/2023 | 300 dias |
| 2021/2023 | 05/02/2024 | 377 dias |
| 2022/2024 | 07/02/2025 | 368 dias |

O ciclo anual de ~360 dias vira 644 dias e depois 154. A Etapa 3 nominal de 2020 só teve resultado
em **outubro de 2021** (o Edital de médias e desvios correspondente, ED_43, é de 19/11/2021), e a
Etapa 3 nominal de 2021 saiu **cinco meses depois**. Uma coorte que esperou quase dois anos pela
prova final, com dois PAS 3 acontecendo em cinco meses e o ENEM/SiSU rodando no meio, perde
candidatos por evasão e por migração de rota. É a explicação com evidência; não é preciso invocar
perda de dado.

---

## 8. Terceira entrega: Etapas em ano afetado pela pandemia, por triênio

Mapeamento nominal `Etapa n do triênio YYYY/(YYYY+2)` → ano da prova, com marcação das que caem na
janela pandêmica (2020–2022). O ano **nominal** é o do subprograma e sai da chave do triênio; a
coluna "publicação do Resultado Final" é medida (seção 7.3) e mostra o quanto a aplicação real
escorregou.

| Triênio | Etapa 1 | Etapa 2 | Etapa 3 | Etapas em ano pandêmico | Observação |
|---|---|---|---|---|---|
| 2016/2018 | 2016 | 2017 | 2018 | **nenhuma** | ciclo normal |
| 2017/2019 | 2017 | 2018 | 2019 | **nenhuma** | ciclo normal |
| 2018/2020 | 2018 | 2019 | **2020** | **E3** | E3 adiada; resultado só em 27/10/2021 |
| 2019/2021 | 2019 | **2020** | **2021** | **E2 e E3** | E2 adiada junto com o PAS 3 de 2018/2020 |
| 2020/2022 | **2020** | **2021** | 2022 | **E1 e E2** | E1 adiada; coorte entrou no PAS já em escola remota |
| 2021/2023 | **2021** | 2022 | 2023 | **E1** | E1 aplicada em calendário já restabelecido, mas ensino médio remoto |
| 2022/2024 | 2022 | 2023 | 2024 | **nenhuma** | coorte com 1º/2º ano do EM em ensino remoto |
| 2023/2025 | 2023 | 2024 | 2025 | **nenhuma** | — |

Sinal empírico de que isso mexe no dado (população limpa: checksum fecha **e** Etapa 1 não zerada):

| Triênio | n limpo | AF média | AF desvio | EB E1 média | EB E2 média | EB E3 média |
|---|---:|---:|---:|---:|---:|---:|
| 2016/2018 | 8.843 | 3,111 | 50,344 | 30,08 | 24,62 | 32,77 |
| 2017/2019 | 8.844 | 4,569 | 49,436 | 33,29 | 27,87 | 28,28 |
| 2018/2020 | 5.384 | 3,404 | 50,029 | 30,95 | 29,79 | 32,41 |
| 2019/2021 | 7.915 | 2,297 | 50,294 | 32,70 | 33,59 | **27,01** |
| 2020/2022 | 6.553 | 3,566 | 49,469 | 29,18 | 28,85 | 30,06 |
| 2021/2023 | 7.122 | 4,705 | 49,440 | 29,08 | 27,41 | 31,51 |
| 2022/2024 | 7.514 | 4,973 | 50,894 | 27,09 | **34,65** | **36,46** |
| 2023/2025 | 7.838 | 4,519 | 50,828 | 30,97 | 34,09 | 35,66 |

O **Argumento Final é estável por construção** (média ~3–5, desvio ~50 em todos os oito) — é uma
soma de z-scores, então a padronização absorve a dificuldade da prova. O **Escore Bruto não é**:
o EB médio da Etapa 3 vai de 27,0 (2019/2021, a Etapa 3 mais pandêmica) a 36,5 (2022/2024) — uma
variação de 35 %. Isso é o aviso operacional para o ticket 08 e para o ticket 04 (alvo canônico):
**um modelo treinado sobre EB cru mistura regimes de dificuldade de prova; um modelo treinado
sobre Argumento Final, não.**

---

## 9. Implicações para o ticket 08 (janela de dados)

**1. A janela pode ir até 2016/2018.** A razão para cortar seria "aqueles triênios são outro
regime de cálculo". Não são: os pesos e a padronização são bit-a-bit os mesmos (seção 3.1), com
resíduo máximo de 0,005 em 8.877 registros de 2016/2018. A pergunta original do dono do produto
("o padrão mudou desde 2018?") tem resposta **não**, no que diz respeito à fórmula.

**2. O critério de corte não é o triênio, é a linha.** Excluindo a população A (Etapa 1 zerada) e
a população B (corrupção grossa), a taxa de checksum residual fica assim:

| Triênio | n | após excluir A e B | % que fecha no restante |
|---|---:|---:|---:|
| 2016/2018 | 9.611 | 8.843 | **100,0 %** |
| 2017/2019 | 9.852 | 8.844 | **100,0 %** |
| 2018/2020 | 5.896 | 5.384 | 100,0 % |
| 2019/2021 | 8.505 | 7.915 | 100,0 % |
| 2020/2022 | 7.228 | 6.553 | 100,0 % |
| 2021/2023 | 8.019 | 7.122 | 100,0 % |
| 2022/2024 | 8.499 | 7.514 | 100,0 % |
| 2023/2025 | 8.703 | 7.838 | 100,0 % |

(A exclusão de Etapa 1 zerada é aplicada a **todos** os triênios, não só aos antigos — daí a queda
de linhas também nos recentes.) O dataset limpo tem **60.013 registros nos 8 triênios**, com
zero falha de checksum. Isso é o candidato natural a dataset de treino do ticket 05.

**3. As linhas com Etapa 1 zerada saem por mérito próprio, não como conserto do checksum.** Um
modelo que prevê a Etapa 3 a partir das Etapas 1 e 2 não tem feature para um Aluno sem Etapa 1:
`eb_pas1 = 0` não é "escore zero", é ausência, e a Volatilidade/CV — que é o que hoje pondera o
ensemble — fica sem sentido: `calculate_volatility` usa `np.std(ddof=0)/mean`, e sobre
`[0, eb_pas2]` isso devolve **exatamente 100 % para todo Aluno**, qualquer que seja o
`eb_pas2`. A feature vira uma constante e o meta-modelo passa a decidir no escuro. São 5.768 linhas
(8,7 % do total) e elas devem virar uma **flag explícita** no dataset do ticket 05, não um
descarte silencioso.

**4. Não confiar em `checksum_fecha` como proxy único de qualidade.** Ele mistura duas coisas
muito diferentes: 569 linhas de dado corrompido (descarte legítimo) e 1.446 linhas de dado
**correto** cuja regra de Etapa ausente é de outro regime (exclusão por ausência de feature, não
por corrupção). O ticket 01 (semântica das flags) deveria registrar essa distinção.

**5. Se alguém quiser cortar a janela, o argumento tem de vir de outro lugar.** Os candidatos
reais estão na seção 8: dificuldade de prova (EB da Etapa 3 variando 35 %) e coortes pandêmicas.
Nenhum deles é o checksum.

---

## 10. Limitações

- **A regra de Etapa ausente de 2016/2018 e 2017/2019 não foi reconstruída.** Sabe-se que existe,
  que é aditiva (não reescala), que não é constante e que é ~+2,7/+3,5 pontos mais generosa que o
  `z` de zero. A pista da "Redação contou" explica 2016/2018 e não explica 2017/2019. Fechar isso
  exigiria o Edital normativo do subprograma daqueles anos, que não está em `data/pdfs/`.
- **A língua estrangeira nunca é impressa.** Toda estimativa que precisa fixar a língua carrega o
  borrão medido na coluna "controle" da seção 5.4. As conclusões usam **medianas**, que o controle
  mostra serem robustas (erro 0,000–0,002), e não médias ou caudas.
- **A contagem de eliminados da seção 7.2 é por âncora de inscrição**, não por parse completo — a
  seção de eliminados tem outro schema e não foi extraída pelo pipeline. Serve para comparar
  ordens de grandeza entre triênios, não como número oficial.
- **As datas de aplicação das provas são inferidas**, não medidas. O que foi lido do PDF é a data
  de *publicação* do Edital de Resultado Final. O mapeamento Etapa → ano nominal na seção 8 vem da
  definição de triênio, e é exato; o quanto cada prova escorregou no calendário real é inferência
  a partir dos intervalos de publicação.
- **Os 3 triênios recentes não têm tabela em `medias_desvios.csv`** (a deles vem da cauda do
  Resultado Final), então o recálculo independente da seção 2 cobre 41.092 dos 66.313 registros.
  Para os recentes usou-se o `checksum_delta` já gravado, cujo valor máximo é 0,005 — não há o que
  reauditar ali.

---

## 11. Glossário

- **Argumento de Etapa (`A1`, `A2`, `A3`)** — soma dos três argumentos padronizados de uma Etapa:
  `[(P1 − média)/desvio]·0,72 + [(P2 − média)/desvio]·8,28 + [(Red − média)/desvio]·1,00`.
- **Argumento Final (AF)** — `1·A1 + 2·A2 + 3·A3`, arredondado a 3 casas. É o número impresso no
  Edital e o que ranqueia na UnB.
- **Checksum do Argumento Final** — verificação do ticket 04 do mapa `pdf-extraction`: recalcula o
  AF das 9 notas com a tabela oficial e compara com o impresso. `checksum_fecha = |delta| ≤ 0,005`.
- **Tolerância 0,005** — não é folga arbitrária: é o arredondamento de 3 casas com que o Edital
  publica *todos* os operandos, propagado na recomposição.
- **Delta com sinal** — `recalculado − impresso`. O CSV grava só o módulo; o sinal foi recuperado
  neste ticket e é o que separa erro sistemático de erro disperso.
- **Falha fina / falha grossa** — corte em `|delta| = 10` usado neste relatório para separar as
  duas populações. Não é um conceito do pipeline; é uma faca de análise, e o histograma da seção
  5.3 mostra que ela cai num vale vazio (nenhuma falha entre 5 e 10 nos triênios antigos).
- **Etapa zerada** — registro em que as três notas de uma Etapa são `0,000`. Aqui sempre a Etapa 1;
  nenhum registro dos 66.313 tem Etapa 2 ou 3 zerada.
- **População limpa** — `checksum_fecha = True` **e** Etapa 1 não zerada. 60.013 dos 66.313.
- **Inferência de língua por Etapa** — a Parte 1 tem média/desvio publicados por língua e a língua
  do Aluno não é impressa; o pipeline testa as 27 combinações (3 línguas × 3 Etapas) e fica com a
  de menor delta. Isso dá ao checksum um grau de liberdade que ele usa para *absorver* erros
  pequenos — motivo pelo qual as línguas gravadas nas linhas que falham não são confiáveis
  (em 2017/2019, 83,5 % das falhas finas saíram com Etapa 3 em francesa, contra 1,1 % na
  população que fecha: o argmin escolheu a língua que minimiza o delta, não a verdadeira).

---

## 12. Onde continuar

- **Ticket 01 (semântica das flags):** registrar que `checksum_fecha = False` tem duas causas
  distintas e que a distinção é `Etapa 1 zerada` vs `campos_formato_invalido`.
- **Ticket 05 (dataset canônico):** materializar a população limpa (60.013 linhas) com uma coluna
  `etapa_1_ausente` explícita, em vez de filtro implícito.
- **Ticket 04 (alvo canônico):** a seção 8 é argumento a favor do Argumento Final como alvo — ele
  é estável entre triênios por construção, o Escore Bruto não é.
- **Ticket 08 (janela):** entra com a resposta "a fórmula não mudou"; o debate que resta é
  dificuldade de prova e coorte pandêmica, não regime de cálculo.
