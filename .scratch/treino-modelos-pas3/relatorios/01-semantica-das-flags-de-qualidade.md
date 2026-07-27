# Relatório — Ticket 01: Semântica das flags de qualidade (descartar, reparar ou ignorar?)

**Ticket:** `.scratch/treino-modelos-pas3/issues/01-semantica-das-flags-de-qualidade.md`
**Status:** concluído — ticket de investigação, nenhum código de produção alterado
**Base medida:** `.scratch/pdf-extraction/saida-nova/resultado_final.csv` (66.313 linhas, 8 triênios)
**Código lido:** `src/pas_extraction/resultado_final.py`, `checksum.py`, `models.py`,
`constants.py`, `csv_writer.py`, `rodada.py`; `src/pas_intelligence/argument_calculator.py`
**Scripts da medição (fora do git, contêm só agregado):**
`scratchpad/analise_flags.py`, `analise_flags2.py`, `analise_flags3.py`, com a saída bruta em
`saida_analise_flags*.txt`

**Privacidade:** nenhum nome, número de inscrição ou linha individual aparece aqui. As colunas
`nome` e `inscricao` são derrubadas na primeira linha de cada script de análise. Todos os
números abaixo são contagens e agregados.

---

## 1. A pergunta

Uma linha marcada com `campos_formato_invalido` não-vazio, ou com `checksum_fecha=False`,
tem valor de nota **confiável ou não**? A resposta define quantos dos 66.313 registros entram
no treino do ticket 05.

**Resposta curta:**

| Sinal | Significado real | Uso no treino |
|---|---|---|
| `campos_formato_invalido` não-vazio | campo **foi corrompido e reparado**; o valor mostrado é o recuperado | **não excluir** — excluir custa 8,7% da base *e* enviesa contra aluno de nota baixa |
| `checksum_fecha=False` | o Argumento Final impresso **não se reproduz** a partir das 9 notas da própria linha | **excluir** — é o único filtro que pega 100% dos valores fisicamente impossíveis |
| `checksum_delta` | dentro da tolerância, é ruído de arredondamento; fora, é rótulo categórico | **não usar como peso contínuo** |

---

## 2. O que o código faz — a ordem é o que resolve a pergunta

### 2.1 `campos_formato_invalido` marca reparo, não erro

Em `resultado_final.py::_montar_registro`, a ordem é literalmente esta:

```python
campos_numericos = campos[2:12]                       # 9 notas + argumento final
valores = [_tentar_float(v) for v in campos_numericos] # 1º REPARA
if any(v is None for v in valores):
    return None                                        # 2º DESCARTA o registro inteiro
...
campos_formato_invalido = tuple(                       # 3º MARCA, contra o texto BRUTO
    nome_campo
    for nome_campo, bruto in zip(_CAMPOS_NUMERICOS, campos_numericos)
    if not _formato_numerico_valido(bruto)
)
```

As três consequências, em ordem de importância:

1. **O reparo acontece ANTES da marcação.** `_tentar_float` faz `_WS.sub("", valor)` — remove
   *todo* espaço interno — e só então tenta o `float`. `"1 7.539"` vira `17.539`,
   `"- 21.683"` vira `-21.683`.
2. **A marcação compara o texto bruto, não o reparado.** `_formato_numerico_valido` roda
   `^-?\d+\.\d{3}$` (`_FORMATO_EXATO_RE`) contra o campo *antes* de tirar o espaço. Por
   construção, um campo reparado com sucesso continua marcado — a marca existe justamente
   para não ficar invisível.
3. **Campo que o reparo não salva nunca chega ao CSV.** Se qualquer um dos 10 valores volta
   `None`, `_montar_registro` devolve `None` e o registro inteiro some. Logo **nenhuma linha
   do CSV tem campo "corrompido e deixado como estava"**: ou o valor foi recuperado, ou a
   linha não existe.

**Decisão:** `campos_formato_invalido` lê-se como *"este campo passou pelo reparo tolerante"*,
não como *"este campo está errado"*. **Porquê:** é o que a ordem do código diz, e é o que a
docstring do módulo já declarava (`"o texto bruto (antes do reparo) é comparado contra o
formato exato — essa comparação é a validação estrutural do ticket 02, que sinaliza o campo em
vez de jogar fora o registro inteiro"`).

### 2.2 `checksum_fecha` e `checksum_delta`

`checksum.py::conferir_argumento_final` recalcula o Argumento Final a partir das 9 notas da
própria linha mais a tabela oficial de médias e desvios do mesmo Edital, testa as **27
combinações** de língua estrangeira (3 línguas × 3 Etapas), e grava o **menor** `|recalculado −
impresso|`.

- **Fórmula:** delegada a `pas_intelligence.argument_calculator.calculate_argument_final` —
  não há segunda cópia dos pesos em `pas_extraction`. `AF = 1×AP1 + 2×AP2 + 3×AP3`, com
  `APn = Σ ((nota − média) / desvio) × peso` e `PESO_P1=0,72`, `PESO_P2=8,28`,
  `PESO_REDACAO=1,00`.
- **Tolerância:** `TOLERANCIA_CHECKSUM = 0.005` em `constants.py`;
  `ChecksumArgumentoFinal.fecha` é a property `delta <= 0.005`.
- **Por que 0,005:** todos os operandos (as 9 notas, as médias, os desvios e o próprio AF) são
  publicados com 3 casas decimais; esse arredondamento se propaga em milésimos quando o valor é
  recomposto. Não é folga escolhida por conveniência.
- **Cobertura:** as três colunas de checksum saem vazias juntas quando não houve tabela oficial.
  **Medido: zero linhas nessa condição** — as 66.313 têm checksum conferido. Não existe o caso
  ambíguo "não conferido gravado como não fecha".

**Decisão:** `checksum_fecha` é o sinal forte de qualidade da linha, e `campos_formato_invalido`
é o sinal fraco. **Porquê:** o checksum é uma verificação *cruzada* — 12 campos verificados por
um número, com a tabela oficial vindo de outra parte do Edital. A flag de formato só olha para
o texto de um campo isolado.

---

## 3. As medições

### 3.1 A flag prediz reparo bem-sucedido — e o relatório 04 se sustenta

| | `fecha=True` | `fecha=False` | total |
|---|---|---|---|
| **com flag** | 5.740 | 736 | 6.476 |
| **sem flag** | 58.558 | 1.279 | 59.837 |
| **total** | 64.298 | 2.015 | 66.313 |

- `P(fecha | tem flag) = 88,63%`
- `P(fecha | sem flag) = 97,86%`

Por triênio:

| triênio | n | com flag | % flag | não fecha | `P(fecha \| flag)` | `P(fecha \| sem flag)` |
|---|---|---|---|---|---|---|
| 2016/2018 | 9.611 | 980 | 10,20% | 734 | 79,49% | 93,82% |
| 2017/2019 | 9.852 | 1.049 | 10,65% | 978 | 77,79% | 91,54% |
| 2018/2020 | 5.896 | 609 | 10,33% | 92 | 85,06% | 99,98% |
| 2019/2021 | 8.505 | 812 | 9,55% | 113 | 86,08% | 100,00% |
| 2020/2022 | 7.228 | 731 | 10,11% | 98 | 86,59% | 100,00% |
| 2021/2023 | 8.019 | 709 | 8,84% | 0 | **100,00%** | 100,00% |
| 2022/2024 | 8.499 | 758 | 8,92% | 0 | **100,00%** | 100,00% |
| 2023/2025 | 8.703 | 828 | 9,51% | 0 | **100,00%** | 100,00% |

**A afirmação do relatório 04 do `pdf-extraction` — "os 758 registros reparados do Ed_38 fecham
todos" — confirma-se sobre o CSV inteiro:** 2022/2024 tem exatamente 758 linhas com flag e
zero falha de checksum. O mesmo vale para 2021/2023 (709) e 2023/2025 (828). Em três triênios
inteiros, **2.295 linhas reparadas e 2.295 confirmadas pelo checksum**.

Mas a hipótese do ticket precisa de uma emenda importante: **nos três triênios do meio, a flag
é condição quase necessária da falha de checksum.** Em 2019/2021 e 2020/2022, *todas* as falhas
(113 e 98) estão em linhas com flag; em 2018/2020, 91 de 92. Ou seja, a flag não é inofensiva —
ela marca o lugar onde a corrupção catastrófica pode ter acontecido. O que separa "reparo certo"
de "reparo errado" é o checksum, não a flag.

### 3.2 Existem DUAS populações de falha de checksum, e elas não se parecem

Separando as 2.015 falhas pela magnitude do delta:

| | Pop. A (`0,005 < delta ≤ 5`) | Pop. B (`delta > 5`) |
|---|---|---|
| linhas | **1.446** | **569** |
| com flag de formato | 168 (11,6%) | 568 (99,8%) |
| Etapa 1 inteira zerada | 1.446 (100%) | 37 (6,5%) |
| valor fisicamente impossível | **0** | **568 (99,8%)** |
| triênios | só 2016/2018 (600) e 2017/2019 (846) | os 5 mais antigos |

O critério de "fisicamente impossível" foi derivado, não chutado: a faixa observada nos três
triênios mais recentes (que não têm falha nenhuma de checksum), alargada em 10% da amplitude
de cada campo. Na base inteira, **568 linhas** têm algum campo fora dela — por exemplo
`eb_p2_e1 = 39.617,919` numa escala cujo máximo real é 85,550, e `argumento_final = 311.102,343`
numa escala real de ±212. É a assinatura do reparo que **colou dois números** ao remover o
espaço interno.

**Achado decisivo:** das 568 linhas com valor impossível, **zero fecham o checksum** e **todas
carregam a flag de formato**. E das 64.298 que fecham, **nenhuma** tem valor impossível.

- **Pop. B é corrupção real.** O reparo recuperou um número, mas o número errado. O checksum
  pega 100% delas.
- **Pop. A não é corrupção.** Zero valores impossíveis; taxa de flag (11,6%) igual à taxa base
  da população (9,8%), isto é, sem enriquecimento; e 100% delas são alunos com a **Etapa 1
  inteira zerada** (P1, P2 e Redação simultaneamente em 0,000). São 1.446 linhas, todas nos
  dois triênios mais antigos.

**Mecanismo provável da Pop. A** (hipótese com base, não certeza): `argumento = ((nota − média)
/ desvio) × peso`. A sensibilidade do resultado a um erro no *desvio-padrão* publicado é
proporcional a `|nota − média|`. Para um aluno típico, `nota ≈ média` e o erro quase se cancela;
para um aluno com a Etapa inteira zerada, `|0 − média|` é o máximo possível e o mesmo erro
aparece com amplitude total. A tabela oficial dos dois Editais mais antigos é a única que vem de
**Edital avulso** e não da cauda do próprio Resultado Final (`rodada.py::_mapa_avulsos_medias_desvios`,
com `Ed_31_2016-2018` documentado como o caso conhecido). Consistente com o padrão: dos 5.768
alunos com Etapa 1 zerada na base, 94,7% e 96,6% falham nos dois triênios mais antigos, contra
1,4%, 1,0%, 1,2% nos três seguintes e **0%** nos três mais recentes.

### 3.3 Excluir as linhas com flag introduziria viés de seleção

A corrupção que dispara a flag não é aleatória: em boa parte dos casos é o **sinal de menos
separado do número** (`"- 21.683"`), padrão que o relatório 02 já tinha medido (302 dos 758
casos do Ed_38). Medido nos três triênios recentes, onde toda linha com flag fecha o checksum e
portanto é dado bom:

- `P(argumento_final < 0 | flag em argumento_final) = 90,79%` (n = 934)
- `P(argumento_final < 0 | sem flag nenhuma) = 54,37%` (n = 22.926)

Como o Argumento Final negativo é o do aluno abaixo da média, a flag é sistematicamente mais
frequente na cauda de baixo. Taxa de flag por decil de Argumento Final (3 triênios recentes):

| decil | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| % com flag | 11,3 | 11,7 | 11,4 | 11,0 | 10,8 | 8,6 | 7,1 | 6,4 | 6,6 | 6,1 |

Gradiente monótono. Descartar as linhas com flag **subiria a média do Argumento Final da base de
0,003 para 1,108** e cortaria quase o dobro de alunos da cauda de baixo em relação à de cima.
Para um modelo cuja utilidade está exatamente em distinguir aluno em risco, isso é um dano
direto ao alvo do produto.

### 3.4 As linhas reparadas que fecham são indistinguíveis das limpas

Correlação entre Escores Brutos de Etapas diferentes — uma medida de coerência interna que não
depende do checksum:

| recorte | n | corr(EB1,EB2) | corr(EB2,EB3) |
|---|---|---|---|
| fecha **e sem** flag | 58.558 | 0,662 | 0,773 |
| fecha **e com** flag | 5.740 | 0,613 | 0,748 |
| **não** fecha | 2.015 | −0,026 | −0,026 |

Linha reparada e confirmada carrega o mesmo sinal que linha limpa. Linha que não fecha carrega
sinal **zero** — a correlação some, exatamente o que se espera de valor aleatório. Esse é o
teste mais forte do relatório: ele não usa o checksum para julgar o checksum.

---

## 4. `checksum_delta` serve como medida contínua de confiança? Não.

**Decisão: usar `checksum_delta` como rótulo de 3 categorias, nunca como peso contínuo de
amostra.**

**Porquê — três evidências:**

**(a) Dentro da tolerância, o delta é quantizado e não carrega informação.** Todo delta da base
é múltiplo exato de 0,001 (medido: 0 exceções em 66.313). Entre os que fecham, ele assume
exatamente 6 valores: `0,000` (17.435), `0,001` (28.024), `0,002` (14.205), `0,003` (4.081),
`0,004` (531), `0,005` (22). Não é um contínuo — é uma grade de arredondamento.

**(b) Nos três triênios comprovadamente limpos, a distribuição do delta é a mesma com e sem
flag.** Se o delta medisse qualidade por linha, as linhas reparadas apareceriam deslocadas:

| delta | 0,000 | 0,001 | 0,002 | 0,003 | 0,004 | 0,005 |
|---|---|---|---|---|---|---|
| com flag (n=2.295) | 26,2% | 43,3% | 22,5% | 7,0% | 1,0% | 0,0% |
| sem flag (n=22.926) | 26,9% | 43,8% | 22,1% | 6,4% | 0,9% | 0,01% |

Praticamente idênticas. Um peso `1/(1+delta)` premiaria a linha de delta 0,000 sobre a de 0,003
sem nenhuma razão além do arredondamento da 3ª casa decimal — introduziria ruído se passando por
sinal.

**(c) Fora da tolerância, o delta é informativo — mas como salto, não como rampa.** A
distribuição das 2.015 falhas é bimodal com um vazio no meio: 1.446 linhas em `(0,005; 5]`,
apenas **3** em `(5; 50]`, e 566 acima de 50 (das quais 523 acima de 500, chegando a 310.999).
As duas modas são as populações da seção 3.2 e têm causas completamente diferentes. Um número
contínuo que mistura "imprecisão de tabela oficial" com "dois números colados" não é uma escala
de confiança; é duas coisas empilhadas no mesmo eixo.

**Uso recomendado — 3 categorias:**

| categoria | condição | n | leitura |
|---|---|---|---|
| **Confirmada** | `delta ≤ 0,005` | 64.298 | notas verificadas por cálculo cruzado |
| **Não reproduzida** | `0,005 < delta ≤ 5` | 1.446 | notas plausíveis; o AF impresso é que não fecha |
| **Corrompida** | `delta > 5` | 569 | valor fisicamente impossível em 568 de 569 |

---

## 5. Limitações conhecidas

1. **O checksum tem 27 tentativas por linha.** Escolher a melhor de 27 combinações de língua
   contra uma tolerância de 0,005 dá alguma chance de um valor errado passar por acaso. Cota
   grosseira de ordem de grandeza: ~3×10⁻⁵ por combinação sobre a faixa útil do AF, × 27, ×
   6.476 linhas com flag ≈ **6 linhas** que poderiam fechar por sorte. É um limite superior
   folgado (as 27 combinações produzem valores agrupados, não espalhados), mas não é zero.
2. **O checksum verifica as 10 colunas numéricas em conjunto, não uma a uma.** Ele não pode
   dizer *qual* campo estava errado — só que o conjunto é consistente. Para o treino isso basta;
   para depuração de extração, não.
3. **`checksum_fecha=True` não valida nada fora das 10 colunas numéricas.** `campus`, `curso`,
   `nome` e as 10 classificações ficam de fora. Os defeitos 13–18 do mapa `pdf-extraction`
   (`nome`, `classificacao_sistema_*`) permanecem, e o mapa já registrou que não tocam nota.
4. **A explicação da Pop. A é hipótese com base, não medição direta.** Não recalculei o
   Argumento Final variando a tabela oficial dos dois Editais mais antigos para confirmar que a
   diferença some. O que está *medido* é a ausência de qualquer marca de corrupção nessas 1.446
   linhas (0 valores impossíveis, taxa de flag na base) e o confinamento perfeito a dois
   triênios e a um único perfil de aluno.
5. **A faixa de plausibilidade é derivada dos 3 triênios recentes.** Se algum ano antigo tivesse
   escala de prova genuinamente diferente, a faixa ficaria estreita demais. Na prática a folga
   é enorme (as violações estão 3 ordens de grandeza fora, não na borda), então o risco é
   desprezível.
6. **Não foi verificado se aluno com Etapa 1 zerada é ausente ou anulado.** O Edital não diz.
   Medido apenas que o padrão só ocorre na Etapa 1 (`Etapa 2 inteira zerada: 0 linhas;
   Etapa 3: 0 linhas`) — compatível com "não fez a primeira prova do triênio".

---

## 6. Glossário — termos novos deste relatório

- **Reparo tolerante** — `_tentar_float` removendo *todo* espaço interno de um campo numérico
  antes do `float()`, para recuperar o valor mesmo com o texto partido pela extração de PDF
  (`"1 7.539"` → `17.539`, `"- 21.683"` → `−21.683`). Vem do ticket 02 do `pdf-extraction`.
- **Flag de formato** (`campos_formato_invalido`) — nomes dos campos cujo **texto bruto** não
  bateu `^-?\d+\.\d{3}$`. Marca que o reparo foi acionado naquele campo, não que o valor esteja
  errado.
- **Checksum do Argumento Final** — recalcular o AF a partir das 9 notas da própria linha e da
  tabela oficial do Edital, e comparar com o AF impresso. Um número verifica 12 campos.
- **Delta** (`checksum_delta`) — `|recalculado − impresso|`, na melhor das 27 combinações de
  língua. **Fecha** quando `≤ 0,005`.
- **Tolerância (0,005)** — o arredondamento de 3 casas com que o Cebraspe publica *todos* os
  operandos do cálculo, propagado em milésimos na recomposição. Não é folga arbitrária.
- **População A / População B** — os dois modos da falha de checksum, nomeados neste relatório.
  A = `delta ≤ 5`, Etapa 1 zerada, notas plausíveis, só nos 2 triênios mais antigos.
  B = `delta > 5`, valor fisicamente impossível, sempre com flag de formato.
- **Valor fisicamente impossível** — nota fora da faixa observada nos 3 triênios recentes
  alargada em 10% da amplitude do campo. Critério derivado do dado, não fixado à mão.
- **Etapa 1 zerada** — linha com `eb_p1_e1 = eb_p2_e1 = red_e1 = 0,000` simultaneamente.
  5.768 linhas (8,70%). Não ocorre nas Etapas 2 e 3.
- **Viés de seleção da flag** — o fato de a flag ser ~2× mais frequente no decil inferior de
  Argumento Final que no superior, porque a corrupção mais comum é o sinal de menos separado do
  número.

---

## 7. Recomendação para o ticket 05

### 7.1 A regra exata de inclusão de linha

```
INCLUIR a linha no dataset de treino  ⇔  checksum_fecha == True
```

Escrita como filtro sobre o CSV:

```python
treino = df[df["checksum_fecha"] == True]   # equivalente: df["checksum_delta"] <= 0.005
```

**E explicitamente NÃO filtrar por `campos_formato_invalido`.** As 5.740 linhas que têm flag e
fecham o checksum **entram**.

**Os três porquês, em ordem:**

1. **`checksum_fecha` é o único filtro necessário para a corrupção real.** Ele remove 568 de
   568 linhas com valor fisicamente impossível, e nenhuma linha impossível sobrevive a ele.
   Custo total: 3,04% da base.
2. **A flag de formato não é critério de exclusão** porque marca reparo, não erro — provado pela
   ordem do código (§2.1) e medido: 100% das linhas com flag fecham o checksum nos três triênios
   recentes; as que fecham têm a mesma estrutura de correlação entre Etapas que as limpas
   (§3.4). Excluí-las custaria mais 8,7% da base **e** enviesaria o treino contra o aluno de
   nota baixa (§3.3) — dano direto ao caso de uso do produto.
3. **`checksum_delta` não vira peso.** Dentro da tolerância é grade de arredondamento sem
   informação (§4a, §4b); fora, é rótulo categórico com um vazio no meio (§4c). Ponderar por ele
   injetaria ruído com cara de sinal.

### 7.2 Contagem resultante por triênio

| triênio | total | excluídas | └ Pop. A (`δ ≤ 5`) | └ Pop. B (`δ > 5`) | **INCLUÍDAS** | % |
|---|---|---|---|---|---|---|
| 2016/2018 | 9.611 | 734 | 600 | 134 | **8.877** | 92,36% |
| 2017/2019 | 9.852 | 978 | 846 | 132 | **8.874** | 90,07% |
| 2018/2020 | 5.896 | 92 | 0 | 92 | **5.804** | 98,44% |
| 2019/2021 | 8.505 | 113 | 0 | 113 | **8.392** | 98,67% |
| 2020/2022 | 7.228 | 98 | 0 | 98 | **7.130** | 98,64% |
| 2021/2023 | 8.019 | 0 | 0 | 0 | **8.019** | 100,00% |
| 2022/2024 | 8.499 | 0 | 0 | 0 | **8.499** | 100,00% |
| 2023/2025 | 8.703 | 0 | 0 | 0 | **8.703** | 100,00% |
| **TOTAL** | **66.313** | **2.015** | **1.446** | **569** | **64.298** | **96,96%** |

Das 64.298 incluídas, **5.740 carregam flag de formato** (8,93%) — e é deliberado que carreguem.
Distribuição por triênio: 779, 816, 518, 699, 633, 709, 758, 828.

### 7.3 Variante opcional, se o ticket 05 quiser as 1.446 de volta

A Pop. A não tem marca nenhuma de corrupção. Recuperá-la é defensável **se** o alvo do modelo
não for o Argumento Final impresso (que é justamente o campo que não fecha nessas linhas):

```
INCLUIR  ⇔  checksum_fecha == True  OR  checksum_delta <= 5
```

→ **65.744 linhas** (+2,18%): 2016/2018 sobe para 9.477 e 2017/2019 para 9.720; os demais
triênios não mudam.

**Recomendação:** ficar com a regra de §7.1 no primeiro corte. É mais simples, tem custo baixo
(2,2%), e as 1.446 linhas são todas de um perfil já anômalo (Etapa 1 zerada) que o ticket 05
provavelmente vai querer tratar à parte de qualquer jeito.

### 7.4 Três coisas para o ticket 05 decidir, que este ticket mediu mas não resolve

1. **Etapa 1 zerada — decisão de modelagem, não de qualidade.** São 5.768 linhas na base, das
   quais **4.285 entram** pela regra recomendada (2016/2018: 34; 2017/2019: 30; 2018/2020: 420;
   2019/2021: 477; 2020/2022: 577; 2021/2023: 897; 2022/2024: 985; 2023/2025: 865). São alunos
   com `EB1 = 0,000` e Argumento Final médio de −38,7 contra +4,7 dos demais. **Alerta
   específico:** a **Volatilidade (CV)** que roteia o ensemble atual é `std/mean × 100` sobre
   `[eb_pas1, eb_pas2]` — com `eb_pas1 = 0` o CV estoura e o meta-modelo roteia essas linhas
   para o ramo errado. Excluí-las levaria o treino a **60.013 linhas** (2016/2018: 8.843;
   2017/2019: 8.844; 2018/2020: 5.384; 2019/2021: 7.915; 2020/2022: 6.553; 2021/2023: 7.122;
   2022/2024: 7.514; 2023/2025: 7.838).
2. **Vazamento entre triênios:** medido, **146 números de inscrição aparecem em mais de um
   triênio** (0,22% de 66.159 inscrições distintas). Zero duplicatas *dentro* do mesmo triênio.
   A proporção é baixa; se o split for por triênio, basta remover as 146 do lado do teste.
3. **`lingua_e1/e2/e3` e `lingua_ambigua` são subproduto do checksum, não dado do Edital.**
   4.498 linhas (todas dentro do conjunto incluído) têm `lingua_ambigua=True` — mais de uma
   combinação fechou. Se a língua virar feature, ela é ruidosa em 7,0% dos casos incluídos; o
   delta continua válido nelas.
