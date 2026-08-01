# Relatório — Ticket 04: alvo canônico, Argumento Final direto ou as 3 notas?

**Ticket:** `.scratch/treino-modelos-pas3/issues/04-alvo-canonico-argumento-ou-tres-notas.md`
**Status:** concluído
**Tipo:** decisão de produto + medição — nenhum código de produção foi alterado nesta sessão
**Dado analisado:** `.scratch/pdf-extraction/saida-nova/resultado_final.csv` (66.313 registros),
`src/pas_intelligence/pas_constants.OFFICIAL_STATS`, e os artefatos em `models/`
**Privacidade:** só agregados e contagens. Nenhum nome, inscrição ou linha individual aparece
aqui nem em arquivo intermediário commitado.
**ADR:** [ADR-0009](../../../docs/adr/0009-alvo-canonico-argumento-da-etapa-3.md)

---

## 1. Veredito

> **Nenhuma das duas opções do ticket. O alvo canônico é uma terceira: `A3`, o Argumento da
> Etapa 3.** O Argumento Final, o Escore Bruto e o escore necessário na P2 saem dele por
> aritmética, e por isso não podem se contradizer.

O ticket ofereceu "prever o Argumento Final direto" (A) contra "prever as 3 notas e derivar" (B).
As duas ignoram o mesmo fato:

```
Argumento Final = 1·A1 + 2·A2 + 3·A3
```

Para o Aluno que já fez PAS 1 e PAS 2, **`A1` e `A2` são aritmética exata** — as notas dele estão
na mão e as médias/desvios daqueles anos são públicos. **Só `A3` é desconhecido.** Prever o
Argumento Final inteiro é gastar capacidade estatística reaprendendo, de forma aproximada, ⅗ do
peso de uma conta que já é exata — e produzir um número que pode ser incompatível com as notas que
o próprio Aluno digitou.

A rota C herda as **duas** conclusões do ticket 02 em vez de escolher entre elas: a estabilidade
da grandeza padronizada (§8 do relatório 02, que empurrava para A) **e** a garantia da fórmula
determinística (§3, que removia a objeção contra B).

---

## 2. As decisões, e o porquê de cada uma

| # | Decisão | Motivo em uma linha |
|---|---|---|
| 1 | **Alvo canônico = `A3`** | `A1` e `A2` são exatos; prever o Argumento Final inteiro é aproximar o que já se sabe |
| 2 | **EB derivado, não previsto** | dois modelos independentes na mesma tela divergem sobre aprovação em 11% dos Alunos (§3) |
| 3 | **Nada de projetar a prova futura — Ano-Âncora** | mostrar 5 anos reais em vez de 1 ano extrapolado; a faixa entre eles **é** a incerteza |
| 4 | **Perguntar a língua estrangeira ao Aluno** | agrupar as três línguas embute viés contra espanhol/francês (§5) |
| 5 | **Por língua onde estável, agrupado onde não** | critério medido e reavaliado, não veredito congelado |
| 6 | **Override só no caminho reverso** | no direto, ele faria "vou tirar 9 na redação" **piorar** o EB previsto (§6) |
| 7 | **Momentum e Volatilidade em Argumento, não em EB** | EB e Argumento discordam sobre subir/cair em 17,2% dos Alunos (§4) |
| 8 | **Volatilidade deixa de ser CV** | a média do par é ~0 e negativa em 49,3% da base — a divisão é impossível e redundante (§4.2) |
| 9 | **EB cru permanece como feature** | barato deixar o ticket 09 medir se sobra sinal; o que não pode é ser a **única** leitura |
| 10 | **Estimador Auxiliar = média ponderada na escala padronizada** | mesma ideia da média ponderada de hoje, sem herdar dificuldade de prova; o ticket 10 mede se ML bate |

### 2.1 Como fica a cadeia inteira

```
Â3                     ← ÚNICA previsão do modelo
P1̂, R̂ed                ← Estimador Auxiliar (média ponderada de z-scores) + override do Aluno
P2                     = resolvido:  z_p2 = (A3 − 0,72·z_p1 − 1,00·z_red) / 8,28
EB                     = P1̂ + P2                    (álgebra)
Argumento Final        = A1 + 2·A2 + 3·Â3            (álgebra; A1 e A2 exatos)
σ(Argumento Final)     = 3 × σ(A3)                   (exato: A1 e A2 têm variância zero)
```

Onde `A3` é `Â3` no caminho direto (previsão) e `A3_necessário = (corte − A1 − 2·A2)/3` no
caminho reverso (Quanto Falta).

---

## 3. O tamanho do problema que isto resolve — medido

**Recorte:** triênio 2023/2025, população limpa (`checksum_fecha == True` e Etapa 1 não zerada),
**n = 7.838**. Modelos `modelo_arg_final.joblib` e `modelo_lgbm.joblib` como estão em `models/`,
alimentados com o vetor de 6 features que `api/services/predict_service.py:26` monta.
**Ressalva:** não se sabe se esta turma estava no treino desses artefatos, então os erros absolutos
da tabela abaixo podem ser otimistas. A **divergência entre as rotas** não depende disso.

### 3.1 As duas rotas que a tela mostra hoje

```
|Argumento(rota A) − Argumento(rota EB)|
    mediana 15,29     p90 29,74     máximo 59,84

divergem mais que o próprio RMSE de 13,49 declarado pelo produto:  57,0% dos Alunos
```

| rota | MAE | RMSE | **viés** |
|---|---:|---:|---:|
| A — `modelo_arg_final` direto | 14,44 | 18,16 | **+9,25** |
| EB — `modelo_lgbm` + proporções fixas (`predict_argument_from_eb`) | 12,54 | 15,72 | **−7,23** |

Não é ruído: a rota A é **sistematicamente otimista** em ~9 pontos e a rota EB **sistematicamente
pessimista** em ~7. São ~16 pontos de desencontro estrutural entre dois números exibidos juntos.

### 3.2 Traduzido para a decisão do Aluno

| corte (percentil do Argumento real da turma) | rota A | rota EB | **discordam sobre passar** |
|---|---:|---:|---:|
| mediano (p50) | 56,0% passam | 44,9% passam | **11,0%** (864) |
| seletivo (p75) | 31,5% | 20,7% | **10,8%** (847) |
| muito seletivo (p90) | 14,3% | 6,9% | **7,4%** (580) |
| Medicina-ish (p97) | 5,9% | 1,4% | **4,5%** (349) |

**Para cerca de 1 em cada 9 Alunos, as duas rotas na mesma tela discordam sobre se ele passa.**
Na rota C esse número vai a zero por construção — existe um número só.

### 3.3 O teste que enfraquece o argumento, reportado assim mesmo

Verificação de coerência aritmética: dado o Argumento previsto pela rota A, qual `A3` ele implica
(`A3_implicado = (AF_previsto − A1 − 2·A2)/3`), e esse `A3` é fisicamente possível?

```
A3 realmente observado nesta turma : [−21,55 ; 34,46]
A3 implicado pela rota A           : [−14,91 ; 38,89]
Alunos exigindo um A3 impossível   : 14  (0,2%)
```

**O teste é fraco** — a faixa observada de `A3` é larga, então quase tudo cabe dentro. A
incoerência da rota A é real e está medida em §3.1, mas **não** se manifesta como impossibilidade
aritmética grosseira. Registrado para que ninguém use "0,2%" como se fosse o tamanho do problema.

---

## 4. Por que Momentum e Volatilidade mudam de escala

### 4.1 O EB mede a prova, não o Aluno

**Recorte:** população limpa dos 8 triênios, **n = 60.013**. Comparação entre `ΔEB = EB2 − EB1` e
`ΔArgumento = A2 − A1` para o mesmo Aluno.

```
correlação  ΔEB × ΔArgumento : 0,874
DISCORDAM NO SINAL           : 10.342 Alunos  (17,2%)
```

| triênio | discordam | ΔEB médio | ΔArgumento médio |
|---|---:|---:|---:|
| 2016/2018 | 16,8% | −5,47 | −0,99 |
| 2017/2019 | 14,2% | −5,42 | −1,43 |
| 2018/2020 | 9,8% | −1,17 | −1,06 |
| 2019/2021 | 11,7% | +0,89 | −0,98 |
| 2020/2022 | 11,8% | −0,33 | −1,22 |
| 2021/2023 | 9,2% | −1,67 | −1,68 |
| **2022/2024** | **39,4%** | **+7,56** | **−1,41** |
| 2023/2025 | 22,4% | +3,12 | −1,36 |

Em 2022/2024 a Etapa 2 ficou muito mais fácil (EB médio +7,56). **Para 4 em cada 10 Alunos daquela
turma, o EB comemora uma subida que o Argumento registra como queda.** O Momentum medido em EB não
estava medindo o Aluno — estava medindo a prova. Como o Momentum é a hipótese central do produto e
sustenta o ADR-0008, medi-lo na escala errada é um defeito de fundação.

*Nota de método:* a língua por Etapa foi lida do CSV. O relatório 02 §11 avisa que ela não é
confiável nas linhas que **falham** o checksum, porque o argmin escolhe a língua que minimiza o
delta. Aqui o recorte é só de linhas que **fecham**, onde a língua é validada pelo próprio checksum.

### 4.2 O CV não sobrevive à troca de escala — e não faz falta

O Coeficiente de Variação é `desvio ÷ média`. A divisão pela média existe para tornar Alunos de
níveis diferentes comparáveis: quem tira 20 e oscila 5 é mais instável que quem tira 60 e oscila 5.

Na escala de Argumento essa divisão quebra:

```
MÉDIA do par de Etapas — o denominador do CV:
  em EB        : mediana 28,66     (mín. −8,92)
  em Argumento : mediana  0,12     (mín. −24,10)

Alunos com média de Argumento NEGATIVA : 29.616  (49,3%)
Alunos com |média| < 0,5               :  3.080  ( 5,1%)

CV calculado na escala de Argumento:
  maior valor absoluto : 2.674.047%
  negativo (sem sentido): 29.616 Alunos (49,3%)
```

Não é caso de borda: **é a fórmula não se aplicando à grandeza.** Um CV negativo não significa
"pouco volátil"; não significa nada.

E o ponto conceitual, que é o que decide: **a divisão do CV é redundante nessa escala.** O
Argumento já é medido em desvios-padrão da turma, então "2 pontos de Argumento" significa a mesma
coisa para o Aluno forte e para o fraco. A padronização da fórmula oficial já fez o trabalho que a
divisão fazia. O que resta é dispersão absoluta — `|A2 − A1|` — e ela basta.

**Consequência que não deve passar em branco:** o CV é o que liga o ensemble
(`ensemble._sigmoid_weight`, com limiares de 10% e 20% que só significam algo em escala de EB).
Trocar a escala do histórico tira o chão do mecanismo de ponderação. Isso não decide o ticket 10 —
mas remove um privilégio que o ensemble tinha por inércia. Ver também o defeito 5 de
`defeitos-pendentes.md`, que já registrava o roteador como cego à direção do Momentum.

---

## 5. A Parte 1 do Aluno vivo — o único buraco genuíno

### 5.1 O que a fonte permite

Fato de domínio confirmado pelo dono do produto: **o Edital de cada Etapa publica a linha de todos
os candidatos** — inscrição, nome, escore bruto na Parte 1, escore bruto na Parte 2, somatório,
nota nos itens tipo D e nota da Redação.

Consequência: média e desvio **não precisam ser publicados**, podem ser **calculados sobre a
população inteira**, que é o mesmo cálculo que o Cebraspe faz. As chaves que hoje faltam para o
Aluno vivo deixam de faltar:

```
Aluno vivo hoje = triênio 2024-2026
  A1 precisa de (2024, Etapa 1)  →  ausente do OFFICIAL_STATS, calculável do Edital da Etapa
  A2 precisa de (2025, Etapa 2)  →  ausente do OFFICIAL_STATS, calculável do Edital da Etapa
```

**Exceto a Parte 1.** O Edital da Etapa publica o escore da Parte 1 mas **não diz qual língua
estrangeira** cada candidato fez — e o Cebraspe normaliza a Parte 1 **por língua**. Dá para
calcular a Parte 1 agrupada; não dá para calcular as três separadas.

### 5.2 O tamanho do buraco

Spread entre a maior e a menor média por língua, nos anos que o `OFFICIAL_STATS` cobre:

| Etapa | spread por ano | leitura |
|---|---|---|
| **Etapa 1** | 0,52 (2022) · 0,58 (2023) | **pequeno e estável** |
| **Etapa 2** | 2,02 (2022) · 0,81 (2023) · **3,86 (2024)** | **grande e instável** |
| Etapa 3 | 1,10 · 0,49 · 1,76 · 0,53 | intermediário |

Em `(2024, 2)`: Inglesa **5,09**, Francesa **3,49**, Espanhola **1,23** — quase 4 pontos num desvio
de ~2,2, ou seja **mais de um desvio-padrão inteiro**. Agrupar naquele ano não produz ruído:
produz **viés sistemático contra quem fez espanhol**, de ordem de 0,7 ponto de Argumento de Etapa,
que a Etapa 2 multiplica por 2 → ~1,4 ponto de Argumento Final, sempre na mesma direção e sempre
sobre a minoria.

Para calibrar: a Parte 1 pesa `0,72` de um total de `10,00` — **7,2%**. Os outros 92,8% (Parte 2 e
Redação) saem **exatos** do Edital da Etapa, porque ali não existe problema de língua.

### 5.3 A decisão

- **Perguntar a língua estrangeira ao Aluno** — um campo de três opções ao lado das seis notas que
  ele já digita. Barato, e elimina o viés contra a minoria.
- **Usar estatística por língua onde ela é estável, agrupada onde não é.** Hoje isso significa
  Etapa 1 por língua e Etapa 2 agrupada — mas registrado como **critério medido**, não veredito:
  usa-se por língua quando o spread daquela Etapa se mantém abaixo do limiar nos anos disponíveis,
  e a regra é reavaliada a cada Edital novo. Se a Etapa 2 voltar a se comportar, ela é promovida
  sem ninguém precisar lembrar.

---

## 6. O override, e a armadilha que a rota C cria para ele

O produto tem uma caixa em que o Aluno corrige na mão o P1 e a Redação previstos, porque — nas
palavras do dono do produto — "os modelos são fracos e às vezes a pessoa sabe que não é real".

Na rota C isso ganha um efeito colateral. Com `Â3` fixo (é a previsão de desempenho **total**), se
o Aluno diz *"vou tirar 9 na Redação, não 7"*, a álgebra é obrigada a concluir que ele tira
**menos** na P2 — o total não mudou, só a repartição. Com os desvios da Etapa 3:

| erro/ajuste no estimador | P2 resolvido muda | EB (`P1+P2`) muda |
|---|---:|---:|
| P1 em +1,0 ponto | −0,60 | **+0,40** |
| Redação em +1,0 ponto | −0,95 | **−0,95** |

Ou seja: subir a Redação em 2 pontos **derruba o EB previsto em 1,9**. Dar uma boa notícia ao app
faria o número da tela piorar.

**Decisão: o override vale só no caminho reverso** (Quanto Falta), onde ele é correto e didático —
*"se você garantir 9 na redação, precisa de 1,9 ponto a menos na P2"*. No caminho direto (Argumento
previsto, probabilidade, EB), usa-se o Estimador Auxiliar e o override não entra.

O motivo é de princípio, não de conveniência: o override é um **"e se" sobre repartição**, não uma
nova crença sobre desempenho. Se ele pudesse mexer na probabilidade, o Aluno digitaria 10 na
Redação e veria a chance subir — o produto viraria máquina de auto-engano e o `Â3` deixaria de ser
alvo canônico, porque um campo de formulário poderia sobrescrevê-lo.

Hoje o código já se comporta assim **por acidente** (`p1_override`/`red_override` só existem em
`calculate_required_score`). Passa a ser assim de propósito, com o porquê escrito.

### 6.1 Um efeito colateral bom

A mesma álgebra que cria a armadilha entrega uma vantagem que a rota B não tem: **com `A3` fixo, o
erro dos estimadores de P1 e Redação não soma — ele redistribui.** Um erro de 1 ponto em P1 custa
0,4 ponto de EB (amortecido em 60%), porque o P2 resolvido se move na direção contrária. Na rota B
esse mesmo erro entraria inteiro no Argumento, multiplicado pelo peso. **A rota C perdoa
justamente a fraqueza que o dono do produto observou na tela.**

O que ela não perdoa é a Redação (~0,95, quase um-pra-um) — e é exatamente ali que a caixa de
override é mais valiosa.

---

## 7. O que cada consumidor passa a usar

### 7.1 `src/pas_intelligence/target_calculator.py`

Fica **mais simples**, não mais complexo:

- `arg_pas3_necessario = (arg_alvo − A1 − 2·A2)/3` — já é a linha 259, continua exata, vira o
  centro do módulo em vez de um passo intermediário.
- **Some o carregamento de `.joblib`**: `_carregar_modelo`, `model_load_error`,
  `_registrar_degradacao`, `ModelLoadError` e o `PAS_STRICT_MODELS` deixam de ser necessários
  neste módulo. O Estimador Auxiliar é aritmética. Isso resolve, por remoção, o defeito 3 de
  `defeitos-pendentes.md` no que toca à calculadora reversa.
- `stats_pas3` deixa de vir de `STATS_PAS3_TREND` e passa a vir do **Ano-Âncora**, uma vez por ano
  exibido.
- `p1_override`/`red_override` continuam, agora com contrato escrito — e o `and` da linha 262 vira
  defeito bloqueante (ver §9).

### 7.2 Camada de probabilidade (`src/pas_intelligence/statistics.py`)

- Consome o **Argumento Final derivado**, não um previsto.
- A incerteza deixa de ser um número solto: como `A1` e `A2` são constantes conhecidas,

  ```
  σ(Argumento Final) = 3 × σ(A3)     — exato
  ```

  O `RMSE = 13.49` de hoje está medido no lugar errado. **O ticket 11 passa a medir a incerteza em
  `A3` e multiplicar por 3.** Isso não é escolha de projeto, é a álgebra da fórmula.
- A Nota de Corte comparada é a do **Ano-Âncora**, e são cinco comparações, não uma.

### 7.3 Artefatos em `models/` que saem do caminho de produção

`modelo_arg_final`, `modelo_lgbm`, `modelo_rf`, `modelo_linear`, `modelo_mlp`, `meta_model`,
`scaler`, `meta_scaler`, `p1_pas3_model`, `red_pas3_model` — **os dez**. O caminho de produção
passa a ter **um** modelo (`A3`) mais aritmética.

Isso **não** decide o ticket 10: o ensemble continua candidato, agora como candidato a prever
`A3`. O que muda é que ele perde o privilégio de estar lá por inércia, e perde o mecanismo de
roteamento (o CV), que não sobrevive à troca de escala.

---

## 8. Limitações

- **Os erros absolutos de §3.1 podem ser otimistas.** Não se sabe se o triênio 2023/2025 estava no
  treino de `modelo_arg_final` e `modelo_lgbm` — os artefatos são de 2026-03-11 e não têm
  manifesto (é o problema que o ticket 03 resolve). A **divergência entre as rotas** não depende
  disso e é o número que sustenta a decisão.
- **A rota EB foi convertida para Argumento com as proporções fixas de
  `predict_argument_from_eb` (`0,25` para P1 e Redação = 7,0)**, que é o que o código oferece
  hoje. Uma conversão melhor mudaria os números de §3.1 para baixo, mas não a conclusão: as
  proporções fixas *são* a rota que existe.
- **O teste de coerência aritmética (§3.3) é fraco** e está declarado como tal.
- **O viés por língua de §5.2 é estimado**, não medido por aluno: a língua não é impressa no
  Edital da Etapa e a proporção de falantes por língua no triênio vivo não é conhecida.
- **A estabilidade do spread por língua da Etapa 1 repousa em dois anos** (2022 e 2023). Dois
  pontos não são uma série. Daí a regra ser critério reavaliado, não constante.
- **Nada foi medido sobre o Aluno sem Etapa 1 neste ticket.** Todos os recortes excluem a classe,
  seguindo o relatório 02. A rota C não muda o ADR-0008: para essa classe `A1` continua sendo o z
  de zero (exato, de 2018/2020 em diante) e o que segue indefinido é o **Momentum**, agora medido
  em Argumento em vez de EB. O ticket 06 mantém a estratificação.

---

## 9. Defeitos novos encontrados nesta sessão

Ambos adicionados a [`defeitos-pendentes.md`](defeitos-pendentes.md).

1. **`TRIENNIUM_STATS` não bate com os Editais.** `api/services/gestao_service.py:36-51` carrega
   médias e desvios que, em 2022-2024, divergem do `OFFICIAL_STATS` (P2 da Etapa 1: 20,7094 contra
   20,406 oficial). Têm cara de calculadas de uma amostra, não lidas do Edital. **A API calcula
   `A1` e `A2` com números que não são os do Cebraspe**, e o Argumento que ela produz não bate com
   o do Edital. Na rota C isso passa de imprecisão a erro na fundação, porque `A1` e `A2` deixam
   de ser aproximação e passam a ser a parte *exata* da conta.
   **Conserto barato:** os números certos já estão em disco para todos os triênios que a API
   serve — `OFFICIAL_STATS` tem as 24 chaves, e `saida-nova/medias_desvios.csv` cobre os cinco
   triênios de Edital avulso. É trocar de dicionário. O que continua faltando são só as chaves do
   triênio **vivo**, `(2024,1)` e `(2025,2)`, que dependem da extração por Etapa.

2. ~~**O override é ignorado em silêncio se só um dos dois campos for preenchido.**~~
   **CORRIGIDO nesta sessão.** `target_calculator.py:262` usava `and`; quem mexia só na Redação —
   o caso mais provável, já que é o estimador mais sensível — tinha o override descartado e via um
   P2 necessário que não correspondia ao número digitado. Cada override passou a valer por si,
   com teste de regressão em `test_override_parcial_e_respeitado` cobrindo os quatro casos e
   verificando que o override sozinho **move o P2 necessário**. Era bloqueante da decisão §6.

---

## 10. Glossário

Termos novos ou redefinidos por este ticket. Os três primeiros entraram no `CONTEXT.md`.

- **Argumento de Etapa (`A1`, `A2`, `A3`)** — pontuação padronizada de **uma** Etapa:
  `[(P1−média)/desvio]×0,72 + [(P2−média)/desvio]×8,28 + [(Redação−média)/desvio]×1,00`. Por ser
  feito de z-scores, já nasce descontado da dificuldade da prova daquele ano. Para quem já fez PAS
  1 e PAS 2, `A1` e `A2` são aritmética exata; só `A3` é desconhecido.
- **Alvo Canônico** — a única grandeza que o modelo prevê e da qual todo número da tela é derivado.
  É o `A3`. Argumento Final, EB e escore necessário saem dele por aritmética e por isso não podem
  se contradizer.
- **Estimador Auxiliar** — regra que estima P1 e Redação da Etapa 3 com o único fim de **repartir**
  o Alvo Canônico entre as três partes, para que o resultado possa ser falado em escore em vez de
  em desvio-padrão. **Não é fonte de verdade**: trocá-lo não muda o Argumento previsto nem a
  probabilidade, só a apresentação. Decidido como média ponderada dos z-scores do próprio Aluno.
- **Ano-Âncora** — ano real e já publicado usado como cenário: *"e se a minha Etapa 3 for como
  2025?"*. Amarra **junto** a Nota de Corte daquele ano (concorrência) e as estatísticas da prova
  daquele ano (dificuldade), nunca uma combinação que não aconteceu. Cinco na tela, o mais recente
  em destaque.
- **Volatilidade** *(redefinida)* — dispersão **absoluta** entre Argumentos de Etapa
  (`|A2 − A1|`). **Não é mais um CV**: dividir pela média é impossível (a média é ~0 e negativa em
  metade da base) e redundante (o Argumento já é comparável entre níveis por construção).
- **Momentum** *(redefinido)* — `A2 − A1`, em Argumento, nunca em EB. Medi-lo em EB confunde "o
  Aluno evoluiu" com "a prova ficou mais fácil", e isso acontece em 17,2% dos Alunos.
- **Dispersão absoluta vs. relativa** — relativa (CV) divide pela média para comparar Alunos de
  níveis diferentes; absoluta não divide. Na escala de Argumento a comparabilidade já vem pronta,
  então a relativa perde a razão de existir.
- **Caminho direto / caminho reverso** — direto: das notas do Aluno para a previsão e a
  probabilidade. Reverso: da Nota de Corte para o escore necessário (o Quanto Falta). O override
  vale só no reverso.

---

## 11. Onde continuar

- **Ticket 05 (dataset canônico):** o alvo a materializar é `A3`, calculado com o
  `OFFICIAL_STATS` e a língua gravada. Guardar também `A1` e `A2`, que passam a ser features
  naturais. Manter a coluna `etapa_1_ausente` do ticket 14.
- **Ticket 06 (régua):** a métrica de erro é medida em `A3`. O erro do Argumento Final é
  exatamente `3×` esse número, então não precisa de régua própria. Estratificação por
  `etapa_1_ausente` do ticket 14 permanece.
- **Ticket 08 (janela):** a pergunta "o padrão mudou desde 2018?" agora tem alvo definido e pode
  ser respondida. O aviso do `map.md` — de que medir a janela com o alvo em aberto produz número
  sem significado — está satisfeito.
- **Ticket 09 (features):** Momentum **com sinal** na escala de Argumento; Volatilidade como
  dispersão absoluta; EB cru permanece como feature candidata, não como leitura única. Proibido
  dropar as features da Etapa 1 (decisão de produto do ticket 14).
- **Ticket 10 (família):** o candidato prevê `A3`. O ensemble entra sem privilégio e **sem o
  roteador**, que dependia do CV. Também mede se um Estimador Auxiliar de ML bate a média
  ponderada de z-scores.
- **Ticket 11 (incerteza):** medir em `A3` e multiplicar por 3. Por classe (`etapa_1_ausente`), no
  mínimo duas, conforme o ticket 14.
- **Ticket novo — extração dos Editais por Etapa.** Estava em *Not yet specified* no `map.md` e
  passa a ser **bloqueante do caminho de produção**: sem `(2024,1)` e `(2025,2)`, `A1` e `A2` do
  Aluno vivo não existem, e a rota C inteira depende deles serem exatos. Nasce com três
  requisitos: (i) calcular média e desvio sobre a população inteira do Edital; (ii) `etapa_1_ausente`
  derivado de evidência cruzada; (iii) a regra de língua estável/agrupada de §5.3.
- **Ticket novo — Ano-Âncora na interface.** Cinco anos, o mais recente em destaque, os outros
  atrás de um botão. Toca Preditor, Painel Multi-Curso e Gestão de Ativos. É trabalho de produto,
  não de modelo.
