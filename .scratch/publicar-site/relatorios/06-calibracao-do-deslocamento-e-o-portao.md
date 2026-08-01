# Relatório — Ticket 06: Calibração do Deslocamento e o portão

**Ticket:** `.scratch/publicar-site/issues/06-calibracao-do-deslocamento-e-o-portao.md`
**Status:** concluído — **o portão aprovou na 2ª rodada** (resíduo máximo 4,366 < 5,009)
**Onde vive o código:** `src/pas_extraction/calibracao_deslocamento.py` (cálculo, 28 testes em
`tests/test_pas_extraction_calibracao_deslocamento.py`) + `scripts/medir_deslocamento.py`
(orquestra a rodada real sobre `data/pdfs` e `resultado_final.csv`).

> **Este ticket teve duas rodadas.** A 1ª reprovou (5,751 ≥ 5,009) e o achado dela — *onde* o
> resíduo morava — foi o que apontou o defeito de forma corrigido na 2ª. As duas estão
> registradas: §3-§4 são a 1ª rodada, §5-§7 são a 2ª. A medição de apoio da 2ª rodada, com
> todos os esquemas testados, está em `.scratch/publicar-site/medicao-portao-06/`.

---

## 1. O que foi pedido, e o resultado

A pergunta do ticket: *o Deslocamento é estável o bastante entre triênios para o Preditor
atender a Turma viva?* Resposta medida: **sim — desde que a correção tenha a forma certa.**
Como uma constante subtraída do Argumento, não; como uma correção de média **e desvio**
aplicada aos stats, sim, com folga de 0,64 ponto.

Critérios de aceite:

- [x] A calibração roda sobre **6 triênios** com as duas fontes (Edital isolado de Etapa 1 e 2
      + Edital/CSV oficial) — acima do mínimo de 4 — e estão nomeados na §2
- [x] A correção é calculada **por Etapa** (1 e 2), com dispersão entre anos (§5)
- [x] O portão é uma **asserção em código**: `calibracao_deslocamento.verificar_portao` levanta
      `PortaoReprovadoError` — não é uma leitura de tabela
- [x] O relatório registra, por triênio: parâmetros medidos, residual após correção, e o `n` de
      cada Edital (§3 e §7)
- [x] **O portão aprovou** — a entrada em `OFFICIAL_STATS` é trabalho do ticket 07, que este
      desbloqueia; `Correcao.aplicar` é a porta que ele chama (§6)
- [x] Este relatório vive em `.scratch/publicar-site/relatorios/`

---

## 2. Os Editais usados

O Passo 1 (`.scratch/publicar-site/medicao-passo-1/`) media com **3 pontos** `(ano, Etapa)`.
Este ticket rodou com **11**, cobrindo 6 triênios — os downloads já estavam em
`data/pdfs/editais-de-etapa/` quando a sessão começou (14 arquivos contra os 6 do Passo 1); o
`INDICE.md` é que estava desatualizado, e `scripts/organizar_pdfs.py --aplicar` bastou.

| Triênio | Edital Etapa 1 | Edital Etapa 2 |
|---|---|---|
| 2018/2020 | `Ed 8 PAS Subprograma 2018 1a etapa...` | `ED_17_PAS_2_2018-2020...` |
| 2019/2021 | `ED_6_PAS_1_2019-2021...` | `ED_20_PAS_2_2019-2021...` |
| 2020/2022 | `ED_8_PAS_1_2020-2022...` | `ED_13_PAS_2_2020-2022...` |
| 2021/2023 | `ED_5_PAS_1_2021-2023...` | `ED_16_PAS_2_2021-2023...` |
| 2022/2024 | `ED_8_PAS_1_2022-2024_Retificação...` | — (não baixado) |
| 2023/2025 | `Ed_7_PAS_1_2023_2025...` (não o `Ed_8`, Retificação **parcial** de 827 registros) | `Ed_15_PAS_2_2023-2025...` |

A "verdade" (nota real + Argumento Final oficial) veio de
`.scratch/pdf-extraction/saida-nova/resultado_final.csv`, filtrado por `checksum_fecha`.

---

## 3. 1ª rodada — a medição que reprovou

Correção = **uma constante por Etapa**, subtraída do Argumento (o *Deslocamento*).

| Etapa | Média | Desvio entre anos | Anos |
|---|---:|---:|---:|
| 1 | 1,808 | 0,769 | 6 |
| 2 | 3,215 | 0,353 | 5 |

Resíduo máximo: **5,751** (triênio 2021/2023) contra o limiar 5,009. **REPROVADO.**

---

## 4. 1ª rodada — o achado que valeu mais que a reprovação

Os 5 piores resíduos eram **todos Alunos do topo** (P2 entre 61 e 69, numa Etapa de média ~21).
A 1ª rodada leu isso como "esperado, o erro de um z-score escala com a distância à média". Está
certo — mas a leitura parou cedo demais. Essa concentração no topo não é uma propriedade
inevitável do problema: é a **assinatura de um defeito na forma da correção**.

O Argumento de Etapa é uma soma de z-scores, `(nota − média) ÷ desvio`. Errar cada parâmetro
produz um erro de formato diferente:

- errar **a média** desloca todo Aluno pela mesma quantidade — é um **degrau**, e uma
  constante o corrige por inteiro;
- errar **o desvio** produz um erro **proporcional à distância do Aluno até a média** — é uma
  **reta**, e nenhuma constante a corrige.

E o Edital isolado erra o desvio, muito:

| Etapa | `desvio_edital ÷ desvio_oficial`, P2 | Redação |
|---|---:|---:|
| 1 | **0,874** (13% menor) | 0,861 |
| 2 | 1,005 | **1,207** (21% maior) |

Um Aluno com P2 = 65 está a 44 pontos da média; com o desvio 13% errado e o peso 8,28, isso
sozinho move o Argumento em ~5 pontos. A 1ª rodada estava corrigindo um degrau e deixando a
reta inteira em pé — e a reta é o termo dominante no topo.

---

## 5. 2ª rodada — a correção com a forma certa

Dois parâmetros por (Etapa, componente), em vez de uma constante por Etapa:

- **Δmédia** = `média_edital − média_oficial`
- **razão de desvio** = `desvio_edital ÷ desvio_oficial`

aplicados **nos stats**, antes do Argumento. Isso não é só mais preciso — é **mais estável**,
que é o que o portão de fato cobra:

| Parâmetro | Média | Desvio entre anos | Variação relativa |
|---|---:|---:|---:|
| Deslocamento da Etapa 1 (1ª rodada) | 1,808 | 0,769 | **43%** |
| Δmédia da P2, Etapa 1 | −2,101 | 1,139 | 54% |
| **razão de desvio da P2, Etapa 1** | **0,874** | **0,018** | **2,0%** |
| **razão de desvio da P2, Etapa 2** | **1,005** | **0,015** | **1,5%** |

O que a Turma viva precisa extrapolar deixa de ser um número que oscila 43% ao ano e passa a
ser um que oscila 2%.

Parâmetros medidos, por (Etapa, ano):

| Etapa | Ano | Δmédia P2 | razão dp P2 | Δmédia Red | razão dp Red |
|---|---:|---:|---:|---:|---:|
| 1 | 2018 | −2,812 | 0,898 | −0,145 | 0,866 |
| 1 | 2019 | −3,677 | 0,872 | −0,344 | 0,901 |
| 1 | 2020 | −2,120 | 0,869 | −0,181 | 0,883 |
| 1 | **2021** | **−0,473** | 0,847 | 0,160 | 0,820 |
| 1 | 2022 | −1,346 | 0,887 | 0,132 | 0,822 |
| 1 | 2023 | −2,180 | 0,872 | −0,064 | 0,872 |
| 2 | 2019 | −4,978 | 1,001 | −0,694 | 1,201 |
| 2 | 2020 | −4,050 | 1,003 | −0,569 | 1,168 |
| 2 | 2021 | −4,229 | 1,026 | −0,652 | 1,180 |
| 2 | 2022 | −3,473 | 0,985 | −0,577 | 1,264 |
| 2 | 2024 | −4,610 | 1,012 | −0,722 | 1,220 |

---

## 6. Decisões da 2ª rodada, e o porquê

**A correção mora nos stats, não no Argumento.** Mesma decisão C que o ticket 07 já tinha
tomado por outro motivo (não duplicar o ajuste em `stats_da_prova`, `model_package`,
`training_dataset`, `target_calculator` e a API). Aqui ela ganhou um segundo motivo: é a única
posição em que dá para corrigir o desvio. `Correcao.aplicar(StatsEmpiricos) → HistoricalStats`
é a porta única, e é o que o ticket 07 chama para produzir a entrada `derivada`.

**A razão de desvio é uma razão, não uma diferença.** É assim que ela entra no z-score, e é
assim que ela é estável (2% de variação contra 54% da diferença de média).

**A Parte 1 não é corrigida — decisão medida, não esquecimento.** O Edital isolado não diz a
língua de ninguém, então a única Parte 1 que ele oferece é a misturada; calibrá-la exigiria
construir uma Parte 1 oficial misturada (mistura das três normais ponderada pelo share de
língua de cada ano) só para servir de alvo. Medido: corrigir a Parte 1 **piora** o resíduo
máximo (4,746 contra 4,366) e o p99,9 (4,32 contra 3,93) — a correção acrescenta ruído
justamente onde o erro de não saber a língua já domina. O preço é um viés de **+0,35 ponto**,
registrado aqui e não escondido.

**A validação é *leave-one-year-out*.** Ao medir o resíduo de um triênio, os parâmetros são
agregados **sem** o ano dele — a situação real da Turma viva, que não tem o próprio Edital
oficial. A 1ª rodada não fazia isso: o 5,751 dela incluía o próprio ano. Refeito com LOO, o
esquema da 1ª rodada sai em **6,101**.

**Mediana entre anos, não média — e o portão não depende disso.** O pool contém a Etapa 1 de
2021, aplicada na volta da pandemia (ensino remoto), com Δmédia da P2 de −0,47 contra −1,35 a
−3,68 em todos os outros anos. Quando se sabe que a amostra tem um choque externo que o modelo
não descreve, o agregado robusto é a escolha certa. **E ela não é o que salva o portão:** com
esta calibração o resíduo máximo sai 4,366 com mediana e 4,761 com média, e as duas passam. Foi
por isso que o esquema sem correção da Parte 1 foi preferido ao que a corrige — este último
passa com mediana (4,746) mas reprova com média (5,129), e deixar o portão pendurado na escolha
do agregado seria o data dredging que a 1ª rodada recusou fazer com a regressão.

**Nada foi excluído do pool.** Tirar 2021 da calibração foi testado e é **pior** (4,921) do que
mantê-lo com mediana. A robustez resolve o que a exclusão não resolveria.

---

## 7. O resultado, por triênio

| Triênio | n Alunos | n Edital E1 | n Edital E2 | Bruto \|erro\| médio | Bruto máx | Corrigido \|erro\| médio | Corrigido p95 | **Corrigido máx** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018/2020 | 5.384 | 22.666 | 21.380 | 9,787 | 12,834 | 2,142 | 2,648 | 3,549 |
| 2019/2021 | 7.915 | 23.803 | 16.079 | 8,983 | 12,883 | 0,892 | 1,352 | 2,552 |
| 2020/2022 | 6.553 | 16.298 | 14.016 | 8,445 | 12,624 | 0,810 | 1,826 | 4,116 |
| 2021/2023 | 7.122 | 15.780 | 15.428 | 6,625 | 13,323 | 2,250 | 3,426 | **4,366** |
| 2023/2025 | 7.838 | 19.506 | 16.340 | 7,864 | 11,452 | 1,227 | 2,013 | 2,574 |

**Resíduo máximo: 4,366. Limiar: 5,009. `PORTÃO: APROVADO`** (`scripts/medir_deslocamento.py`
sai com código 0).

**E o resíduo mudou de lugar, que é a notícia melhor que o número.** Resíduo por faixa do
Argumento Final verdadeiro, no esquema da 1ª rodada contra o da 2ª:

| Faixa | 1ª rodada: viés / % otimista | **2ª rodada: viés / % otimista** |
|---|---:|---:|
| 0-25% | −0,682 / 41,0% | −0,033 / 62,4% |
| 90-99% | +1,444 / 90,2% | +0,015 / 63,1% |
| **top 1%** | **+2,285 / 95,7%** | **−0,257 / 53,0%** |

Na 1ª rodada o topo saía inflado em 2,3 pontos e 95,7% dos Alunos do top 1% recebiam um
Argumento otimista — o pior lugar possível, porque é o topo que a
[[project_lista_maiores_argumentos]] enumera e é lá que a nota de corte decide. Na 2ª o viés do
topo some e o máximo do topo cai de 5,085 para 2,933. O que sobra é a cauda **de baixo** de
2021/2023, errando para menos — a direção segura.

---

## 8. O que NÃO foi feito, de propósito

- **Nada foi escrito em `OFFICIAL_STATS`.** As duas entradas derivadas são o ticket 07, que
  este desbloqueia. `Correcao.aplicar` é a porta pronta.
- **Nenhum Edital novo foi baixado.** Três ainda valeriam a pena (§9), mas o portão fechou sem
  eles e a rodada não ficou refém de download.
- **O limiar não foi mexido.** Havia argumento para isso — 5,009 é um erro *típico* (o RMSE do
  modelo) cobrado contra o **máximo entre 34.812 Alunos**, um erro *extremo*, e a língua
  misturada sozinha já consome 3,207 desse orçamento. Ficou registrado como observação (§9), e
  não como mudança: o portão passou na régua original, e afrouxá-la depois de passar seria
  gratuito.

---

## 9. Recomendações para quem vier depois

1. **Três Editais que ainda valem:** a Etapa 2 de 2022/2024 (o par que falta) e as Etapas 1 e 2
   de 2016/2018 e 2017/2019 — os dois triênios já têm Edital oficial. Levaria a validação de 5
   para 8 triênios e diria se 2021 é exceção ou regra. Zero código.
2. **Declarar o resíduo na Largura de Incerteza.** O resíduo tem RMSE 1,633; o Argumento Final
   já carrega 3 × 5,009 = 15,03 de incerteza vinda do `Â3`. Em quadratura, a calibração
   acrescenta 0,56% à incerteza do Aluno. A Largura já é por classe e já vive no manifesto
   (ADR-0012) — uma classe "Aluno servido por estatística derivada" põe esse resíduo na conta
   da probabilidade de aprovação em vez de deixá-lo mudo.
3. **A métrica do portão, se ele for reusado.** Um máximo sobre a população só cresce com mais
   evidência: cada triênio novo pode reprovar um portão que já passou, sem que nada tenha
   piorado. Se este portão virar regressão em CI, vale trocar o máximo por um quantil alto
   (p99,9 = 3,930) ou pelo RMSE, que compõem com o tamanho da amostra em vez de brigar com ele.

---

## 10. Glossário

- **Deslocamento:** a diferença sistemática entre o Argumento de Etapa calculado com a
  média/desvio do Edital isolado e com os oficiais do Cebraspe. Era a *correção* na 1ª rodada;
  na 2ª virou só **número de relatório** — o tamanho do erro que a calibração precisa vencer.
- **Correção de estatística (`CorrecaoComponente`):** o par `(Δmédia, razão de desvio)` que
  descreve o erro do Edital isolado num componente. Substituiu o Deslocamento como a coisa que
  de fato corrige.
- **Componente:** cada uma das três partes de uma Etapa — P1 (língua estrangeira), P2, Redação.
  A correção é por componente porque o erro tem tamanho diferente em cada uma.
- **Edital isolado de Etapa:** o "Resultado final nos itens do tipo D e na prova de redação" de
  uma Etapa 1 ou 2 sozinha. Lista nota por candidato, mas **não a língua estrangeira** de
  ninguém, e cobre todos os **inscritos**, não só os concluintes.
- **Argumento de Etapa (`A1`, `A2`):** a nota de uma Etapa já padronizada pela média/desvio
  daquele ano. `Argumento Final = A1 + 2·A2 + 3·A3`; este ticket só mede `A1` e `A2`.
- **z-score:** `(nota − média) ÷ desvio`. O motivo de a correção precisar de dois parâmetros:
  a média entra somando (erro vira degrau) e o desvio entra dividindo (erro vira reta).
- **Leave-one-year-out (LOO):** ao validar o triênio X, calibrar sem o ano de X. É o que separa
  "o método funciona" de "o método decora o próprio ano".
- **Resíduo:** o quanto o Argumento Final de um Aluno ainda erra **depois** da correção.
- **Portão:** a asserção em código (`verificar_portao`): pelo menos 4 triênios medidos, e o
  maior resíduo abaixo de `LIMIAR_PORTAO` (5,009 — o RMSE do modelo de `A3`, uma vez, não três).
- **Turma viva:** o triênio 2024-2026, que ainda não tem Edital oficial de nenhuma Etapa. É
  quem o ticket 07 atende, agora que o portão aprovou.
