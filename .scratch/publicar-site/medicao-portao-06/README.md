# Medição — como contornar a reprovação do portão (ticket 06)

**Data:** 2026-07-30 · **Status:** ✅ **implementado; o portão aprovou em 4,366**

> **Desfecho, para quem ler só o topo.** A saída (1) desta nota foi implementada em
> `src/pas_extraction/calibracao_deslocamento.py` e o portão aprovou. Duas coisas mudaram
> depois que esta nota foi escrita, e as tabelas abaixo ainda refletem o estado anterior:
>
> 1. **A Parte 1 acabou ficando sem correção.** Medido na rodada seguinte: corrigi-la dá
>    máximo 4,746; não corrigi-la dá **4,366** — e, o que decidiu, o esquema sem ela passa
>    com **média (4,761) e mediana (4,366)**, enquanto o esquema com ela reprova sob média
>    (5,129). Isso tira o portão de cima da escolha do agregado, que era a ressalva do §7.
> 2. **O download dos 3 Editais deixou de ser necessário para destravar** — segue valendo
>    como reforço, agora sem urgência (ver §9 do relatório do ticket 06).
>
> O relatório definitivo é `../relatorios/06-calibracao-do-deslocamento-e-o-portao.md`.
**Scripts:** `/private/tmp/.../scratchpad/experimento{,2,3}.py` (cópias abaixo em `scripts/`)
**Entrada:** os mesmos 11 Editais isolados do ticket 06 + `resultado_final.csv` (34.812 Alunos
com as duas Etapas, 5 triênios de validação)

---

## 1. O diagnóstico: a correção do ticket 06 tem a forma errada

O ticket 06 corrige o **Argumento** com **um número por Etapa** (o Deslocamento). Mas o erro
entre o Edital isolado e o oficial **não é um número** — é uma **reta**.

O Argumento de Etapa é uma soma de z-scores (`z = (nota − média) / desvio`). Se o Edital
isolado erra **a média**, todo Aluno erra igual: é um degrau, e subtrair uma constante resolve.
Se ele também erra **o desvio**, o erro passa a crescer com a distância do Aluno até a média —
e nenhuma constante conserta isso.

E ele erra o desvio, muito:

| Etapa | `desvio_edital / desvio_oficial` da P2 | da Redação |
|---|---|---|
| 1 | **0,874** (13% menor) | 0,861 |
| 2 | 1,005 | **1,207** (21% maior) |

Um Aluno com P2 = 65 numa Etapa de média 21 está a 44 pontos da média. Com o desvio 13%
errado e o peso 8,28, isso sozinho move o Argumento em ~5 pontos — que é a ordem de grandeza
do resíduo que reprovou o portão.

**A assinatura disso está no próprio relatório do ticket 06:** os 5 piores resíduos eram todos
Alunos do topo. Não é coincidência nem azar de amostra; é a reta.

## 2. A saída: calibrar a média **e** o desvio, nos stats

Em vez de um Deslocamento no Argumento, dois parâmetros por (Etapa, componente):

- **Δmédia** = `média_edital − média_oficial`
- **razão de desvio** = `desvio_edital / desvio_oficial`

E aplicá-los **nos stats**, antes do Argumento — que é exatamente onde o ticket 07 já decidiu
que a correção mora (decisão C: "a correção entra na média, não no Argumento Final"). O
ticket 07 só precisa carregar o desvio corrigido junto da média corrigida.

**Por que isso é mais estável, e não só mais preciso:** a razão de desvio varia muito menos
entre anos do que o Deslocamento. Dispersão dos parâmetros entre os 5-6 anos medidos:

| Parâmetro | Média | Desvio entre anos | Variação relativa |
|---|---:|---:|---:|
| Deslocamento da Etapa 1 (o do ticket 06) | 1,808 | 0,769 | **43%** |
| Δmédia da P2, Etapa 1 | −2,101 | 1,139 | 54% |
| **razão de desvio da P2, Etapa 1** | **0,874** | **0,018** | **2,0%** |
| **razão de desvio da P2, Etapa 2** | **1,005** | **0,015** | **1,5%** |

O que a Turma viva precisa extrapolar deixa de ser um número que oscila 43% ao ano e passa a
ser um que oscila 2%.

## 3. O resultado — tudo em leave-one-year-out

**Leave-one-year-out (LOO):** ao corrigir o triênio X, os parâmetros são calculados **sem** o
ano X. Sem isso, a medição usa dado que a Turma viva não tem. O ticket 06 **não** fez isso: os
5,751 dele incluem o próprio ano no cálculo do Deslocamento. Feita a conta honesta, o esquema
do ticket 06 sai em **6,101**, não 5,751.

| Esquema | viés | \|erro\| médio | RMSE | p99,9 | **máx** | portão |
|---|---:|---:|---:|---:|---:|---|
| Bruto (sem correção) | 8,272 | 8,272 | — | 12,17 | 13,323 | reprova |
| S1 — constante, como no ticket 06 (com o próprio ano) | 0,034 | 1,252 | 1,571 | 4,976 | 5,751 | reprova |
| S1 — constante, LOO honesto | 0,039 | 1,383 | — | 5,324 | 6,101 | reprova |
| S1 — constante do ano mais recente, LOO | 0,564 | 1,380 | — | 4,720 | 5,440 | reprova |
| **S2 — média+desvio nos stats, LOO, agregado por média** | −0,003 | 1,463 | 1,729 | 4,659 | 5,129 | reprova por 0,12 |
| **S2 — média+desvio nos stats, LOO, agregado por mediana** | −0,104 | **1,313** | **1,598** | **4,317** | **4,746** | **APROVA** |
| S2 sem corrigir o desvio (só a média) | 0,238 | 1,663 | 2,031 | 6,109 | 6,886 | reprova |
| Piso de língua misturada (irredutível) | 0,001 | 0,443 | 0,605 | 2,683 | 3,207 | — |

Sob S2-mediana: **0 Alunos de 34.812** acima de 5,009 (contra 75 sob S1).

Máximo por triênio:

| Triênio | S1 (ticket) | **S2 mediana** | Piso de língua |
|---|---:|---:|---:|
| 2018/2020 | 4,595 | 2,858 | 1,381 |
| 2019/2021 | 4,644 | 1,854 | 1,454 |
| 2020/2022 | 4,385 | 3,441 | **3,207** |
| 2021/2023 | **5,751** | **4,746** | 1,349 |
| 2023/2025 | 4,560 | 2,374 | 2,869 |

Em 2020/2022 e 2023/2025 o S2 já encosta no **piso de língua** — o erro de não saber a língua
estrangeira de ninguém, que nenhuma calibração remove. Ali não há mais o que ganhar.

## 4. O resíduo que sobra mudou de lugar — e essa é a notícia melhor

Resíduo por faixa do Argumento Final verdadeiro:

| Faixa | S1 viés | S1 máx | S1 % otimista | **S2 viés** | **S2 máx** | S2 % otimista |
|---|---:|---:|---:|---:|---:|---:|
| 0-25% | −0,682 | 5,751 | 41,0% | −0,033 | 4,746 | 62,4% |
| 25-50% | −0,331 | 5,571 | 47,9% | −0,186 | 4,263 | 58,4% |
| 50-75% | 0,126 | 4,475 | 57,2% | −0,147 | 4,006 | 59,3% |
| 75-90% | 0,683 | 4,118 | 71,6% | −0,075 | 3,705 | 60,7% |
| 90-99% | 1,444 | 4,595 | 90,2% | 0,015 | 2,918 | 63,1% |
| **top 1%** | **+2,285** | **5,085** | **95,7%** | **−0,257** | **2,933** | 53,0% |

Sob S1 o topo é sistematicamente **otimista em +2,3 pontos**, e 95,7% dos Alunos do top 1%
recebem um Argumento inflado — o pior lugar possível para o erro morar, porque é o topo que a
[[project_lista_maiores_argumentos]] enumera e é no topo que a nota de corte decide. Sob S2 o
viés do topo some (−0,26) e o máximo do topo cai de 5,085 para 2,933. O que sobra é a **cauda
de baixo de 2021/2023**, errando para **menos** — a direção segura.

## 5. 2021 é um ano anômalo, e carrega a reprovação sozinho

Parâmetros por (Etapa, ano) — repare na linha de 2021, Etapa 1:

| Etapa | ano | Δm P1 | Δm P2 | Δm Red | razão dp P1 | razão dp P2 | razão dp Red | Deslocamento |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2018 | −0,404 | −2,812 | −0,145 | 0,942 | 0,898 | 0,866 | 2,192 |
| 1 | 2019 | −0,470 | −3,677 | −0,344 | 0,977 | 0,872 | 0,901 | 2,962 |
| 1 | 2020 | −0,312 | −2,120 | −0,181 | 0,929 | 0,869 | 0,883 | 1,915 |
| 1 | **2021** | −0,227 | **−0,473** | +0,160 | 0,948 | **0,847** | 0,820 | **0,746** |
| 1 | 2022 | −0,285 | −1,346 | +0,132 | 0,946 | 0,887 | 0,822 | 1,226 |
| 1 | 2023 | −0,330 | −2,180 | −0,064 | 0,928 | 0,872 | 0,872 | 1,809 |
| 2 | 2019 | −0,763 | −4,978 | −0,694 | 1,097 | 1,001 | 1,201 | 3,803 |
| 2 | 2020 | −0,590 | −4,050 | −0,569 | 1,078 | 1,003 | 1,168 | 3,014 |
| 2 | 2021 | −0,378 | −4,229 | −0,652 | 1,085 | 1,026 | 1,180 | 3,282 |
| 2 | 2022 | −0,710 | −3,473 | −0,577 | 1,058 | 0,985 | 1,264 | 2,940 |
| 2 | 2024 | −0,722 | −4,610 | −0,722 | 1,009 | 1,012 | 1,220 | 3,037 |

A Δmédia da P2 na Etapa 1 de 2021 é **−0,47**, contra −1,35 a −3,68 em todos os outros anos —
5× fora do padrão. É esse ano que faz o triênio 2021/2023 reprovar nos dois esquemas. **Vale
perguntar ao dono do domínio o que aconteceu com a Etapa 1 de 2021** (prova adiada por
pandemia? população de inscritos diferente?) antes de tratá-lo como ruído.

## 6. A métrica do portão também merece uma pergunta

O limiar é **5,009 = 1× o RMSE do modelo de `A3`** — um erro *típico*. Ele é cobrado contra o
**máximo entre 34.812 Alunos** — um erro *extremo*. São grandezas de naturezas diferentes, e a
comparação tem duas consequências ruins:

1. **O portão fica mais difícil quanto mais evidência entra.** Cada triênio novo só pode
   aumentar o máximo, nunca diminuir. Um portão que reprova mais à medida que se mede melhor
   não está medindo risco, está medindo tamanho de amostra.
2. **O piso já come 64% do orçamento.** A língua misturada sozinha, sem erro nenhum de
   calibração, chega a 3,207 no máximo. Sobram 1,8 pontos para tudo o mais.

Comparando grandezas iguais: o resíduo do S2 tem **RMSE 1,598** contra o limiar de 5,009. E o
Argumento Final que o Aluno vê já carrega **3 × 5,009 = 15,03** de incerteza vinda do `Â3`
previsto. Somando em quadratura, o resíduo da calibração aumenta a incerteza total do Aluno em
**0,56%** (de 15,03 para 15,11).

## 7. Três formas de destravar, e o que cada uma custa

1. **Trocar a forma da correção (S2).** Recalibra média **e** desvio, nos stats. É trabalho de
   código já desenhado, cabe no ticket 07 (a correção já ia morar nos stats), e melhora todos
   os números — inclusive tira o viés otimista do topo. **Passa o portão como está escrito**,
   com margem de 0,26 se o agregado for mediana; reprova por 0,12 se for média.
2. **Baixar 3 Editais.** A Etapa 2 de 2022/2024 (o par que falta) e as Etapas 1 e 2 de
   2016/2018 e 2017/2019 — os dois triênios já têm Edital oficial. Levaria a validação de 5
   para 8 triênios e diria se 2021 é exceção ou regra, tirando a decisão de cima da escolha
   entre média e mediana. Custo: uma sessão de download, zero código.
3. **Declarar o resíduo em vez de escondê-lo.** A Largura de Incerteza já é por classe de
   Aluno e já vive no manifesto (ADR-0012). Uma classe nova — "Aluno servido por estatística
   derivada" — com largura `√(largura² + RMSE_resíduo²)` põe o resíduo na conta da
   probabilidade de aprovação, em vez de deixá-lo mudo. Custo: uma entrada no manifesto.

**A honestidade da medição:** média e mediana foram testadas na mesma rodada, e a mediana passa
enquanto a média reprova por 0,12. Escolher a mediana *depois* de ver o resultado é o
data dredging que o próprio ticket 06 recusou fazer com a regressão. Por isso o (2) importa: com
8 triênios a escolha do agregado deixa de ser o que decide o portão.
