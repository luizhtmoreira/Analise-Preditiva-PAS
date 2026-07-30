# 06 — Calibração do Deslocamento e o portão

**What to build:** a resposta, medida e com critério de reprovação, para a pergunta *"o Deslocamento
é estável o bastante entre triênios para o Preditor atender a Turma viva?"*.

⚠ **Este ticket pode reprovar.** Ele é o único risco vivo do mapa, e existe separado justamente para
falhar barato — antes de qualquer código de produto ser escrito em cima da resposta.

## O que é o Deslocamento

Estimando média e desvio a partir dos **Editais isolados de Etapa** (a única fonte disponível para a
Turma viva), o Argumento Final sai **+7,87 pontos acima** do verdadeiro. Isso caberia no limiar
frouxo de `3 × RMSE = 15,03`, mas com pouca folga e no sentido perigoso: o Preditor ficaria
**otimista**, dizendo ao Aluno que ele está melhor do que está.

**O que salva é que não é ruído, é um degrau.** A média do erro e a média do valor absoluto são o
mesmo número — 7,867 e 7,867. Todo Aluno erra na mesma direção e quase na mesma quantidade:

| | \|erro\| médio | p95 | máx |
|---|---:|---:|---:|
| Bruto | 7,87 | 9,85 | 11,46 |
| **Corrigido pelo deslocamento** | **1,14** | 2,62 | **4,19** |

**A causa está localizada** — o erro não se distribui entre as Etapas, está quase todo na Etapa 2:
−1,35 em (2022, Etapa 1), −2,18 em (2023, Etapa 1), **−4,61** em (2024, Etapa 2). O Edital isolado
de Etapa 2 de 2024 tem 16.339 candidatos; os concluintes daquele triênio são 8.703. O Cebraspe
calcula a média da Etapa 2 sobre os **concluintes** — ele só publica o Edital de média e desvio
depois do PAS 3, quando já sabe quem chegou ao fim. Estimando sobre os 16.339 pegamos metade a mais
de gente, e essa metade é mais fraca: 0,31 desvio-padrão de diferença.

**Não tente resolver por filtro — já foi testado e não dá.** Sete recortes da lista do Edital
(tirando faltoso, nota zero, redação zero, tipo D zero) e nenhum reproduz o oficial. O desvio erra
por −1,5 em todos: para o desvio subir seria preciso gente com mais dispersão, para a média subir
seria preciso gente com nota mais alta, e as duas direções se contradizem. Existe uma população que
o Cebraspe usa e que não temos, e **explicá-la está fora desta rodada**.

## O problema com o número atual

Os +7,87 estão medidos em **um** triênio para a Etapa 2 e dois para a Etapa 1 — e os dois da Etapa 1
já divergem entre si (+1,23 em 2022, +1,81 em 2023). Com tão poucos pontos a correção é um número
solto, sem média nem dispersão próprias.

## O portão

A calibração roda sobre **pelo menos quatro triênios** que tenham *tanto* o Edital isolado de Etapa
*quanto* o Edital oficial de média e desvio. Aplicando o Deslocamento médio por Etapa, o **maior**
erro residual em Argumento Final, sobre todos os triênios de validação, tem que ficar **abaixo de
5,009** — um RMSE do modelo, não três.

Hoje esse máximo está em **4,19** com os três pontos existentes.

O portão é apertado de propósito: o limiar frouxo de 15,03 já foi atendido pelo erro **bruto** de
7,87, então passar por ele não é evidência de nada.

**Se o portão não fechar,** o Preditor volta a recusar para a Turma viva, as entradas derivadas não
entram, e o ticket 07 não acontece. Isso reordena o mapa e é decisão do dono do produto, não do
código — o portão só a torna visível antes de publicar, em vez de depois.

## ⚠ Pré-condição que não é trabalho de agente

**Alguém precisa baixar os PDFs.** A calibração precisa dos Editais isolados de Etapa 1 e 2 de mais
três ou quatro triênios fechados, em `data/pdfs`. Não há automação de download aqui, e o extrator do
ticket 02 já roda nessa família de documento sem alteração — é só o arquivo que falta.

Os 6 que já estão em disco: `(2022, Etapa 1)`, `(2023, Etapa 1)`, `(2024, Etapa 2)` servem de
validação; `(2024, Etapa 1)` e `(2025, Etapa 2)` são as entradas de produção; e o Edital 8 de 2023
é o parcial que não serve.

**Ao baixar, conte os registros antes de usar** — "Retificação" no nome não diz se o documento é
parcial ou completo (ver ticket 02).

**Blocked by:** 02 (Extrator de Editais de Etapa vira módulo).

**Status:** ready-for-agent

- [ ] A calibração roda sobre ≥ 4 triênios com as duas fontes, e os triênios usados estão nomeados
      no relatório
- [ ] O Deslocamento é calculado **por Etapa**, com média e dispersão entre triênios, não como um
      número global
- [ ] O portão é uma **asserção em código**, não uma leitura de relatório: residual máximo em
      Argumento Final < 5,009
- [ ] O relatório registra, por triênio: offset medido, residual após correção, e o `n` de cada
      Edital — no padrão de `relatorio_official_stats.py`
- [ ] Se o portão reprovar, o ticket entrega **a medição e a reprovação**, sem escrever entrada
      nenhuma no `OFFICIAL_STATS`, e o mapa é atualizado com o achado
- [ ] O relatório vai para `.scratch/publicar-site/relatorios/`, com decisões, porquês e glossário
