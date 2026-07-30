# Relatório — Ticket 06: Calibração do Deslocamento e o portão

**Ticket:** `.scratch/publicar-site/issues/06-calibracao-do-deslocamento-e-o-portao.md`
**Status:** concluído — **o portão reprovou**
**Onde vive o código:** `src/pas_extraction/calibracao_deslocamento.py` (cálculo, testado em
`tests/test_pas_extraction_calibracao_deslocamento.py`) + `scripts/medir_deslocamento.py`
(orquestra a rodada real sobre `data/pdfs` e `resultado_final.csv`).

---

## 1. O que foi pedido, e o resultado

A pergunta do ticket: *o Deslocamento é estável o bastante entre triênios para o Preditor
atender a Turma viva?* Resposta medida: **não, ainda não** — com os dados hoje em disco, o
portão reprova.

Critérios de aceite:

- [x] A calibração roda sobre **6 triênios** com as duas fontes (Edital isolado de Etapa 1 e 2
      + Edital/CSV oficial) — acima do mínimo de 4 — e estão nomeados na §2
- [x] O Deslocamento é calculado **por Etapa** (1 e 2), com média e dispersão entre triênios
- [x] O portão é uma **asserção em código**: `calibracao_deslocamento.verificar_portao` levanta
      `PortaoReprovadoError` — não é uma leitura de tabela
- [x] O relatório registra, por triênio: offset (Deslocamento) medido, residual após correção,
      e o `n` de cada Edital (§3)
- [x] **O portão reprovou** — nenhuma entrada foi escrita em `OFFICIAL_STATS`; a medição e a
      reprovação são a entrega (§4); o mapa foi atualizado (`map.md`, "O que pode reordenar
      tudo", item 2)
- [x] Este relatório vive em `.scratch/publicar-site/relatorios/`

---

## 2. O que mudou desde o Passo 1

O Passo 1 (`.scratch/publicar-site/medicao-passo-1/`) media o Deslocamento com **3 pontos**
`(ano, Etapa)`: `(2022, Etapa 1)`, `(2023, Etapa 1)`, `(2024, Etapa 2)` — só um triênio
(2023/2025) tinha Edital isolado de Etapa 1 **e** 2 ao mesmo tempo, o mínimo para montar um
Argumento Final completo (`1×A1 + 2×A2`) e comparar contra o oficial.

Este ticket teve uma pré-condição que não é trabalho de agente: baixar Editais isolados de
Etapa de triênios mais antigos. Os downloads **já estavam em `data/pdfs/editais-de-etapa/`**
quando a sessão começou (14 arquivos, contra os 6 do Passo 1) — o `INDICE.md` é que estava
desatualizado; `scripts/organizar_pdfs.py --aplicar` só precisou ser rodado de novo para
refletir o que já tinha sido baixado. Isso deu **4 triênios novos com Etapa 1 e 2 completas**:
2018/2020, 2019/2021, 2020/2022, 2021/2023 — mais 2023/2025 (já tinha as duas) e 2022/2024
(só Etapa 1; entra no Deslocamento por Etapa mas não na tabela por triênio, que exige as duas).

| Triênio | Edital Etapa 1 | Edital Etapa 2 |
|---|---|---|
| 2018/2020 | `Ed 8 PAS Subprograma 2018 1a etapa...` | `ED_17_PAS_2_2018-2020...` |
| 2019/2021 | `ED_6_PAS_1_2019-2021...` | `ED_20_PAS_2_2019-2021...` |
| 2020/2022 | `ED_8_PAS_1_2020-2022...` | `ED_13_PAS_2_2020-2022...` |
| 2021/2023 | `ED_5_PAS_1_2021-2023...` | `ED_16_PAS_2_2021-2023...` |
| 2022/2024 | `ED_8_PAS_1_2022-2024_Retificação...` | — (não baixado) |
| 2023/2025 | `Ed_7_PAS_1_2023_2025...` (não o `Ed_8`, que é a Retificação **parcial** de 827 registros) | `Ed_15_PAS_2_2023-2025...` |

A "verdade" (nota real + Argumento Final oficial) para todos os 8 triênios fechados já existia
em `.scratch/pdf-extraction/saida-nova/resultado_final.csv` (extraído para o mapa de treino do
modelo) — este ticket não precisou extrair Resultado Final nenhum, só os Editais isolados de
Etapa que faltavam.

---

## 3. A medição

**Método.** Para cada `(ano, Etapa)` com Edital isolado, `pas_extraction.etapa` (ticket 02)
extrai média/desvio *empíricos* (o que a Turma viva veria). Para o mesmo `(ano, Etapa, língua)`,
`training_dataset.stats_da_prova` devolve o oficial (o que o Cebraspe publicou). Para cada Aluno
de `resultado_final.csv`, o Argumento de Etapa é calculado duas vezes com
`argument_calculator.calculate_argument_etapa` — a mesma função de produção, nunca reimplementada
— uma vez com cada fonte; a diferença é `dA`. Isso reproduz o cenário realista do Passo 1 (o
"LP": língua misturada **e** população de inscritos, não de concluintes), agora com dado real em
vez de simulado em 5 dos 6 triênios.

**Deslocamento por Etapa** (média entre os anos com Edital isolado daquela Etapa; desvio é
**amostral**, porque os triênios medidos são uma amostra dos possíveis):

| Etapa | Média | Desvio entre triênios | Triênios | Por ano |
|---|---:|---:|---:|---|
| 1 | 1,808 | 0,769 | 6 | 2018=2,19 · 2019=2,96 · 2020=1,91 · 2021=0,75 · 2022=1,23 · 2023=1,81 |
| 2 | 3,215 | 0,353 | 5 | 2019=3,80 · 2020=3,01 · 2021=3,28 · 2022=2,94 · 2024=3,04 |

O desvio da Etapa 1 (0,77) é mais de duas vezes o da Etapa 2 (0,35) — a Etapa 2 é a mais
otimista na média (o que já se sabia do Passo 1), mas é a Etapa 1 que **varia mais** de ano para
ano, e é essa variação que o portão vai cobrar.

**Por triênio, aplicando a média acima como correção única** (nenhum triênio usa o próprio
Deslocamento — isso seria simular ter o dado que a Turma viva não tem):

| Triênio | n Alunos | n Edital E1 | n Edital E2 | Bruto \|erro\| médio | Bruto p95 | Bruto máx | Corrigido \|erro\| médio | Corrigido p95 | **Corrigido máx** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018/2020 | 5.384 | 22.666 | 21.380 | 9,787 | 11,130 | 12,834 | 1,555 | 2,891 | 4,595 |
| 2019/2021 | 7.915 | 23.803 | 16.079 | 8,983 | 10,716 | 12,883 | 0,979 | 2,477 | 4,644 |
| 2020/2022 | 6.553 | 16.298 | 14.016 | 8,445 | 10,180 | 12,624 | 0,822 | 2,035 | 4,385 |
| **2021/2023** | 7.122 | 15.780 | 15.428 | 6,625 | 9,195 | 13,323 | 1,859 | 3,932 | **5,751** |
| 2023/2025 | 7.838 | 19.506 | 16.340 | 7,864 | 9,846 | 11,452 | 1,127 | 2,860 | 4,560 |

**Resíduo máximo, sobre todos os 5 triênios: 5,751. Limiar do portão: 5,009. `PORTÃO:
REPROVADO`** (`scripts/medir_deslocamento.py` sai com código 1).

O bruto e o corrigido de 2023/2025 (7,864 / 4,560) ficam perto do que o Passo 1 media (7,87 /
4,19) — a pequena diferença vem de o Deslocamento aqui ser a média de **6** anos, não só dos 2-3
que o Passo 1 tinha para a Etapa 1.

---

## 4. Por que reprovou, e por que não é um artefato de dado

O culpado é **2021/2023**: sozinho, passa do limiar em 0,74 ponto. Investigando quem carrega
esse resíduo (`erro_corrigido` por Aluno naquele triênio): os 5 piores são todos alunos do
**topo** — P2 entre 61 e 69 (a média da Etapa é ~21), redação 9 a 10. Isso é esperado, não um
bug: o erro de um z-score escala com a distância à média, então o mesmo desvio-padrão errado
pesa mais em quem está longe da média — e é justamente o extremo superior que
`[[project_lista_maiores_argumentos]]` (outro ticket desta frente) tenta enumerar. O portão
aperta exatamente onde o produto mais precisa de precisão.

A causa de fundo é a mesma do Passo 1 — Etapa 2 estimada sobre inscritos, não concluintes — mas
agora com um segundo efeito visível: o **Deslocamento da Etapa 1 varia entre 0,75 e 2,96** ponto
conforme o ano (desvio amostral 0,77), e usar a média de 6 anos para corrigir um ano específico
deixa sobra justamente quando esse ano se afasta da média — como 2021, que está no extremo baixo
(0,75) enquanto a média é 1,81.

**O que não é a causa:** falha de extração. As 6 novas extrações rodaram os 12 Editais sem
`EditalParcialError`, e os `n` batem com o que se espera de um Edital completo (14-24 mil
registros cada). O parser do ticket 02 continua funcionando sem alteração nesta família de
documento, exatamente como o ticket previa.

---

## 5. Decisões tomadas e o porquê

**Deslocamento como uma média simples entre triênios, não uma projeção temporal.** Havia a opção
de ajustar uma tendência (regressão linear, como `argument_calculator.project_historical_stats`
já faz para outra finalidade) em vez de uma média fixa — talvez a Etapa 1 esteja subindo ano a
ano, e extrapolar reduzisse o resíduo de 2021. Decisão: **não** fazer isso neste ticket. Com 5-6
pontos e um padrão que não parece monotônico (2,19 → 2,96 → 1,91 → 0,75 → 1,23 → 1,81), ajustar
uma tendência seria data dredging — encontrar a curva que faz o portão passar, não a que
descreve o fenômeno. Fica registrado como uma saída possível para quem decidir o próximo passo
(§7), não como algo já tentado e descartado.

**O portão também cobra `≥ 4` triênios em código, não só o resíduo.** `verificar_portao` levanta
`PortaoReprovadoError` tanto por poucos triênios quanto por resíduo alto — são as duas formas do
mesmo risco (Deslocamento medido em poucos pontos não generaliza), e a asserção não deveria
aprovar por acidente se um Edital futuro sumir e a contagem cair para 3.

**A correção usa a mesma fórmula duas vezes, nunca uma reimplementação.** `calcular_delta_por_etapa`
chama `argument_calculator.calculate_argument_etapa` tanto para o Argumento "verdade" (stats
oficiais, por língua) quanto para o "empírico" (stats do Edital isolado, língua misturada) — o
mesmo cuidado que `training_dataset._calcular_argumentos_etapa` já toma. Duas fórmulas
seriam o começo de uma divergência silenciosa entre o que este ticket mede e o que o Preditor
calcula de verdade.

**`stats_oficiais` é injetável, não importado direto.** Mesmo padrão de
`relatorio_official_stats.comparar`: os testes montam `HistoricalStats` na mão em vez de
depender do `OFFICIAL_STATS` de produção, para não quebrar toda vez que um Edital novo entrar em
`pas_constants.py`.

**2022/2024 entra no Deslocamento por Etapa mas fica fora da tabela por triênio.** Só tem Edital
isolado de Etapa 1 (a Etapa 2 desse triênio não foi baixada) — sem as duas Etapas não há
Argumento Final para comparar. Baixar essa Etapa 2 aumentaria a amostra do Deslocamento da Etapa
2 de 5 para 6 pontos e adicionaria um sétimo triênio à tabela de validação; não foi feito porque
o portão já tinha dado o número final desta rodada sem ela.

---

## 6. O que NÃO foi feito, de propósito

- **Nada foi escrito em `OFFICIAL_STATS`.** O ticket é explícito: se o portão reprova, a entrega
  é a medição e a reprovação, sem entrada nova.
- **`map.md` foi atualizado** ("O que pode reordenar tudo", item 2, e a linha do ticket 06 na
  tabela de rota) — não reescrito. Reordenar o mapa inteiro (o que entra no lugar dos tickets 07
  e 14) é decisão do dono do produto, não deste ticket.
- **Nenhuma tentativa de "consertar" o resíduo por filtro ou tendência** (ver §5) — o ticket pede
  a medição honesta, e o Passo 1 já registrou que filtrar a população do Edital isolado para
  bater com o oficial não funciona (7 recortes testados, nenhum reproduziu).

---

## 7. Próximo passo — decisão do dono do produto

O portão reprovou; os tickets 07 (Preditor responde para a Turma viva) e 14 (publicação) ficam
bloqueados até uma decisão. Três saídas visíveis na medição, nenhuma escolhida aqui:

1. **Deslocamento por triênio mais próximo**, em vez de uma média global — usar o ano mais
   recente disponível de cada Etapa (ou os últimos 2-3) em vez da média dos 6, testando se isso
   reduz o resíduo de 2021/2023 sem piorar os demais.
2. **Aceitar o risco residual** — o pior caso medido (5,751) passa do limiar por 0,74 ponto, não
   por uma ordem de grandeza; decidir se `LIMIAR_PORTAO` (1× RMSE) é conservador demais para este
   uso, ou se o produto tolera esse resíduo com um aviso na tela do Aluno.
3. **Baixar mais triênios** (só restam os anteriores a 2018, se existirem em formato compatível)
   para engordar a amostra do Deslocamento antes de decidir 1 ou 2.

Nenhuma das três foi tentada nesta rodada — são chamadas de produto, e o ticket pede
explicitamente que o código só torne a decisão visível, não que a tome no lugar do dono.

---

## 8. Glossário

- **Deslocamento:** a diferença sistemática entre o Argumento de Etapa calculado com a média/
  desvio do Edital isolado (empírico, o que a Turma viva vê) e com a média/desvio oficial do
  Cebraspe (o que só sai depois do PAS 3). Medido por Etapa (1 ou 2), porque o erro não se
  distribui igual entre elas.
- **Edital isolado de Etapa:** o "Resultado final nos itens do tipo D e na prova de redação" de
  uma Etapa 1 ou 2 sozinha, publicado no ano seguinte à prova. Lista nota por candidato, mas
  **não a língua estrangeira** de ninguém, e cobre todos os **inscritos**, não só os concluintes.
- **Argumento de Etapa (`A1`, `A2`):** a nota de uma Etapa já padronizada pela média/desvio
  daquele ano (`calculate_argument_etapa`). `Argumento Final = A1 + 2·A2 + 3·A3`; este ticket só
  mede `A1` e `A2` — `A3` é previsto pelo modelo e está fora do escopo.
- **Resíduo (corrigido):** `1·(A1_empírico − Deslocamento₁) + 2·(A2_empírico − Deslocamento₂)`
  menos o mesmo cálculo com as stats oficiais — o quanto o Argumento Final de um Aluno erraria
  mesmo depois de aplicar a correção.
- **Portão:** a condição em código (`verificar_portao`) que decide se a calibração está boa o
  bastante: pelo menos 4 triênios medidos, e o maior resíduo corrigido abaixo de `LIMIAR_PORTAO`
  (5,009 — o RMSE do modelo de `A3`, uma vez, não três).
- **Turma viva:** o triênio 2024-2026, que ainda não tem Edital oficial de nenhuma Etapa — só
  Editais isolados. É quem o ticket 07 atenderia, se o portão tivesse aprovado.
- **Triênio de validação:** um triênio fechado (já tem Argumento Final oficial) com Edital
  isolado de Etapa 1 **e** 2 em disco — só esses entram na tabela do §3; 2022/2024 (só Etapa 1)
  fica de fora dela.
