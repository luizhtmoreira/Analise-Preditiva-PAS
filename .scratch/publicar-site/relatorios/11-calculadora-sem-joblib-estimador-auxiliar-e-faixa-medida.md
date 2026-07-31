# Relatório — Ticket 11: Calculadora sem `.joblib`, Estimador Auxiliar e a faixa medida da P2

**Ticket:** `.scratch/publicar-site/issues/11-calculadora-sem-joblib-estimador-auxiliar-e-faixa-medida.md`
**Status:** concluído
**Onde vive o código:** `src/pas_intelligence/target_calculator.py` (reescrito, menor que
antes) + `api/services/predict_service.py` (`predict_strategy`, reordenado para resolver as
estatísticas antes do Estimador Auxiliar) + testes em `tests/test_pas_intelligence.py`
(`TestTargetCalculator`).

---

## 1. O que foi pedido, e o resultado

Dois bloqueadores impediam a Calculadora de Estratégia de ir ao ar: os dois `.joblib` de P1 e
Redação não carregam sob o `sklearn` atual, e a faixa `[−100, 100]` da Parte 2 era um chute sem
procedência. A saída dos dois foi por **remoção e medição**, não por consertar o que já existia:

- `p1_pas3_model.joblib` / `red_pas3_model.joblib`, `_carregar_modelo`, `model_load_error`,
  `_registrar_degradacao`, `ModelLoadError` e `PAS_STRICT_MODELS` saíram do módulo inteiro. No
  lugar entrou o **Estimador Auxiliar**: P1 e Redação da Etapa 3 saem de uma média ponderada
  (1:2) dos z-scores de Etapa 1 e 2, reconvertida para a escala da Etapa 3 — aritmética pura.
- `P2_MAXIMO`/`P2_MINIMO` passaram de `100`/`−100` (chute) para `85.6`/`0.24` (medidos, 8
  triênios de Etapa 3, ~64 mil Alunos), e o status `'impossível'` deixou de ser uma constante:
  agora é `100 − P1̂`, porque é `P1 + P2` **juntos** que não pode passar de 100.

## 2. As decisões, e o porquê de cada uma

| # | Decisão | Motivo |
|---|---|---|
| 1 | Estimador Auxiliar = média ponderada de **z-scores**, não de notas cruas | compara Etapa 1 e Etapa 2 sem herdar a dificuldade de cada prova (relatório 04 §2.1/§10) |
| 2 | Peso 1 (Etapa 1) : 2 (Etapa 2) | o mesmo peso da média ponderada que ele substitui — comportamento não muda quando as três `HistoricalStats` coincidem (é o que mantém `test_override_parcial_e_respeitado` passando sem alteração) |
| 3 | `stats_pas3` (para reconverter o z) usa a língua da **Etapa 2** | o Aluno ainda não sentou a Etapa 3; a troca de língua no PAS é de mão única (72% das trocas vão de inglesa para espanhola); documentado como assunção explícita no docstring de `predict_stable_components`, não decidido em silêncio |
| 4 | `'impossível'` = `p2_necessario > 100 − P1̂`, não uma constante | é aritmética: o EB (`P1+P2`) não pode passar de 100. Esse limite cai **abaixo** de `P2_MAXIMO` (85,6) sempre que a P1 estimada passa de 14,4 pontos — por isso a ordem dos `if` importa: `impossível` é checado antes de `improvável` |
| 5 | Bounds físicos (`P1 ∈ [−20,20]`, `Red ∈ [0,10]`) continuam no Estimador Auxiliar | mesmo com z-score, a reconversão pode estourar a escala física da prova; o ticket não pediu para tirá-los e removê-los trocaria overflow silencioso por um número impossível na tela |
| 6 | `predict_stable_components` passou a exigir `stats_pas1/2/3` | consequência direta de virar z-score: sem as três estatísticas não dá para padronizar nem reconverter. Isso empurrou uma reordenação em `predict_service.predict_strategy` (as estatísticas do triênio precisam resolver **antes** do Estimador Auxiliar, não depois) |

## 3. O que o teste prova

`test_override_parcial_e_respeitado` (regressão do defeito 7) continua passando **sem alterar
os valores esperados** — só removi duas linhas mortas (`calc.model_p1 = None`) que não faziam
mais sentido. Isso não é coincidência: o teste usa a mesma `HistoricalStats` para as três
Etapas, e sob essa condição a média ponderada de z-scores reduz algebricamente à antiga média
ponderada de notas cruas (mesma média/desvio cancelam nos dois lados da conta).

Os dois testes de fronteira mudaram, como o ticket previu:

- `test_guaranteed_scenario` — mesmo cenário e `arg_alvo`, mas agora comparado contra
  `P2_MINIMO = 0.24` em vez de `−100`. Continua `'garantido'`.
- `test_alvo_baixo_mas_dentro_da_faixa_ainda_e_possivel_nao_garantido` — **mudou de cenário**,
  não só de valor esperado. Com a faixa antiga, `arg_alvo=-100` dava `p2≈-99,4`, ainda dentro do
  piso antigo (`-100`), logo `'possivel'`. Com o piso medido (`0.24`), esse mesmo cenário passou
  a cair em `'garantido'` — exatamente a correção de comunicação que o ticket documenta (a
  mensagem feia "*você precisa de -99.4 pts*" deixa de aparecer para esse Aluno, porque ele para
  de cair no ramo `'possivel'`). O teste foi reescrito com um novo `arg_alvo=150.0`, escolhido
  para deixar `p2_necessario ≈ 1.2` — dentro da faixa medida, ainda `'possivel'` — com a conta
  documentada no docstring do teste.

Dois testes novos (`test_estimador_auxiliar_pondera_z_scores_1_para_2`,
`test_estimador_auxiliar_pesa_etapa_2_o_dobro_da_etapa_1`) cobrem o comportamento novo
diretamente: reconversão de escala com estatísticas diferentes por Etapa, e a proporção 1:2 do
peso. Um teste negativo (`test_calculadora_nao_usa_joblib`) prova a ausência — `model_p1`,
`model_red`, `model_load_error`, `ModelLoadError` e o módulo `joblib` não existem mais no
módulo.

Cinco testes que só existiam para cobrir o carregamento de `.joblib`
(`test_ml_model_integration`, `test_degradacao_nao_e_silenciosa`,
`test_modo_estrito_derruba_em_vez_de_degradar`, `test_modo_estrito_desligado_por_padrao`) saíram
— a superfície que testavam não existe mais.

## 4. `defeitos-pendentes.md`

Defeito 1 (faixa sem procedência) e defeito 3 (`.joblib` não carrega) marcados como resolvidos —
o 1 por medição, o 3 por remoção — com referência a este relatório.

## 5. O que ficou de fora, de propósito

O modelo de correção item a item (110 itens, tipos A/B/C/D) não entra aqui — alimenta o
**Simulador de Itens**, uma tela diferente, que depende de dado que não sai em Edital.
Confundir os dois foi o que manteve a Calculadora bloqueada por engano (ver ticket).

## 6. Estado do repositório ao terminar

Havia trabalho não commitado de outro ticket (13 — língua por Etapa) já em andamento na árvore
de trabalho, tocando `api/schemas/predict.py`, `api/schemas/gestao.py`,
`api/services/gestao_service.py`, `src/pas_intelligence/model_package.py`,
`src/pas_intelligence/pas_constants.py`, `src/pas_intelligence/training_dataset.py` e dois
arquivos de teste. Esse trabalho é anterior a esta sessão, não está completo (8 testes falhando
antes de eu tocar em qualquer coisa, relacionados a `OFFICIAL_STATS` da turma viva — ticket 07 —
e não à Calculadora) e não foi tocado por este ticket, exceto onde os dois se cruzam em
`predict_service.py` (a assinatura de `entrada_de_previsao` já usava `lingua_e1`/`lingua_e2`; a
mudança deste ticket, em `predict_strategy`, é uma seção separada da função e não depende
disso). O commit deste ticket inclui só os arquivos do escopo do ticket 11; o resto permanece
não commitado, como estava.

## 7. Checklist do ticket

- [x] Nenhum `joblib.load` no caminho da Calculadora
- [x] Estimador Auxiliar por média ponderada de z-scores (1:2), reconvertida para a Etapa 3
- [x] Estatística da Parte 1 da Etapa 3 usa a língua da Etapa 2, assunção documentada no docstring
- [x] `P2_MAXIMO`/`P2_MINIMO` = faixa medida, origem na docstring
- [x] Os quatro status disparam pela faixa medida, `'impossível'` a partir de `100 − P1̂`
- [x] Mensagem de `'garantido'` sem valor negativo truncado
- [x] `test_override_parcial_e_respeitado` passa sem alteração de valores
- [x] Os dois testes de fronteira atualizados com valor novo e justificativa escrita
- [x] Defeito 3 de `defeitos-pendentes.md` marcado resolvido por remoção
- [x] `pytest tests/` — mesmos 8 falhos pré-existentes (não relacionados, ticket 07/13), 0 falhos novos
