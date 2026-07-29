# 10 — Merge do portal para cima do modelo

**What to build:** um único tronco que tem ao mesmo tempo o modelo novo e o portal — Preditor,
Calculadora de Estratégia, header público, recuperação de senha e página de perfil.

Hoje há duas realidades. A `feat/pdf-extraction` tem o modelo promovido, o pipeline de treino e a
extração. A `feat/nextjs-frontend` tem o portal, mas está **52 commits atrás da `main`**.

## A direção do merge não é simétrica

`feat/nextjs-frontend` vem para cima de `feat/pdf-extraction`, e não o contrário. A branch do portal
**deleta** `model_package.py`, `training_dataset.py`, `training_pipeline.py`, `validation.py` e
`dataset_pas3.py` — não porque discorde deles, mas porque é velha e eles não existiam. E
**ressuscita `ensemble.py`**, que o ADR-0011 aposentou (o ensemble batia seu próprio melhor
componente por 0,10%, dentro do ruído de dobra para dobra) e que, segundo o defeito 10 de
`defeitos-pendentes.md`, **nunca chegou a rodar em produção** — `predict_with_dynamic_ensemble` não
é chamado em lugar nenhum.

Se o merge for feito na direção errada, ele desfaz a rodada inteira do treino em silêncio.

## Cinco conflitos, dois de verdade

Os dois reais são o **serviço do Preditor** e a **página do Preditor**, porque os dois lados
reescreveram o Preditor por motivos diferentes:

- o lado do modelo trocou o miolo para `A3` + Largura de Incerteza vinda do manifesto;
- o lado do portal acrescentou semestre, curso alvo e persistência do Aluno logado.

**As duas mudanças são para manter** — resolver escolhendo um lado perde metade do trabalho. Os
outros três conflitos são mecânicos.

## O que a branch do portal traz e é para preservar

- `CalculadoraPage.tsx` e o endpoint `/api/predict/strategy` com schema e serviço — a base da
  Calculadora, que os tickets 11 e 12 reescrevem por dentro;
- `PublicHeader.tsx`, os fluxos de esqueci-senha / redefinir-senha, a página de perfil do Aluno;
- a reescrita da landing e as mudanças da Análise Temporal.

`simulador_itens.py` vem junto mas **fica fora desta rodada** (ver a spec, *Out of Scope*) — ele
depende de dado que só existe no caderno de questões.

**Blocked by:** 05 (`TRIENNIUM_STATS` e `STATS_PAS3_TREND` saem) — resolver o conflito do serviço do
Preditor exige saber de onde a média e o desvio vêm, e o `get_strategy_prediction` que chega no
merge consome justamente o dicionário que o 05 apaga.

**Status:** ready-for-agent

- [ ] `feat/nextjs-frontend` está integrada sobre a `feat/pdf-extraction`, sem perder nenhum dos
      módulos do pipeline de treino
- [ ] `ensemble.py` **não** volta
- [ ] O serviço e a página do Preditor têm as duas evoluções: `A3` + Largura do manifesto **e**
      semestre, curso alvo e persistência do Aluno logado
- [ ] `/api/predict/strategy`, o header público, a recuperação de senha e a página de perfil
      funcionam no tronco unificado
- [ ] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
- [ ] Um Aluno de teste passa pelo Preditor de ponta a ponta na branch unificada
