# 10 — Família de modelo

**Type:** task
**Status:** concluído — 2026-07-28
**Blocked by:** 04, 07, 08, 09

> **Resolvido.** LightGBM único (`n_estimators=400, learning_rate=0,01, num_leaves=15`), com
> `NaN` nativo nas colunas derivadas da Etapa 1 do Aluno sem Etapa 1 em vez de zero literal.
> RMSE 5,014 em `A3` (+2,97% sobre o baseline do ticket 07), bate o Portão 1 nas três pernas. O
> **ensemble por volatilidade é aposentado** — reimplementado sobre o alvo e a régua novos, ganha
> só 0,10% do melhor componente sozinho, dentro do ruído entre dobras (±0,37). A volatilidade
> também morre como feature (−0,01%). **Dois modelos por classe foi medido e perde** para o
> modelo único com `NaN` nativo (RMSE minoritária 5,379 contra 5,158) — a dobra 1 treina o
> submodelo da minoria com só 64 exemplos, dado de menos para um modelo dedicado valer a pena.
> → [relatório](../relatorios/10-familia-de-modelo.md) ·
> [ADR-0011](../../../docs/adr/0011-lightgbm-unico-com-faltante-nativo-substitui-o-ensemble.md)

## Question

Qual modelo, com quais hiperparâmetros? Decidido por **medição** sobre o holdout do ticket 06,
contra os baselines do ticket 07 — não por herança.

**O incumbente está em julgamento.** A arquitetura atual é incomum: quatro modelos base
(LGBM, RandomForest, LinearRegression, MLP) mais um `meta_model` (RandomForestClassifier) que
escolhe o melhor por aluno, com peso guiado pela **volatilidade do aluno** (CV de
`[eb_pas1, eb_pas2]` — CV baixo tende ao linear, CV alto ao LGBM/RF, transição por sigmoide).
A intuição é boa: aluno regular é previsível por reta, aluno errático não. Mas a intuição nunca
foi testada contra a alternativa óbvia — **um único GBM bem calibrado, que já modela
interações e não-linearidade sozinho**, incluindo a interação com a própria volatilidade se ela
entrar como feature.

Se o Luiz quiser mantê-la, que seja porque ela ganhou de um GBM único no mesmo holdout.

**Candidatos a comparar:**

- GBM único (LightGBM ou similar) com tuning honesto — o padrão de mercado para tabular;
- o ensemble atual, reimplementado sobre o dataset e o split novos;
- modelo linear regularizado, como piso não-trivial (às vezes ganha, em dado com 6 features e
  relação quase-linear);
- modelo multi-saída, se o ticket 04 escolher prever as 3 notas — prever P1, P2 e Redação
  conjuntamente respeita a correlação entre elas, que três modelos independentes ignoram.

**Cuidados que separam medição de teatro:**

- **Tuning dentro do split, nunca sobre o teste.** Selecionar hiperparâmetro olhando o holdout
  final contamina o número que vai para o critério de aceite.
- **Complexidade tem custo de manutenção.** Um ensemble de 5 artefatos que ganha 0,2 de RMSE de
  um GBM único não vale o que cobra em versionamento, depuração e retreino. O critério de
  desempate é explícito: ganho material, ou o mais simples vence.
- **A volatilidade não some se o ensemble perder** — ela vira candidata a feature (ticket 09)
  em vez de mecanismo de arquitetura.

- [x] Ao menos três famílias comparadas sobre o mesmo holdout, com o mesmo conjunto de features
- [x] Tuning feito dentro do split de validação, com o procedimento registrado
- [x] O ensemble por volatilidade tem veredito explícito: mantido ou aposentado, com o número
- [x] Se aposentado, a volatilidade foi avaliada como feature
- [x] Modelo escolhido com hiperparâmetros registrados e reprodutíveis por semente
- [x] Ganho contra o baseline do ticket 07 declarado, e comparado ao critério de aceite do 06
- [x] Relatório em `relatorios/10-familia-de-modelo.md`
