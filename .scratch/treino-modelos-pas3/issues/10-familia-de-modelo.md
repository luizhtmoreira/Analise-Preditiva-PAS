# 10 — Família de modelo

**Type:** task
**Status:** open
**Blocked by:** 04, 07, 08, 09

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

- [ ] Ao menos três famílias comparadas sobre o mesmo holdout, com o mesmo conjunto de features
- [ ] Tuning feito dentro do split de validação, com o procedimento registrado
- [ ] O ensemble por volatilidade tem veredito explícito: mantido ou aposentado, com o número
- [ ] Se aposentado, a volatilidade foi avaliada como feature
- [ ] Modelo escolhido com hiperparâmetros registrados e reprodutíveis por semente
- [ ] Ganho contra o baseline do ticket 07 declarado, e comparado ao critério de aceite do 06
- [ ] Relatório em `relatorios/10-familia-de-modelo.md`
