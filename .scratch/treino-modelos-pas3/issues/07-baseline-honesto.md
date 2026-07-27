# 07 — Baseline honesto e medição do ensemble atual

**Type:** task
**Status:** open
**Blocked by:** 05, 06

## Question

Quanto erra um modelo **burro**, e quanto erra o **ensemble atual**, sobre o mesmo holdout?

Nenhum dos dois números existe hoje. Sem eles, qualquer resultado dos tickets 08–10 é
incomparável: um RMSE de 12 pode ser excelente ou constrangedor dependendo de onde está o piso.

**Os baselines burros a medir** (todos triviais, todos possivelmente difíceis de bater):

- média dos EB das Etapas 1 e 2 do próprio aluno, projetada para a Etapa 3;
- a Etapa 2 sozinha, repetida (a mais recente é frequentemente a melhor previsora);
- regressão linear simples sobre as 6 features atuais;
- média do curso/turma, ignorando o aluno.

**A medição que importa mais:** rodar os `.joblib` **atuais** (`modelo_lgbm`, `modelo_rf`,
`modelo_linear`, `modelo_mlp`, `meta_model`, `modelo_arg_final`) sobre o holdout do ticket 06 e
registrar o erro. Isso responde uma pergunta que ninguém no projeto consegue responder hoje:
*o ensemble por volatilidade se justifica?* O `RMSE = 13.49` cravado em `statistics.py` não tem
proveniência registrada — não se sabe sobre qual dado ou qual split foi medido.

**Cuidado metodológico:** os modelos atuais foram treinados no
`data/banco_alunos_pas_final.csv`, que provavelmente **contém** boa parte dos alunos do holdout
novo. Medi-los sobre esse holdout é medir dado que eles já viram — o número sai bom por
vazamento, não por qualidade. O ticket precisa quantificar a sobreposição entre as duas bases e
reportar o número com essa ressalva explícita, ou construir um holdout limpo da sobreposição.

- [ ] Medidos ao menos três baselines triviais sobre o holdout do ticket 06
- [ ] Medidos os `.joblib` atuais sobre o mesmo holdout
- [ ] Quantificada a sobreposição entre `banco_alunos_pas_final.csv` e o holdout; número dos
      modelos atuais reportado com a ressalva de vazamento, ou medido sobre subconjunto limpo
- [ ] Veredito preliminar: o ensemble por CV de volatilidade bate o baseline trivial?
- [ ] Registrada a proveniência do `RMSE = 13.49` — de onde veio, e o quanto difere do medido
- [ ] Uma tabela de referência única, citável por todos os tickets seguintes
- [ ] Relatório em `relatorios/07-baseline-honesto.md`
