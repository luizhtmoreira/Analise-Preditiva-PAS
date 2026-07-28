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

- [x] Medidos ao menos três baselines triviais sobre o holdout do ticket 06 — **sete**, o melhor
      com RMSE **5,167** em `A3`
- [x] Medidos os `.joblib` atuais sobre o mesmo holdout
- [x] Quantificada a sobreposição entre `banco_alunos_pas_final.csv` e o holdout — **95,2%**
      (36.034 de 37.844); tudo reportado nos dois recortes, com as **1.810 linhas limpas** ao lado
- [x] Veredito preliminar: **não** — e o ensemble por volatilidade nem está em uso; o arranjo real
      é o meta-modelo roteador, que manda 75% dos Alunos para o modelo que memorizou
- [x] Registrada a proveniência do `RMSE = 13.49` — é um **MAE**, de `calculate.py:81`, medido no
      triênio 2023/2025 sobre a base de treino do próprio modelo. O RMSE real é **16,26**
- [x] Uma tabela de referência única, citável por todos os tickets seguintes — §1 do relatório
- [x] Relatório em `relatorios/07-baseline-honesto.md`

**Fechado junto:** a faixa de decisão foi congelada em **15,500**, e a limitação de cobertura que
o §10 do ticket 06 deixou em aberto foi resolvida — os 34% eram artefato do filtro `sistema == 1`,
não defeito de casamento de curso; a cobertura real é 90,0%.
