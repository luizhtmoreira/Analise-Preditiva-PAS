# 09 — Conjunto de features: o que o modelo pode ver, e o que o aluno consegue informar

**Type:** task
**Status:** open
**Blocked by:** 06, 07

## Question

Quais colunas entram como feature? A base nova traz muito mais do que as **6 features atuais**
(`[eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red]`), e cada candidata tem um custo que não é
estatístico.

**Candidatas disponíveis no `resultado_final.csv`:**

| Candidata | Por que pode ajudar | O custo |
|---|---|---|
| `curso` | trajetórias diferem muito entre MEDICINA e PEDAGOGIA | ~100 categorias; e o aluno do app pode ainda não ter escolhido curso |
| `campus` / `turno` | proxies grosseiros de perfil | idem, mais fracos |
| `lingua_e1/e2/e3` | a Parte 1 tem média e desvio **oficialmente distintos por língua** — ignorar isso é erro conhecido | o aluno sabe informar; barato |
| `perfil_cota` e as 5 booleanas | perfil socioeconômico correlaciona com trajetória | dado sensível; pedir isso ao aluno tem custo de produto e ético |
| `trienio` | captura o efeito de ano | não existe para o aluno futuro — só como efeito temporal, nunca como categoria direta |
| derivadas de trajetória | delta e aceleração entre Etapas 1 e 2 | grátis, já derivável das 6 atuais |

**A restrição que manda:** uma feature só serve se o aluno da escola parceira conseguir
fornecê-la no app **no momento da previsão**. Uma feature que melhora o holdout mas não existe
em produção é uma melhoria fantasma. Este ticket precisa medir o ganho *e* confirmar a
disponibilidade com o produto — se a feature exige mudar a tela de entrada, o ganho tem que
pagar por isso.

**Armadilha específica do `trienio`:** ele é o mais tentador e o mais perigoso. Como feature
categórica, o modelo aprende o ano e não generaliza para o ano seguinte — que é o único ano que
importa. Se entrar, entra como variável temporal contínua ou como ponderação (ticket 08), nunca
como categoria.

**Armadilha do `curso`:** ele pode estar codificando a Nota de Corte por vias tortas, o que
aproxima o modelo do alvo por um caminho que não é aprendizado sobre o aluno. Verificar antes
de aceitar o ganho.

- [ ] Ganho de cada bloco de feature medido isoladamente sobre o holdout do ticket 06, contra a
      linha de base de 6 features do ticket 07
- [ ] Disponibilidade de cada feature vencedora confirmada com o produto — o aluno consegue
      informá-la hoje, ou o que precisa mudar
- [ ] `lingua_e*` avaliada especificamente contra a normalização de P1 por língua do
      `pas_constants.py`
- [ ] `trienio` tratado como efeito temporal ou ponderação, nunca como categoria
- [ ] Verificado se `curso` está atuando como proxy da Nota de Corte
- [ ] Decisão sobre features de cota tomada explicitamente, incluindo a dimensão ética de pedir
      esse dado ao aluno
- [ ] Conjunto final de features declarado, com o ganho de cada uma
- [ ] Relatório em `relatorios/09-conjunto-de-features.md`
