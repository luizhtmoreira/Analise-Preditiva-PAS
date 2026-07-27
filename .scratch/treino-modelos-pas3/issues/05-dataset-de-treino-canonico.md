# 05 — Dataset de treino canônico

**Type:** task
**Status:** open
**Blocked by:** 01, 02, 14

## Question

Produzir **o** dataset de treino a partir do `resultado_final.csv` — um artefato único,
reproduzível e contado — para que todo ticket seguinte meça sobre a mesma coisa.

Hoje os modelos foram treinados no `data/banco_alunos_pas_final.csv`, uma base ad-hoc sem
proveniência. A base nova tem 66.313 registros com origem rastreável (arquivo, Edital, página) e
flags de qualidade. Trocar a base é metade do valor deste mapa; mas só depois que 01 e 02
disserem quais linhas são confiáveis e se os triênios antigos são o mesmo regime.

**O que este ticket resolve, além de filtrar linha:**

- **Duplicata de inscrição.** O mapa `pdf-extraction` já achou 10 inscrições com nome divergente
  entre Editais. Um aluno que aparece em mais de um triênio (repetente) é vazamento direto entre
  treino e teste. Medir a proporção; se for material, escalar para o mapa.
- **Escala e unidade.** `eb_p2` chega a ~44 e `red` a ~9,4 nas amostras; `argumento_final` vai a
  negativo (`-74.793`). Confirmar que as escalas são comparáveis entre triênios — uma mudança de
  escala entre anos seria quebra de regime tão real quanto mudança de fórmula.
- **A língua estrangeira.** O `pas_constants.py` foi mudado no ticket 12 do `pdf-extraction`
  para ter média/desvio de Parte 1 **por língua**, porque o Edital nunca publica um valor único.
  O `resultado_final.csv` tem `lingua_e1`/`e2`/`e3`. Isso precisa estar no dataset, porque
  qualquer normalização de P1 que ignore a língua está errada por construção.
- **Privacidade.** `nome` e `inscricao` são PII. Definir se o dataset guarda um identificador
  pseudonimizado ou nenhum, e garantir que o artefato fique fora do git
  ([[project_parser_privacy]]).

- [ ] Existe um script determinístico que vai do `resultado_final.csv` ao dataset de treino,
      com semente fixa e versão registrada
- [ ] Regra de inclusão/exclusão de linha aplicada conforme a recomendação do ticket 01, com
      contagem de descarte por motivo e por triênio
- [ ] Medida a proporção de inscrições repetidas entre triênios, com veredito sobre vazamento
- [ ] Escalas conferidas entre triênios; qualquer descontinuidade documentada
- [ ] `lingua_e*` presente e utilizável
- [ ] `nome` fora do dataset; identificador pseudonimizado ou ausente, decidido e justificado
- [ ] Artefato fora do git, com o caminho e a forma de regerá-lo documentados
- [ ] Relatório em `relatorios/05-dataset-de-treino-canonico.md`
