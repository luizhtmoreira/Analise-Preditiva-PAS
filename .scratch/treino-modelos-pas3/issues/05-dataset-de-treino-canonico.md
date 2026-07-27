# 05 — Dataset de treino canônico

**Type:** task
**Status:** concluído — 2026-07-27
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

- [x] Existe um script determinístico que vai do `resultado_final.csv` ao dataset de treino,
      com semente fixa e versão registrada — `scripts/build_training_dataset.py`; sem passo
      aleatório, então não há semente a fixar (filtro + aritmética pura)
- [x] Regra de inclusão/exclusão de linha aplicada conforme a recomendação do ticket 01, com
      contagem de descarte por motivo e por triênio — `checksum_fecha == True`, 64.298/66.313
      linhas; `scripts/build_training_dataset.py` imprime a contagem de descarte por triênio a
      cada rodada, conferida bit a bit contra a tabela §7.2 do relatório 01 (motivo — Pop. A/B —
      permanece só no relatório 01, para não duplicar aquela medição)
- [x] Medida a proporção de inscrições repetidas entre triênios, com veredito sobre vazamento —
      144 alunos / 296 linhas (0,46%) no dataset final; flag `inscricao_repetida_entre_trienios`
      entregue ao ticket 06, sem ticket próprio (proporção pequena demais)
- [x] Escalas conferidas entre triênios; qualquer descontinuidade documentada — `A3` estável
      (média ~0, desvio ~9,1) nos 8 triênios; nenhuma descontinuidade
- [x] `lingua_e*` presente e utilizável — usada para calcular `A1`/`A2`/`A3`, presente no
      dataset sem transformação
- [x] `nome` fora do dataset; identificador pseudonimizado ou ausente, decidido e justificado —
      `id_pseudonimo` = SHA-256(inscricao) sem sal, truncado; `inscricao` nunca sai do build
- [x] Artefato fora do git, com o caminho e a forma de regerá-lo documentados —
      `data/training/pas3_dataset.parquet` (`data/` já no `.gitignore`);
      `python scripts/build_training_dataset.py` regenera
- [x] Relatório em `relatorios/05-dataset-de-treino-canonico.md`
