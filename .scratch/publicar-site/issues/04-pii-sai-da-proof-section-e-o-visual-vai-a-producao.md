# 04 — PII sai da `feat/proof-section` e o visual da landing vai a produção

**What to build:** o visual novo da landing entra em produção, e nenhum nome de Aluno real vai
junto.

A `feat/proof-section` tem 14 commits de visual da landing, já contém a `main` inteira e **não
conflita com nada** — pode ir a produção sozinha, sem esperar o resto do mapa. É o ganho mais
barato desta rodada.

**Mas ela carrega PII, e pior do que o registrado.** O commit-raiz da PII (SHA não citado aqui de propósito — ver ticket 15) cria
`docs/notas/calibracao-modelo-arg-final.md`, com uma tabela contendo o **nome completo de Alunos
reais** e a chance de aprovação calculada para cada um. O registro em `defeitos-pendentes.md`
(defeito 8) dizia que o arquivo não existia mais na árvore de trabalho — isso vale para a
`feat/pdf-extraction`. **Verificado em 2026-07-29: na `feat/proof-section` e na
`origin/feat/proof-section` o arquivo está na árvore**, não só no histórico. Mergear essa branch
como está publica os nomes na `main`.

Isso viola a única restrição do mapa marcada como **dura**: nenhum dado de Aluno vai para arquivo
commitado, relatório, teste ou exemplo. Ela sobreviveu às 4 rodadas de expurgo de 2026-07-25.

**Nada de valor se perde.** O conteúdo técnico da nota — a descoberta de que o `13,49` era um MAE
consumido como se fosse RMSE — já está preservado sem PII no §6 de
`.scratch/treino-modelos-pas3/relatorios/07-baseline-honesto.md`.

**O force-push está autorizado** (decidido pelo dono do produto em 2026-07-29): o ticket reescreve
o histórico da branch e sobrescreve o remoto. Confira antes que ninguém mais tenha a branch clonada.

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** done — ver `.scratch/publicar-site/relatorios/04-pii-sai-da-proof-section-e-o-visual-vai-a-producao.md`

- [x] `docs/notas/calibracao-modelo-arg-final.md` não existe na árvore de trabalho da
      `feat/proof-section`
- [x] O arquivo não é alcançável por nenhum commit da `feat/proof-section` nem da
      `origin/feat/proof-section` — histórico reescrito e remoto sobrescrito
- [x] Uma varredura no histórico das quatro branches (`main`, `feat/pdf-extraction`,
      `feat/proof-section`, `feat/nextjs-frontend`) não acha nenhum outro arquivo com nome de Aluno
      real, e o resultado fica registrado
- [x] A `feat/proof-section` é mergeada na `main` e o deploy da Vercel fica verde
- [x] O defeito 8 de `defeitos-pendentes.md` é marcado como corrigido, com a correção de que o
      arquivo estava na **árvore** e não só no histórico
- [ ] `[[project_parser_privacy]]` (a watch-list de privacidade) é atualizada com esta rodada —
      ver nota no relatório: o objeto ainda é servível por SHA direto no GitHub (ticket 15), então
      a memória só deve fechar o item 6/7 depois do GC confirmado
