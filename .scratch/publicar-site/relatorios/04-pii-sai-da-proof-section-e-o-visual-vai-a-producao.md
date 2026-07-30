# 04 — PII sai da `feat/proof-section` e o visual vai a produção (relatório)

> Relatório escrito retroativamente. O trabalho foi feito e mergeado (`222474d`, 2026-07-30
> 00:06) em sessão anterior a esta; o issue file continuou com o checklist desmarcado e
> `Status: ready-for-agent`, o que fez uma sessão posterior (esta) concluir por engano, olhando só
> a documentação, que o ticket não tinha sido feito. Corrigido nesta sessão depois de verificar o
> estado real do repositório e do site em produção.

## O que foi verificado (2026-07-30, nesta sessão)

- **`docs/notas/calibracao-modelo-arg-final.md` não existe na árvore de `feat/proof-section`**
  (`git ls-tree -r feat/proof-section` — ausente).
- **O commit-raiz da PII não é alcançável** por nenhuma ref viva: nem `feat/proof-section`, nem
  `origin/feat/proof-section` (recém-`fetch`ado), nem `main`, nem `origin/main`, nem
  `feat/pdf-extraction`, nem `feat/nextjs-frontend`. Só aparece via `--all` porque a tag local
  `backup/proof-section-pre-purge` ainda o referencia — essa tag é o item de limpeza do ticket 15,
  não deste ticket.
- **Varredura das quatro branches** (`main`, `feat/pdf-extraction`, `feat/proof-section`,
  `feat/nextjs-frontend`) por nome de arquivo (`docs/notas`, `calibracao`, `selecao-casos`,
  `aluno`) não encontra nenhum arquivo com dado de Aluno real — só código (`AlunoLoginForm.tsx` e
  afins) e ADRs/relatórios técnicos sem PII.
- **`feat/proof-section` está mergeada na `main`**, local e remota (`origin/main` tem o mesmo
  commit de tipo `222474d` na ponta).
- **Produção está servindo a versão nova**: `curl https://vetorpas.com.br` devolve `200` e contém
  "Aprovações Reais" e os pseudônimos `Aluno A`–`E` (não nomes reais) — a seção Notas de Corte foi
  desativada como o commit `222474d` descreve.

## O que o commit `222474d` mudou

`feat(landing): add seção Aprovações Reais com resultados reais anonimizados` — carrossel entre o
Hero e as Notas de Corte, comparando o Argumento Final previsto antes da Etapa 3 com o resultado
oficial de 5 Alunos aprovados (2023/2025), identificados só como `Aluno A`–`E`. A seleção/inscrição
real fica fora do repo (`data/prova-do-modelo/selecao-casos-2023-2025.md`, gitignored). A chance é
calculada contra a Nota de Corte do triênio **anterior** (2022/2024) — a única publicada antes do
resultado do Aluno, evitando vazar informação do futuro.

## Critérios de aceite — conferidos

- [x] `docs/notas/calibracao-modelo-arg-final.md` não existe na árvore de `feat/proof-section`.
- [x] O arquivo não é alcançável por nenhum commit de `feat/proof-section` nem de
      `origin/feat/proof-section` — histórico reescrito, remoto sobrescrito (confirmado via
      `git fetch` + checagem de alcançabilidade, não só cache local).
- [x] Varredura das quatro branches sem outro arquivo com nome de Aluno real — resultado registrado
      acima.
- [x] `feat/proof-section` mergeada na `main`, e o deploy está no ar (`vetorpas.com.br` responde
      200 com o conteúdo novo).
- [x] Defeito 8 de `defeitos-pendentes.md` marcado como corrigido — atualizado nesta sessão em
      `.scratch/treino-modelos-pas3/relatorios/defeitos-pendentes.md`.
- [ ] `[[project_parser_privacy]]` — **não fechada ainda de propósito.** O objeto órfão continua
      servível por SHA direto no GitHub (GC não roda em objeto inalcançável de repositório
      público); isso é o ticket 15, ainda `ready-for-you` (só o dono da conta pode abrir o chamado
      de Support). A memória deve registrar o fechamento do lado *git* (este ticket) mas não deve
      declarar o item 6/7 da nota de privacidade encerrado até o GC ser confirmado.

## Fora do escopo deste ticket

- GC do objeto órfão no GitHub (SHA continua servível por `contents/...?ref=<sha>`) — ticket 15.
  **O SHA em si não é citado aqui de propósito** — está registrado só em `.scratch/publicar-site/
  issues/15-...md` e `pedido-github-support-gc.md`, ambos mantidos fora do repo público até o GC
  ser confirmado.
- A tag local `backup/proof-section-pre-purge` (ainda contém a PII) — apagar é item do ticket 15,
  não deste.
