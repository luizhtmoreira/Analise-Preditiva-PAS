# 17 — Fixture sintética ponta a ponta para Nota de Corte

**What to build:** um teste automatizado, executável em CI sem depender de `data/pdfs` local,
que roda o pipeline inteiro — extração → Resultado Final → Convocação → Nota de Corte — sobre
dados 100% inventados e confere que o corte final sai correto.

O ticket 10 (`.scratch/pdf-extraction/relatorios/
10-notas-de-corte-por-sistema-de-concorrencia.md`, seção de limitações conhecidas) documentou
que não existe teste ponta a ponta para Nota de Corte: exigiria duas fixtures do mesmo triênio
(uma Resultado Final, uma Convocação) com inscrições que se cruzam, e uma fixture de Edital real
carrega dado de prova de aluno identificável — proibido de commitar por
[[project_parser_privacy]]. Hoje a cobertura é 41 testes sintéticos da *regra* de derivação
(números inventados direto em Python, sem passar por PDF) mais a rodada manual sobre os 77
Editais reais, que só é reproduzível por quem tem o corpus local.

**A saída:** as duas fixtures deste ticket usam nomes e números **inteiramente inventados**
(nenhum aluno real) — só respeitando a mesma estrutura de schema declarado que os PDFs reais,
o que as torna comitáveis sem violar a política de privacidade.

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** concluído — ver
`.scratch/pdf-extraction/relatorios/17-fixture-sintetica-ponta-a-ponta-nota-de-corte.md`

- [ ] Duas fixtures sintéticas do mesmo triênio — uma Resultado Final, uma Convocação — com
      inscrições que se cruzam, dados inteiramente inventados
- [ ] As fixtures são comitáveis no repositório sem violar [[project_parser_privacy]]
      (confirmar explicitamente que não reproduzem nenhum dado de aluno real)
- [ ] O teste roda o pipeline ponta a ponta sobre essas fixtures (`extrair_edital` → Resultado
      Final + Convocação → derivação de Nota de Corte) e confere o corte esperado, calculado
      manualmente a partir dos dados inventados
- [ ] Cobre o caso de empate/múltiplos alunos do mesmo sistema na maior chamada (mesmo cenário
      testado sinteticamente no ticket 10)
- [ ] O teste roda em CI sem exigir `data/pdfs` local
- [ ] `defeitos-pendentes.md` (item 5) e a seção de limitações do relatório do ticket 10 são
      atualizados removendo essa limitação, com referência a este ticket
