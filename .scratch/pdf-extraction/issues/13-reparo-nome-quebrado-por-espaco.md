# 13 — Reparo de nome quebrado por espaço

**What to build:** o nome do aluno sai correto no `resultado_final.csv` (e em `convocacao.csv`,
se a mesma lógica se aplicar) mesmo quando o extrator de texto do PDF injeta um espaço espúrio
no meio de uma palavra do campo `nome`.

Achado ao comparar `resultado_final.csv` contra a base antiga `data/banco_alunos_pas_final.csv`
por `(inscrição, triênio)` em 2026-07-26 (ver `.scratch/pdf-extraction/relatorios/
defeitos-pendentes.md`, item 1): 1.796 de 66.313 registros (2,71%) têm o nome quebrado por um
espaço inserido no meio de uma palavra (ex. `"Isabella"` → `"Isabell a"`) ou um espaço duplicado
sem quebra de palavra — em 100% dos casos do lado do extrator novo, presente nos 8 de 8 Editais
de Resultado Final do corpus, não concentrado num arquivo isolado.

É a mesma classe de corrupção já catalogada em `scripts/NOTES.md` ("ARMADILHA B(c) — números
partidos por whitespace") e já reparada para os 9 campos numéricos de nota (tickets 02/04) —
o campo `nome` nunca recebeu o reparo equivalente. `convocacao.py` só colapsa espaço duplicado
(`_ESPACOS_RE`), o que não resolve a quebra de palavra.

**Cuidado central:** a heurística de reparo não pode fundir partículas curtas legítimas do
português (`de`, `da`, `do`, `dos`, `das`, `e`) que aparecem sozinhas entre nomes — essas são
nomes válidos, não um defeito de extração.

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** ready-for-agent

- [ ] Nome com uma palavra partida por um espaço espúrio (ex. `"Isabell a"` → `"Isabella"`) é
      reparado no `resultado_final.csv`
- [ ] Espaço duplicado entre palavras (sem quebra de palavra) também é normalizado
- [ ] Partículas curtas legítimas (`de`, `da`, `do`, `dos`, `das`, `e`) nunca são fundidas com a
      palavra vizinha pelo reparo
- [ ] O reparo é sinalizado por uma coluna de proveniência no CSV (não silencioso), no mesmo
      espírito de `cota_padrao_suspeito`/`checksum_fecha`
- [ ] Teste sintético cobre: quebra de palavra real, espaço duplicado sem quebra, e nome com
      partícula curta legítima que não deve ser alterado
- [ ] Rodando sobre o corpus real de 77 Editais, a taxa de nomes reparados é registrada no
      relatório de validação (baseline conhecido: ~2,71%)
- [ ] Avaliado (e decidido, com registro do porquê) se `convocacao.py` reusa a mesma lógica de
      reparo de nome
