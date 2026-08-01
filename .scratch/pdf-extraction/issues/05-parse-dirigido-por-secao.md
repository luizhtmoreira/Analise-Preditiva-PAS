# 05 — Parse dirigido por seção

**What to build:** dos Editais que contêm duas listas com schemas diferentes no mesmo arquivo,
apenas a seção de não eliminados é extraída.

Os Editais de *resultado final tipo D + redação* têm duas seções. Medido no Ed_27 (2021/2023,
317 páginas): páginas 0–98 trazem registros de 4 campos, e a partir da página 99 registros de 22
campos. Um parser que assumisse um schema por arquivo produziria lixo em metade do documento
**sem levantar erro** — por isso a unidade de parse passa a ser a seção, não o arquivo.

A transição é detectada pelo cabeçalho numerado do próprio documento — `"2 DO RESULTADO FINAL DOS
CANDIDATOS NÃO ELIMINADOS"` — e não por número de página fixo, para que funcione em qualquer
Edital da família e não só no Ed_27.

Custo aceito conscientemente: perdem-se ~1.449 Alunos eliminados por Edital. Eles só têm 2 notas
e não formam o vetor de 9, então não servem para o treino dos modelos. A seção de itens do tipo D
e redação está fora de escopo por decisão registrada.

**Blocked by:** 01 — Costura `extrair_edital` + classificador de família + CSV de Resultado Final.

**Status:** ready-for-agent

- [ ] A transição entre seções é detectada pelo cabeçalho numerado, não por número de página
- [ ] Apenas a seção de não eliminados é extraída dos Editais de duas seções
- [ ] Um Edital de seção única continua sendo extraído normalmente, sem regressão
- [ ] Existe fixture com a transição entre as duas seções, gerada localmente (não commitada, ver ticket 01)
- [ ] Um teste verifica que nenhum registro da primeira seção aparece na saída
