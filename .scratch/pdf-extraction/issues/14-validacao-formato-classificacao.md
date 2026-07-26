# 14 — Validação de formato do campo de classificação

**What to build:** um valor implausível no campo de classificação (o ranking do aluno em cada
Sistema de Concorrência) passa a ser sinalizado no próprio registro — `campos_formato_invalido`
— em vez de só ser neutralizado como efeito colateral na camada de detecção de buracos.

Achado originalmente no ticket 08 (`.scratch/pdf-extraction/relatorios/
08-rodada-completa-deterministica.md`, §3.3): o campo de classificação não passa pela mesma
validação de formato exato que os 9 campos numéricos de nota (`_formato_numerico_valido`,
ticket 02) — é só `_WS.sub("", v)` seguido de `int()`. Um dígito colado (mesma classe de
corrupção do ticket 06 — ver ticket 15) produz uma posição implausível; no corpus real isso
gerou uma posição de 6 dígitos no Edital 36 (2017/2019, MEDICINA), que sem limite explodia o
CSV de saída para 6,4 GB. O ticket 08 mitigou o sintoma (limite de plausibilidade em
`_buracos_por_sistema`, `3× observado + 50`), mas não corrigiu a raiz — ambos os relatórios (08
e 10) registram isso como follow-up pendente.

**Evidência de que ainda está ativo:** comparando `notas_corte.csv` (novo) contra
`data/notas_corte_pas.csv` (antigo) em 2026-07-26, apareceu o mesmo padrão — MEDICINA, Darcy
Ribeiro, Universal, 2020/2022 → `nota_corte = 199.162,872`. O pipeline já marca esse registro
com `checksum_fecha=False`, mas o valor absurdo ainda vai para o CSV de saída.

**Blocked by:** Nenhum — pode começar imediatamente. (Nota: toca a mesma área de código do
ticket 15 — `resultado_final._montar_registro`/`_separar_registro` — sem dependência lógica
entre os dois; vale sequenciar a implementação pra evitar conflito de merge.)

**Status:** ready-for-agent

- [ ] Existe uma validação de formato para o campo de classificação, simétrica a
      `_formato_numerico_valido`, aplicada em `_montar_registro`
- [ ] Um valor de classificação com formato inválido marca `campos_formato_invalido` nesse
      registro específico
- [ ] O caso real conhecido (Edital 36, 2017/2019, MEDICINA) é reproduzido numa fixture
      sintética de teste e sai marcado como formato inválido
- [ ] O limite de plausibilidade em `_buracos_por_sistema` (ticket 08) continua funcionando sem
      regressão, como segunda camada de defesa
- [ ] Rodando sobre o corpus real, o outlier de Nota de Corte (MEDICINA, Darcy Ribeiro,
      2020/2022, corte=199.162,872) some do `notas_corte.csv` ou fica corretamente excluído/
      marcado como suspeito
