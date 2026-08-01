# 15 — GC de objetos órfãos com PII no GitHub

**What to build:** os 6 nomes reais de Aluno que ficaram órfãos pelo force-push do ticket 04 param
de ser servíveis por qualquer meio no `luizhtmoreira/Analise-Preditiva-PAS`.

**Por que isto é um ticket à parte.** O ticket 04 fechou o lado que um `git push` consegue
resolver: nenhuma ref (branch ou tag) alcança mais o commit com PII. Mas GitHub **não coleta
objeto inalcançável** em repositório público — ele continua servível por SHA direto, e isso foi
verificado depois do force-push:

```
GET /repos/luizhtmoreira/Analise-Preditiva-PAS/commits/2690f0c…                     → 200
GET /…/contents/docs/notas/calibracao-modelo-arg-final.md?ref=2690f0c…              → 200, 5344 bytes
grep dos 6 nomes no conteúdo baixado                                                 → 6 ocorrências
```

Os três SHAs sujos, todos de `feat/proof-section`: `2690f0c9ac…`, `12898405d4…`, `020ac687cf…`.

**Isto invalida a métrica com que quatro rodadas de expurgo de julho se declararam concluídas**
("0 hits em `git rev-list --all`") — ela só inspeciona objeto alcançável, e o objetivo do expurgo é
justamente tornar os ruins inalcançáveis. A verificação correta é remota e por SHA:
`gh api ".../contents/<path>?ref=<sha-órfão>"` tem que dar 404, não a ausência local.

Os SHAs sujos das quatro rodadas de julho são **irrecuperáveis** — `git filter-repo` reescreveu as
tags de backup daquelas rodadas também, então não há como enumerá-los. Por isso o pedido não pode
ser "apague estes SHAs": tem que pedir GC de todo objeto inalcançável do repositório.

**Isto não bloqueia nada do mapa.** É trabalho fora do fluxo normal de agente: só o dono da conta
`luizhtmoreira` pode abrir o chamado.

**Blocked by:** Nenhum ticket do mapa. Mas **você** é o único que pode executá-lo — nenhum agente
tem acesso à conta do GitHub Support.

**Status:** ready-for-you — o rascunho já está pronto, falta só enviar

- [ ] O pedido em `.scratch/publicar-site/pedido-github-support-gc.md` é enviado ao GitHub Support
      (`support.github.com/contact` → *Privacy / Data removal*, logado como `luizhtmoreira`)
- [ ] O Support confirma o GC dos objetos inalcançáveis
- [ ] Reverificação por SHA: os três `GET .../contents/...?ref=<sha>` acima devolvem 404
- [ ] A tag local `backup/proof-section-pre-purge` (que ainda contém a PII) é apagada
- [ ] `[[project_parser_privacy]]` é atualizada registrando o fechamento do item 6/7 daquela nota
