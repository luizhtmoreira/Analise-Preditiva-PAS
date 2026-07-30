# Pedido ao GitHub Support — GC de commits órfãos com PII

**Como enviar:** https://support.github.com/contact → categoria *Privacy / Data removal*
(ou *Account & profile → Other*). Precisa estar logado como `luizhtmoreira`.

**Contexto pra você, antes de colar:** o force-push de hoje tirou os commits das refs, mas o
GitHub não coleta objeto inalcançável na hora — em repo público ele continua servível por SHA.
Verificado hoje (2026-07-30): `GET /repos/luizhtmoreira/Analise-Preditiva-PAS/contents/docs/notas/calibracao-modelo-arg-final.md?ref=2690f0c…`
devolveu os 5344 bytes com os 6 nomes dentro. Só o Support faz o GC.

**Sobre as quatro rodadas de julho:** não dá pra listar os SHAs. O `git filter-repo` reescreveu
também as tags de backup (`backup-antes-scrub-pii-2`, `backup-antes-scrub-ip-2`,
`backup-antes-remove-app-assets`), então elas estão limpas e os commits originais sujos não
existem mais em nenhuma ref local — os SHAs são irrecuperáveis deste lado. Por isso o pedido
abaixo cita os três SHAs de hoje **e** pede GC de tudo inalcançável, cobrindo julho sem depender
de enumeração.

---

## Texto para colar

**Subject:** Request to garbage-collect unreachable commits containing personal data — luizhtmoreira/Analise-Preditiva-PAS

Hello,

I need unreachable (orphaned) commits permanently removed from a public repository I own, because
they contain personal data about real people and are still being served by the API after a
force-push.

**Repository:** `luizhtmoreira/Analise-Preditiva-PAS` (public, 0 forks)

**What the data is.** Full legal names of six real secondary-school students, each paired with a
computed university-admission probability. They are minors' or young adults' identifiable academic
records, collected under a privacy commitment that they would never be published. They were
committed by mistake.

**Where it is.** The following commits are no longer referenced by any branch or tag — I rewrote
the branch `feat/proof-section` and force-pushed it on 2026-07-30 — but they remain retrievable by
SHA:

- `2690f0c9ac5e64afc09e2e2691e587b096c01e97`
  — `docs/notas/calibracao-modelo-arg-final.md` and
    `landing-page/components/public/landing/LandingPage.tsx`
- `12898405d4857b1425b919534cec8952725c1605`
  — `landing-page/components/public/landing/LandingPage.tsx`
- `020ac687cfb8d8591a52376296087afa831440ed`
  — `landing-page/components/public/landing/LandingPage.tsx`

I confirmed on 2026-07-30 that both the commit metadata and the file contents are still downloadable
at those SHAs via the REST API, e.g.
`GET /repos/luizhtmoreira/Analise-Preditiva-PAS/contents/docs/notas/calibracao-modelo-arg-final.md?ref=2690f0c9ac5e64afc09e2e2691e587b096c01e97`
returns the file (5344 bytes) with the six names in it.

**What I'm asking for:**

1. Garbage-collect the unreachable objects in this repository so those commits, trees and blobs
   stop being served, and
2. purge any cached views or references to them (API responses, diff/blob views, search indexes,
   and the events/activity feed if the pushes appear there).

**Please also GC any other unreachable objects in this repository, not only the three SHAs above.**
Between 2026-07-24 and 2026-07-25 I ran four earlier history rewrites on this same repository to
remove other personal data and proprietary files (student names and enrolment numbers in Python
scripts, trained model binaries, PDF-parsing notebooks). Those rewrites were also followed by
force-pushes. Because the rewrites also rewrote my local backup tags, the original commit SHAs from
those rounds no longer exist on my side and I cannot enumerate them for you — so a full GC of
everything currently unreachable is what I need, rather than a SHA-by-SHA removal.

There are no forks of this repository (`forks_count: 0`), and I am the only person with a clone, so
no other copy should need to be reconciled.

I can verify from my side once the GC has run. Thank you.

---

## Depois que o Support confirmar

- [ ] Re-testar os três SHAs — os dois endpoints (`/commits/{sha}` e `/contents/{path}?ref={sha}`)
      devem dar 404
- [ ] `git tag -d backup/proof-section-pre-purge` — é a última cópia local dos nomes
- [ ] Avaliar deletar `backup-antes-scrub-pii`, `backup-antes-scrub-pii-2`,
      `backup-antes-scrub-ip-2`, `backup-antes-remove-app-assets` (já verificadas limpas hoje,
      mas não servem mais como backup de nada)
- [ ] Registrar o resultado no defeito 8 de
      `.scratch/treino-modelos-pas3/relatorios/defeitos-pendentes.md` e na watch-list de privacidade
