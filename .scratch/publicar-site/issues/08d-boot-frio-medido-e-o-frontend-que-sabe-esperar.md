# 08d — Boot Frio medido, e o frontend que sabe esperar

**What to build:** o Aluno que chega primeiro depois de um período ocioso vê o produto acordando, e
não um erro.

O **Boot Frio** é o comportamento **normal** da hospedagem gratuita, não uma anomalia: acontece a
cada deploy, restart da plataforma, OOM kill e período ocioso. Nenhuma quantidade de keep-alive o
elimina — o `08e` só o torna raro (ADR-0014).

**O risco concreto, e o motivo de este ticket vir antes do keep-alive.** Metade das chamadas do
frontend é **server-side** (Gestão, Analytics, a série temporal) e passa por uma Function da Vercel,
que tem timeout próprio no plano Hobby. Se o Boot Frio exceder esse timeout, essas páginas **quebram
com erro**, não ficam lentas — e só em produção, só depois de ociosidade. As chamadas client-side
(Preditor, Calculadora) saem do navegador direto e não têm esse problema. **São dois
comportamentos diferentes e precisam de dois tratamentos.**

**O que se sabe antes de medir:** a API entra com **297 MB de 512 MB** do teto gratuito em repouso
(213 MB só de importar `pandas + scipy + lightgbm`; os 4,5 MB do Derivado economizam imagem e
download, **não RAM**), com **0,1 vCPU**. Os 30–50 s de spin-up que terceiros medem em apps comuns
são o piso, não a estimativa — este app importa bibliotecas científicas com um décimo de núcleo.

**Este ticket pode devolver uma decisão, e isso não é falha dele.** Se o Boot Frio medido estourar
o timeout da Function e não houver configuração que resolva, a saída é antecipar a camada 3 do
ADR-0014 — o Starter de $7/mês, sem spin-down e com 0,5 vCPU. Nesse caso **o ADR ganha uma nota e o
dono do produto decide**; não trate como bug de frontend nem contorne com gambiarra de timeout.

## Nota para quem implementar

**Este ticket é medição e julgamento, não código — e é o único da série que pode derrubar uma
decisão.**

Se o Boot Frio medido estourar o timeout da Function da Vercel, a resposta certa é **escalar**: nota
no ADR-0014 e decisão do dono do produto sobre antecipar o Starter de $7/mês. Não é consertar.

O modo de falha esperado aqui é atraente e silencioso: aumentar um timeout, embrulhar a chamada num
retry, mover a chamada para client-side só para fugir do limite — e entregar tudo verde, tendo
escondido que o plano gratuito não serve para este produto. **Entregar verde escondendo isso é pior
que entregar vermelho**, porque a descoberta vem depois, com Alunos na frente.

Um número medido que contraria o plano é **entrega bem-sucedida deste ticket**, não fracasso dele.

**Blocked by:** 08c (serviço no ar).

**Status:** concluído para Preditor/Calculadora/série temporal pública; **Gestão/Escola/Comparação
seguem sem verificação ao vivo** (item aberto abaixo). Relatório em
`.scratch/publicar-site/relatorios/08d-boot-frio-medido-e-o-frontend-que-sabe-esperar.md`

- [x] O Boot Frio real está **medido** contra o serviço no ar, com o número registrado — não
      estimado a partir de benchmark de terceiro (**32,4 s**, API real hibernada 16 min antes)
- [x] O timeout das Functions da Vercel neste plano está **conferido** e comparado com esse número
      (Hobby: 300 s fixo, ~10× de folga — confirmado na documentação oficial, não de memória)
- [x] As chamadas client-side mostram estado de "acordando" na primeira requisição, em vez de
      travar sem sinal (`useWakingUp`, Preditor e Calculadora)
- [x] `/temporal` (série temporal pública) verificado com a API efetivamente hibernada — 200 em
      produção, sem card de erro — **mas** com ressalva registrada: a resposta veio do cache ISR
      (`revalidate: 3600`), não de uma chamada bloqueante à API fria; ver relatório §4
- [ ] **Gestão, Escola, Comparação — sem teste ao vivo.** Essas telas exigem sessão autenticada e
      dados de alunos carregados antes de chamarem a API; não dá para provocar isso de linha de
      comando. A confiança de que não quebram vem de inferência do mesmo mecanismo medido
      (`fetch()` sem timeout, Function de 300 s, API que leva 32,4 s fria) — não de observação ao
      vivo. Registrado como lacuna aberta no relatório §4, não como resolvido
- [x] Se o número medido inviabilizar o plano gratuito, isso vira nota no ADR-0014 — **não
      inviabilizou**; nota registrada mesmo assim, fechando a pergunta que o ADR deixava em aberto
