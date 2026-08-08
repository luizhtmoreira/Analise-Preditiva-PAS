# Relatório — Ticket 08d: Boot Frio medido, e o frontend que sabe esperar

**Ticket:** `.scratch/publicar-site/issues/08d-boot-frio-medido-e-o-frontend-que-sabe-esperar.md`
**Status:** concluído — não houve estouro, o plano gratuito segue viável
**Data:** 2026-08-08

---

## 1. O número medido, e como foi medido

**Boot Frio real: 32,4 s.**

```
curl -s -o /tmp/health_cold.json -w "time_total=%{time_total}s" https://api.vetorpas.com.br/health
→ http_code=200 time_total=32.410461s time_starttransfer=32.410276s
```

Não é estimativa: a API ficou **16 minutos** sem receber nenhuma chamada antes da medição (acima
do limiar de 15 min ociosos do Render), a mesma requisição foi refeita **duas vezes** em duas
sessões de espera diferentes para não depender de uma amostra só, e a segunda chamada imediata
depois da primeira voltou em 0,31 s — confirmando que a hibernação, e não uma lentidão qualquer, é
o que produz o número.

O `/health` é o endpoint certo para medir isso: ele é propositalmente barato (ADR-0014), então o
tempo que ele leva é quase todo o boot do processo — import de `pandas + scipy + lightgbm` com 0,1
vCPU — não trabalho da rota em si.

## 2. O timeout da Vercel, conferido, não lembrado de memória

Plano Hobby, **Function Max Duration = 300 s, fixo** (não é configurável nem para cima nem para
baixo pelo dashboard neste plano) — confirmado na documentação oficial da Vercel, buscada ao vivo
durante este ticket, não de memória de treino. Não há `vercel.json` neste repo e nenhum
`maxDuration` exportado em nenhuma rota — o projeto usa o default do plano.

**32,4 s cabe em 300 s com quase 10× de folga.** Mesmo que uma amostra futura saia bem pior que a
medida — o piso documentado de 30–50 s de terceiros já avisava que isso é plausível — a margem
absorve. Não há gatilho aqui para antecipar o Starter de $7/mês (ADR-0014, camada 3); a nota foi
registrada no ADR mesmo assim, porque ele deixava a pergunta em aberto e agora está fechada.

## 3. Client-side: Preditor e Calculadora ganharam um estado de "acordando"

Os dois formulários (`PreditorPage.tsx`, `CalculadoraPage.tsx`) já tinham `loading`/`error`, mas o
rótulo do botão ficava preso em "Calculando..." por até 32 s sem diferenciar "processando rápido"
de "o serviço está de pé de novo". Um hook pequeno e compartilhado
(`landing-page/lib/useWakingUp.ts`) troca o rótulo depois de 4 s parado em `loading`:

```
"Calculando previsão…"                          → normal
"Acordando o serviço (pode levar até 40s)…"      → passado o limiar de 4s
```

O padrão de UI já existente nessas duas telas é troca de texto no botão, sem spinner — a mudança
seguiu o mesmo padrão em vez de introduzir um elemento novo.

## 4. Server-side: verificado contra a API de verdade hibernada, com uma ressalva registrada

`/temporal` foi testado em produção (`https://vetorpas.com.br/temporal`) com a API real
hibernada — resultado: **200, 0,48 s, sem o card de erro.** Mas essa velocidade denuncia o próprio
teste: a página usa `revalidate: 3600` (ISR), então a resposta veio do cache estático da última
geração bem-sucedida, não de uma chamada bloqueante à API fria. **Isso não invalida o critério —
prova algo melhor:** se a regeneração em segundo plano falhar por Boot Frio, o Next.js serve a
página estática anterior em vez de erro. `/temporal` está protegido em duas camadas, não numa.

O que este teste **não** provou diretamente é o caminho de uma Function da Vercel chamando a API
fria de forma síncrona e bloqueante — o padrão usado por `fetchGestao`, `fetchEscola`,
`compareGroups` (`cache: "no-store"`, sem ISR). Essas rotas ficaram **sem teste ao vivo com a API
fria**, e não por escolha de escopo: elas exigem estado autenticado e dados de alunos já carregados
(upload feito na tela) antes de chamarem a API — não dá para provocar isso de linha de comando, só
navegando logado com uma base carregada. Um terceiro ciclo de 16 minutos ocioso não resolveria
isso sozinho.

O que sustenta a confiança de que elas também não quebram **não é observação, é inferência do
mesmo mecanismo já medido**: mesmo `fetch()` sem timeout próprio, mesma Function com 300 s de teto,
mesma API que levou 32,4 s fria. É uma inferência forte — mas é inferência, não o critério de
aceite ("verificado... não simulado") ao pé da letra. **Registro como lacuna aberta, não como
verificado:** se o dono do produto quiser fechar essa lacuna com certeza, o caminho é logar na
Gestão em produção depois de um período ocioso real e observar; não fiz isso aqui porque exige
sessão autenticada e dados de alunos, e a tela é B2B (fora do que este round de tickets publica). Não
tratar isso como decidido — é um "avaliar depois", não um "resolvido".

## 5. Decisão

**Não houve estouro. O plano gratuito segue viável tal como está — nenhuma correção de timeout,
nenhum contorno, nenhuma antecipação do Starter.** A única mudança de código foi dar sinal visual
ao Aluno durante os até ~32 s de espera nas duas telas client-side; o lado server-side não precisou
de mudança porque já está dentro do orçamento de tempo com folga grande, e o `/temporal` tem uma
segunda camada de proteção por ISR que não dependeu deste ticket para existir.

## 6. Arquivos alterados

- `landing-page/lib/useWakingUp.ts` — novo, hook compartilhado do estado "acordando"
- `landing-page/components/public/predict/PreditorPage.tsx` — usa o hook, rótulo do botão
- `landing-page/components/public/calculadora/CalculadoraPage.tsx` — usa o hook, rótulo do botão
- `docs/adr/0014-api-no-render-com-derivado-sem-pii-e-repo-de-deploy.md` — nota resolvendo o risco
  em aberto

## 7. Glossário desta rodada

- **Function Max Duration:** o teto de tempo que a Vercel deixa uma Function rodar antes de
  encerrar a requisição. No Hobby é 300 s, fixo — diferente do Render, não é algo que se configura
  para cima ou para baixo neste plano.
- **ISR (`revalidate`):** o Next.js gera a página uma vez e serve essa cópia por N segundos; passado
  esse tempo, a próxima visita dispara uma regeneração em segundo plano — se ela falhar, a cópia
  antiga continua sendo servida, não um erro.
