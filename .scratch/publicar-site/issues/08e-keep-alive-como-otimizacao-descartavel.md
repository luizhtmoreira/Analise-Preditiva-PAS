# 08e — Keep-alive, como otimização descartável

**What to build:** o Boot Frio deixa de acontecer na maior parte do tempo, sem que nada no produto
passe a **depender** disso.

Um ping externo a cada poucos minutos mantém a instância acordada, e a aritmética fecha: ~730 horas
de uso contra as 750 horas mensais do plano gratuito.

**Isto é workaround, não contrato.** O Render **não suporta oficialmente** manter serviço gratuito
acordado — a posição deles é "migre para plano pago". Funciona na prática e muita gente roda assim,
mas é exatamente a mesma classe de coisa que matou o ADR-0004: uma política externa não escrita, que
vale até parar de valer sem aviso. Por isso ele entra **depois** do `08d` e não antes: se o keep-alive
sumir amanhã, o produto degrada para o comportamento do `08d` — não quebra.

**E por isso a ordem importa também na outra direção:** ligar o keep-alive antes do `08d` esconderia
o fenômeno que o `08d` precisa observar. Não ligue nada aqui até o Boot Frio estar medido.

**O `/health` precisa ser barato de propósito.** Ele passa a ser batido a cada poucos minutos, para
sempre, nos seus 0,1 vCPU. Se ele tocar no pacote de modelo ou nos CSVs, você paga CPU de graça o
mês inteiro — e num plano onde CPU é o recurso escasso.

**Cuidado com a cota:** sempre-acordado consome ~730 das 750 horas/mês do **workspace**. Isso
confirma o limite do `08c` — um único serviço gratuito, e um segundo estoura tudo até virar o mês.

**Blocked by:** 08d (Boot Frio medido e o frontend que sabe esperar).

**Status:** ready-for-agent

- [x] O `/health` responde sem tocar no pacote de modelo nem nos CSVs — verificado, não presumido
      (código lido: `api/main.py`, é `{"status": "ok"}` puro; carregar recursos é `lifespan`, roda
      uma vez no boot do processo, não por requisição)
- [x] O ping externo está de pé e o serviço permanece acordado ao longo de um período ocioso real —
      verificado com o serviço em produção, não simulado. Primeira tentativa (GitHub Actions) **falhou
      na prática e foi medido falhando** (gaps de 50-90 min contra os 10 declarados); a correção foi
      cron-job.org, verificado depois com duas janelas de ~25 min sem nenhuma chamada minha:
      `/health` em 1,25 s e 0,35 s, nunca subiu para a casa dos 30 s do Boot Frio
- [x] Desligar o ping **não quebra** o produto: ele volta ao comportamento entregue no `08d` (nada no
      código depende do ping — é infraestrutura externa ao produto)
- [x] O consumo mensal de horas está estimado contra as 750 do plano, com a folga registrada
      (ADR-0014: 744 h num mês de 31 dias, 6 h de folga — a conta de ~730 h citada antes era
      otimista, corrigida)
- [x] Está escrito, onde quem for mexer vai ler, que o keep-alive é otimização descartável e não
      pode virar dependência (ADR-0014 e comentário no topo de `.github/workflows/keep-alive.yml`)
