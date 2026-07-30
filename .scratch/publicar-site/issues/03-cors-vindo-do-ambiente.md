# 03 — CORS vindo do ambiente

**What to build:** o navegador de um Aluno em `vetorpas.com.br` consegue chamar a API. Hoje não
conseguiria.

**CORS** é a regra que diz de quais endereços o navegador pode chamar a API. Se o endereço do site
não estiver na lista que a API declara, o navegador **recusa a chamada antes de fazê-la** — e do
lado do frontend isso lê como "API indisponível", sem nada no log do servidor.

A lista atual tem três entradas: dois `localhost` e `"https://*.vercel.app"`. A terceira é um
**texto literal** — o Starlette não trata `*` como curinga nessa lista, então nenhum deploy de
preview da Vercel passa. E `vetorpas.com.br` não está lá de forma alguma.

O Preditor e a Calculadora chamam a API **do navegador**, então isso falha na primeira requisição
em produção, para todo mundo.

Depois deste ticket, as origens vêm do ambiente em vez de estarem cravadas no código: a lista
explícita para os domínios de produção, e o mecanismo que o Starlette realmente oferece para
padrões (expressão regular) para os previews da Vercel. DEV e PROD deixam de ser a mesma lista
editada à mão.

**Este ticket não sobe nada** — ele conserta a regra. O deploy é o ticket 08, que verifica isso num
navegador de verdade, que é o único lugar onde CORS falha de verdade.

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** ready-for-agent

- [ ] As origens permitidas vêm de variável de ambiente, com um padrão seguro para DEV
      (`localhost`) quando ela não está definida
- [ ] `https://vetorpas.com.br` e `https://www.vetorpas.com.br` são aceitos em PROD
- [ ] Um deploy de preview da Vercel é aceito por expressão regular, não por texto literal com `*`
- [ ] Uma origem qualquer não listada é **recusada**
- [ ] Teste cobrindo os três casos (produção aceito, preview aceito, desconhecido recusado) —
      novo, sem prior art no repo
- [ ] `pytest tests/` continua verde
