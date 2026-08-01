# 02 — Validações estruturais por registro

**What to build:** cada linha do CSV passa a carregar o resultado da sua própria validação, para
que o consumidor filtre por confiança em vez de confiar cegamente.

O problema que isso resolve: a extração de PDF corrompe dados **silenciosamente**. Números
partem no meio (`56.29 1` vira 56.29), cabeçalhos de curso são engolidos dentro de registros, e
registros se colam perdendo o número de inscrição. Nada disso levanta exceção — produz um número
plausível e errado. Ninguém pode conferir 122 mil registros no olho.

Três verificações automáticas, todas dentro do CSV:

1. **Formato numérico exato.** Todo campo numérico casa `^-?\d+\.\d{3}$`. Pega número partido por
   espaço, que é invisível para qualquer parse tolerante.
2. **Classificação como sequência `1..N`** dentro de cada curso e cada Sistema de Concorrência,
   ordenada por Argumento Final decrescente. Buraco na sequência é registro que o parser perdeu, e
   se sabe exatamente qual posição faltou. Esta é a única camada que pega o ponto cego de todas as
   outras: um registro que nunca foi extraído não deixa nada para conferir.
3. **Ordem alfabética** dentro do curso. Quebra de ordem indica registros colados.

Fixa como testes de regressão os quatro casos de corrupção real já identificados no protótipo,
para que uma correção de parser não reintroduza um problema já resolvido: o número `56.29 1` que
deve virar 56.291; o cabeçalho `ENGENHARIA DE REDES DE COMUNICAÇÃO (BACHARELADO)` engolido no
meio de um registro; o par de registros colados em que o segundo perdeu o número de inscrição; o
negativo `- 58.570` com o sinal separado.

**Blocked by:** 01 — Costura `extrair_edital` + classificador de família + CSV de Resultado Final.

**Status:** ready-for-agent

- [ ] Cada linha do CSV carrega o resultado da sua própria validação
- [ ] Campo numérico que não case `^-?\d+\.\d{3}$` exatamente é sinalizado
- [ ] Buraco na sequência `1..N` de classificação é detectado por curso e por Sistema de Concorrência, indicando qual posição faltou
- [ ] Quebra de ordem alfabética dentro do curso é sinalizada
- [ ] Os quatro casos de corrupção do protótipo estão fixados como testes de regressão
- [ ] Os testes exercitam a costura `extrair_edital`, não a estrutura interna do parser
