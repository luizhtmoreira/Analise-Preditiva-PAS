# 14 — Publicação: a `main` no ar contra a API hospedada

**What to build:** o destino do mapa. Um Aluno abre `vetorpas.com.br`, usa o Preditor e a
Calculadora de Estratégia sem criar conta, e recebe números calculados pelo modelo novo contra Notas
de Corte novas, servidos por uma API que não está no `localhost` de ninguém.

Este ticket não constrói funcionalidade — ele **integra e verifica**. Todo o comportamento já foi
entregue pelos treze anteriores; o que falta é o tronco chegar à `main` e o conjunto funcionar
junto, em produção, contra um usuário de verdade.

## O que tem que estar de pé ao mesmo tempo

- **Preditor PAS 3** respondendo para um Aluno da Turma viva (2024-2026) — o que ele recusa hoje;
- **Calculadora de Estratégia** com o Estimador Auxiliar, a faixa medida da P2 e os cinco
  Anos-Âncora;
- **Análise Temporal** pública, com a série oficial e a evolução de Nota de Corte por curso;
- a **landing** de pé, sem regressão;
- a **API** hospedada, com o pacote de modelo dentro da imagem.

## As verificações que não acontecem em `pytest`

Estas são o miolo do ticket, porque são exatamente as que falharam por não terem sido feitas antes:

- **num navegador de verdade**, não em teste de servidor — é onde CORS falha, antes da requisição
  sair;
- **numa máquina limpa**, sem `models/` e sem `data/` — é onde a dívida do domicílio aparece;
- **com um Aluno real da Turma viva**, ponta a ponta: seis notas, duas línguas, curso alvo, e o
  número que sai.

## Uma coisa a conferir na saída, não só na entrada

O Preditor precisa deixar claro, na tela, que a previsão de um Aluno 2024-2026 se apoia em
estatística **derivada** — porque quando o Edital de verdade sair em 2026 esses números serão
substituídos e a previsão vai mexer. O ticket 07 entrega o dado e a exibição; aqui se confere que
sobreviveram ao merge e ao deploy.

**Blocked by:** 04 (PII fora e visual em produção), 07 (Preditor responde para a Turma viva), 08
(API hospedada), 09 (CSVs novos), 12 (Ano-Âncora) e 13 (Língua por Etapa).

**Status:** ready-for-agent

- [ ] A `main` contém Preditor, Calculadora, Análise Temporal e landing, e o deploy da Vercel está
      verde
- [ ] O frontend em produção aponta para a URL pública da API, não para `localhost`
- [ ] Um Aluno da Turma viva completa o fluxo do Preditor **num navegador**, em produção, e recebe
      Argumento Final e chance por curso
- [ ] A Calculadora completa o fluxo em produção e mostra os cinco Anos-Âncora
- [ ] A tela indica que a previsão de 2024-2026 se apoia em estatística derivada
- [ ] A Análise Temporal mostra a série oficial e a evolução de Nota de Corte por curso, sem
      regressão
- [ ] `/health` responde na URL pública
- [ ] Nenhum nome nem inscrição de Aluno é servido por nenhum endpoint público
- [ ] O mapa `.scratch/publicar-site/map.md` é fechado, com o relatório da rodada em
      `.scratch/publicar-site/relatorios/`
