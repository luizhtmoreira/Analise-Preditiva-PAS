# 12 — Pipeline de treino reproduzível

**Type:** task
**Status:** open
**Blocked by:** 03, 08, 09, 10

## Question

Um comando que vai do `resultado_final.csv` ao artefato de modelo — determinístico, com todas
as decisões dos tickets anteriores codificadas nele em vez de espalhadas em relatórios.

Hoje não existe: os 10 `.joblib` em `models/` apareceram por um processo que não está no
repositório. Ninguém consegue regerá-los, nem saber que dado os produziu. **Esse é o defeito
que este mapa não pode repetir** — se o modelo novo nascer de um notebook perdido, o próximo
retreino recomeça do zero e o mapa inteiro terá sido consumo, não investimento.

**O que o pipeline codifica** (cada item é a decisão de um ticket, não uma escolha nova):

- filtro de linha e construção do dataset (ticket 05);
- janela de triênios ou ponderação por idade (ticket 08);
- conjunto de features (ticket 09);
- família de modelo e hiperparâmetros (ticket 10);
- avaliação pelo esquema do ticket 06, com o critério de aceite verificado automaticamente;
- serialização e metadados no formato do ticket 03.

**Requisitos que o tornam confiável:**

- **Determinismo.** Duas execuções sobre a mesma entrada produzem artefatos equivalentes.
  Semente fixa em tudo que amostra — split, modelo, tuning.
- **Registro automático.** Cada execução grava versão do dado, commit do código,
  hiperparâmetros, métricas de holdout e versões das bibliotecas. Sem passo manual: metadado que
  depende de disciplina humana some.
- **Falha ruidosa.** Se o dataset de entrada mudou de forma inesperada, ou o critério de aceite
  não foi batido, a execução falha em vez de publicar um modelo pior em silêncio.
- **Privacidade.** Nenhum dado de aluno vaza para log, métrica ou artefato commitado
  ([[project_parser_privacy]]).

- [ ] Um comando único treina do CSV ao artefato, com as decisões dos tickets 03, 05, 06, 08,
      09 e 10 codificadas
- [ ] Duas execuções com a mesma entrada produzem artefatos equivalentes
- [ ] Metadados de proveniência gravados automaticamente a cada execução
- [ ] Critério de aceite verificado pelo próprio pipeline; falha impede publicação
- [ ] Testes cobrem o pipeline sem depender de `data/pdfs` nem de dado de aluno real
- [ ] O script está no repositório, e o caminho para regerar os modelos está documentado
- [ ] Relatório em `relatorios/12-pipeline-de-treino-reproduzivel.md`
