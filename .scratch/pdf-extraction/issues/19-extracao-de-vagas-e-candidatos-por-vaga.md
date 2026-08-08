# 19 — Extração de vagas ofertadas e candidatos por vaga

**What to build:** um extrator novo para o documento de **candidatos por vaga** (Edital de
Abertura ou equivalente) que produz, por (triênio, curso, campus, turno, Sistema de
Concorrência): número de vagas ofertadas e número de candidatos inscritos. Alimenta a página
pública de curso (`/nota-de-corte/<curso>`, [[project_alcance_organico]]) com "vagas: N,
candidatos: M, concorrência: M/N por vaga" — dado que hoje não existe em nenhum CSV extraído
(`notas_corte.csv` e `resultado_final.csv` só têm `convocados_na_chamada`, contagem por chamada
já filtrada, não o total de inscritos nem o total de vagas do curso).

## Por que é ticket separado do 16

O ticket 16 é o spike que confirma **se** esse documento serve como fonte — de N, para fechar o
ponto cego de `_buracos_por_sistema` (ticket 02). Este ticket é o consumo **de produto**: extrair
o documento e publicar vagas/candidatos na página de curso, mesmo que o resultado do 16 seja "não
serve para fechar o ponto cego de validação" (as duas perguntas são independentes — um documento
pode não ter a granularidade certa para validar contagem de posição e ainda assim ter vagas e
candidatos por curso, que é tudo que a página pública precisa).

**Confirmado pelo dono do produto em 2026-08-03:** o documento existe e é publicado pela banca —
deixou de ser uma suposição.

## O que resolver

1. **Localizar o documento.** Já está entre os 77 PDFs em `data/pdfs`, ou precisa ser baixado à
   parte? Esta pergunta é compartilhada com o ticket 16 — não duplicar o trabalho, usar a
   conclusão de lá.
2. **Confirmar a granularidade.** Vagas e candidatos aparecem por (curso, campus, turno, Sistema
   de Concorrência) — o mesmo grão de `notas_corte.csv` — ou só agregado por curso? Isso decide
   se a página de curso consegue mostrar concorrência por cota ou só um número único.
3. **Extrator novo em `src/pas_extraction/`**, seguindo o padrão das famílias existentes
   (`resultado_final.py`, `convocacao.py`) — parser + schema + validação estrutural + CSV de
   saída próprio (ex. `vagas_candidatos.csv`), integrado ao `rodada.py` como uma família nova.
4. **Exposição na API.** Novo campo ou endpoint em `api/services/` (provavelmente perto de
   `get_course_chamadas`/`predict_service.py` ou de onde a página de curso já lê nota de corte)
   para a landing consumir sem duplicar lógica de agregação no frontend.

## O que este ticket não é

Não é a extração de médias/desvios (já existe, `medias_desvios`) nem duplica a Nota de Corte
(ticket 10, já fechado). É uma quarta família de documento, com dado que nenhuma das três atuais
carrega.

**Blocked by:** Nenhum — ticket 16 concluído (`relatorios/16-spike-documento-candidatos-por-vaga.md`).

**Status:** ready-for-agent

**Resposta do ticket 16 aplicada aqui:** o documento existe em
`data/pdfs/candidatos-por-vaga/` (4 PDFs, cobrindo os triênios 2021/2023, 2022/2024 e
parcialmente 2023/2025 — os 5 triênios de 2016/2018 a 2020/2022 não têm esse documento em mãos
ainda). Reporta candidatos e vagas por (curso, campus, turno, Sistema de Concorrência) — grão
igual ou mais fino que o pedido no item 3 abaixo. **Ressalva de definição, herdada do spike:**
a coluna `Inscritos` mede quem se inscreveu antes da eliminação da Etapa 3, não quem chegou ao
fim do processo — é sistematicamente maior (dezenas a centenas por curso) que o total de
`resultado_final.csv`. Rotular a página pública como "candidatos inscritos por vaga", não
"candidatos classificados por vaga", para não sugerir relação com a Nota de Corte.

- [x] Ticket 16 concluído e sua resposta de granularidade aplicada aqui
- [ ] Extrator novo em `src/pas_extraction/` para o documento, com validação estrutural
      equivalente às famílias existentes (checksum ou verificação de plausibilidade cabível)
- [ ] CSV de saída com vagas e candidatos por (triênio, curso, campus, turno, Sistema de
      Concorrência) — ou grão menor, se o documento não permitir esse nível
- [ ] Sincronizado com o backup de `src/pas_extraction/` (ver `CLAUDE.md`, seção de backup dos
      parsers) antes de encerrar a tarefa
- [ ] API expõe vagas/candidatos por curso para a página pública consumir
- [ ] Página de curso mostra número de vagas e candidatos por vaga, com a mesma ausência de PII
      já aplicada ao resto da página (sem nome de candidato)
