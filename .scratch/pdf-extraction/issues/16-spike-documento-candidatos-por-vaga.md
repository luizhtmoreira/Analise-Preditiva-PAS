# 16 — Spike: documento de candidatos por vaga como fonte de N

**What to build:** não é ticket de implementação — é uma resposta documentada a uma pergunta,
no mesmo formato de relatório dos demais tickets.

O ticket 02 (`.scratch/pdf-extraction/relatorios/02-validacoes-estruturais-por-registro.md`,
§3.4) documentou um ponto cego: `_buracos_por_sistema` infere N (o total esperado de candidatos
por curso/sistema) como `max(posições observadas)`, porque nenhum Edital de Resultado Final ou
Convocação declara esse número. Se o registro perdido for justamente o de classificação N (o
último colocado), o `max` encolhe junto com ele e a checagem não vê buraco nenhum — foi
documentado como limitação permanente da técnica, sem fonte independente disponível.

O Luiz apontou que existe um documento de **candidatos por vaga**, que mostra quantos
candidatos há por curso — possivelmente a fonte independente de N que faltava.

**Confirmado em 2026-08-03:** o documento existe e é publicado pela própria banca ("documento
exclusivo disponibilizado por eles") — deixa de ser hipótese. Ganhou um segundo consumidor além
do ponto cego do ticket 02: o ticket 19 quer número de vagas e candidatos por vaga como conteúdo
da página pública de curso, e depende da conclusão deste spike (mesmo documento, mesma pergunta
de granularidade e definição de "candidato").

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** concluído — ver `relatorios/16-spike-documento-candidatos-por-vaga.md`

- [x] Identificado se o documento de "candidatos por vaga"/quantitativo já está entre os 77
      PDFs baixados em `data/pdfs`, ou se precisa ser obtido separadamente (e de onde) —
      não estava; 4 PDFs foram obtidos separadamente e adicionados em
      `data/pdfs/candidatos-por-vaga/` (cobrindo 2021/2023 e 2022/2024, parcialmente 2023/2025)
- [x] Confirmado se esse documento reporta candidatos na mesma granularidade usada pela
      checagem de buracos — por (curso, Sistema de Concorrência) — ou só um total agregado por
      curso (o que resolveria só parcialmente o ponto cego) — reporta por (curso, campus,
      turno, Sistema de Concorrência), grão mais fino que o necessário
- [x] Avaliado se o número desse documento é comparável ao `max(posições observadas)` da
      extração (mesma definição de "candidato" — inclui eliminados? inclui todos os sistemas?)
      — não é comparável: `Inscritos` mede quem se inscreveu (antes da eliminação), tipicamente
      dezenas a centenas a mais que o total do Resultado Final, por curso
- [x] Relatório de decisão registrado no mesmo formato dos demais tickets, com uma conclusão
      clara: viável fechar o ponto cego com esse documento (e como, como ticket de
      implementação separado), ou a limitação do ticket 02 continua de pé e por quê — não é
      viável; limitação do ticket 02 continua de pé (definição de candidato incompatível)
- [x] `defeitos-pendentes.md` atualizado com o resultado do spike (item 4)
