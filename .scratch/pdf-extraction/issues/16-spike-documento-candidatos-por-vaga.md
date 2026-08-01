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

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** ready-for-agent

- [ ] Identificado se o documento de "candidatos por vaga"/quantitativo já está entre os 77
      PDFs baixados em `data/pdfs`, ou se precisa ser obtido separadamente (e de onde)
- [ ] Confirmado se esse documento reporta candidatos na mesma granularidade usada pela
      checagem de buracos — por (curso, Sistema de Concorrência) — ou só um total agregado por
      curso (o que resolveria só parcialmente o ponto cego)
- [ ] Avaliado se o número desse documento é comparável ao `max(posições observadas)` da
      extração (mesma definição de "candidato" — inclui eliminados? inclui todos os sistemas?)
- [ ] Relatório de decisão registrado no mesmo formato dos demais tickets, com uma conclusão
      clara: viável fechar o ponto cego com esse documento (e como, como ticket de
      implementação separado), ou a limitação do ticket 02 continua de pé e por quê
- [ ] `defeitos-pendentes.md` atualizado com o resultado do spike (item 4)
