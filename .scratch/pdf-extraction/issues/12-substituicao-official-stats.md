# 12 — Substituição do `OFFICIAL_STATS` pelos valores oficiais

**What to build:** o Argumento Final calculado pelo sistema para de carregar erro de estimativa.
Os valores inferidos do `banco_alunos_pas_final.csv` são substituídos pelos valores oficiais
extraídos dos Editais, revisados no relatório do ticket 11.

A estrutura também muda: o `ExamStats` atual tem um `m_p1`/`dp_p1` único, mas o Edital publica a
Parte 1 separada por língua estrangeira. A mudança de forma precisa acomodar isso **sem quebrar
quem consome hoje** — hoje há um consumidor, `api/services/analytics_service.py`, que lê `s.m_p1`
e `s.m_p1 + s.m_p2`.

Esta é a única alteração fora do pacote `src/pas_extraction/`. Nada mais em `pas_intelligence` e
nada no app Streamlit muda.

**Blocked by:** 11 — Relatório de diferenças do `OFFICIAL_STATS`.

**Status:** ready-for-agent

- [ ] Os valores do `OFFICIAL_STATS` são os oficiais extraídos dos Editais, e o comentário sobre geração via `banco_alunos_pas_final.csv` sai
- [ ] O `ExamStats` acomoda a Parte 1 por língua estrangeira
- [ ] `api/services/analytics_service.py` continua funcionando sem alteração de comportamento observável
- [ ] `tests/test_pas_intelligence.py` continua passando
- [ ] Nenhuma outra alteração em `pas_intelligence` e nenhuma no app Streamlit
