# 07 — As entradas derivadas entram e o Preditor responde para a Turma viva

**What to build:** um Aluno do triênio 2024-2026 abre o Preditor, digita as seis notas das Etapas 1
e 2, e recebe o Argumento Final previsto e a chance por curso. Hoje ele recebe uma recusa.

Este é o **tracer bullet central** da rodada: atravessa do dado (`OFFICIAL_STATS`) ao cálculo
(`A1`, `A2`, `Â3`) à API (`/api/predict`) e à tela.

## Por que ele recusa hoje

`Argumento Final = A1 + 2·A2 + 3·Â3`, e `A1` e `A2` são **conta exata**, não previsão: dependem da
média e do desvio que o Cebraspe publica por Etapa. O Cebraspe só publica depois do PAS 3 — um
Edital por triênio, com as três Etapas juntas. Para 2024-2026 isso sai em 2026.

Então `(2024, Etapa 1)` e `(2025, Etapa 2)` não existem no `OFFICIAL_STATS`, `stats_da_prova`
levanta `EstatisticaOficialAusenteError`, e o serviço traduz isso em `modelo_disponivel: false`.
**A recusa é proposital e correta** — aproximar `A1` e `A2` destruiria a parte exata da conta, que é
a fundação do ADR-0009. O que este ticket muda não é a política de recusa; é o dado deixar de faltar.

## O que entra

Duas entradas novas no `OFFICIAL_STATS`, marcadas como **derivadas** (ticket 01), com a Parte 1 na
forma **misturada** (o Edital isolado não diz a língua de ninguém) e a média já corrigida pelo
Deslocamento calibrado no ticket 06:

| | `(2024, Etapa 1)` | `(2025, Etapa 2)` |
|---|---:|---:|
| candidatos | 19.127 | 16.990 |
| `m_p2` / `dp_p2` (bruto, antes da correção) | 23,906 / 11,398 | 27,644 / 14,752 |
| `m_red` / `dp_red` | 6,471 / 2,292 | 6,316 / 2,251 |
| Parte 1 misturada | 2,787 / 2,466 | 3,066 / 3,100 |

O `OFFICIAL_STATS` passa de **24 para 26** entradas.

**A correção entra na média, não no Argumento Final.** Essa é a decisão C da spec: a causa está
localizada em `m_p2`, e corrigir na origem significa que `stats_da_prova`, `model_package`,
`training_dataset`, `target_calculator` e a API **não mudam uma linha** — herdam a correção de
graça, e não existe um segundo lugar onde a conta é ajustada. (Corrigir o Argumento Final no fim
obrigaria o caminho reverso da Calculadora a somar de volta, criando duas correções para manter em
sincronia.)

## O que a marca de "derivada" precisa alcançar

Quando o Edital de verdade sair em 2026, esses dois números serão substituídos e **as previsões vão
mexer**. Isso precisa estar registrado no dado e legível na tela — não descoberto depois por um
Aluno que viu a própria previsão mudar sem explicação.

## O que já não é mais problema

A **língua estrangeira**, que parecia o bloqueador, custa **0,46 ponto de Argumento Final em média**
(máx 3,21) e tem **viés zero** — é ruído, não erro sistemático. A Parte 1 pesa 0,72 numa conta que
soma 10, e a média misturada cai praticamente em cima da média da inglesa, que é 66% a 73% da
população.

**Blocked by:** 01 (`ExamStats` com Parte 1 misturada e procedência) e 06 (Calibração do
Deslocamento e o portão — se ele reprovar, este ticket não acontece).

**Status:** ready-for-agent

- [ ] `OFFICIAL_STATS` tem 26 entradas; as duas novas têm `origem = derivada` e Parte 1 misturada
- [ ] As médias das entradas novas estão corrigidas pelo Deslocamento calibrado no ticket 06, e o
      valor bruto (não corrigido) está registrado em algum lugar rastreável
- [ ] `POST /api/predict` com `trienio = "2024-2026"` devolve `modelo_disponivel: true`, com `a1`,
      `a2`, `a3_previsto`, `arg_previsto` e `largura_incerteza` preenchidos
- [ ] Um triênio que continua sem Edital segue recebendo `modelo_disponivel: false` com motivo — a
      política de recusa não foi afrouxada, só deixou de disparar para 2024-2026
- [ ] A resposta da API carrega a informação de que a previsão se apoia em estatística derivada, e a
      tela do Preditor a exibe
- [ ] O teste `tests/test_api_predict.py` inverte o sentido de `TRIENIO_DA_TURMA_VIVA`: ele passa a
      afirmar que 2024-2026 **responde**, e um triênio sem Edital continua recusando
- [ ] `pytest tests/` continua verde
