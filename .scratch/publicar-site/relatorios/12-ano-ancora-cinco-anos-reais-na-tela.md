# Relatório — Ticket 12: Ano-Âncora, cinco anos reais na tela

**Ticket:** `.scratch/publicar-site/issues/12-ano-ancora-cinco-anos-reais-na-tela.md`
**Status:** concluído
**Onde vive o código:** `src/pas_intelligence/pas_constants.py` (`anos_ancora`) +
`api/schemas/predict.py` (`AnoAncoraResultado`, campos novos de `StrategyInput`/`StrategyResponse`)
+ `api/services/predict_service.py` (`_resultados_ano_ancora`, `get_course_cutoff`,
`predict_strategy` reescrito) + `api/routers/predict.py` (dedup de `/courses/cutoff`) +
`landing-page/components/public/calculadora/CalculadoraPage.tsx` + testes em
`tests/test_api_predict.py`.

---

## 1. O que foi pedido, e o resultado

O ticket 11 já tinha trocado `STATS_PAS3_TREND` (regressão sobre uma prova que ainda não
aconteceu) por um único Ano-Âncora — a Etapa 3 real e já publicada mais recente — sempre que a
Etapa 3 do próprio triênio do Aluno ainda não saiu (turma viva). Este ticket troca esse **um**
ano por **cinco**: a resposta da Calculadora passa a carregar uma lista de cinco cenários, um por
Ano-Âncora, cada um varrendo junto a estatística da Etapa 3 daquele ano e a Nota de Corte do curso
no triênio correspondente (Ano-Âncora 2025 → triênio 2023-2025).

## 2. Quando os cinco cenários aparecem — e quando não

**Decisão de escopo, não explícita no ticket:** os cinco cenários só existem quando a Etapa 3 do
próprio triênio do Aluno **ainda não foi publicada** (o mesmo gatilho que já existia para o
Ano-Âncora único do ticket 11, em `_stats_do_ciclo`). Quando o triênio do Aluno já tem sua própria
Etapa 3 real em `OFFICIAL_STATS` (ex.: `ciclo_aluno="2023-2025"`, cuja Etapa 3 é 2025, já
publicada), a resposta continua exata e `anos_ancora` vem vazio — simular "e se a minha Etapa 3
for como 2023?" não faz sentido quando a Etapa 3 real do Aluno **é** conhecida.

Por quê: o Ano-Âncora (relatório 04 §7.1) é definido como "ano real e já publicado usado como
**cenário**" para uma prova que o Aluno **ainda não sentou**. Isso só é verdade quando a Etapa 3
dele é futura. Gatilho reaproveitado, não reinventado: `_stats_do_ciclo` já sabia distinguir os
dois casos (bloco `try/except EstatisticaOficialAusenteError`); este ticket só faz o `except`
devolver cinco estatísticas em vez de uma, através de um quarto valor de retorno booleano
(`e3_e_ancora`) que `predict_strategy` usa para decidir qual caminho seguir.

## 3. As decisões, e o porquê de cada uma

| # | Decisão | Motivo |
|---|---|---|
| 1 | `anos_ancora()` em `pas_constants.py`, derivada de `OFFICIAL_STATS` | "a lista é derivada do dado, não uma constante" (ticket, linha 33) — quando o Edital de 2026 entrar, o quinto ano cai fora sozinho, sem editar código |
| 2 | Campos de topo (`p1_estimado`, `p2_necessario`, ...) replicam `anos_ancora[0]` quando os cinco existem | retrocompatibilidade: um cliente que não sabe do ticket 12 continua lendo a mesma forma de resposta de sempre — só o `frontend` novo lê `anos_ancora` |
| 3 | `StrategyInput` ganha `curso_alvo`/`cota`/`semestre`, opcionais | é o requisito do próprio ticket — "Cada Ano-Âncora usa... a Nota de Corte do triênio correspondente" (linha 60) — sem o curso e o sistema de concorrência não dá para buscar cinco cortes diferentes. Opcionais com fallback para não quebrar um cliente antigo (ver decisão 4) |
| 4 | Sem `curso_alvo`, os cinco cenários caem de volta no único `nota_alvo` do cliente | fallback deliberado: um cliente que só manda `nota_alvo` (como os três testes que já existiam antes deste ticket) continua funcionando, só que sem a Nota de Corte por triênio — a estatística de Etapa 3 ainda varia por ano, só a Nota de Corte fica presa. Documentado no docstring de `_resultados_ano_ancora`, não escondido |
| 5 | `get_course_cutoff` extraído do router para `predict_service.py` | o router e os cinco cenários do Ano-Âncora precisam exatamente da mesma regra (última chamada do 1º semestre = piso; 1ª chamada do 2º semestre = teto) — duas cópias seria o começo de duas regras |
| 6 | Um `prob_hist`/Reality Check só, sobre o cenário mais recente | o ticket pede cinco **rotas**, não cinco Reality Checks; a coorte histórica não muda por Ano-Âncora, é a mesma pergunta ("quantos Alunos com este EB passaram?") |
| 7 | `AnoAncoraResultado` carrega `arg_pas3_necessario` próprio | achado na revisão (`/code-review`): a primeira versão zerava esse campo no topo da resposta em vez de replicar o do cenário mais recente — um cliente antigo lendo `arg_pas3_necessario` receberia um `0.0` fabricado, silenciosamente, sem `status` avisando. Corrigido antes do commit |

## 4. O que o teste prova

- `test_calculadora_le_official_stats_e_nao_um_dicionario_proprio` (já existia) — ganhou a
  asserção `e3_e_ancora is False` para o triênio `2023-2025` (Etapa 3 = 2025, já publicada) e
  `anos_ancora == []` na resposta: confirma que o caminho antigo (uma resposta exata) continua
  intacto quando a Etapa 3 do Aluno já é real.
- `test_calculadora_turma_viva_traz_os_cinco_anos_ancora_mais_recentes` (novo) — para o triênio
  vivo (`2024-2026`), `anos_ancora` tem exatamente 5 entradas, nos anos de `anos_ancora()`
  (2025, 2024, 2023, 2022, 2021), e os campos de topo replicam o primeiro. Também prova que o P2
  necessário varia entre os cinco (cada um usa a estatística da sua própria Etapa 3), mesmo sem
  `curso_alvo` (fallback da decisão 4).
- `test_calculadora_ano_ancora_varia_a_nota_de_corte_por_trienio` (novo) — com `curso_alvo`
  informado e `get_course_cutoff` substituído por um dublê que devolve um valor distinto por
  triênio, prova que cada cenário usa a Nota de Corte do **seu** triênio (`trienio_corte =
  f"{ano-2}-{ano}"`), não um valor único repetido.
- `_stats_do_ciclo` passou a devolver 4 valores (`stats_p1, stats_p2, stats_p3, e3_e_ancora`); o
  teste que já desempacotava 3 foi ajustado.

`pytest tests/` — 444 testes, 0 falhas. `tsc --noEmit` — limpo. `eslint` nos arquivos tocados —
0 erros (1 warning pré-existente e não relacionado, `BrandMark` não usado em
`CalculadoraPage.tsx`, já presente antes deste ticket).

## 5. O que ficou de fora, de propósito

- **Preditor e Gestão de Ativos.** O relatório 04 menciona um "ticket novo — Ano-Âncora na
  interface" que tocaria também o Preditor e a Gestão de Ativos — mas o ticket 12 desta rodada
  (`publicar-site/issues/12-...md`) só descreve a Calculadora de Estratégia. Os outros dois
  continuam com o Ano-Âncora único (`gestao_service._stats_pas3_ancora`), sem ticket próprio
  ainda.
- **Cinco Reality Checks.** Só o cenário mais recente alimenta o `prob_hist`/`amostra` — decisão
  6 acima.
- **Bundle de `curso_alvo`/`cota`/`semestre` num tipo próprio.** A revisão (`/code-review`,
  eixo Standards) apontou que esses três parâmetros já viajam juntos em `get_course_chamadas` e
  `get_course_cutoff`, e este ticket cria um terceiro ponto de chamada com a mesma tripla — um
  candidato a "Data Clump". Deixado como está: é o mesmo padrão já presente no módulo antes deste
  ticket, e introduzir um tipo novo por três parâmetros que só um outro ponto do código
  compartilha seria a generalidade especulativa que o CLAUDE.md do projeto pede para evitar.

## 6. Checklist do ticket

- [x] A resposta da Calculadora carrega uma lista de cinco resultados, um por Ano-Âncora —
      quando a Etapa 3 do próprio triênio ainda não aconteceu; exata e sem lista quando já aconteceu (§2)
- [x] Os Anos-Âncora saem das cinco chaves `(ano, Etapa 3)` mais recentes do `OFFICIAL_STATS`,
      não de uma lista cravada (`anos_ancora()`)
- [x] Cada Ano-Âncora usa a estatística da Etapa 3 **e** a Nota de Corte do triênio
      correspondente — quando `curso_alvo` é enviado; fallback documentado quando não é (§3.4)
- [x] A tela mostra os cinco com o mais recente em destaque, faixa legível como incerteza
- [x] Nenhuma projeção linear de prova futura sobra no caminho (`base_projecao` seguiu sem
      efeito, nenhuma regressão nova)
- [x] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
