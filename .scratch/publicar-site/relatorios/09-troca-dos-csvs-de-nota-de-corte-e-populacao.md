# 09 — Troca dos CSVs de Nota de Corte e da base populacional (relatório)

## O que mudou

`api/services/gestao_service.py:load_resources` e `api/services/analytics_service.py:_load_population`
passam a ler `data/notas_corte.csv` e `data/resultado_final.csv` (saída da frente de extração),
em vez de `data/notas_corte_pas.csv` e `data/banco_alunos_pas_final.csv` (base ad-hoc antiga). Os
dois arquivos novos foram copiados de `.scratch/pdf-extraction/saida-nova/` para `data/`
(gitignored, igual aos antigos).

Os arquivos antigos (`notas_corte_pas.csv`, `banco_alunos_pas_final.csv`) **não foram apagados** —
ficaram em `data/` sem uso, por segurança (são gitignored, então apagar seria irreversível fora do
disco local).

## Critério do recorte — meça, não herde

O mapa registrava "4.786 cortes". Medido:

- `notas_corte.csv`: 5.225 linhas totais → **4.986** com `checksum_fecha == True` → **4.559** depois
  de descartar `semestre == "desconhecido"` (a linha não cabe em nenhum dos dois mapas de corte
  sem saber o semestre — perda de 427 linhas, toda concentrada no triênio 2018-2020, que some do
  corte por essa razão específica; o *fallback* existente em `_build_cutoff_maps` cobre isso).
- `resultado_final.csv`: 66.313 linhas → **64.298** com `checksum_fecha == True` (bate exatamente
  com o número do ticket).

`checksum_fecha == True` é o único filtro de plausibilidade necessário nos dois arquivos: o caso
MEDICINA/Darcy Ribeiro/Universal/2020-2022 (`nota_corte = 199.162,872`) já sai marcado
`checksum_fecha = False` e foi conferido ausente do resultado carregado. Nenhum corte que sobra
fica fora da faixa de Argumento Final observada por triênio em `resultado_final.csv` (verificado
programaticamente, não por amostragem).

## PII

`inscricao` e `nome` nunca são lidos — `pd.read_csv(..., usecols=...)` já exclui as duas colunas
antes de qualquer linha entrar em memória, nos dois arquivos. (`Inscricao`, sem PII de nome, é
mantida do lado da população porque `analyze_escola` depende dela pra casar aluno de escola com o
banco — mesmo padrão que o arquivo antigo já usava.)

## Bug encontrado durante a troca (fora do checklist original)

`Campus` no arquivo novo vem com a sigla da faculdade entre parênteses (`"UnB CEILÂNDIA (FCE)"`,
`"UNB GAMA (FCTE)"`, etc.), diferente do antigo (`"CEILÂNDIA"`, `"GAMA"`). A chave de curso que o
serviço monta é `f"{Curso} - {Turno} ({Campus})"` — com Campus já parentetizado, ela ganhava
parênteses aninhados, e `predict_service._parse_course_key` (que localiza o **último** `(` pra
separar Campus) cortava no parêntese errado. Resultado observado ao testar `/api/predict` de
verdade: cursos de Ceilândia/Gama/Planaltina voltavam com `turno` e `campus` embaralhados (ex.
`"turno": "DIURNO (UNB CEILÂNDIA"`, `"campus": "FCTS)"`).

Corrigido normalizando `Campus` na carga (`_normaliza_campus`, `gestao_service.py`): tira o sufixo
`" (SIGLA)"` e o prefixo `"UnB "/"UNB "`, devolvendo o nome de Campus simples que o resto do
sistema sempre assumiu. A sigla da faculdade (FCE, FGA, FUP, FCTS, FCTE) é descartada — não existe
consumidor dela hoje.

## Outras traduções de schema (ponto único: `load_resources`)

- `trienio` (`"2016/2018"`) → `Trienio` (`"2016-2018"`).
- `semestre` (`"1"`/`"2"`/`"desconhecido"`) → `Semestre` (`"1°"`/`"2°"`, linha descartada se
  desconhecido).
- `sistema_nome` — o rótulo do Sistema Universal mudou de `"Universal"` (novo) pra
  `"Sistema Universal"` (o que o resto do serviço, os schemas da API e os `Query(default=...)`
  ainda esperam); normalizado na carga. As outras cotas (`"Cota para Negros"`,
  `"EP / Baixa Renda / PPI"`, etc.) não têm nenhum lugar com comparação exata — ficam como estão e
  dependem do fuzzy match (`_find_best_match`) que já existia pra essa finalidade.
- `turno` — cursos de período integral (Enfermagem, Farmácia, Fisioterapia, Fonoaudiologia, Saúde
  Coletiva, Terapia Ocupacional) não têm Turno no Edital novo (`NaN`); virou `"Integral"` em vez de
  vazar um `"nan"` literal pra chave do curso.
- `resultado_final.csv`: `eb_p1_eN`/`eb_p2_eN`/`red_eN` → `P{1,2}_PAS{1,2,3}`; `argumento_final` →
  `Arg_Final` (analytics) / `ARG_FINAL_REAL` (cohort do Reality Check, mesmo nome que o código já
  usava).

## Verificado

- `pytest tests/` (356 testes, incluindo 8 novos em
  `tests/test_notas_corte_resultado_final_load.py`) — verde.
- `POST /api/predict` com `trienio="2023-2025"` responde `200` com cursos e probabilidades (o
  triênio que a base antiga não cobria bem).
- Chave de curso pós-normalização de Campus bate de volta com `_parse_course_key` (curso, turno e
  campus corretos), testado com dado sintético e conferido contra a API de verdade.

## Fora do escopo deste ticket

- A sigla da faculdade (FCE/FGA/FUP/FCTS/FCTE) descartada na normalização de Campus — se algum dia
  for útil na tela, precisa de um campo próprio, não uma extração do nome.
- `SUB JUDICE` aparece como pseudo-curso em alguns registros (categoria judicial, não curso real) —
  já existia na fonte antes desta troca; não filtrado aqui por não ter sido pedido.
