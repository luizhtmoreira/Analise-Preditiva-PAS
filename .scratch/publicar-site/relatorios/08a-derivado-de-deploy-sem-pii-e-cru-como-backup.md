# Relatório — Ticket 08a: Derivado de Deploy sem PII, cru como backup separado

**Ticket:** `.scratch/publicar-site/issues/08a-derivado-de-deploy-sem-pii-e-cru-como-backup.md`
**Status:** concluído em código; **um passo manual pendente** antes do próximo deploy (ver §4)
**Onde vive o código:** `src/pas_intelligence/derivado_deploy.py` (novo — fonte única das
colunas) + `deploy/publicar_pacote.py` (reescrito) + `api/services/gestao_service.py` e
`api/services/analytics_service.py` (usecols agora importados) + `deploy/ponteiro.json` +
`deploy/README.md` + testes em `tests/test_derivado_deploy.py`.

---

## 1. O que foi pedido, e o resultado

`resultado_final.csv` e `notas_corte.csv` iam inteiros (com `nome`) para dentro da imagem
hospedada; o `usecols` dos serviços protegia a leitura, não o arquivo. A saída: o **Derivado de
Deploy** — os mesmos dois CSVs, reduzidos às colunas que a API de fato lê, sem `nome` em nenhum
dos dois — passa a ser o que sai do disco de quem publica, não o que entra na leitura.

A lista de colunas nasceu com dono: `pas_intelligence.derivado_deploy.COLUNAS_DERIVADO` (por
arquivo) é a fonte única, lida tanto por `publicar_pacote.py` (que constrói o Derivado antes de
subir) quanto pelos dois serviços que já liam os CSVs em runtime — as duas listas que existiam
duplicadas nos serviços saíram, e não abriu-se uma terceira cópia no publicador.

O Domicílio Versionado ganhou o segundo papel que lhe faltava: `Luiz1912/vetor-pas-dados`
(existente) vira o cru privado — backup explícito, fora do `ponteiro.json`, que nenhuma etapa de
build lê — e um repositório novo, `Luiz1912/vetor-pas-dados-derivado`, é o que o Ponteiro passa a
apontar.

## 2. As decisões, e o porquê de cada uma

| # | Decisão | Motivo |
|---|---|---|
| 1 | Colunas moram em `src/pas_intelligence/derivado_deploy.py`, não em `deploy/` | é o mesmo lugar onde já mora `pas_constants.py`/`training_dataset.py` — a fronteira de produto que tanto `api/` quanto `deploy/` já importam, em vez de inventar uma dependência nova de `api/` para `deploy/` ou vice-versa |
| 2 | `build_derivado()` só corta coluna, nunca linha | o ticket é explícito: "cortar coluna é trivial" — o filtro `checksum_fecha == True` continua sendo decisão de quem lê (`gestao_service`, `analytics_service`), não do Derivado. Cortar linha também mudaria a superfície do que os testes de `test_notas_corte_resultado_final_load.py` já provam |
| 3 | `COLUNAS_RESULTADO_FINAL` é a união das colunas que `gestao_service` e `analytics_service` precisam (10 colunas), não duas listas menores | os dois leem o mesmo arquivo; manter subconjuntos diferentes por serviço reabriria a duplicação que o ticket pede para fechar. O custo é ler 1-2 colunas a mais em cada serviço — desprezível |
| 4 | `cru` é um alvo novo e explícito em `publicar_pacote.py` (`python deploy/publicar_pacote.py cru`), não roda por padrão | preserva "o cru continua existindo... backup por decisão, não por acidente" (a mesma lição da invariante dos parsers no `CLAUDE.md`) sem forçar upload do CSV com PII em toda promoção |
| 5 | `cru` nunca aparece em `ponteiro.json` | `buscar_artefatos.py` itera **todas** as chaves do ponteiro e baixa cada uma no build — se `cru` entrasse ali, a fronteira que este ticket existe para criar quebraria na primeira reconstrução da imagem |
| 6 | Repo do Derivado ganhou nome novo (`vetor-pas-dados-derivado`), não reaproveitou `vetor-pas-dados` | o cru já vivia em `vetor-pas-dados`; sobrescrevê-lo com o Derivado destruiria o backup que o ticket pede para preservar |
| 7 | Publicar o Derivado publicamente ficou de fora | escopo explícito do ticket ("Não é escopo... publicar é irreversível") — só o repositório privado do Derivado foi criado |

## 3. O que o teste prova

`tests/test_derivado_deploy.py` prova sobre o **arquivo escrito**, não sobre a intenção do
script (like o ticket pede):

- `test_nome_nao_sobrevive_em_nenhum_dos_dois_csvs` — lê os CSVs que `build_derivado` escreveu e
  confirma `"nome" not in df.columns` nos dois.
- `test_colunas_do_derivado_batem_exatamente_com_a_fonte_unica` — as colunas do arquivo escrito
  são exatamente `COLUNAS_NOTAS_CORTE`/`COLUNAS_RESULTADO_FINAL`, nem mais nem menos.
- `test_nenhuma_linha_e_descartada_no_derivado` — corte de coluna não descarta linha.
- `test_arquivo_ausente_na_origem_e_ignorado` — publicar só `notas_corte.csv` (sem
  `resultado_final.csv` em disco) não quebra.

`tests/test_notas_corte_resultado_final_load.py` (pré-existente) continua passando sem alteração
de valores esperados — prova que trocar os `usecols` inline pelas constantes importadas não mudou
o schema que os serviços produzem. Suíte completa: 436 passed.

## 4. O que ficou pendente, e por quê é pendente por natureza

`deploy/ponteiro.json`'s `dados.repo_id` já aponta para `vetor-pas-dados-derivado`, mas
`dados.revision` ficou `""` — nenhum agente de código deve rodar `hf auth login` e publicar dados
reais de Alunos num repositório do Hugging Face sem o dono do produto no controle dessa ação.
`deploy/README.md` documenta o passo exato (`python deploy/publicar_pacote.py cru` +
`python deploy/publicar_pacote.py dados`, depois commitar o ponteiro) com um aviso explícito no
topo do arquivo. Até isso rodar, `buscar_artefatos.py` falha no build com mensagem clara
("ponteiro.json não tem revisão para 'dados'") — não com crash genérico, mas ainda bloqueia
deploy. O ticket 08c (serviço no ar) já está `Blocked by: 08a` — este passo manual é o que destrava.

## Glossário desta rodada

- **Derivado de Deploy:** os CSVs reduzidos, sem `nome`, que o Ponteiro aponta e o build baixa —
  o que a API hospedada de fato lê.
- **Cru:** os CSVs completos, com `nome`, mantidos como backup explícito fora do Ponteiro.
- **Domicílio Versionado:** os repositórios privados no Hugging Face Hub onde vivem os artefatos
  gitignored (modelo e dados) — nome cunhado no mapa `treino-modelos-pas3`.
