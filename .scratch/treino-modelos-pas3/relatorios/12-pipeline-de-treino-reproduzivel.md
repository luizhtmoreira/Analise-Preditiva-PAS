# Relatório — Ticket 12: Pipeline de treino reproduzível

**Ticket:** `.scratch/treino-modelos-pas3/issues/12-pipeline-de-treino-reproduzivel.md`
**Status:** concluído
**Tipo:** engenharia mecânica (Sonnet, médio — delegável)
**Código:** `src/pas_intelligence/training_pipeline.py`, `scripts/treinar_pipeline.py`
**Testes:** `tests/test_training_pipeline.py` (6 testes, dados 100% sintéticos)

---

## 1. O que foi pedido

Um comando único que vai do `resultado_final.csv` ao pacote do modelo, com todas as decisões dos
tickets 03, 05, 06, 08, 09 e 10 codificadas em vez de espalhadas em relatório — o defeito que o
mapa não podia repetir era o dos `.joblib` atuais em `models/`, que "apareceram por um processo
que não está no repositório".

## 2. O que foi construído

`treinar(caminho_csv, diretorio_saida, **opções)` em `training_pipeline.py` — uma função, não um
notebook, que:

1. `training_dataset.load_and_build` (ticket 05) — filtro `checksum_fecha == True`, calcula
   `A1`/`A2`/`A3`, remove PII;
2. monta o vetor de features do ticket 09 (`dataset_pas3.FEATURES_CANONICAS`: as 6 legadas +
   `A1`/`A2` + as 3 derivadas de trajetória) com o `NaN` nativo do ticket 10 nas colunas da
   Etapa 1 do Aluno sem Etapa 1;
3. avalia via `validation.avaliar()` (ticket 06), janela expansiva (`janela=None`, ticket 08),
   LightGBM com os hiperparâmetros fechados no ticket 10
   (`n_estimators=400, learning_rate=0.01, num_leaves=15`, com `deterministic=True` e
   `force_row_wise=True` para reprodutibilidade bit-a-bit);
4. confere o resultado contra o **Portão 1** (ticket 07, congelado: RMSE geral ≤5,167,
   majoritária ≤5,038, minoritária ≤6,028, \|viés\| ≤0,5) — se não bater, levanta
   `PortaoFechadoError` **antes de escrever qualquer coisa em disco**;
5. treina o artefato final sobre todos os triênios **exceto o lacrado**, salva em texto nativo do
   LightGBM (gatilho da Decisão 1 do ticket 03, acionado porque o ticket 10 fechou em "um
   LightGBM só, sem scaler");
6. escreve `manifest.json` com os cinco blocos da Decisão 5 do ticket 03 (`dado`, `codigo`,
   `ambiente`, `modelos`, `avaliacao`), preenchidos automaticamente — hash do CSV, commit e árvore
   limpa via `git`, versões instaladas via `importlib.metadata`, nome e ordem das features, e as
   métricas com o recorte que as produziu.

`scripts/treinar_pipeline.py` é só a CLI (`argparse` + chamada a `treinar`) — a regra fica toda em
`training_pipeline.py`, para não repetir o defeito que motivou o ticket 07/09 (`baseline_avaliacao.py`
tinha o vetor de features errado justamente porque a regra vivia num script solto).

### Refatoração pequena, para não duplicar a decisão do ticket 10

`scripts/familia_de_modelo_ticket10.py` tinha sua própria cópia de "trocar zero por `NaN` na
Etapa 1 ausente". Movida para `dataset_pas3.com_faltante_nativo_etapa1` (com a lista
`FEATURES_ETAPA1` nomeada), e o script do ticket 10 passou a importar de lá. Consequência do
próprio motivo do ticket 12: se essa lógica existisse em dois lugares, o dia que alguém mudar uma
cópia e esquecer a outra é o próximo `ADR-0007`.

## 3. Requisitos de confiabilidade — como cada um foi atendido

- **Determinismo.** Testado (`test_duas_execucoes_com_mesma_entrada_produzem_artefatos_equivalentes`):
  duas execuções com a mesma semente sobre o mesmo CSV produzem o **mesmo arquivo de modelo,
  byte a byte**, e o mesmo `manifest.json` (exceto o timestamp). Depende de `deterministic=True`
  + `force_row_wise=True` no LightGBM — sem eles, paralelismo interno pode reordenar somas de
  ponto flutuante entre execuções.
- **Registro automático.** Nenhum bloco do manifesto depende de alguém digitar algo à mão — commit
  e árvore limpa vêm de `git rev-parse`/`git status`, versões de `importlib.metadata`, métricas do
  próprio `ResultadoValidacao` que gerou o gate.
- **Falha ruidosa.** `PortaoFechadoError` interrompe antes do `mkdir` do diretório de saída — não
  existe pacote parcial no disco se o portão não foi batido. A chave de força
  (`forcar=True, motivo_forca=...`) exige o motivo por assinatura da função: chamar sem motivo é
  `ValueError`, não publicação silenciosa.
- **Privacidade.** `load_and_build` já remove `nome`/`inscricao` (ticket 05); o manifesto só
  recebe o hash do arquivo fonte, contagem de linhas e métrica agregada — nunca uma linha.

## 4. O que este pipeline decide **não** fazer (fronteira com o ticket 13)

O `TRIENIO_LACRADO` nunca é lido para treino nem para teste — a régua já garante isso
estruturalmente (`gerar_dobras` nunca o produz), e o treino do artefato final filtra
explicitamente `trienio != TRIENIO_LACRADO` antes do `fit`, por redundância deliberada.

Isso significa que o pacote que este pipeline produz é o **gêmeo** medido até 2022/2024, não
necessariamente o artefato de produção. A decisão do ticket 06 de "embarcar o modelo treinado nos
8 triênios" (isto é, incluindo o lacrado, para produção) exige abrir o lacre via
`validation.holdout_final_use_uma_vez` — e essa função só pode ser chamada pelo ticket 13, uma
única vez. Este pipeline não a chama e não a antecipa: rodá-lo de novo nunca corrói o lacre. Fica
para o ticket 13 decidir se retreina uma vez mais sobre os 8 triênios para o artefato definitivo,
usando o número medido aqui como a estimativa honesta a registrar.

Publicar o pacote num repositório privado do Hugging Face, o workflow do GitHub Actions e o
portão de build (Decisões 3, 4, 6 e 7 do ticket 03) também não entraram — são infraestrutura de
promoção (ticket 13), não do treino em si, e não fazem sentido gastar antes que o ticket 11
(incerteza) feche e o ticket 13 rode a comparação lado a lado com o dono do produto.

## 5. Testes

Seis testes em `tests/test_training_pipeline.py`, todos sobre um `resultado_final.csv` sintético
gerado em `tmp_path` (nenhum dado de aluno real, nenhum PDF):

| Teste | O que prova |
|---|---|
| `test_pipeline_do_csv_ao_pacote` | o comando único roda do CSV ao pacote; manifesto com os 5 blocos, features na ordem certa, triênio lacrado descrito mas não usado no treino |
| `test_pipeline_nunca_toca_o_trienio_lacrado` | nenhuma previsão de teste vem do triênio lacrado |
| `test_duas_execucoes_com_mesma_entrada_produzem_artefatos_equivalentes` | determinismo bit-a-bit do modelo e do manifesto |
| `test_portao_fechado_recusa_publicar_sem_forcar` | gate fecha a publicação; nada escrito em disco |
| `test_forcar_publica_com_motivo_gravado_no_manifesto` | a chave de força funciona e é visível no manifesto |
| `test_forcar_sem_motivo_e_erro_de_uso` | forçar sem motivo é erro, não omissão |

Rodada completa (`pytest tests/`): **304 passaram**, incluindo os testes já existentes de
`scripts/familia_de_modelo_ticket10.py` após a extração de `com_faltante_nativo_etapa1` para
`dataset_pas3.py`.

Smoke test manual da CLI (`scripts/treinar_pipeline.py`) sobre o mesmo tipo de CSV sintético: sem
`--forcar`, recusa publicar (RMSE ~8,7 contra o Portão 1 de 5,167 — esperado, dado sintético
aleatório não carrega o sinal real de PAS 1/PAS 2 → PAS 3); com `--forcar`, publica e grava o
motivo. **Não foi executado contra o `resultado_final.csv` real** — o arquivo não está presente
neste ambiente de trabalho (é `data/`, fora do git); a próxima execução com o CSV real fica para
o ticket 13, que é quem de fato promove.

## 6. Critérios de aceite

| Critério | Onde foi atendido |
|---|---|
| Comando único com as decisões dos tickets 03, 05, 06, 08, 09 e 10 codificadas | §2 |
| Duas execuções produzem artefatos equivalentes | §3, testado |
| Metadados de proveniência automáticos | §3, testado |
| Critério de aceite verificado pelo pipeline; falha impede publicação | §3, testado |
| Testes sem `data/pdfs` nem dado real | §5 |
| Script no repositório e caminho de regeração documentado | `scripts/treinar_pipeline.py` + `DEVELOPER_HANDBOOK.md` §9.5 |
| Relatório | este arquivo |

## 7. Entrega a outros tickets

- **Ticket 13** herda `treinar()` pronta para rodar sobre o CSV real, mais a fronteira explícita
  da §4: quando quiser o artefato de produção com os 8 triênios, é ele quem chama
  `holdout_final_use_uma_vez` e decide se retreina.
- **Ticket 11 (incerteza)**: o manifesto já tem o bloco `avaliacao.metricas` com RMSE/MAE/viés por
  classe, pronto para ser a fonte que substitui o `RMSE = 13,49` cravado — só falta a forma da
  distribuição por Aluno, que é o próprio escopo do 11.

## 8. Glossário

Termos novos desta sessão (para `.scratch/treino-modelos-pas3/glossario.md`): *gate/portão
automático*, *falha ruidosa vs. silenciosa* (retomado do ticket 03), *chave de força*,
*determinismo bit-a-bit*, *`force_row_wise`/`deterministic` do LightGBM*, *texto nativo de
modelo* (retomado do ticket 03), *provenance/proveniência*.
