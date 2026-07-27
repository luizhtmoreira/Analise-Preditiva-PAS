# Mapa — Treino dos modelos de previsão do PAS 3

**Label:** `wayfinder:map`
**Criado:** 2026-07-26

## Destination

Os modelos que preveem a Etapa 3 do PAS estão **retreinados sobre o `resultado_final.csv`**
(66.313 registros, 8 triênios, com proveniência), substituindo os `.joblib` atuais em `models/`
— e cada escolha por trás deles é uma decisão registrada, não herdada:

- **quais linhas** entram no treino (janela de triênios e tratamento das flags de qualidade);
- **quais features** e **qual alvo** (Argumento Final direto ou as 3 notas + fórmula oficial);
- **qual família de modelo**, escolhida por medição contra um baseline honesto — o ensemble
  atual (LGBM/RF/linear/MLP + meta-modelo por volatilidade) entra como candidato, sem privilégio;
- **de onde a camada de probabilidade tira a incerteza**, no lugar do `RMSE = 13.49` fixo;
- **como o artefato é empacotado, versionado e promovido**, no padrão que um MLOps sênior usaria.

O mapa fecha quando esses artefatos existem, batem um critério de aceite escrito, e a API
continua funcionando com eles.

## Notes

**Este mapa carrega execução.** Override explícito do "plan, don't do" do wayfinder, pedido pelo
dono do produto: há tickets de decisão *e* tickets que produzem dataset, medição e modelo
treinado. Um ticket de execução só existe depois que a decisão que ele materializa está fechada.

**Domínio.** PAS/UnB. Ler `CONTEXT.md` e `DEVELOPER_HANDBOOK.md` antes do primeiro ticket.
Glossário essencial: **EB** (Escore Bruto, `P1 + P2`), **Argumento Final** (nota ponderada
acumulada das 3 Etapas que ranqueia na UnB — pesos oficiais `PESO_P1=0.72`, `PESO_P2=8.28`,
`PESO_REDACAO=1.00`), **Etapa** (1, 2 ou 3 — uma por ano do triênio), **Triênio** (`2023/2025`
= PAS 1 em 2023, PAS 3 em 2025), **Volatilidade/CV** (`std/mean * 100` sobre `[eb_pas1,
eb_pas2]`, o que hoje pondera o ensemble), **Nota de Corte** (Argumento Final mínimo aprovado
num curso na última chamada, por Sistema de Concorrência).

**Privacidade — restrição dura.** O `resultado_final.csv` contém nome + notas de aluno real
identificável. Vale [[project_parser_privacy]]: nenhum dado de aluno vai para arquivo commitado,
relatório, teste ou exemplo. Datasets derivados e artefatos ficam fora do git, como `models/` e
`data/`. Relatórios citam agregado e contagem, nunca linha.

**Defeitos pendentes.** Registro consolidado do que está documentado e ainda não corrigido em
`src/pas_intelligence/` e na avaliação de modelos:
[`relatorios/defeitos-pendentes.md`](relatorios/defeitos-pendentes.md).

**Relatório por ticket.** Cada ticket resolvido gera um relatório em
`.scratch/treino-modelos-pas3/relatorios/NN-<slug>.md` com as decisões e o *porquê* de cada uma,
mais glossário dos termos novos ([[feedback_ticket_completion_reports]]).

**Skills.** `/grilling` + `/domain-modeling` para os tickets HITL; `/research` para os AFK de
leitura. Investigação pesada vai para sub-agente com saída bruta em arquivo
([[feedback_context_hygiene]]).

**Convenção de medição.** Nenhum número entra num relatório sem o recorte que o produziu
(quais linhas, qual split, qual semente). Comparação entre modelos só vale sobre o *mesmo*
holdout.

## Rota até produção

Ordem de execução recomendada, uma sessão por ticket, limpando o contexto entre elas. **O
número da rota não é o número do arquivo** — os arquivos são numerados por ordem de criação,
esta lista é por ordem de execução. O status de cada ticket vive no próprio arquivo, nunca
aqui: esta seção só ordena, não rastreia.

| Rota | Ticket | Você na sala? |
|---|---|---|
| 1 | [Alunos com Etapa 1 ausente](issues/14-alunos-com-etapa-1-ausente.md) | **sim** — é pergunta de produto |
| 2 | [Alvo canônico: Argumento ou 3 notas](issues/04-alvo-canonico-argumento-ou-tres-notas.md) | **sim** |
| 3 | [Dataset de treino canônico](issues/05-dataset-de-treino-canonico.md) | não — delegável |
| 4 | [Esquema de validação](issues/06-esquema-de-validacao.md) | **sim** — é a régua |
| 5 | [Baseline honesto](issues/07-baseline-honesto.md) | não |
| 6 | [Janela de dados: 2018 vale?](issues/08-janela-de-dados-2018-vale.md) | não |
| 7 | [Conjunto de features](issues/09-conjunto-de-features.md) | em parte — confirmar disponibilidade com o produto |
| 8 | [Família de modelo](issues/10-familia-de-modelo.md) | não |
| 9 | [Incerteza calibrada](issues/11-incerteza-calibrada.md) | **sim** |
| 10 | [Pipeline de treino reproduzível](issues/12-pipeline-de-treino-reproduzivel.md) | não |
| 11 | [Treinar, avaliar e promover](issues/13-treinar-avaliar-e-promover.md) | **sim** — revisar antes de promover |

**Duas ordenações que não são arbitrárias e não devem ser trocadas:**

- **O alvo (2) vem antes da janela (6).** O ticket 02 mostrou que o EB da Etapa 3 varia ~35%
  entre triênios enquanto o Argumento Final é estável. Logo "o padrão mudou desde 2018?" tem
  resposta *diferente conforme o alvo* — medir a janela com o alvo em aberto produz um número
  sem significado.
- **A régua (4) vem antes de tudo que mede (5, 6, 7, 8).** Escolher o esquema de validação
  depois de ver resultados é escolher a conclusão.

**Espere a rota crescer.** Duas coisas em *Not yet specified* devem virar ticket quando o alvo
canônico fechar: o encaixe no `target_calculator` reverso, e os limites chutados de P1/P2
(`[-20,20]`, e o teto de 100 aplicado a P2 isolado quando provavelmente é do EB combinado).

## Decisions so far

<!-- uma linha por ticket fechado: gist + link -->

- [01 — Semântica das flags de qualidade](issues/01-semantica-das-flags-de-qualidade.md) —
  `campos_formato_invalido` marca **reparo bem-sucedido**, não erro; descartar essas linhas
  enviesaria o treino contra o aluno de nota baixa. Filtro do dataset é `checksum_fecha == True`
  sem filtrar por flag → **64.298 linhas (96,96%)**. `checksum_delta` não serve como peso
  contínuo. As falhas de checksum são **duas populações**, e a maior (1.446) é composta
  inteiramente de alunos com a Etapa 1 zerada, não de corrupção.
- [02 — Checksum antigo: extração ou mudança de fórmula?](issues/02-checksum-antigo-extracao-ou-mudanca-de-formula.md)
  — **a fórmula não mudou**: os pesos `0,72/8,28/1,00` e `1/2/3` são recuperados exatamente dos
  Editais de 2016/2018 e 2017/2019 (resíduo máx. 0,005 em 8.877 e 8.874 linhas), e o
  `OFFICIAL_STATS` cobre as 24 chaves `(ano, etapa)` dos 8 triênios. O degrau é a **regra da Etapa 1
  ausente** (`0,000/0,000/0,000`), mais generosa que o z de zero nos dois triênios antigos
  (mediana +2,704 e +3,549) e literal de 2018/2020 em diante. O resto são ~100 linhas/Edital de
  corrupção de extração, auto-sinalizadas e em taxa constante. **A janela pode ir até 2016/2018;
  o corte é por linha, não por triênio** → população limpa de **60.013** linhas.
  O déficit de 2018/2020 é coorte menor (12.740 vs 18.726), não perda de extração.
  → [relatório](relatorios/02-checksum-antigo-extracao-ou-mudanca-de-formula.md)
- [03 — Formato, versionamento e promoção do artefato](issues/03-formato-e-versionamento-do-artefato.md)
  — **`.joblib` fica**, com gatilho escrito para o texto nativo do LightGBM se o ticket 10 der GBM
  único; o que ataca a fragilidade não é o formato, é manifesto + versões cravadas + falha
  barulhenta. A unidade versionada é o **pacote** de uma rodada, num repositório **privado no
  Hugging Face** (100 GB grátis), **assado na imagem no build** — nunca baixado no boot, porque o
  Space hiberna em 48h. Manifesto de 5 blocos: dado (hash, nunca o dado), código, ambiente,
  modelos **com os nomes das features**, avaliação com o recorte. **Promoção é commit** de um
  ponteiro no GitHub, reversão é `git revert`. Portão bloqueante no build (carrega / versões batem
  / features batem); qualidade pior que produção só passa com chave de força **gravada no
  manifesto**. Sem reversão automática.
  Evidência que reordenou o ticket: `p1_pas3_model` e `red_pas3_model` **já não carregam**
  (`ModuleNotFoundError: No module named '_loss'`) e `target_calculator.py:66` engole o erro,
  respondendo por média ponderada em silêncio.
  → [relatório](relatorios/03-formato-e-versionamento-do-artefato.md)

## Not yet specified

Névoa reconhecida, ainda sem nitidez para virar ticket:

- **Encaixe do modelo novo em `api/services/`.** O `target_calculator.py` faz o caminho
  reverso (dado um corte, qual P2 o aluno precisa) usando `p1_pas3_model.joblib` e
  `red_pas3_model.joblib`. Se o ticket 04 escolher prever o Argumento Final direto, esse
  reverso perde a base — mas a forma do conserto só fica clara depois do 04 e do 10.
- **Retreino periódico.** Um Edital de Resultado Final novo sai por ano. Com que gatilho e
  frequência o modelo é retreinado, e quem decide promover. **Destravado pelo ticket 03** — o
  mecanismo existe (promoção é commit de um ponteiro, portão bloqueante no build, quem decide é o
  dono do produto). Falta só a cadência, e ela não depende de mais nenhuma decisão.
- **XAI: explicar a previsão por aluno.** O LightGBM entrega `predict(X, pred_contrib=True)` —
  quanto cada nota puxou a previsão para cima ou para baixo, cálculo exato (SHAP) para árvores e
  praticamente de graça. Matéria-prima de produto: "sua previsão é 68; a queda no PAS 2 custou 4
  pontos". Levantado no ticket 03 ao medir o formato nativo. Não é escopo deste mapa — só ganha
  forma depois que o ticket 10 fixar a família.
- **Monitoramento em produção.** Detectar que a distribuição dos alunos do app descolou da
  base de treino. Só se especifica depois que existir uma linha de base medida (ticket 07).
- **Efeito da Nota de Corte contaminada na avaliação.** A tabela de cortes ainda carrega os
  defeitos dos tickets 14 e 15 do mapa `pdf-extraction` (ex. `MEDICINA/Darcy/Universal/
  2020-2022 = 199.162,872`). Isso não afeta treinar, mas afeta medir "probabilidade de
  aprovação" ponta a ponta. Revisitar quando aqueles fecharem.
- **Alunos que aparecem em mais de um triênio.** Medido pelo ticket 01: **146 inscrições
  (0,22%)**. Pequeno demais para ticket próprio, grande demais para ignorar — o ticket 06
  decide se o split agrupa por aluno. Fica aqui só até aquela decisão ser tomada.

## Out of scope

- **Corrigir os defeitos de extração.** Tickets 13–18 do mapa `pdf-extraction` seguem em
  paralelo. Verificado em 2026-07-26 que nenhum deles toca os 9 campos de nota nem o
  `argumento_final` — só `nome` (13, 18) e `classificacao_sistema_*` (14, 15) — então o treino
  não espera por eles.
- **Carregar dado em Supabase e mudanças de frontend/API além de continuar funcionando.**
- **Reescrever a camada de probabilidade como modelo.** `P(X > corte)` continua sendo conta
  analítica; o ticket 11 troca só de onde vem a incerteza, não o mecanismo.
