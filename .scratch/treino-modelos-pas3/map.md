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

| Rota | Ticket | Você na sala? | Modelo / esforço |
|---|---|---|---|
| 1 | [Alunos com Etapa 1 ausente](issues/14-alunos-com-etapa-1-ausente.md) | **sim** — é pergunta de produto | Opus, alto |
| 2 | [Alvo canônico: Argumento ou 3 notas](issues/04-alvo-canonico-argumento-ou-tres-notas.md) | **sim** | Opus, alto |
| 3 | [Dataset de treino canônico](issues/05-dataset-de-treino-canonico.md) | não — delegável | Sonnet, médio |
| 4 | [Esquema de validação](issues/06-esquema-de-validacao.md) | **sim** — é a régua | Opus, alto |
| 5 | [Baseline honesto](issues/07-baseline-honesto.md) | não | Sonnet, alto |
| 6 | [Janela de dados: 2018 vale?](issues/08-janela-de-dados-2018-vale.md) | não | Sonnet, médio |
| 7 | [Conjunto de features](issues/09-conjunto-de-features.md) | em parte — confirmar disponibilidade com o produto | Sonnet, médio |
| 8 | [Família de modelo](issues/10-familia-de-modelo.md) | não | Sonnet, alto |
| 9 | [Incerteza calibrada](issues/11-incerteza-calibrada.md) | **sim** | Opus, alto |
| 10 | [Pipeline de treino reproduzível](issues/12-pipeline-de-treino-reproduzivel.md) | não | Sonnet, médio |
| 11 | [Treinar, avaliar e promover](issues/13-treinar-avaliar-e-promover.md) | **sim** — revisar antes de promover | Opus, alto |

**⚠ Timebox nos tickets 08, 09 e 10 — decidido no ticket 06, com medição.** A régua mediu o teto
de acurácia antes de o mapa gastar sessões procurando por ele, e ele está a 0,2% de uma regressão
linear de duas variáveis (dobra 5, classe majoritária):

| preditor | RMSE em `A3` |
|---|---:|
| repete a Etapa 2 (`A3 = A2`) | 5,187 |
| **regressão linear em (A1, A2)** | **4,690** |
| LightGBM, 400 árvores, + os 6 EBs crus | **4,681** |

`A1` e `A2` já contêm quase toda a informação existente sobre a Etapa 3; os ~26% restantes da
variância são o ano do próprio Aluno e não estão no dado. **Estes três tickets entram com prazo
curto e expectativa de empate**, não como exploração aberta — a regra de parada do ticket 06
(menos de 1% relativo em dois tickets seguidos) provavelmente dispara já no 08. O ganho deste
mapa não está em acurácia; está em coerência (ticket 04) e em incerteza honesta (ticket 11).
Ressalva: não foram medidos `curso`/`campus`/`turno` nem língua — é o ticket 09 que mede, e o
teto não está declarado sobre feature não medida.

**Critério da coluna "Modelo / esforço":** **Você na sala = sim** (grilling/decisão de produto,
sem métrica que resolva sozinha) → **Opus, alto**. **Não/delegável** → **Sonnet**, com esforço
**médio** quando é engenharia mecânica de resultado previsível (filtrar linha, montar pipeline,
comparar duas janelas) e **alto** quando é medição/comparação cujo resultado numérico vira a
base de uma decisão maior a jusante (baseline que define "honesto", família de modelo).

**Onde dá para paralelizar.** A rota é sequencial por dependência (`Blocked by` de cada issue),
não por convenção — a maior parte dela é uma corrente de um elo só porque cada ticket consome o
resultado do anterior. Só existem dois pontos onde dois tickets têm exatamente as mesmas
dependências satisfeitas ao mesmo tempo e não dependem um do outro:

```
05 → 06 → 07 ─┬─→ 08 ─┐
              └─→ 09 ─┴─→ 10 → 11 ─┐
                                    ├─→ 13
                              12 ───┘
                    (12 também espera 08 e 09)
```

- **Depois que 07 fechar:** 08 (janela) e 09 (features) podem rodar em sessões paralelas — as
  duas só pedem 06 e 07, nenhuma lê a saída da outra. Ambas são Sonnet médio, então é literalmente
  abrir duas sessões.
- **Depois que 10 fechar:** 11 (incerteza, Opus alto, você na sala) e 12 (pipeline, Sonnet médio,
  delegável) também satisfazem as próprias dependências ao mesmo tempo — e aqui o paralelismo é
  ainda mais natural, porque um é HITL e o outro não: você grilling o 11 numa sessão enquanto o
  12 roda delegado noutra.

Fora desses dois pontos, rodar em paralelo é gastar sessão sem ganhar tempo — o ticket seguinte
vai ficar bloqueado esperando o anterior de qualquer forma.

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

**⏳ Uma pergunta esperando resposta do dono do produto** (2026-07-28): **quantos itens tem a
Parte 2, e como o desconto por erro fecha o piso?** Os limites da P2 viraram constantes nomeadas
(`P2_MAXIMO` / `P2_MINIMO` em `target_calculator.py`), mas o valor segue sendo chute herdado — e
ele decide sozinho quando o produto diz "impossível" e quando diz "garantido". Se o piso real for
`−N` para `N` itens, bem mais raso que −100, o status `'garantido'` é **código morto** e nenhum
Aluno o alcança. Ele vai olhar junto das outras pendências; até lá, defeito 1 de
[`relatorios/defeitos-pendentes.md`](relatorios/defeitos-pendentes.md).

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

- [14 — Alunos com Etapa 1 ausente](issues/14-alunos-com-etapa-1-ausente.md) — **é ausência, não
  zero real**, fechado pela regra do PAS (que prevê as três células da tabela de etapas zeradas:
  5.768 / 0 / 0) sem precisar do Edital normativo. **O produto atende a classe** — nenhum Aluno
  recebe recusa —, e ela está no funil comercial. A previsão exige **função própria** porque o
  **Momentum** é indefinido para ela; o **Quanto Falta já está correto hoje**, por aritmética
  (`calculate_argument_etapa(0,0,0)` = z de zero = o que o Cebraspe aplica). Ausência passa a ser
  **declarada, nunca inferida** de notas zeradas, porque na fonte por etapa ela é um silêncio e
  "não encontrado" tem causa conhecida de defeito (tickets 13/18 do `pdf-extraction`). Dataset:
  **64.298** = 60.013 com Etapa 1 + **4.285** sem, numa tabela só com coluna `etapa_1_ausente`.
  Modelo único só com features da Etapa 2 **rejeitado**: apagaria o Momentum de 91% para acomodar
  9%. **Atender ≠ treinar em todos** — o treino ficou para medição do ticket 10.
  → [relatório](relatorios/14-alunos-com-etapa-1-ausente.md) ·
  [ADR-0008](../../docs/adr/0008-aluno-sem-etapa-1-atendido-com-funcao-propria.md)

- [04 — Alvo canônico: Argumento ou 3 notas?](issues/04-alvo-canonico-argumento-ou-tres-notas.md)
  — **nenhuma das duas: o alvo é `A3`, o Argumento da Etapa 3.** Para quem já fez PAS 1 e 2,
  `A1` e `A2` são aritmética exata; prever o Argumento Final inteiro aproxima ⅗ do peso de uma
  conta que já se sabe. **Argumento Final, EB e escore necessário saem por álgebra do mesmo `A3`**
  — a P2 é *resolvida*, não prevista. Tamanho do problema medido: as duas rotas que a tela mostra
  hoje divergem **15,29** na mediana, acima do RMSE declarado em **57%** dos Alunos, com viés
  oposto (`+9,25` e `−7,23`) — **11% discordam sobre passar** (n = 7.838, triênio 2023/2025).
  Decidido junto: **nada de projetar a prova futura** (`STATS_PAS3_TREND` e
  `project_historical_stats` saem) — entra o **Ano-Âncora**, 5 anos reais com o corte e a prova
  daquele ano casados. **Momentum e Volatilidade passam para a escala de Argumento** (EB e
  Argumento discordam sobre subir/cair em **17,2%** dos 60.013; **39,4%** em 2022/2024), e a
  **Volatilidade deixa de ser CV** — a média do par é ~0 e negativa em 49,3% da base. P1 e Redação
  viram **Estimadores Auxiliares** (média ponderada de z-scores), com **override só no caminho
  reverso**. O Aluno passa a **informar a língua estrangeira**.
  → [relatório](relatorios/04-alvo-canonico-argumento-ou-tres-notas.md) ·
  [ADR-0009](../../docs/adr/0009-alvo-canonico-argumento-da-etapa-3.md)

- [05 — Dataset de treino canônico](issues/05-dataset-de-treino-canonico.md) — **materializado**:
  `python scripts/build_training_dataset.py` → `data/training/pas3_dataset.parquet`, **64.298
  linhas**, uma por aluno-triênio. Aplica a regra do ticket 01 (`checksum_fecha == True`) e
  calcula `A1`, `A2`, `A3` (ticket 04) com validação cruzada contra o `argumento_final` impresso
  — divergência acima de 0,01 faz o build falhar em vez de publicar alvo errado. `nome` e
  `inscricao` nunca saem do build; sobrevive só `id_pseudonimo` (SHA-256 sem sal, truncado).
  Duplicata entre triênios: **144 alunos (0,46% das linhas)**, flag
  `inscricao_repetida_entre_trienios` entregue ao ticket 06, sem ticket próprio. Escala do alvo
  conferida estável nos 8 triênios (`A3`: média ~0, desvio ~9,1, sem degrau). `lingua_e1/e2/e3`
  presentes, usadas no cálculo. Formato Parquet (não `.joblib` — é dataset, não modelo treinado;
  a decisão do ticket 03 não se aplica aqui). `data/` já estava no `.gitignore`.
  → [relatório](relatorios/05-dataset-de-treino-canonico.md)

- [06 — Esquema de validação](issues/06-esquema-de-validacao.md) — **validação deslizante**
  (5 dobras, treina no passado e prevê o futuro) com **2023/2025 lacrado**, aberto uma vez no
  ticket 13; abrir o lacre produz **um número, não uma decisão** — reajustar depois de ver queima
  o lacre até o Edital de 2027. Embarca o modelo treinado nos 8 triênios, com o número do gêmeo
  medido até 2022/2024 escrito no manifesto. **Não agrupa por aluno**: os 144 repetentes ficam,
  porque sob split temporal a resposta do teste não existe em treino nenhum — e porque o Aluno
  vivo pode ser um deles (mesmo princípio do ADR-0008). Quatro métricas com papéis distintos:
  **RMSE decide**, MAE se fala, **viés** valida o RMSE como σ, erro de decisão **veta
  conversando**. Corte pelo **menor corte entre os sistemas do Aluno** (32,3% não são Universal;
  só 0,01% de `cota_padrao_suspeito`). Achado que atravessa o mapa: **a classe
  `etapa_1_ausente` é 0,36% nos dois triênios antigos e ~10% nos recentes** — é o filtro do
  ticket 01, não deriva do mundo; 1.465 linhas recuperáveis foram testadas e **não fecham**.
  Quatro travas mecânicas impedem o ticket 08 de ler isso como deriva. Critério de aceite virou
  **não-regressão + coerência + incerteza honesta**, não melhoria de acurácia — ver o timebox
  acima.
  → [relatório](relatorios/06-esquema-de-validacao.md)

- [07 — Baseline honesto](issues/07-baseline-honesto.md) — a régua existe
  (`src/pas_intelligence/validation.py`, com teste) e produziu a **tabela de referência única**.
  Melhor baseline trivial: **RMSE 5,167 em `A3`** (linear em `A1`,`A2` + as 6 legadas); a linear
  de duas variáveis sozinha dá 5,185 e repetir a Etapa 2 dá 5,661 — **o teto do ticket 06 se
  confirma sobre 37.844 linhas**. `modelo_arg_final` dá **5,420 e perde para a regressão linear**,
  mesmo tendo visto 95,2% do teste. **O ensemble por volatilidade é código morto** — não é chamado
  pela API nem pelo Streamlit; o arranjo real é o **meta-modelo roteador**, que manda 75% dos
  Alunos para o `modelo_rf`, o único que **memorizou** (RMSE 5,198 no visto contra 8,422 no
  limpo). Nas linhas limpas nenhum arranjo bate usar o MLP sozinho: **nem ensemble nem roteador
  pagam**. Proveniência do `13,49` fechada: é um **MAE** de `calculate.py:81`, medido em 2023/2025
  sobre a base de treino do próprio modelo — o RMSE real é **16,26**, **20,5% maior** que a
  constante que o produto declara. Faixa de decisão **congelada em 15,500**; erro de
  decisão do baseline **7,4%** (34,3% dentro da faixa, contra o piso matemático de 31,6%).
  Duas premissas caíram: **não existe "em qual semestre o Aluno concorreu"** (todos concorrem de
  uma vez e quem não entra no 1º disputa o 2º — corte do 1º maior em **1.317 de 1.317** chaves), e
  a regra dos **3 convocados foi relaxada para 1**, porque `convocados_com_argumento = 1` é a
  definição literal da Nota de Corte e não ruído (cobertura 55,8% → **90,0%**).
  → [relatório](relatorios/07-baseline-honesto.md)

- [08 — Janela de dados: 2018 vale?](issues/08-janela-de-dados-2018-vale.md) — **usa 2018**: a
  curva de erro contra número de triênios de treino **cai monotonicamente até N=6** (todo o
  histórico fora do lacre), sem mínimo em N=4 — não existe horizonte de validade. Cortar é sempre
  igual ou pior: janela de 3 triênios custa **+0,65% de RMSE**, janela de 1 custa **+5,5%** (e
  ainda perde 3 de 5 dobras para a trava 1). **Ponderar o dado velho por idade também não bate
  treinar em tudo sem peso** — testado com decaimento geométrico em 3 bases, nenhuma venceu.
  Nenhum dos três candidatos a quebra de regime (pandemia, fórmula, deriva gradual) aparece na
  escala do Argumento: média, desvio e correlação `(A1+A2)/2`×`A3` ficam estáveis nos 8 triênios
  — a normalização por ano (ticket 04) absorve o que existiria na escala de EB crua. **A régua
  ganhou o parâmetro `pesos` em `avaliar()`** para essa comparação. Melhor RMSE obtido (5,053,
  majoritária) fica **0,3% acima do Portão 1** (5,038) — a distância não mora na janela; a régua
  de parada do mapa provavelmente dispara no ticket 10.
  → [relatório](relatorios/08-janela-de-dados-2018-vale.md)

- [09 — Conjunto de features](issues/09-conjunto-de-features.md) — só um bloco paga o custo: as
  **derivadas de trajetória** (`|Cresc_EB|/|EB_PAS1|`, `|Cresc_Red|/|Red_PAS1|`,
  `sign(Cresc_EB)` — as mesmas razões que o `meta_scaler.joblib` de hoje já usa), **+2,13% de
  RMSE, grátis** (derivadas das 6 legadas, nada novo do Aluno). `curso` (+0,43%) e `lingua_e*`
  (+0,35%) ganham, mas pouco, e nenhum dos dois está disponível no schema de previsão hoje
  (`curso` não é campo de entrada; `lingua_e*` seria fácil de coletar mas o ganho não justifica) —
  **decisão fechada sem precisar consultar o produto**, porque nenhuma feature vencedora exige
  mudar a tela. `perfil_cota` + as 5 booleanas de cota **pioram** (−0,07%): a dimensão ética do
  checklist não precisou decidir, o número já rejeitou. `trienio` como numérica contínua (nunca
  categoria) **piora 1,43%**, confirmando por medição a armadilha que o ticket previu por
  argumento. Verificado que `curso` não é proxy da Nota de Corte (correlação do resíduo com o
  corte médio: **0,126**, fraca) — descartado por custo, não por leak. Conjunto final: as 6
  legadas + `A1`/`A2` + as 3 derivadas, **RMSE 5,057** — bate as três pernas do Portão 1.
  → [relatório](relatorios/09-conjunto-de-features.md)

## Restrições que o ticket 06 deixou nos tickets seguintes

- ~~**07 (baseline):** a faixa de decisão é **congelada** no RMSE do melhor baseline trivial deste
  ticket.~~ **FEITO** — largura **15,500**. O `13,49` não é "quase o baseline burro": é um **MAE**
  medido no próprio treino, e o RMSE real é **16,26**.
- ~~**08 (janela):** a curva de erro contra número de triênios é medida **só na classe
  majoritária**.~~ **FEITO** — usa 2018, janela expansiva, sem peso. Ver decisão acima.
- ~~**09, 10:** timeboxados — ver acima.~~ **09 FEITO** — ver decisão acima; moveu 2,13%, dentro
  da faixa "afinação". Comparação sempre **pareada dentro da dobra**, com a janela segurada
  igual; nunca modelo A na dobra 2 contra modelo B na dobra 4.
- **11 (incerteza):** o viés do baseline deu **+0,00** e a razão RMSE/MAE **1,26** (teórico 1,25)
  — a forma normal se sustenta no baseline, então o trabalho do 11 é largura **por Aluno**, não
  trocar a forma.
- **13 (promoção):** único lugar do repositório autorizado a chamar
  `holdout_final_use_uma_vez()`.

## Restrições que o ticket 07 deixou nos tickets seguintes

- **Todos:** medir é `avaliar()` de `src/pas_intelligence/validation.py`, com semente explícita.
  Nada de laço próprio — as travas do ticket 06 só são travas porque existem num lugar só. A
  fábrica devolve modelo **novo** por dobra, e a régua recusa quem devolver o mesmo objeto duas
  vezes. Features da Etapa 3 são recusadas na entrada.
- **08, 09, 10 — o que precisam bater (Portão 1):** RMSE `A3` agrupado **≤ 5,167** no geral,
  **≤ 5,038** na majoritária, **≤ 6,028** na minoritária, com **|viés| ≤ 0,5**. Ruído entre dobras
  do baseline: **±0,37**.
- **08 (janela):** a janela é o parâmetro `janela` de `gerar_dobras`/`avaliar`; os triênios de
  teste não mudam com ela, de propósito, para a comparação continuar pareada dentro da dobra.
- **09 (features):** o teto **não** está declarado sobre `curso`/`campus`/`turno`/língua. A média
  do curso sozinha dá 8,464 — carrega pouca informação isolada, o que não é o mesmo que carregar
  pouca **a mais**.
- **10 (família):** o veredito sobre arranjo de vários modelos já está dado — nem ensemble nem
  roteador batem o melhor componente sozinho nas linhas limpas. O que resta ao 10 é a família
  única.
- **11 (incerteza):** a largura honesta de hoje é **16,26** em Argumento Final (5,420 × 3), não
  13,49. RMSE/MAE entre 1,25 e 1,27 em tudo que foi medido — a forma normal serve, o trabalho é
  largura por Aluno.
- **Erro de decisão:** faixa **congelada em 15,500**, e o corte de um Aluno é o **menor entre os
  sistemas e os semestres** em que ele concorre. Não recalcular a faixa por modelo.

## Restrições que o ticket 04 deixou nos tickets seguintes

- **05 (dataset):** o alvo a materializar é `A3`; guardar também `A1` e `A2`, que viram features
  naturais. A língua por Etapa tem que entrar — sem ela não se calcula `A3`.
- **06 (régua):** o erro é medido em `A3`. O erro do Argumento Final é exatamente `3×` esse
  número, então não existe régua separada para ele.
- **08 (janela):** destravado — a pergunta "o padrão mudou desde 2018?" agora tem alvo definido.
- **09 (features):** Momentum **com sinal em Argumento**; Volatilidade como dispersão absoluta;
  EB cru permanece candidato, nunca leitura única.
- **10 (família):** o candidato prevê `A3`. O ensemble entra **sem o roteador** — o CV que o
  ponderava não existe nesta escala. Mede também se um Estimador Auxiliar de ML bate a média
  ponderada de z-scores.
- **11 (incerteza):** medir em `A3` e multiplicar por 3. O `RMSE = 13,49` está no lugar errado.

## Restrições que o ticket 14 deixou nos tickets seguintes

- ~~**04 (alvo):** o alvo **decide o custo** desta classe.~~ **RESOLVIDO** — a rota `A3` não
  cobra o custo que este item temia: `A1` sai por aritmética também para esta classe (o z de zero,
  exato de 2018/2020 em diante), e o conflito com a §8 do relatório 02 se dissolveu, porque `A3`
  é padronizado como o Argumento Final. O que segue exigindo função própria é só o **Momentum**,
  agora indefinido na escala de Argumento em vez de na de EB. ADR-0008 intacto.
- **06 (régua):** holdout estratificado por `etapa_1_ausente`; todo candidato reporta **dois
  números**, um por classe.
- **09 (features):** proibido dropar as features da Etapa 1 como simplificação — decisão de
  produto. O Momentum precisa estar representado **com sinal**.
- **10 (família):** "aceita valor faltante nativamente" é critério **com peso**, não desempate
  (linear/MLP fecham a porta da classe). Medir um-modelo-com-faltante vs. dois-modelos.
- **11 (incerteza):** incerteza **por classe**, no mínimo duas. RMSE emprestado da maioria produz
  probabilidade errada mesmo com previsão pontual certa.

## Not yet specified

Névoa reconhecida, ainda sem nitidez para virar ticket:

- ~~**Encaixe do modelo novo em `api/services/`.**~~ **RESOLVIDO pelo ticket 04** — o reverso não
  perde base, fica **mais simples**: `A3_necessário = (corte − A1 − 2·A2)/3` já é
  `target_calculator.py:259` e é exato. O módulo deixa de carregar `.joblib` (some
  `_carregar_modelo`, some o `ModuleNotFoundError`, some a degradação silenciosa) e o
  `stats_pas3` passa a vir do Ano-Âncora. Ver §7 do relatório 04.

- **Ano-Âncora na interface.** Cinco anos reais por curso, o mais recente em destaque e os outros
  atrás de um botão, no lugar da projeção por regressão. Toca Preditor, Painel Multi-Curso e
  Gestão de Ativos. Decidido no ticket 04; é trabalho de produto, não de modelo, e por isso não
  entrou na rota deste mapa. Vira ticket quando o modelo novo existir.
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
- **Extração dos Editais por etapa (PAS 1 e PAS 2 isolados). ⚠ PASSOU A SER BLOQUEANTE
  (ticket 04).** Sem `(2024,1)` e `(2025,2)`, `A1` e `A2` do Aluno vivo não existem — e a rota
  canônica inteira se apoia neles serem **exatos**, não aproximados. Duas coisas que o ticket 04
  acrescenta ao escopo abaixo: (i) média e desvio não precisam ser publicados, são **calculáveis
  sobre a população inteira** do Edital, que lista todos os candidatos; (ii) a Parte 1 é a
  exceção — o Edital da Etapa não diz a língua, então vale a regra "por língua onde o spread é
  estável, agrupado onde não", com o Aluno informando a própria língua.
  Levantada no ticket 14. Não é
  treino — é a fonte das notas do **Aluno vivo**, que está no meio do triênio e cujo Resultado
  Final do PAS 3 ainda não existe. Precisa trazer **duas** coisas: as notas dos Alunos vivos *e* as
  médias/desvios das etapas vivas — `OFFICIAL_STATS` tem 24 chaves e faltam `(2024,1)`, `(2025,1)`,
  `(2025,2)`, porque a tabela foi montada dos Editais de PAS 3 e o do triênio 24-26 só sai em 2027.
  Sem `(2024,1)`, o Quanto Falta não calcula o A1 do Aluno sem Etapa 1 do triênio vivo. Nasce com
  `etapa_1_ausente` como campo de primeira classe, derivado de evidência **cruzada** (presente no
  Edital da Etapa 2, ausente no da Etapa 1), nunca de notas zeradas. **Teste de aceite pronto:**
  dos 865 registros com Etapa 1 Ausente em 2023/2025, quantos estão ausentes do Edital da Etapa 1
  de 2023? Fecha a leitura de ausência com prova documental *e* mede a taxa de acerto do casamento
  por nome numa população de resposta conhecida.
- ~~**Alunos que aparecem em mais de um triênio.**~~ **RESOLVIDO pelo ticket 06** — 144 Alunos,
  todos nos quatro triênios recentes, 83 com uma perna no lacre. **Não se agrupa e não se
  remove**: sob split temporal a resposta do teste não existe em treino nenhum, então não há
  vazamento; e tirar repetentes do teste o tornaria menos parecido com a produção, não mais.
  Ver §3 do [relatório 06](relatorios/06-esquema-de-validacao.md).

## Out of scope

- **Corrigir os defeitos de extração.** Tickets 13–18 do mapa `pdf-extraction` seguem em
  paralelo. Verificado em 2026-07-26 que nenhum deles toca os 9 campos de nota nem o
  `argumento_final` — só `nome` (13, 18) e `classificacao_sistema_*` (14, 15) — então o treino
  não espera por eles.
- **Carregar dado em Supabase e mudanças de frontend/API além de continuar funcionando.**
- **Reescrever a camada de probabilidade como modelo.** `P(X > corte)` continua sendo conta
  analítica; o ticket 11 troca só de onde vem a incerteza, não o mecanismo.
