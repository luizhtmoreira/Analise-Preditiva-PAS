# Relatório — Ticket 03: Formato, versionamento e promoção do artefato de modelo

**Ticket:** `.scratch/treino-modelos-pas3/issues/03-formato-e-versionamento-do-artefato.md`
**Tipo:** grilling (decisão — não produz código)
**Status:** concluído
**Data:** 2026-07-26
**Glossário dos termos:** `.scratch/treino-modelos-pas3/glossario.md`

---

## 1. O que foi pedido

Como um modelo treinado sai da máquina de quem treinou e chega na API em produção, de forma que
se saiba **qual** modelo está rodando, **que dado** o gerou, e **como voltar** para o anterior.
Padrão de mercado, dentro de duas restrições duras: a stack é gratuita, e `models/` está fora do
git por ser IP do produto.

Critérios de aceite (todos atendidos — ver seção 8):

- [x] Decidido o formato de serialização, com o motivo e o que se perde na escolha
- [x] Decidido onde o artefato mora e como a API o obtém em produção
- [x] Decidido o que acompanha cada artefato como metadado mínimo
- [x] Decidido como se promove um modelo novo e como se reverte para o anterior
- [x] Confirmado que a escolha cabe na stack gratuita e na restrição de IP de `models/`
- [x] Relatório escrito

---

## 2. O que o levantamento achou antes de decidir

Três medições reordenaram o ticket. A primeira transforma a hipótese dele em fato, e a segunda
tem consequência fora dele.

### 2.1 Dois modelos já não carregam — e a falha é silenciosa

No ambiente atual (`python 3.14.3`, `scikit-learn 1.9.0`, `lightgbm 4.6.0`, `numpy 2.4.6`,
`joblib 1.5.3`):

```
p1_pas3_model.joblib   → FAIL  ModuleNotFoundError: No module named '_loss'
red_pas3_model.joblib  → FAIL  ModuleNotFoundError: No module named '_loss'
modelo_lgbm / modelo_rf / meta_model / scaler → OK
```

Os dois `HistGradientBoostingRegressor` foram salvos quando o `sklearn` ainda tinha um módulo
interno `_loss` no lugar de onde ele está hoje. Os números seguem intactos no arquivo; a receita
de remontagem é que aponta para o vazio.

O agravante não é a quebra — é o tratamento dela. `src/pas_intelligence/target_calculator.py:66`:

```python
except Exception as e:
    print(f"Aviso: Não foi possível carregar modelos ML: {e}")
```

A exceção é engolida, `self.model_p1` e `self.model_red` ficam `None`, e a calculadora reversa
passa a responder por média ponderada em vez de ML — entregando número ao aluno como se nada
tivesse acontecido. O ADR-0007 já tinha registrado essa falha em 2026-07-20
(`⚠ Modelo não carregado — incompatibilidade de versão do sklearn`) e ela passou como nota de
rodapé.

**Isto é a tese do ticket materializada.** Não é risco projetado: está em produção.

### 2.2 O `modelo_lgbm` foi treinado com features diferentes das que o ADR-0007 mediu

```
booster.feature_name()  →  ['EB_PAS1', 'Red_PAS1', 'EB_PAS2', 'Red_PAS2', 'Cresc_EB', 'Cresc_Red']
```

```
scripts/baseline_avaliacao.py:55
FEATURE_COLS_BASE =        ["EB_PAS1", "EB_PAS2", "Cresc_EB", "Media_EB", "Std_EB", "CV_EB"]
```

Só a posição 1 bate. As outras cinco estão trocadas: o script passa `EB_PAS2` na casa de
`Red_PAS1`, `Cresc_EB` na casa de `Red_PAS2`, `Media_EB` na casa de `Cresc_EB`, e assim por
diante. O `CLAUDE.md` documenta o vetor correto (`[eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red]`),
o que confirma que quem está errado é o script de avaliação, não o modelo.

O LightGBM aceitou em silêncio porque recebeu um array NumPy — array não carrega nome de coluna,
só posição.

Isso explica os números impossíveis do ADR-0007: `R² = -83.4`, `MAPE = 1.25e+19`,
`MaxErr = 31293` num alvo que vai de 0 a 92. Não era modelo ruim; era modelo alimentado com
lixo. **Consequência: o ADR-0007 é inválido, e ele é a linha de base contra a qual o mapa inteiro
prometeu se comparar.** Entregue ao ticket 07 (seção 6).

### 2.3 Nada no ambiente está cravado

`requirements.txt`: 22 dependências, **zero** com versão fixada. Não existe registro de "a versão
que gerou o artefato" — a informação que teria evitado 2.1 simplesmente nunca foi gravada.

### 2.4 Medições auxiliares

| Medição | Valor |
|---|---|
| `modelo_rf.joblib` | 353.982.401 bytes (354 MB) — 99% do peso de `models/` |
| `modelo_lgbm.joblib` × mesmo booster em texto nativo | 282.600 × **281.165** bytes (empate) |
| Dependências ML instaladas | scipy 98 MB · pandas 71 MB · sklearn 47 MB · numpy 33 MB · lightgbm 7 MB |
| Armazenamento privado gratuito no Hugging Face | **100 GB** |
| `models/` versionado? | Não — `.gitignore` bloqueia `models/` e `*.joblib`; `git ls-files models` volta vazio |

---

## 3. As sete decisões

### Decisão 1 — Formato: `.joblib`, com gatilho escrito para o texto nativo

O artefato continua em `.joblib`. **A fragilidade não é atacada pelo formato**, e sim por três
propriedades decididas adiante: manifesto com as versões, ambiente que as reproduz, e falha
barulhenta ao carregar.

**Por quê.** O que quebrou o `p1_pas3_model` não foi "pickle é ruim" — foi ninguém saber com que
`sklearn` o artefato nasceu, somado a um `except` que engoliu o erro. Trocar de serializador sem
gravar a versão não resolve; gravar a versão sem trocar de serializador resolve. A própria
documentação do scikit-learn autoriza `joblib` sob exatamente essa condição: garantia de
funcionamento apenas com a mesma versão de biblioteca e de Python com que foi salvo. O projeto
usava a ferramenta certa ignorando o manual.

**Alternativas rejeitadas, e o que elas tinham de bom:**

*Texto nativo do LightGBM.* Custo zero medido (281 KB × 283 KB), legível sem executar código, e
estabilidade que é promessa do fornecedor em vez de disciplina do time. O argumento mais forte a
favor: se o ticket 10 concluir "um GBM só", morrem junto o `linear`, o `mlp`, o `scaler`, o
`meta_scaler` e o `meta_model` — sobra um booster, e aí não é "dois formatos convivendo", é um
arquivo de texto sem uma linha de pickle no sistema. Foi por isso que virou gatilho em vez de
rejeição.

*ONNX.* É a única opção que **elimina a classe de falha** em vez de administrá-la: o servidor não
instala `sklearn` nem `lightgbm`, então não existe versão para divergir. Derrubaria ~100 MB de
~250 MB de dependências, o que importa num Space que hiberna. E o argumento decisivo a favor
dele, que não tem resposta boa: disciplina apodrece, arquitetura não — o
`ModuleNotFoundError` de hoje é essa história já contada uma vez, e ninguém *decidiu* quebrar o
`p1_pas3_model`, ele quebrou porque o tempo passou.

Foi rejeitado assim mesmo por dois motivos. O antídoto contra a decadência da disciplina é barato
(Decisão 7: se não carrega, a imagem não fica pronta) e falha de um jeito que se descobre em
segundos. E — o que pesou mais — **o ONNX não teria pego o bug de 2.2**: a conversão típica
produz um tensor único de seis floats sem nome, e teria aceitado o vetor trocado exatamente como
o `.joblib` aceitou. Quem pega esse erro é o nome das features, que o manifesto carrega em
qualquer formato.

*(Duas objeções minhas ao ONNX foram retiradas por serem fracas: a perda de precisão numérica
aparece na terceira casa decimal, irrelevante numa nota 0–100 com RMSE ≈ 13; e a lógica do
ensemble não precisa morar no artefato, ela é código versionado em `ensemble.py`.)*

**Gatilho registrado:** se o ticket 10 concluir *modelo único LightGBM sem scaler*, o pipeline do
ticket 12 emite texto nativo em vez de `.joblib`, sem reabrir esta discussão — a condição já está
escrita.

**O que se perde:** o artefato continua sendo código executável e continua acoplado à versão do
framework. A mitigação é procedimento, não propriedade do formato.

### Decisão 2 — A unidade versionada é o pacote, não o arquivo

Uma rodada de treino produz **um** conjunto — todos os modelos que ela gerou, mais o manifesto,
com uma versão só. Trocar em produção significa trocar o conjunto inteiro.

**Por quê.** Versionar arquivo a arquivo permite o defeito que já existe duas vezes no
repositório: `modelo_linear` só está correto com o `scaler` ajustado no mesmo dado, e
`meta_model` com o `meta_scaler`. Separe o par e a API carrega os dois, não dá erro, e responde
errado — a mesma família de falha silenciosa de 2.2. O pacote torna o descasamento impossível por
construção: não existe meio pacote.

**O que se perde:** corrigir um único modelo obriga a republicar o conjunto. Com um retreino por
ano, irrelevante. Se o `modelo_rf` de 354 MB sobreviver ao ticket 10, cada versão carrega esses
354 MB — o que é argumento contra o RF, não contra o pacote.

### Decisão 3 — Domicílio: repositório privado no Hugging Face Hub

Um repositório de modelos **privado**, separado do repositório do Space. Git com LFS por baixo:
cada publicação é um commit com SHA, o conteúdo é endereçado por hash, dá para marcar tag, e o
`manifest.json` é commitado junto.

**Por quê.** Entrega as quatro propriedades que faltam hoje — imutabilidade, checksum, linhagem e
histórico auditável — sem dependência nova paga e sem servidor para operar. É a mesma plataforma
do Space, então o download é interno. E é a execução do que o ADR-0004 já prometia e nunca
aconteceu.

**Duas alternativas ficaram fora por fato, não por gosto:**

- **Modelo no repositório da aplicação:** impossível. O repositório é público no GitHub e
  `models/` é IP.
- **Modelo no repositório do Space:** impossível pelo mesmo motivo. `landing-page/lib/api.ts:1-5`
  chama a API do lado do cliente via `NEXT_PUBLIC_API_URL`, ou seja, do navegador do aluno — API
  chamada do navegador precisa ser publicamente alcançável, e num Space público tudo que está no
  repositório é público.
- **GitHub privado + Releases:** funciona, mas a cota gratuita de LFS é de 1 GB de armazenamento e
  1 GB de banda por mês — três downloads do `modelo_rf` e o mês acabou. Contornável via assets de
  Release, ao custo de um passo manual.
- **Supabase Storage:** já está na stack e tem controle de acesso, mas é armazenamento de objeto
  puro: sem commit, sem histórico, sem tag. Trocaria um lugar sem versão por outro lugar sem
  versão.

**Avisos operacionais:** o histórico de LFS conta na cota (versão antiga de arquivo grande
continua ocupando espaço), o que torna o `modelo_rf` gordo uma dívida acumulativa. E existe um
bug conhecido e aberto no Hub — erro 403 `"Private repository storage limit reached"` com cota
sobrando (`huggingface_hub` issues #3048 e #3049). Não é bloqueador; é para reconhecer a
mensagem quando aparecer.

### Decisão 4 — O pacote é assado na imagem, no build

O `Dockerfile` do Space baixa o pacote na revisão fixada durante a construção da imagem, usando
`HF_TOKEN` como secret de build. Em produção a API sobe lendo do disco local.

**Por quê.** O Space gratuito hiberna após 48h e armazenamento persistente é add-on pago — baixar
no boot significaria rebaixar o pacote inteiro a cada despertar, com o aluno esperando. É
justamente o cold start que o keep-alive do ADR-0004 tenta evitar. Assar na imagem também remove
a rede do caminho crítico do boot, elimina a possibilidade de subir com o pacote errado, e —
principal — permite que o teste de carregamento rode **no build**, enquanto alguém está olhando.

**O que se perde:** promover exige reconstruir a imagem (minutos) em vez de reiniciar. Com um
retreino por ano, preço nenhum.

### Decisão 5 — Manifesto com cinco blocos

`manifest.json` commitado junto do pacote:

```json
{
  "pacote": "pas3-2026-07-27",
  "criado_em": "2026-07-27T14:03:00-03:00",

  "dado":     { "arquivo": "resultado_final.csv", "sha256": "9f2c…",
                "linhas": 66313, "trienios": ["2016/2018", "…", "2023/2025"] },

  "codigo":   { "commit": "342bcd8", "arvore_limpa": true },

  "ambiente": { "python": "3.14.3", "scikit-learn": "1.9.0",
                "lightgbm": "4.6.0", "numpy": "2.4.6", "joblib": "1.5.3" },

  "modelos":  [{ "arquivo": "modelo_lgbm.joblib", "sha256": "a71e…",
                 "alvo": "EB_PAS3",
                 "features": ["EB_PAS1","Red_PAS1","EB_PAS2","Red_PAS2","Cresc_EB","Cresc_Red"],
                 "hiperparametros": { "n_estimators": 100, "learning_rate": 0.1 },
                 "semente": 42 }],

  "avaliacao": { "holdout": "trienio 2023/2025 inteiro", "semente": 42,
                 "metricas": { "MAE": 9.8, "RMSE": 12.4, "R2": 0.41 } }
}
```

Cada bloco existe por um motivo verificado nesta sessão:

- **`dado`** — responde "que dado gerou isto". Entra o **hash, não o dado**: o
  `resultado_final.csv` tem nome e nota de aluno real e não sai da máquina do treino
  ([[project_parser_privacy]]). Hash é um número; não revela nada.
- **`codigo`** — `arvore_limpa: false` significa que havia alteração não commitada na máquina
  durante o treino, ou seja, o commit registrado **não** descreve o código de verdade. É a
  diferença entre reproduzível e "quase".
- **`ambiente`** — é o bloco que ataca 2.1 e 2.3 diretamente.
- **`modelos.features`** — o mais importante, e só está aqui por causa de 2.2. Com a lista de
  nomes na ordem certa gravada no artefato, a API confere o vetor antes de prever e recusa se não
  bater.
- **`avaliacao`** — a métrica viaja **dentro** do artefato, com o recorte que a produziu, então
  nunca se separa dele. Atende a convenção de medição do `map.md`.

### Decisão 6 — Promoção é commit; reversão é desfazer o commit

Um arquivo versionado no repositório público do GitHub aponta para a revisão em produção:

```json
{ "repositorio": "luizhtmoreira/pas3-modelos",
  "revisao": "7c41ab9…",
  "pacote": "pas3-2026-07-27" }
```

Um workflow do GitHub Actions observa esse arquivo: quando ele muda em `main`, empurra para o
Space e dispara a reconstrução da imagem.

- **Promover** = mudar uma linha e commitar.
- **Reverter** = `git revert`. O workflow reconstrói com a revisão antiga.
- O histórico do git vira o histórico de trocas de modelo — quem, quando, para qual — de graça.
- Não vaza IP: o arquivo carrega o nome de um repositório privado e um SHA, que não dizem nada
  sobre o modelo.

**Por que não as alternativas.** Uma **tag móvel** (`producao`) no repositório de modelos parece
elegante e quebra a reprodutibilidade: duas construções da mesma receita, em dias diferentes,
podem produzir imagens diferentes porque a tag andou no meio — e o registro de quem a moveu fica
longe do histórico de código. Uma **variável no painel do Hugging Face** é o pior dos três: o
estado de produção passa a morar numa tela de configuração, sem histórico, sem autoria e sem
desfazer — o problema do Dropbox mudando de endereço.

**O que se perde:** meia hora configurando o workflow uma vez. Sem ele, o push para o Space seria
manual, e GitHub e Space poderiam divergir sem ninguém notar — que é exatamente o problema que
esta decisão existe para matar.

### Decisão 7 — Portão de promoção: três travas mecânicas e uma trava de julgamento com registro

Rodam na construção da imagem; se qualquer uma falhar, a imagem não fica pronta e o pacote nunca
chega perto de produção:

1. **O pacote carrega.** Abre cada modelo e faz uma previsão de fumaça. É o que
   `target_calculator.py:66` deixa de fazer hoje.
2. **As versões batem.** O bloco `ambiente` do manifesto contra o que está instalado.
3. **As features batem.** `modelos.features` contra o vetor que a API passa. É o portão que teria
   pego 2.2.

E a quarta, sobre qualidade: **o pipeline recusa publicar um pacote com desempenho pior que o de
produção; forçar é possível, mas a chave de força grava no manifesto que aquele pacote entrou por
cima do portão, com o motivo digitado.**

**Por quê.** Bloqueio puramente automático por métrica não sabe julgar: um modelo pode errar menos
na média e errar mais justamente nos alunos de nota alta, que são os que disputam Medicina.
Decisão puramente humana é onde a disciplina apodrece numa noite corrida, sem deixar registro de
que foi exceção. A terceira via não confia na disciplina nem finge que um número resolve
julgamento — ela só garante que passar por cima seja **visível** em vez de silencioso.

**Dependência:** as três primeiras travas funcionam de imediato. A quarta só ganha régua quando
existir medição válida do modelo atual — ou seja, depois do ticket 07, hoje inválido por 2.2.

**Decidido explicitamente que NÃO haverá reversão automática.** Se a API errar em produção, quem
decide voltar é o dono do produto. Na escala deste projeto, sistema que se reverte sozinho é mais
uma coisa que pode dar errado sozinha.

---

## 4. O desenho de ponta a ponta

```
   máquina de treino                    Hugging Face (privado)          GitHub (público)
   ─────────────────                    ──────────────────────          ────────────────
   ticket 12: pipeline
        │
        ├─ treina sobre resultado_final.csv
        ├─ mede no holdout
        ├─ escreve manifest.json
        ├─ portão: pior que produção? ──► recusa (ou força, e grava)
        │
        └─ publica o pacote ───────────► commit no repo de modelos
                                            │  pas3-2026-07-27/
                                            │    modelo_*.joblib
                                            │    manifest.json
                                            │  SHA 7c41ab9…
                                            │
                                                        você commita o SHA ─┐
                                                                            ▼
                                                                    modelo.lock.json
                                                                            │
                                            ┌───────────────────────────────┘
                                            ▼                     GitHub Actions
                                    Space (público) — build
                                       ├─ baixa o pacote na revisão 7c41ab9 (HF_TOKEN)
                                       ├─ trava 1: carrega?      falhou → imagem não sai
                                       ├─ trava 2: versões batem? falhou → imagem não sai
                                       ├─ trava 3: features batem? falhou → imagem não sai
                                       └─ imagem pronta, modelo dentro
                                            │
                                            ▼
                                    boot: lê do disco local, sem rede
```

**Reverter:** `git revert` no `modelo.lock.json` → Actions reconstrói com o SHA anterior.

---

## 5. Cabe na stack gratuita e na restrição de IP

| Peça | Custo | Verificação |
|---|---|---|
| Repositório de modelos privado no HF | R$ 0 | 100 GB privados no plano gratuito — ~270 versões do pacote no pior caso (354 MB cada), com um retreino por ano |
| Space (FastAPI) | R$ 0 | CPU Basic gratuito, ADR-0004 |
| GitHub Actions | R$ 0 | minutos ilimitados em repositório público, e o da aplicação é público |
| UptimeRobot (anti-hibernação) | R$ 0 | já previsto no ADR-0004 |

**IP:** o modelo mora em repositório privado; o repositório público carrega apenas o nome do
repositório e um SHA. `models/` continua fora do git como está hoje. O `resultado_final.csv`
nunca sai da máquina de treino — só o hash entra no manifesto.

---

## 6. O que este ticket descobriu e entrega a outros

**→ Ticket 07 (baseline honesto): o ADR-0007 está inválido, e a causa está localizada.**
`scripts/baseline_avaliacao.py:55` define `FEATURE_COLS_BASE` com cinco das seis posições
trocadas em relação ao que os modelos foram treinados (evidência em 2.2). Os `R² = -83` e
`MAPE = 1e+19` são artefato disso, não desempenho. O ticket 07 não começa por "rodar o script" e
sim por "consertar o vetor e só então rodar". Aviso para não criar expectativa errada: refazer
não vai necessariamente inverter o ranking — `mlp`, `arg_final` e `linear` podem ser ruins mesmo.
O que muda é que passa a haver como saber.

**→ Ticket 10 (família de modelo): a escolha lá aciona ou não o gatilho da Decisão 1.** Se der
"um LightGBM só, sem scaler", o formato passa a texto nativo automaticamente. E o `modelo_rf` de
354 MB é dívida acumulativa de armazenamento, o que conta como custo no julgamento dele.

**→ Ticket 11 (incerteza calibrada): o `RMSE = 13.49` de `statistics.py` fica no manifesto.** O
bloco `avaliacao` passa a ser a fonte da incerteza, em vez de um número cravado no código que
envelhece em silêncio quando o modelo troca.

**→ Ticket 12 (pipeline de treino): recebe a especificação de saída completa.** Formato, unidade,
manifesto, portão e destino de publicação estão todos definidos aqui.

**→ Dívida fora deste mapa:** `target_calculator.py:66` engole exceção de carregamento. O conserto
é a Decisão 7, mas o `except` continua lá até o ticket 12 passar por ele.

**→ Névoa nova, registrada no `map.md`:** XAI via SHAP. O LightGBM entrega
`predict(X, pred_contrib=True)` — quanto cada nota puxou a previsão de cada aluno, cálculo exato
para árvores. Matéria-prima de produto ("sua previsão é 68; a queda no PAS 2 custou 4 pontos").
Não é escopo deste mapa.

---

## 7. O que este ticket NÃO decide

- **Qual família de modelo** (ticket 10). O gatilho da Decisão 1 depende dela, mas a decisão de
  formato padrão não esperou.
- **Qual o alvo canônico** (ticket 04). O manifesto tem campo `alvo` e aceita qualquer resposta.
- **Qual o holdout** (ticket 06). O manifesto tem campo `holdout` e aceita qualquer recorte.
- **Com que gatilho e frequência se retreina.** Continua em *Not yet specified* no `map.md`; agora
  destravado, já que dependia do mecanismo de versionamento definido aqui.
- **Monitoramento em produção.** Depende de linha de base medida (ticket 07).

---

## 8. Critérios de aceite

| Critério | Onde foi atendido |
|---|---|
| Formato decidido, com motivo e o que se perde | Decisão 1 |
| Onde o artefato mora e como a API o obtém | Decisões 3 e 4 |
| Metadado mínimo que acompanha o artefato | Decisão 5 |
| Como se promove e como se reverte | Decisões 6 e 7 |
| Cabe na stack gratuita e na restrição de IP | Seção 5 |
| Relatório | este arquivo |

---

## 9. Glossário

Os termos deste ticket foram para `.scratch/treino-modelos-pas3/glossario.md`, documento de
estudo organizado por tema, com o caso real deste repositório em cada verbete. Os que apareceram
aqui: *serializar*, *pickle*, *joblib*, *formato nativo*, *ONNX*, *opset*, *unidade versionável*,
*manifesto*, *imutabilidade*, *checksum*, *linhagem*, *commit/SHA/revisão/tag*, *pin*, *Git LFS*,
*promoção*, *rollback*, *registry*, *imagem*, *build × boot*, *hibernação*, *cold start*,
*secret*, *smoke test*, *falha silenciosa × barulhenta*, *holdout*, *MAE*, *RMSE*, *R²*,
*baseline*, *família de modelo*, *GBM*, *ensemble*, *meta-modelo*, *features*, *hiperparâmetro*,
*XAI × transparência do artefato*, *SHAP*.
