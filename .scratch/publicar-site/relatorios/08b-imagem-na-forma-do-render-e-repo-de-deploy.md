# Relatório — Ticket 08b: A imagem na forma do Render, entregue por um Repo de Deploy

**Ticket:** `.scratch/publicar-site/issues/08b-imagem-na-forma-do-render-e-repo-de-deploy.md`
**Status:** concluído em código; **um passo manual pendente** antes do primeiro deploy real (ver §4)
**Onde vive o código:** `Dockerfile` (reescrito) + `deploy/publicar_space.py` (reescrito) +
`deploy/README.md` (atualizado) + testes em `tests/test_publicar_repo_deploy.py`.

---

## 1. O que foi pedido, e o resultado

Duas metades da mesma fatia: o Dockerfile deixava de fazer sentido no Render (porta cravada em
`7860`, comentários que assumiam Hugging Face Spaces), e `publicar_space.py` publicava pela API do
Hub — um destino que não existe mais para a API.

**Dockerfile:** o `CMD` passou de exec-form fixo (`--port 7860`) para shell-form que lê `$PORT`
com default `10000` (`sh -c "... --port ${PORT:-10000}"` — exec-form puro não expande variável de
ambiente, por isso a troca de forma). O secret de build passou a ser montado e lido de
`/etc/secrets/HF_TOKEN` — o caminho real do Render (verificado nos docs, §2) — em vez do
`/run/secrets/` genérico do BuildKit. O `useradd -u 1000` ficou, com o comentário reescrito: era
"convenção do HF Spaces", passa a ser "boa prática, não exigência de plataforma". O download dos
artefatos continua em build-time; o comentário que justifica isso ficou mais forte, citando a
hibernação de 15 min do Render contra as 48h do HF.

**Publicador:** `publicar_space.py` trocou de destino — em vez de `HfApi.upload_folder` para um
Space, agora clona o **Repo de Deploy** (URL em `DEPLOY_REPO_URL`), substitui seu conteúdo pelos
arquivos de `PERMITIDOS` (a mesma lista de 8 padrões de antes, inalterada) e sobe um commit de
snapshot via `git push`. A função central (`publicar_snapshot`) é pura o bastante para testar
contra um bare repo local, sem precisar de rede nem de um repositório GitHub real.

## 2. As decisões, e o porquê de cada uma

| # | Decisão | Motivo |
|---|---|---|
| 1 | `dst=/etc/secrets/HF_TOKEN` no `--mount=type=secret`, não o `/run/secrets/` default do BuildKit | fato verificado nos docs do Render (`render.com/docs/docker-secrets`): Secret Files ficam disponíveis em `/etc/secrets/<nome>` — o ticket pedia explicitamente para não responder de memória aqui, depois do 402 do ADR-0004 |
| 2 | `CMD` em shell-form (`sh -c "... ${PORT:-10000}"`), não exec-form com valor cravado | exec-form (`CMD ["uvicorn", ..., "--port", "7860"]`) não faz expansão de variável — teria de ser `${PORT}` literal, quebrado. Testado com build real: default 10000 e `-e PORT=8080` funcionam (ver §3) |
| 3 | `publicar_snapshot` sempre parte de um clone fresco do remoto antes de montar o snapshot | garante que o `git push` seguinte seja sempre fast-forward — "publicar duas vezes seguidas produz um segundo commit, não conflito" é uma consequência do desenho, não um caso especial tratado à parte |
| 4 | `git commit --allow-empty` em toda publicação | o contrato do ticket é "publicar duas vezes seguidas produz um **segundo commit**" — literal, mesmo sem diferença de conteúdo entre as duas rodadas. Um `commit` condicional a haver diff quebraria esse critério |
| 5 | Ponto de partida do branch local decidido por `origin/main` (quando existe), nunca pelo HEAD simbólico do clone | armadilha real encontrada testando: um repositório recém-criado (seja um `git init --bare` local ou um repo remoto que ainda não recebeu push) não tem HEAD apontando para `main`, e `git checkout -B main` a partir de um HEAD "unborn" cria um branch **órfão** — perderia a história já publicada a cada rodada. Ver §3 |
| 6 | `PERMITIDOS` não mudou | o ticket foi explícito: "a lista já existe, testada e revisada — ela não muda". A nota do ticket ("9 padrões") não bate com a contagem real (8) — não inventei um 9º padrão para fechar a conta, porque isso seria alterar comportamento sem pedido |
| 7 | `DEPLOY_REPO_URL` como variável de ambiente, não uma URL cravada no script | o Repo de Deploy ainda não existe (é o dono do produto quem cria, ticket 08c) — cravar um nome de repositório que pode nem existir ainda seria inventar um fato externo, o mesmo erro que este ticket existe para evitar |

## 3. O que foi verificado, e como

**Nada foi respondido de memória** — o ticket pedia verificação explícita de porta e caminho de
secret contra a documentação do Render, depois do ADR-0004 ter custado um `402 Payment Required`
por confiar numa premissa desatualizada.

- **Porta:** `render.com/docs/environment-variables` — default `10000`. Testado com build Docker
  real (stage runtime isolado, sem depender do `fetch` que precisa de `HF_TOKEN`): `docker run`
  sem `PORT` responde em `10000`; com `-e PORT=8080` responde em `8080`.
- **Secret File:** `render.com/docs/docker-secrets` — `/etc/secrets/<nome>`, sintaxe
  `--mount=type=secret,id=NOME,dst=/etc/secrets/NOME`, disponível em todos os planos (a doc não
  restringe por tier). Testado com build real: secret fornecido → build passa e o valor é lido;
  secret ausente com `required=true` → build falha com `secret HF_TOKEN: not found` (mensagem
  clara do próprio BuildKit, não um crash do Python dentro do container).
- **`tests/test_publicar_repo_deploy.py`** (6 testes, todos contra um bare repo Git local fazendo
  o papel do remoto): `PERMITIDOS` filtra exatamente o esperado (arquivo fora da lista não
  aparece; `deploy/publicar_pacote.py`, que não está em `PERMITIDOS`, some); o Repo de Deploy não
  compartilha nenhum SHA de commit com o monorepo de origem; o `README.md` gerado está presente;
  publicar duas vezes seguidas produz dois commits distintos, sem erro de push; um arquivo que
  saiu de `PERMITIDOS` entre duas publicações some do Repo de Deploy na segunda (o script
  substitui o conteúdo, não acumula).
- Suíte completa: 442 passed (436 pré-existentes + 6 novos).

**Não verificado nesta rodada:** o build completo de ponta a ponta contra um `HF_TOKEN` real e um
`DEPLOY_REPO_URL` real — isso depende de segredos e de um repositório que só o dono do produto
pode criar (ver §4), e é também onde o ticket 08c (serviço no ar) recomeça o trabalho.

## 4. O que ficou pendente, e por quê é pendente por natureza

O **Repo de Deploy ainda não existe**. Nenhum agente de código deve criar um repositório GitHub em
nome do dono do produto sem confirmação — é uma ação com efeito em sistema compartilhado. Antes do
próximo deploy:

1. Criar um repositório Git privado vazio (ex.: `vetor-pas-api-deploy`).
2. `export DEPLOY_REPO_URL=git@github.com:<usuario>/vetor-pas-api-deploy.git`
3. `python deploy/publicar_space.py` — a primeira publicação nasce o repositório vazio.

O ticket 08c (`Blocked by: 08a, 08b` — agora ambos entregues em código) é quem cria o serviço no
Render a partir deste repositório, cadastra o Secret File e as variáveis, e verifica `/health` e o
Preditor num navegador de verdade.

## Glossário desta rodada

- **Repo de Deploy:** repositório Git dedicado que recebe só um snapshot curado do código, nunca a
  árvore nem a história do monorepo — gerado por script, nunca editado à mão.
- **Secret File (Render):** mecanismo do Render para segredos de build, montado em
  `/etc/secrets/<nome>` — o equivalente ao `--mount=type=secret` do BuildKit, com caminho fixo por
  plataforma.
- **Boot Frio:** ver glossário do ADR-0014 e do ticket 08 — a justificativa para o download dos
  artefatos continuar em build-time ficou mais forte no Render (hibernação a cada 15 min ocioso).
