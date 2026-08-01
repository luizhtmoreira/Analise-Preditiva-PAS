# Deploy da API (ticket 08, ADR-0014)

A API roda no [Render](https://render.com), plano gratuito, buildando um Dockerfile a partir do
**Repo de Deploy** (ticket 08b) — um repositório Git dedicado que recebe só um snapshot curado do
código, nunca a árvore nem a história deste monorepo. **O passo de criar o serviço no Render, o
domínio próprio e a verificação num navegador de verdade são o ticket 08c** — este documento cobre
o que já está pronto: os artefatos versionados e o publicador do Repo de Deploy.

O pacote de modelo (`models/pas3/`) e os dois CSVs que a API lê (`data/notas_corte.csv`,
`data/resultado_final.csv`) não viajam com o código: são assados na imagem no build, buscados de
repositórios privados no Hugging Face Hub, na revisão exata gravada em
[`ponteiro.json`](./ponteiro.json) — a Decisão 3/4 do
`.scratch/treino-modelos-pas3/issues/03-formato-e-versionamento-do-artefato.md`, construída
aqui pela primeira vez.

> **Pendente após o ticket 08a:** `ponteiro.json` da chave `dados` já aponta para o repositório
> novo do Derivado (`vetor-pas-dados-derivado`), mas com `"revision": ""` — ninguém rodou o
> publicador contra um token real ainda. **Não faça deploy até rodar isto uma vez:**
> `python deploy/publicar_pacote.py cru` (backup do cru) seguido de
> `python deploy/publicar_pacote.py dados` (publica o Derivado e grava a revisão), depois commite
> o `ponteiro.json` atualizado. Sem isso, `buscar_artefatos.py` falha no build com "ponteiro.json
> não tem revisão para 'dados'" — mensagem clara, não crash genérico, mas ainda um bloqueio real.

`assets/` (templates whitelabel) fica fora desta rodada — não entra na imagem.

## As quatro peças

| Repositório HF | Tipo | Conteúdo | Script que publica |
|---|---|---|---|
| `Luiz1912/vetor-pas3-modelo` | model | `modelo_pas3.txt` + `manifest.json` | `publicar_pacote.py modelo` |
| `Luiz1912/vetor-pas-dados-derivado` | dataset | `notas_corte.csv` + `resultado_final.csv` **sem `nome`**, reduzidos às colunas de `pas_intelligence.derivado_deploy.COLUNAS_DERIVADO` — o que o Ponteiro aponta e o build baixa | `publicar_pacote.py dados` |
| `Luiz1912/vetor-pas-dados` | dataset | os mesmos dois CSVs **crus** (com `nome`) — backup explícito, fora do Ponteiro, nenhuma etapa de build o lê | `publicar_pacote.py cru` |
| Repo de Deploy (Git, nome a escolher) | repositório Git dedicado | snapshot de `api/`, `src/pas_intelligence/`, `Dockerfile` e o resto de `PERMITIDOS` | `publicar_space.py` |

Os três primeiros existem só para os artefatos gitignored. O quarto é o repositório que o Render
de fato clona e builda — e ele nunca recebe um `git push` **deste** monorepo: a história deste
repositório já passou por PII em commits órfãos (ticket 15), e replicá-la para um terceiro host
seria repetir o problema. `publicar_space.py` (ver o docstring no topo do arquivo) clona o estado
atual do Repo de Deploy, substitui seu conteúdo pelos arquivos de `PERMITIDOS` e sobe um commit de
snapshot novo — nunca reescreve a história que já está lá, então publicar várias vezes nunca gera
conflito.

**Por que dois repositórios de dados, e não um.** `Luiz1912/vetor-pas-dados` acumulava dois
papéis — backup dos CSVs de extração e insumo de deploy. Cortar `nome` na origem destruiria o
backup, então o Derivado passou a ter um lar próprio (ADR-0014, ticket 08a). `cru` nunca aparece
em `ponteiro.json`: se aparecesse, `buscar_artefatos.py` o baixaria no build — exatamente o que
este desenho existe para impedir. Rode `publicar_pacote.py cru` quando `data/` mudar; ele não faz
parte do alvo default (`publicar_pacote.py` sem argumento sobe só `modelo` + `dados`).

## Setup (uma vez) e promoção (sempre que o modelo ou os CSVs mudam)

```bash
pip install -r deploy/requirements.txt   # huggingface_hub + pandas, só para estes scripts
hf auth login                            # uma vez, com um token de ESCRITA da conta Luiz1912

# 0. Uma vez, ou sempre que data/ mudar: backup do cru (fora do Ponteiro, nunca lido por build)
python deploy/publicar_pacote.py cru

# 1. Sobe o(s) artefato(s) — "dados" primeiro reduz aos CSVs sem `nome` (o Derivado) — e grava a
# revisão nova em deploy/ponteiro.json
python deploy/publicar_pacote.py            # modelo + dados
# ou, para promover só um dos dois:
python deploy/publicar_pacote.py modelo
python deploy/publicar_pacote.py dados

# 2. O commit de ponteiro — isto é "promover" (decisão 6 do ticket 03 do mapa de treino)
git add deploy/ponteiro.json
git commit -m "deploy: promove <o que mudou>"
git push

# 3. Propaga pro Repo de Deploy — nasce vazio na primeira vez; nas seguintes, sobe um commit de
# snapshot novo com o código + a revisão nova do ponteiro. O Render observa este repositório e
# reconstrói a imagem sozinho a cada push (setup do serviço: ticket 08c).
export DEPLOY_REPO_URL=git@github.com:<usuario>/vetor-pas-api-deploy.git   # uma vez
python deploy/publicar_space.py
```

Rode o passo 3 de novo depois de qualquer mudança em `api/`, `src/pas_intelligence/` ou no
`Dockerfile`, mesmo sem trocar o ponteiro.

## Reverter

Nada é apagado dos repositórios de artefato — a revisão antiga continua lá. O Repo de Deploy
também guarda todos os commits de snapshot anteriores (reverter lá é o `git revert` de sempre, se
for preciso; normalmente basta reverter o ponteiro e publicar de novo).

```bash
git revert <sha-do-commit-de-ponteiro>
git push
python deploy/publicar_space.py
```

## Secret de build e variáveis de ambiente

Cadastro é manual no dashboard do Render (ticket 08c cobre o passo a passo completo de criar o
serviço); o que muda em código está aqui:

- **Secret File `HF_TOKEN`** — o Dockerfile lê o secret de build de onde o Render monta Secret
  Files (`/etc/secrets/HF_TOKEN`, disponível em todos os planos, inclusive o gratuito — ver o
  comentário na etapa `fetch` do `Dockerfile`). Use um token de leitura nos dois repositórios
  privados de artefato — não precisa ser o mesmo de escrita usado para publicar.
- **Environment Variable `CORS_ALLOW_ORIGINS`** = `https://vetorpas.com.br,https://www.vetorpas.com.br` —
  lida em runtime por `api/main.py` (ticket 03). Sem ela a API cai nos defaults de DEV
  (`localhost`) e o navegador do Aluno em produção recebe CORS.

## Apontar o frontend para a API hospedada

O endereço final é `api.vetorpas.com.br` (nunca um domínio `onrender.com` — ver ADR-0014) e o
CNAME, a criação do serviço no Render e a verificação num navegador de verdade são o ticket 08c.
Depois disso, na Vercel → Project Settings → Environment Variables (Production):

- `NEXT_PUBLIC_API_URL` = `https://api.vetorpas.com.br` — é o que o navegador usa (Preditor,
  Calculadora: `landing-page/lib/api.ts`, chamadas client-side).
- `API_URL` = a mesma URL — é o que o Next.js server-side usa (Gestão de Ativos, Analytics).

Redeploy manual da landing page depois (a integração Git da Vercel não está ativa —
`project_deploy_vercel_manual`): `cd landing-page && vercel --prod`.

## Verificação — checklist do ticket 08b

- [ ] Máquina limpa reproduz o build do zero:
  ```bash
  git clone <este-repo> /tmp/vetor-pas-clean && cd /tmp/vetor-pas-clean
  # confirma que não há cópia local de nada
  test -d models || test -d data || echo "sem models/ nem data/ — como esperado"
  export HF_TOKEN=$(hf auth token)   # token de LEITURA nos dois repos de artefato
  docker buildx build --secret id=HF_TOKEN,src=<(echo -n "$HF_TOKEN") -t vetor-pas-api .
  docker run --rm -p 8000:10000 vetor-pas-api
  curl http://localhost:8000/health
  # sem PORT: escuta em 10000 (default do Render). Com -e PORT=8080 -p 8000:8080, escuta em 8080.
  ```
  Sem `models/`, sem `data/`, sem cópia manual — só o clone e os comandos acima.
- [ ] `/health` no ar contra o serviço real e o Preditor num navegador de verdade —
      critério do ticket 08c, não deste.
