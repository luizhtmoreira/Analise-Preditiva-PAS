# Deploy da API (ticket 08)

A API roda num [Hugging Face Space](https://huggingface.co/spaces) privado (Docker SDK, CPU
Basic — ADR-0004). O pacote de modelo (`models/pas3/`) e os dois CSVs que a API lê
(`data/notas_corte.csv`, `data/resultado_final.csv`) não viajam com o código: são assados na
imagem no build, buscados de dois repositórios privados no Hugging Face Hub, na revisão exata
gravada em [`ponteiro.json`](./ponteiro.json) — a Decisão 3/4 do
`.scratch/treino-modelos-pas3/issues/03-formato-e-versionamento-do-artefato.md`, construída
aqui pela primeira vez.

`assets/` (templates whitelabel) fica fora desta rodada — não entra na imagem.

## As três peças

| Repositório HF | Tipo | Conteúdo | Script que publica |
|---|---|---|---|
| `Luiz1912/vetor-pas3-modelo` | model | `modelo_pas3.txt` + `manifest.json` | `publicar_pacote.py modelo` |
| `Luiz1912/vetor-pas-dados` | dataset | `notas_corte.csv` + `resultado_final.csv` | `publicar_pacote.py dados` |
| `Luiz1912/vetor-pas-api` | space (Docker) | snapshot de `api/`, `src/pas_intelligence/`, `Dockerfile` | `publicar_space.py` |

Os dois primeiros existem só para os artefatos gitignored. O terceiro é o Space de verdade —
ele nunca recebe um `git push` deste monorepo (ver o porquê no topo de `publicar_space.py`): a
história deste repositório já passou por PII em commits órfãos (ticket 15), e replicá-la para um
terceiro host seria repetir o problema. Cada publicação sobe só os arquivos permitidos, direto
pela API do Hub.

## Setup (uma vez) e promoção (sempre que o modelo ou os CSVs mudam)

```bash
pip install -r deploy/requirements.txt   # huggingface_hub, só para estes scripts
hf auth login                            # uma vez, com um token de ESCRITA da conta Luiz1912

# 1. Sobe o(s) artefato(s) e grava a revisão nova em deploy/ponteiro.json
python deploy/publicar_pacote.py            # modelo + dados
# ou, para promover só um dos dois:
python deploy/publicar_pacote.py modelo
python deploy/publicar_pacote.py dados

# 2. O commit de ponteiro — isto é "promover" (decisão 6 do ticket 03 do mapa de treino)
git add deploy/ponteiro.json
git commit -m "deploy: promove <o que mudou>"
git push

# 3. Propaga pro Space — cria o Space na primeira vez; nas seguintes, só sobe o snapshot novo
# de código + a revisão nova do ponteiro, e a Hugging Face reconstrói a imagem sozinha.
python deploy/publicar_space.py
```

Rode o passo 3 de novo depois de qualquer mudança em `api/`, `src/pas_intelligence/` ou no
`Dockerfile`, mesmo sem trocar o ponteiro.

## Reverter

Nada é apagado dos repositórios de artefato — a revisão antiga continua lá.

```bash
git revert <sha-do-commit-de-ponteiro>
git push
python deploy/publicar_space.py
```

## Segredos e variáveis do Space

`publicar_space.py` já configura os dois automaticamente na primeira publicação:

- **Secret `HF_TOKEN`** (Settings → Repository secrets) — reusa o token com que você rodou
  `hf auth login`, porque é quem já tem leitura nos dois repositórios privados de artefato. Só
  é lido durante o build (`Dockerfile`, etapa `fetch`, via `--mount=type=secret`) — nunca fica
  em nenhuma camada da imagem final. Para restringir o escopo, crie um token
  [fine-grained](https://huggingface.co/settings/tokens) com leitura só nos dois repositórios
  de artefato e rode `huggingface_hub.HfApi().add_space_secret("Luiz1912/vetor-pas-api",
  "HF_TOKEN", "<esse token>")` uma vez para sobrescrever.
- **Variable `CORS_ALLOW_ORIGINS`** (Settings → Variables) = `https://vetorpas.com.br,https://www.vetorpas.com.br` —
  lida em runtime por `api/main.py` (ticket 03). Sem ela a API cai nos defaults de DEV
  (`localhost`) e o navegador do Aluno em produção recebe CORS.

## Apontar o frontend para a API hospedada

Depois que `/health` responder na URL pública do Space (copie a URL exata da página do Space —
o formato costuma ser `https://<usuário>-<nome-do-space>.hf.space`, normalizado em minúsculas):

- Vercel → Project Settings → Environment Variables (Production):
  - `NEXT_PUBLIC_API_URL` = a URL do Space — é o que o navegador usa (Preditor, Calculadora:
    `landing-page/lib/api.ts`, chamadas client-side).
  - `API_URL` = a mesma URL — é o que o Next.js server-side usa (Gestão de Ativos, Analytics).
- Redeploy manual da landing page (a integração Git da Vercel não está ativa —
  `project_deploy_vercel_manual`): `cd landing-page && vercel --prod`.

## Verificação — checklist do ticket

- [ ] `curl https://<url-do-space>/health` → `{"status":"ok"}`
- [ ] O Preditor, aberto no navegador contra a URL do Space (não localhost), devolve uma
      previsão sem erro de CORS no console
- [ ] Máquina limpa reproduz do zero:
  ```bash
  git clone <este-repo> /tmp/vetor-pas-clean && cd /tmp/vetor-pas-clean
  # confirma que não há cópia local de nada
  test -d models || test -d data || echo "sem models/ nem data/ — como esperado"
  export HF_TOKEN=$(hf auth token)   # token de LEITURA nos dois repos de artefato
  docker buildx build --secret id=HF_TOKEN,env=HF_TOKEN -t vetor-pas-api .
  docker run --rm -p 8000:7860 vetor-pas-api
  curl http://localhost:8000/health
  ```
  Sem `models/`, sem `data/`, sem cópia manual — só o clone e os dois comandos acima.
