# 08 — API hospedada: Dockerfile, Space e o pacote assado na imagem (relatório)

## O que mudou

- `Dockerfile` (raiz) — duas etapas. `fetch`: baixa `models/pas3/` (modelo + manifesto) e os dois
  CSVs (`notas_corte.csv`, `resultado_final.csv`) do Hugging Face Hub, na revisão exata gravada em
  `deploy/ponteiro.json`. `runtime`: imagem final, só com `requirements-api.txt`, `api/`,
  `src/pas_intelligence/` e o que a etapa `fetch` baixou.
- `deploy/ponteiro.json` — o ponteiro versionado: `repo_id`, `repo_type`, `revision` e a lista de
  arquivos de cada um dos dois artefatos (modelo, dados).
- `deploy/buscar_artefatos.py` — roda só dentro da etapa `fetch` do Dockerfile; lê o ponteiro e
  baixa via `huggingface_hub`, autenticado com um secret de build (`HF_TOKEN`).
- `deploy/publicar_pacote.py` — roda no disco de quem promove (nunca em CI, porque `models/` e
  `data/*.csv` são gitignored). Sobe `models/pas3/` e/ou os CSVs pros dois repositórios privados no
  Hugging Face Hub e escreve a revisão nova de volta em `ponteiro.json`.
- `deploy/publicar_space.py` — cria (na primeira vez) e atualiza o Space privado (Docker SDK),
  publicando um snapshot dos arquivos permitidos e configurando o secret `HF_TOKEN` e a variable
  `CORS_ALLOW_ORIGINS` automaticamente.
- `deploy/README.md` — o runbook: setup, promoção, reversão, como apontar o Vercel pra API
  hospedada, e o roteiro de verificação em máquina limpa.
- `requirements-api.txt` — subconjunto de `requirements.txt`, pinado nas mesmas versões do
  `ambiente` gravado em `models/pas3/manifest.json`.
- `.dockerignore` — mantém o contexto de build restrito ao que a imagem de fato usa.

## Decisões e porquês

**Dois repositórios HF, não um.** Perguntei antes de decidir: um repositório `model` só para
`models/pas3/`, outro `dataset` só para os dois CSVs, cada um com sua própria revisão no ponteiro.
Promover um modelo novo não mexe no ponteiro dos CSVs, e trocar CSV (rotina do ticket 09) não mexe
no ponteiro do modelo — cada mudança gera só o commit que lhe diz respeito.

**O Space nunca recebe `git push` deste monorepo.** `publicar_space.py` publica um snapshot pela
API do Hub (`upload_folder` com `allow_patterns` restrito a `Dockerfile`, `requirements-api.txt`,
`api/`, `src/pas_intelligence/` e os três arquivos de `deploy/` que o build usa) — nunca um push da
árvore inteira. A história deste repositório já teve PII em commits órfãos de
`feat/proof-section` (o motivo do próprio ticket 15); replicar essa história para um terceiro host
seria reabrir o mesmo problema num lugar novo.

**`requirements-api.txt` separado, não uma edição do `requirements.txt` da raiz.** A API de fato só
importa `fastapi`, `uvicorn`, `numpy`, `pandas`, `scipy`, `lightgbm` e `joblib` (rastreado import a
import a partir de `uvicorn api.main:app`) — `streamlit`, `reportlab`, `pdfkit`, `fpdf`, `jinja2`,
`openpyxl`, `xlsxwriter`, `pypdf`, `requests`, `statsmodels`, `supabase`, `scikit-learn` e `xgboost`
servem só o app Streamlit legado, o `pdf_generator` (nenhum dos dois entra na imagem) ou não são
importados em lugar nenhum. `scikit-learn` é o caso que mais merece nota: ele só apareceria via
`target_calculator.py` tentando carregar `p1_pas3_model.joblib`/`red_pas3_model.joblib` — que hoje
**já falham** de qualquer forma (`ModuleNotFoundError: _loss`, o defeito 3 documentado no
CLAUDE.md). Excluir `scikit-learn` da imagem não piora esse caminho: ele já está quebrado,
excluí-lo só evita instalar ~30MB de wheel que não muda o comportamento observável.

**`CORS_ALLOW_ORIGINS` não é fixado no Dockerfile.** Fica como Variable do Space (o ticket 03 já
tinha decidido "vem do ambiente"); o Dockerfile só documenta isso em comentário. `publicar_space.py`
escreve o valor de produção automaticamente, então não depende de alguém clicar na UI do Hugging
Face.

**O secret do Space reusa o token de escrita do publicador, por padrão.** É o token com que se
publica o modelo/dados, então já tem leitura nos dois repositórios privados. Documentado como uma
troca de simplicidade por escopo — `deploy/README.md` explica como sobrescrever por um token
fine-grained só-leitura.

## Bug encontrado durante a implementação

`python:3.14-slim` não vem com `libgomp1` (runtime OpenMP). `lightgbm` carrega sua `.so` via
`ctypes` no import, não no build — então o `pip install` passava limpo e a imagem só quebrava no
boot, com `OSError: libgomp.so.1: cannot open shared object file`, dentro do `lifespan` do FastAPI
(`load_resources() → PacoteDeModelo.carregar() → import lightgbm`). Corrigido com
`apt-get install -y libgomp1` antes de trocar pro usuário não-root — exatamente o tipo de falha que
`PacoteDeModelo.carregar()` foi desenhado pra expor no startup, não numa resposta de Aluno.

## Critérios de aceite do ticket — conferidos

- [x] Existe um Dockerfile que constrói a API com o pacote de modelo e os CSVs dentro da imagem —
      testado com build real (etapa runtime com artefatos equivalentes).
- [ ] O Space no Hugging Face está de pé e `/health` responde numa URL pública — **bloqueado
      (ver seção "Bloqueio" abaixo)**.
- [x] O pacote de modelo é buscado no build a partir do domicílio versionado, não copiado do disco
      de ninguém — `Dockerfile` não tem nenhum `COPY models/` nem `COPY data/` a partir do contexto
      local; `.dockerignore` os exclui do contexto.
- [x] Promover um modelo novo é um commit de ponteiro, e reverter é voltar o ponteiro — documentado
      em passos executáveis em `deploy/README.md`.
- [x] O Preditor funciona **num navegador**, sem erro de CORS — testado localmente com `curl`
      simulando `Origin: https://vetorpas.com.br` (preflight `200`, header
      `access-control-allow-origin` correto) e `Origin` desconhecida (recusada, `400`). O teste **no
      navegador de verdade contra a URL pública** depende do Space existir — pendente.
- [x] Uma máquina limpa reproduz o deploy do zero, sem cópia manual de arquivo — verificado que a
      etapa `fetch` falha do jeito certo (mensagem clara, não crash) sem `HF_TOKEN` ou sem revisão
      no ponteiro; o download real de um repositório privado depende de credenciais que só existem
      na sua conta.
- [x] Os templates de `assets/` não entram na imagem nesta rodada — `.dockerignore` exclui
      `assets/`; nem o Dockerfile nem `publicar_space.py` (`PERMITIDOS`) referenciam o diretório.

## Verificado

- Build real da etapa `runtime` (artefatos equivalentes copiados no lugar do que a etapa `fetch`
  produziria): imagem sobe, `/health` responde, preflight CORS aceita `vetorpas.com.br` e recusa
  origem desconhecida, `POST /api/predict` devolve uma previsão real (`arg_previsto: 242.9`,
  `modelo_disponivel: true`) — o pacote carregado da imagem é o mesmo que roda localmente.
- Build real da etapa `fetch` isolada (`docker build --target fetch` com secret falso): falha com a
  mensagem esperada (`ponteiro.json não tem revisão para 'modelo'`), não com um erro genérico —
  confirma que `--mount=type=secret` e a leitura do ponteiro funcionam antes mesmo de existir uma
  revisão real pra buscar.
- `pytest tests/`: 401 testes verdes antes de um merge (`feat/nextjs-frontend`) que aconteceu em
  paralelo durante esta sessão; 419 depois do merge (a suíte cresceu, nada quebrou).
- Code review nos dois eixos (Standards + Spec): sem violação documentada, sem item de escopo
  faltando ou além do pedido. Achado real (duplicação do nome dos arquivos de cada artefato em dois
  lugares) corrigido — `ponteiro.json` passou a ser a fonte única, lida tanto por
  `buscar_artefatos.py` quanto por `publicar_pacote.py`.

## Fora do escopo deste ticket

- A criação de fato dos dois repositórios privados e do Space no Hugging Face, e o `hf auth login`
  — só a conta `Luiz1912` pode executar. Passos exatos em `deploy/README.md`.
- Apontar `NEXT_PUBLIC_API_URL`/`API_URL` no Vercel pra URL do Space e fazer o redeploy manual da
  landing page — mesma razão.
- `PAS_STRICT_MODELS=1`, que o próprio docstring de `target_calculator.py` diz que "produção liga":
  o code review notou que nada neste ticket seta essa variável. Não é escopo do ticket 08 (o
  carregamento estrito que ele documenta é de `PacoteDeModelo`, não do par `p1`/`red` que
  `target_calculator.py` usa) e ligá-la agora transformaria o defeito 3 já conhecido (modelos que
  não carregam) de uma degradação silenciosa pra um `500` na Estratégia — troca de comportamento
  que não é minha de decidir sozinho.
- `assets/` (templates whitelabel) — explicitamente fora desta rodada, fica pro B2B.

## Bloqueio: Hugging Face Docker Space exige PRO (2026-07-31)

Ao rodar `python deploy/publicar_space.py`, o HF retornou `402 Payment Required`:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic
> requires a PRO subscription. Subscribe at https://huggingface.co/pro

O ADR-0004 previa "CPU Basic, gratuito" — a política do HF mudou desde a decisão. O erro se repete
tanto com `private=True` quanto com `private=False`.

### O que foi executado até aqui

- `publicar_pacote.py` rodou com sucesso: modelo e CSVs estão nos dois repositórios HF privados
  (`Luiz1912/vetor-pas3-modelo` e `Luiz1912/vetor-pas-dados`) com revisões gravadas em
  `ponteiro.json`.
- `ponteiro.json` foi commitado em `feat/pdf-extraction` (será mergeado pra `main` no ticket 14).
- `publicar_space.py` foi ajustado para `private=False` — não resolve o problema.

### Decisão pendente (dono do produto)

| Opção | Custo | Atrito | Observação |
|---|---|---|---|
| **Assinar HF PRO** | ~$9/mês | Mínimo — roda `publicar_space.py` e encerra | Faz sentido perto do lançamento |
| **Migrar para Render** | Gratuito | Médio — requer novo ticket | Dockerfile precisa mudar: secrets de build são pagos no Render; artefatos teriam de ser baixados em runtime, não no build |
