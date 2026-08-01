# syntax=docker/dockerfile:1
#
# API hospedada do Vetor PAS (ticket 08 do mapa `publicar-site`), buildada pelo Render a partir do
# Repo de Deploy (ADR-0014). Duas etapas:
#
#   1. `fetch`   — busca `models/pas3/` (modelo + manifesto) e os dois CSVs de `data/` no
#                  domicílio versionado do Hugging Face Hub (Decisão 3/4 do ticket 03 do mapa
#                  `treino-modelos-pas3`), na revisão exata gravada em `deploy/ponteiro.json`.
#                  Roda só no build — nunca no boot. Isso vale ainda mais no Render que valia no
#                  Hugging Face Spaces: lá hibernar era raro (48h), aqui é rotina (15 min
#                  ociosos), então um boot dependente de rede transformaria o Hugging Face em
#                  dependência de cada Boot Frio.
#   2. runtime   — imagem final: só o código e as libs que `api/` de fato importa
#                  (`requirements-api.txt`), mais o que a etapa `fetch` baixou.
#
# `assets/` (templates whitelabel) não entra aqui — fora de escopo desta rodada.

FROM python:3.14-slim AS fetch

WORKDIR /fetch
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/ponteiro.json deploy/buscar_artefatos.py ./

# O secret de build `HF_TOKEN` vem de um Secret File do Render, montado em runtime de build sob
# /etc/secrets/<nome> (docs: Using Secrets with Docker) — disponível em todos os planos,
# inclusive o gratuito. Só é lido dentro desta RUN — não vaza para nenhuma camada da imagem
# final. Ver deploy/README.md para como cadastrá-lo.
RUN --mount=type=secret,id=HF_TOKEN,dst=/etc/secrets/HF_TOKEN,mode=0444,required=true \
    HF_TOKEN="$(cat /etc/secrets/HF_TOKEN)" python buscar_artefatos.py


FROM python:3.14-slim

# `lightgbm` carrega sua libgomp.so em runtime (OpenMP) — ausente na imagem slim e sem ela
# `import lightgbm` derruba o startup com OSError, não um erro óbvio de "faltou pacote Python".
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Nasceu como convenção do Hugging Face Docker Spaces; fica por mérito próprio no Render — não
# rodar como root é boa prática, não uma exigência de plataforma.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user api/ ./api/
COPY --chown=user src/pas_intelligence/ ./src/pas_intelligence/
COPY --chown=user --from=fetch /fetch/models ./models
COPY --chown=user --from=fetch /fetch/data ./data

# `CORS_ALLOW_ORIGINS` não é fixado aqui de propósito (ticket 03: vem do ambiente). É uma
# Environment Variable do Render, não um valor cravado na imagem — ver deploy/README.md.
EXPOSE 10000

# O Render injeta $PORT (default 10000) e espera o processo escutar nela — não é uma porta fixa
# como no Hugging Face Spaces. Forma shell (via `sh -c`) porque o CMD exec puro não expande
# variáveis de ambiente; `${PORT:-10000}` cobre também `docker run` local, sem a variável.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
