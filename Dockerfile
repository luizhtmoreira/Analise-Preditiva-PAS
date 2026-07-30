# syntax=docker/dockerfile:1
#
# API hospedada do Vetor PAS (ticket 08 do mapa `publicar-site`). Duas etapas:
#
#   1. `fetch`   — busca `models/pas3/` (modelo + manifesto) e os dois CSVs de `data/` no
#                  domicílio versionado do Hugging Face Hub (Decisão 3/4 do ticket 03 do mapa
#                  `treino-modelos-pas3`), na revisão exata gravada em `deploy/ponteiro.json`.
#                  Roda só no build — nunca no boot, porque o Space hiberna em 48h e o boot não
#                  pode depender de rede.
#   2. runtime   — imagem final: só o código e as libs que `api/` de fato importa
#                  (`requirements-api.txt`), mais o que a etapa `fetch` baixou.
#
# `assets/` (templates whitelabel) não entra aqui — fora de escopo desta rodada.

FROM python:3.14-slim AS fetch

WORKDIR /fetch
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/ponteiro.json deploy/buscar_artefatos.py ./

# O secret `HF_TOKEN` vem do Space (Settings → Repository secrets) e é montado só nesta RUN —
# não vaza para nenhuma camada da imagem final. Ver deploy/README.md para como criá-lo.
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" python buscar_artefatos.py


FROM python:3.14-slim

# `lightgbm` carrega sua libgomp.so em runtime (OpenMP) — ausente na imagem slim e sem ela
# `import lightgbm` derruba o startup com OSError, não um erro óbvio de "faltou pacote Python".
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Convenção do Hugging Face Docker Spaces: o container roda como uid 1000, não root.
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
# Variable do Space, não um valor cravado na imagem — ver deploy/README.md.
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
