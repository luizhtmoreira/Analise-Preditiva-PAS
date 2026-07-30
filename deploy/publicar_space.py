"""Cria (se preciso) e atualiza o Space privado do Hugging Face que hospeda a API — passo 3 da
promoção, ver `publicar_pacote.py`. Também é o comando a rodar depois de qualquer mudança em
`api/`, `src/pas_intelligence/` ou no Dockerfile.

Publica um **snapshot** dos arquivos permitidos direto pela API do Hub — nunca um `git push` do
monorepo inteiro. O Space é um repositório git à parte, e mandar a árvore deste repositório
mandaria junto a história que o ticket 15 está tentando conter (os SHAs órfãos com PII de
`feat/proof-section`). Cada rodada deste script sobe um commit novo no Space com o estado atual
dos arquivos da lista abaixo; a Hugging Face reconstrói a imagem sozinha a cada commit.

Uso:
    hf auth login          # uma vez, com um token de ESCRITA (o mesmo do publicar_pacote.py)
    python deploy/publicar_space.py
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token

RAIZ = Path(__file__).resolve().parent.parent
SPACE_ID = "Luiz1912/vetor-pas-api"

PERMITIDOS = [
    "Dockerfile",
    ".dockerignore",
    "requirements-api.txt",
    "api/*",
    "src/pas_intelligence/*",
    "deploy/ponteiro.json",
    "deploy/buscar_artefatos.py",
    "deploy/requirements.txt",
]
IGNORADOS = ["*__pycache__*", "*.pyc"]

# CORS_ALLOW_ORIGINS não é fixado no Dockerfile de propósito (ticket 03: vem do ambiente). É
# uma Variable do Space, escrita aqui para que o passo não dependa de clicar na UI.
CORS_PROD = "https://vetorpas.com.br,https://www.vetorpas.com.br"

README_SPACE = """---
title: Vetor PAS API
emoji: \U0001F4C8
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Backend FastAPI do Vetor PAS. Consumido pelo frontend em vetorpas.com.br — não é uma interface
para uso direto.
"""


def main() -> None:
    api = HfApi()
    create_repo(SPACE_ID, repo_type="space", space_sdk="docker", private=True, exist_ok=True)

    api.upload_file(
        path_or_fileobj=README_SPACE.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Atualiza README/metadados do Space",
    )

    commit = api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=str(RAIZ),
        allow_patterns=PERMITIDOS,
        ignore_patterns=IGNORADOS,
        commit_message="Publica snapshot do backend",
    )

    # O secret de build (`HF_TOKEN`) reusa o token com que você rodou `hf auth login` — o
    # mesmo que já tem leitura nos dois repositórios privados de artefato, porque foi quem os
    # criou. Ver deploy/README.md para trocar por um token de leitura mais restrito.
    token = get_token()
    if token:
        api.add_space_secret(SPACE_ID, "HF_TOKEN", token)
    api.add_space_variable(SPACE_ID, "CORS_ALLOW_ORIGINS", CORS_PROD)

    print(f"Space publicado: https://huggingface.co/spaces/{SPACE_ID} ({commit.oid})")
    print("A Hugging Face está reconstruindo a imagem agora. `/health` deve responder em alguns minutos.")


if __name__ == "__main__":
    main()
