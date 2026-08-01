"""Busca o pacote de modelo e os CSVs no Hugging Face Hub — roda só na etapa `fetch` do
Dockerfile (ticket 08), nunca no boot do serviço.

Lê `ponteiro.json` (repo_id + revisão exata de cada artefato) e baixa os arquivos daquela
revisão para o diretório de destino, na mesma estrutura que `model_package.DIRETORIO_PADRAO` e
`api/services/gestao_service.py` esperam em runtime (`models/pas3/`, `data/`).

`ponteiro.json` é o "commit de ponteiro" do ADR do ticket 03 (mapa treino-modelos-pas3,
decisão 6): promover um modelo novo é escrever a revisão nova aqui e commitar; reverter é
`git revert` desse commit. `deploy/publicar_pacote.py` é quem escreve a revisão depois de subir
um pacote novo.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

AQUI = Path(__file__).parent
RAIZ_DESTINO = Path(os.environ.get("DESTINO_ARTEFATOS", "/fetch"))


def _token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN ausente. No Render isto vem do Secret File de build (montado em "
            "/etc/secrets/HF_TOKEN); localmente, exporte um token de leitura antes de "
            "`docker build` (ver deploy/README.md)."
        )
    return token


def _baixar(nome: str, alvo: dict, token: str) -> None:
    if not alvo.get("revision"):
        sys.exit(
            f"ponteiro.json não tem revisão para {nome!r} — rode "
            "`python deploy/publicar_pacote.py` antes do primeiro deploy."
        )

    destino = RAIZ_DESTINO / alvo["destino"]
    destino.mkdir(parents=True, exist_ok=True)

    for arquivo in alvo["arquivos"]:
        hf_hub_download(
            repo_id=alvo["repo_id"],
            repo_type=alvo["repo_type"],
            revision=alvo["revision"],
            filename=arquivo,
            token=token,
            local_dir=destino,
        )
    # `local_dir` também grava metadado interno do huggingface_hub junto dos arquivos pedidos;
    # a imagem final só quer os arquivos do artefato.
    shutil.rmtree(destino / ".cache", ignore_errors=True)
    print(f"{nome}: {alvo['repo_id']}@{alvo['revision']} → {destino}")


def main() -> None:
    ponteiro = json.loads((AQUI / "ponteiro.json").read_text(encoding="utf-8"))
    token = _token()
    for nome, alvo in ponteiro.items():
        _baixar(nome, alvo, token)


if __name__ == "__main__":
    main()
