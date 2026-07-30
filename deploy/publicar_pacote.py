"""Publica `models/pas3/` e/ou os CSVs de `data/` nos repositórios privados do Hugging Face
Hub, e grava a revisão nova em `deploy/ponteiro.json`. Roda no disco de quem promove — nunca em
CI — porque `models/` e `data/*.csv` são gitignored (IP do produto / PII, ver
project_parser_privacy) e só existem localmente.

Isto é o passo 1 da promoção (ADR do ticket 03 do mapa treino-modelos-pas3, decisão 6):
    1. este script sobe o artefato e atualiza `ponteiro.json`           ← você está aqui
    2. `git add deploy/ponteiro.json && git commit && git push`         ← o commit de ponteiro
    3. `python deploy/publicar_space.py`                                ← propaga pro Space

Reverter é `git revert` do commit do passo 2, seguido de novo do passo 3 — nada é apagado do
repositório HF, a revisão antiga continua lá.

Uso:
    hf auth login          # uma vez, com um token de ESCRITA
    python deploy/publicar_pacote.py modelo
    python deploy/publicar_pacote.py dados
    python deploy/publicar_pacote.py            # os dois (default)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

RAIZ = Path(__file__).resolve().parent.parent
PONTEIRO_PATH = Path(__file__).parent / "ponteiro.json"

# Onde cada artefato mora no disco de quem publica. Só isto — repo_id, arquivos e destino em
# runtime são fatos versionados, e vivem só em `ponteiro.json` (fonte única, também lida por
# `buscar_artefatos.py`).
ORIGEM_LOCAL = {
    "modelo": RAIZ / "models" / "pas3",
    "dados": RAIZ / "data",
}


def publicar(nome: str, ponteiro: dict, api: HfApi) -> None:
    alvo = ponteiro[nome]
    origem = ORIGEM_LOCAL[nome]

    faltando = [a for a in alvo["arquivos"] if not (origem / a).exists()]
    if faltando:
        sys.exit(f"{nome}: faltam {faltando} em {origem} — nada para publicar.")

    create_repo(alvo["repo_id"], repo_type=alvo["repo_type"], private=True, exist_ok=True)

    commit = api.upload_folder(
        repo_id=alvo["repo_id"],
        repo_type=alvo["repo_type"],
        folder_path=str(origem),
        allow_patterns=alvo["arquivos"],
        commit_message=f"Publica {nome} ({', '.join(alvo['arquivos'])})",
    )

    alvo["revision"] = commit.oid
    print(f"{nome} publicado em {alvo['repo_id']}@{commit.oid}")


def main() -> None:
    alvos = [a for a in sys.argv[1:] if a] or list(ORIGEM_LOCAL)
    desconhecidos = [a for a in alvos if a not in ORIGEM_LOCAL]
    if desconhecidos:
        sys.exit(f"alvo desconhecido: {desconhecidos} (use {list(ORIGEM_LOCAL)})")

    ponteiro = json.loads(PONTEIRO_PATH.read_text(encoding="utf-8"))
    api = HfApi()
    for nome in alvos:
        publicar(nome, ponteiro, api)

    PONTEIRO_PATH.write_text(json.dumps(ponteiro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{PONTEIRO_PATH} atualizado. Revise `git diff` e commite — esse commit é a promoção.")


if __name__ == "__main__":
    main()
