"""Publica `models/pas3/` e/ou o Derivado de Deploy dos CSVs de `data/` nos repositórios
privados do Hugging Face Hub, e grava a revisão nova em `deploy/ponteiro.json`. Roda no disco de
quem promove — nunca em CI — porque `models/` e `data/*.csv` são gitignored (IP do produto / PII,
ver project_parser_privacy) e só existem localmente.

Isto é o passo 1 da promoção (ADR do ticket 03 do mapa treino-modelos-pas3, decisão 6):
    1. este script sobe o artefato e atualiza `ponteiro.json`           ← você está aqui
    2. `git add deploy/ponteiro.json && git commit && git push`         ← o commit de ponteiro
    3. `python deploy/publicar_space.py`                                ← propaga pro Repo de Deploy

Reverter é `git revert` do commit do passo 2, seguido de novo do passo 3 — nada é apagado do
repositório HF, a revisão antiga continua lá.

**`dados` publica o Derivado, não o cru.** `resultado_final.csv` e `notas_corte.csv` em disco têm
a coluna `nome`; o alvo `dados` primeiro reduz os dois às colunas de
`pas_intelligence.derivado_deploy.COLUNAS_DERIVADO` (fonte única, também lida pelos serviços que
consomem os CSVs em runtime) num diretório temporário, e só esse diretório sobe — nunca o cru. É o
que `ponteiro.json` aponta e o que `buscar_artefatos.py` baixa no build (ADR-0014).

**`cru` é o backup explícito, separado do Ponteiro.** Sobe `data/` inteiro (com `nome`) para
`CRU_REPO_ID` — um repositório que nenhuma etapa de build lê, porque não aparece em
`ponteiro.json` (`buscar_artefatos.py` baixa cada entrada de lá). Rode-o quando `data/` mudar,
pela mesma razão que a invariante dos parsers no `CLAUDE.md` versiona backup por decisão, não por
acidente. Sem tracking de revisão: é backup, não artefato consumido por versão exata.

Uso:
    hf auth login          # uma vez, com um token de ESCRITA
    python deploy/publicar_pacote.py modelo
    python deploy/publicar_pacote.py dados
    python deploy/publicar_pacote.py cru
    python deploy/publicar_pacote.py            # modelo + dados (default; cru é sempre explícito)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

RAIZ = Path(__file__).resolve().parent.parent
PONTEIRO_PATH = Path(__file__).parent / "ponteiro.json"

if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from pas_intelligence.derivado_deploy import build_derivado  # noqa: E402

# Onde cada artefato mora no disco de quem publica. Só isto — repo_id, arquivos e destino em
# runtime são fatos versionados, e vivem só em `ponteiro.json` (fonte única, também lida por
# `buscar_artefatos.py`).
ORIGEM_LOCAL = {
    "modelo": RAIZ / "models" / "pas3",
    "dados": RAIZ / "data",
}

# O cru privado: backup explícito de `data/` (com `nome`), separado do Derivado que o Ponteiro
# aponta. Nunca entra em `ponteiro.json` — se entrasse, `buscar_artefatos.py` o baixaria no
# build, violando a fronteira que este script existe para manter (ADR-0014).
CRU_REPO_ID = "Luiz1912/vetor-pas-dados"


def publicar(nome: str, ponteiro: dict, api: HfApi) -> None:
    alvo = ponteiro[nome]

    if nome == "dados":
        origem_crua = ORIGEM_LOCAL[nome]
        faltando = [a for a in alvo["arquivos"] if not (origem_crua / a).exists()]
        if faltando:
            sys.exit(f"{nome}: faltam {faltando} em {origem_crua} — nada para publicar.")

        with tempfile.TemporaryDirectory() as tmp:
            derivado_dir = Path(tmp)
            escritos = build_derivado(origem_crua, derivado_dir)
            for arquivo, (linhas, colunas) in escritos.items():
                print(f"  Derivado: {arquivo} → {linhas} linhas × {colunas} colunas (sem nome)")
            _publicar_pasta(nome, alvo, derivado_dir, api)
        return

    origem = ORIGEM_LOCAL[nome]
    faltando = [a for a in alvo["arquivos"] if not (origem / a).exists()]
    if faltando:
        sys.exit(f"{nome}: faltam {faltando} em {origem} — nada para publicar.")
    _publicar_pasta(nome, alvo, origem, api)


def _publicar_pasta(nome: str, alvo: dict, origem: Path, api: HfApi) -> None:
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


def publicar_cru(api: HfApi) -> None:
    origem = ORIGEM_LOCAL["dados"]
    arquivos = ["notas_corte.csv", "resultado_final.csv"]
    faltando = [a for a in arquivos if not (origem / a).exists()]
    if faltando:
        sys.exit(f"cru: faltam {faltando} em {origem} — nada para publicar.")

    create_repo(CRU_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=CRU_REPO_ID,
        repo_type="dataset",
        folder_path=str(origem),
        allow_patterns=arquivos,
        commit_message="Backup do cru (com nome) — nunca lido por build",
    )
    print(f"cru publicado em {CRU_REPO_ID}@{commit.oid} (backup; fora do Ponteiro, de propósito)")


def main() -> None:
    alvos = [a for a in sys.argv[1:] if a] or list(ORIGEM_LOCAL)
    desconhecidos = [a for a in alvos if a not in ORIGEM_LOCAL and a != "cru"]
    if desconhecidos:
        sys.exit(f"alvo desconhecido: {desconhecidos} (use {list(ORIGEM_LOCAL) + ['cru']})")

    api = HfApi()

    if "cru" in alvos:
        publicar_cru(api)
        alvos = [a for a in alvos if a != "cru"]
        if not alvos:
            return

    ponteiro = json.loads(PONTEIRO_PATH.read_text(encoding="utf-8"))
    for nome in alvos:
        publicar(nome, ponteiro, api)

    PONTEIRO_PATH.write_text(json.dumps(ponteiro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{PONTEIRO_PATH} atualizado. Revise `git diff` e commite — esse commit é a promoção.")


if __name__ == "__main__":
    main()
