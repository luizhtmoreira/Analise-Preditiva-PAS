"""Publica um snapshot do backend no Repo de Deploy que o Render builda — passo 3 da promoção,
ver `publicar_pacote.py`. Também é o comando a rodar depois de qualquer mudança em `api/`,
`src/pas_intelligence/` ou no Dockerfile.

O Repo de Deploy nasce vazio e nunca compartilha história com este monorepo: a árvore deste
repositório já teve PII em commits órfãos (ticket 15), e o force-push não despublica — mandar a
história junto para um terceiro host reabriria o mesmo problema num lugar novo. Este script clona
o estado atual do Repo de Deploy, substitui seu conteúdo pelos arquivos de `PERMITIDOS` e sobe um
commit novo — nunca reescreve a história que já está lá.

O Repo de Deploy nunca é editado à mão: um `README.md` de uma linha, escrito por este script a
cada publicação, diz isso — mesma razão pela qual `.scratch/parser-backup/` carrega a instrução
dele no `CLAUDE.md`: dois lugares com o mesmo código convidam alguém a editar o errado.

Uso:
    export DEPLOY_REPO_URL=git@github.com:<usuario>/vetor-pas-api-deploy.git   # uma vez
    python deploy/publicar_space.py
"""
from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Os arquivos que o Dockerfile de fato usa — a mesma lista testada e revisada desde o ticket 08,
# que não muda com a troca de destino (HF Space → Repo de Deploy).
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

README_REPO_DEPLOY = (
    "Gerado por `deploy/publicar_space.py` a cada publicação — não edite este repositório à mão, "
    "edite o monorepo e publique de novo.\n"
)

AUTOR_COMMIT = ("Vetor PAS Deploy Bot", "deploy@vetorpas.com.br")


def _coletar_arquivos(raiz: Path, permitidos: list[str], ignorados: list[str]) -> list[Path]:
    """Todo arquivo sob `raiz` cujo caminho relativo casa com `permitidos` e não com `ignorados`."""
    arquivos = []
    for caminho in sorted(raiz.rglob("*")):
        if not caminho.is_file():
            continue
        relativo = caminho.relative_to(raiz).as_posix()
        if any(fnmatch.fnmatch(relativo, padrao) for padrao in ignorados):
            continue
        if any(fnmatch.fnmatch(relativo, padrao) for padrao in permitidos):
            arquivos.append(caminho)
    return arquivos


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _preparar_clone(repo_url: str, destino: Path) -> None:
    """Clona o Repo de Deploy — funciona tanto num repositório que já tem commits quanto num que
    nasceu vazio (git não falha ao clonar um repositório remoto sem nenhum commit).

    Um `git clone` simples não é suficiente: o HEAD simbólico de um repositório recém-criado nem
    sempre aponta para `main` (varia por host, e um `git init --bare` local nunca aponta), e nesse
    caso o clone não faz checkout de nada — `git checkout -B main` partiria então de um HEAD
    "unborn" e criaria um branch **órfão**, perdendo a história já publicada. Por isso o ponto de
    partida do branch local é decidido explicitamente a partir de `origin/main`, não do HEAD do
    clone.
    """
    subprocess.run(["git", "clone", repo_url, str(destino)], check=True, capture_output=True, text=True)
    _git("config", "user.name", AUTOR_COMMIT[0], cwd=destino)
    _git("config", "user.email", AUTOR_COMMIT[1], cwd=destino)

    tem_main_remoto = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main"], cwd=destino, capture_output=True, text=True
    ).returncode == 0
    if tem_main_remoto:
        _git("checkout", "-B", "main", "origin/main", cwd=destino)
    else:
        _git("checkout", "-B", "main", cwd=destino)


def _substituir_conteudo(destino: Path, raiz: Path, arquivos: list[Path]) -> None:
    for item in destino.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for arquivo in arquivos:
        relativo = arquivo.relative_to(raiz)
        alvo = destino / relativo
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arquivo, alvo)

    (destino / "README.md").write_text(README_REPO_DEPLOY, encoding="utf-8")


def publicar_snapshot(raiz: Path, repo_url: str) -> str:
    """Publica um snapshot de `PERMITIDOS` no Repo de Deploy em `repo_url`. Devolve o SHA do
    commit de snapshot. Sempre parte de um clone fresco do estado atual do remoto, então o push
    é sempre fast-forward — publicar duas vezes seguidas nunca produz conflito."""
    arquivos = _coletar_arquivos(raiz, PERMITIDOS, IGNORADOS)
    if not arquivos:
        sys.exit("nenhum arquivo casou com PERMITIDOS — nada para publicar.")

    with tempfile.TemporaryDirectory() as tmp:
        destino = Path(tmp) / "repo-deploy"
        _preparar_clone(repo_url, destino)
        _substituir_conteudo(destino, raiz, arquivos)

        _git("add", "-A", cwd=destino)
        # `--allow-empty`: mesmo sem diferença de conteúdo, cada publicação grava um commit de
        # snapshot — é o contrato do ticket, não uma otimização de "só commita se mudou".
        _git("commit", "--allow-empty", "-m", "Publica snapshot do backend", cwd=destino)
        _git("push", "origin", "main", cwd=destino)

        return _git("rev-parse", "HEAD", cwd=destino).stdout.strip()


def main() -> None:
    repo_url = os.environ.get("DEPLOY_REPO_URL")
    if not repo_url:
        sys.exit(
            "DEPLOY_REPO_URL ausente. Exporte a URL do Repo de Deploy (ex.: "
            "git@github.com:<usuario>/vetor-pas-api-deploy.git) antes de publicar — ver "
            "deploy/README.md."
        )

    sha = publicar_snapshot(RAIZ, repo_url)
    print(f"Repo de Deploy publicado: {repo_url} @ {sha}")
    print("O Render builda a imagem a partir deste commit — confira o deploy no dashboard.")


if __name__ == "__main__":
    main()
