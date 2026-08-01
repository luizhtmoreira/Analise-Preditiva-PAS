"""O Repo de Deploy é o que o Render clona para buildar a imagem (ticket 08b, ADR-0014): recebe
só o snapshot de `PERMITIDOS`, nunca a árvore nem a história do monorepo, e nunca é editado à
mão. Estes testes provam isso sobre um repositório Git de verdade (um bare local fazendo o papel
do remoto) — não sobre a intenção do script.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent / "deploy"))

from publicar_space import PERMITIDOS, publicar_snapshot  # noqa: E402


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo_deploy_vazio(tmp_path: Path) -> Path:
    """Um Repo de Deploy que nasce vazio, como o real na primeira publicação."""
    bare = tmp_path / "repo-deploy.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    return bare


@pytest.fixture
def monorepo_falso(tmp_path: Path) -> Path:
    """Uma árvore com a mesma forma do monorepo real: alguns arquivos em PERMITIDOS, alguns fora,
    e uma história git própria (para provar que ela não viaja para o Repo de Deploy)."""
    raiz = tmp_path / "monorepo"
    (raiz / "api").mkdir(parents=True)
    (raiz / "src" / "pas_intelligence").mkdir(parents=True)
    (raiz / "deploy").mkdir(parents=True)
    (raiz / "landing-page").mkdir(parents=True)  # fora de PERMITIDOS

    (raiz / "Dockerfile").write_text("FROM python:3.14-slim\n")
    (raiz / ".dockerignore").write_text(".git/\n")
    (raiz / "requirements-api.txt").write_text("fastapi\n")
    (raiz / "api" / "main.py").write_text("app = None\n")
    (raiz / "src" / "pas_intelligence" / "model_package.py").write_text("# pacote\n")
    (raiz / "deploy" / "ponteiro.json").write_text("{}\n")
    (raiz / "deploy" / "buscar_artefatos.py").write_text("# busca\n")
    (raiz / "deploy" / "requirements.txt").write_text("huggingface_hub\n")
    (raiz / "deploy" / "publicar_pacote.py").write_text("# não é PERMITIDO\n")
    (raiz / "landing-page" / "package.json").write_text("{}\n")

    _git("init", cwd=raiz)
    _git("config", "user.email", "monorepo@teste.local", cwd=raiz)
    _git("config", "user.name", "Monorepo Teste", cwd=raiz)
    _git("add", "-A", cwd=raiz)
    _git("commit", "-m", "PII que não pode viajar para o Repo de Deploy", cwd=raiz)
    return raiz


def _arquivos_publicados(bare: Path) -> set[str]:
    """Clona `main` explicitamente — um bare repo criado por `git init --bare` não tem HEAD
    apontando para `main` (só passa a apontar sozinho quando um host como o GitHub recebe o
    primeiro push num repositório vazio), então um `git clone` simples não faria checkout de
    nada."""
    checkout = bare.parent / f"checkout-verificacao-{len(list(bare.parent.glob('checkout-verificacao-*')))}"
    subprocess.run(
        ["git", "clone", "--branch", "main", str(bare), str(checkout)], check=True, capture_output=True
    )
    return {
        str(p.relative_to(checkout))
        for p in checkout.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }


def test_publica_so_os_padroes_de_permitidos(monorepo_falso: Path, repo_deploy_vazio: Path) -> None:
    publicar_snapshot(monorepo_falso, str(repo_deploy_vazio))

    publicados = _arquivos_publicados(repo_deploy_vazio)

    assert "landing-page/package.json" not in publicados
    assert "deploy/publicar_pacote.py" not in publicados
    assert "Dockerfile" in publicados
    assert "api/main.py" in publicados
    assert "src/pas_intelligence/model_package.py" in publicados
    assert "deploy/ponteiro.json" in publicados


def test_repo_de_deploy_nao_carrega_historia_do_monorepo(
    monorepo_falso: Path, repo_deploy_vazio: Path
) -> None:
    publicar_snapshot(monorepo_falso, str(repo_deploy_vazio))

    log_monorepo = _git("log", "--format=%H", cwd=monorepo_falso).stdout.split()
    log_repo_deploy = subprocess.run(
        ["git", "log", "--format=%H", "main"], cwd=repo_deploy_vazio, check=True, capture_output=True, text=True
    ).stdout.split()

    assert not set(log_monorepo) & set(log_repo_deploy)
    assert len(log_repo_deploy) == 1


def test_repo_de_deploy_diz_que_nao_deve_ser_editado_a_mao(
    monorepo_falso: Path, repo_deploy_vazio: Path
) -> None:
    publicar_snapshot(monorepo_falso, str(repo_deploy_vazio))

    checkout = repo_deploy_vazio.parent / "checkout-conteudo-readme"
    subprocess.run(
        ["git", "clone", "--branch", "main", str(repo_deploy_vazio), str(checkout)],
        check=True,
        capture_output=True,
    )
    conteudo = (checkout / "README.md").read_text(encoding="utf-8").lower()

    assert "gerado" in conteudo
    assert "não edite" in conteudo or "nao edite" in conteudo


def test_publicar_duas_vezes_produz_segundo_commit_sem_conflito(
    monorepo_falso: Path, repo_deploy_vazio: Path
) -> None:
    publicar_snapshot(monorepo_falso, str(repo_deploy_vazio))
    sha2 = publicar_snapshot(monorepo_falso, str(repo_deploy_vazio))

    log = subprocess.run(
        ["git", "log", "--format=%H", "main"], cwd=repo_deploy_vazio, check=True, capture_output=True, text=True
    ).stdout.split()

    assert len(log) == 2
    assert log[0] == sha2


def test_arquivo_fora_de_permitidos_some_apos_republicar(
    monorepo_falso: Path, repo_deploy_vazio: Path
) -> None:
    """Um arquivo que casava com PERMITIDOS numa rodada e some da árvore de origem some também do
    Repo de Deploy na próxima publicação — o script sempre substitui o conteúdo, não acumula."""
    extra = monorepo_falso / "api" / "temporario.py"
    extra.write_text("# vai sumir\n")

    publicar_snapshot(monorepo_falso, str(repo_deploy_vazio))
    assert "api/temporario.py" in _arquivos_publicados(repo_deploy_vazio)

    extra.unlink()
    publicar_snapshot(monorepo_falso, str(repo_deploy_vazio))
    assert "api/temporario.py" not in _arquivos_publicados(repo_deploy_vazio)


def test_permitidos_continua_com_a_mesma_lista_do_ticket_08() -> None:
    assert PERMITIDOS == [
        "Dockerfile",
        ".dockerignore",
        "requirements-api.txt",
        "api/*",
        "src/pas_intelligence/*",
        "deploy/ponteiro.json",
        "deploy/buscar_artefatos.py",
        "deploy/requirements.txt",
    ]
