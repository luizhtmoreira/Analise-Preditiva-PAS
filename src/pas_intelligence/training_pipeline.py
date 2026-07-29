"""Pipeline de treino reproduzível do PAS 3 (ticket 12 do mapa `treino-modelos-pas3`).

Um comando: do `resultado_final.csv` até o pacote do modelo, com as decisões dos tickets
anteriores codificadas aqui em vez de espalhadas em relatório — cada bloco abaixo aponta de volta
para o ticket que o decidiu:

- filtro de linha e construção do dataset → `training_dataset.py` (ticket 05);
- janela de triênios (`janela=None`, expansiva) → ticket 08;
- conjunto de features (`FEATURES_CANONICAS`) → ticket 09;
- família de modelo e hiperparâmetros (`HIPERPARAMETROS_LGBM`, `NaN` nativo na Etapa 1 ausente)
  → ticket 10;
- avaliação e critério de aceite (`PORTAO_1`) → esquema do ticket 06, números do ticket 07;
- formato de artefato (texto nativo do LightGBM) → gatilho da Decisão 1 do ticket 03, acionado
  porque o ticket 10 concluiu "um LightGBM só, sem scaler";
- largura da incerteza (duas larguras fixas por classe de `etapa_1_ausente`, escala `A3`) →
  ticket 11, ADR-0012.

**Nunca toca o triênio lacrado.** `TRIENIO_LACRADO` é excluído da avaliação (pela própria régua)
e do treino do artefato final, explicitamente, antes de chamar `fit`. Decidir se o artefato de
produção embarca o triênio lacrado é decisão do ticket 13, através de
`validation.holdout_final_use_uma_vez` — este módulo não a chama e não a antecipa.

Privacidade: lê `resultado_final.csv` só para montar o dataset em memória; nenhuma linha (nome,
inscrição, nota individual) sai para o manifesto ou para qualquer arquivo publicado — só hash,
contagem e métrica agregada ([[project_parser_privacy]]).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.stats import norm  # type: ignore

from .dataset_pas3 import FEATURES_CANONICAS, montar_features
from .training_dataset import load_and_build
from .validation import ALVO_CANONICO, COLUNA_CLASSE, TRIENIO_LACRADO, ResultadoValidacao, avaliar

SEMENTE_PADRAO = 20260728
"""A mesma semente dos tickets 07-10 — trocá-la muda o número por acaso de dado aleatório, não
por uma decisão."""

HIPERPARAMETROS_LGBM = dict(n_estimators=400, learning_rate=0.01, num_leaves=15)
"""Fechados no ticket 10 por busca em grade sobre uma dobra de ajuste dedicada
(`scripts/familia_de_modelo_ticket10.py`). O pipeline não busca de novo a cada execução — buscar
de novo a cada retreino tornaria o artefato dependente de qual grade rodou por último, não da
medição fechada."""

# Portão 1, congelado no ticket 07 §8: o que qualquer candidato precisa bater para não regredir
# contra o baseline honesto. `avaliar()` mede; este dicionário decide.
PORTAO_1 = dict(geral=5.167, majoritaria=5.038, minoritaria=6.028, vies_abs=0.500)

NOME_ARQUIVO_MODELO = "modelo_pas3.txt"
NOME_ARQUIVO_MANIFESTO = "manifest.json"


class PortaoFechadoError(RuntimeError):
    """O modelo treinado não bateu o Portão 1 — falha ruidosa em vez de publicar em silêncio.

    Só é evitável passando `forcar=True` com um `motivo`, que fica gravado no manifesto: a chave
    de força do ticket 03 Decisão 7 é visível, nunca discreta.
    """


def _fabrica_lgbm(hiperparametros: dict[str, Any]):
    from lightgbm import LGBMRegressor  # type: ignore

    def construir(semente: int):
        return LGBMRegressor(
            random_state=semente,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
            **hiperparametros,
        )

    return construir


def _bate_portao(resultado: ResultadoValidacao, portao: dict[str, float]) -> tuple[bool, str]:
    geral = resultado.agrupado
    maj = resultado.agrupado_majoritaria
    minoria = resultado.agrupado_minoritaria

    falhas = []
    if geral.rmse > portao["geral"]:
        falhas.append(f"RMSE geral {geral.rmse:.3f} > {portao['geral']:.3f}")
    if maj.rmse > portao["majoritaria"]:
        falhas.append(f"RMSE majoritária {maj.rmse:.3f} > {portao['majoritaria']:.3f}")
    if minoria is not None and minoria.rmse > portao["minoritaria"]:
        falhas.append(f"RMSE minoritária {minoria.rmse:.3f} > {portao['minoritaria']:.3f}")
    if abs(geral.vies) > portao["vies_abs"]:
        falhas.append(f"|viés| {abs(geral.vies):.3f} > {portao['vies_abs']:.3f}")

    return (not falhas), "; ".join(falhas)


NIVEIS_COBERTURA = (0.50, 0.80, 0.90, 0.95)
"""Os níveis prometidos que o relatório 11 §3.2 verificou contra a cobertura empírica."""


def _bloco_incerteza(resultado_validacao: ResultadoValidacao) -> dict[str, Any]:
    """O bloco `incerteza` do manifesto — ADR-0012 e relatório 11 §7.1.

    Duas larguras fixas por classe de `etapa_1_ausente`, tiradas do `ResultadoValidacao` que
    `treinar()` já tem em mãos (RMSE agrupado das 5 dobras, nunca recalculado nem otimizado aqui
    — o número é o que a régua do ticket 06 mede, não um parâmetro deste módulo). Escala `A3`,
    nunca Argumento Final (relatório 11 §7.1).
    """
    sigma_com_etapa_1 = resultado_validacao.agrupado_majoritaria.rmse
    minoritaria = resultado_validacao.agrupado_minoritaria
    sigma_sem_etapa_1 = minoritaria.rmse if minoritaria is not None else sigma_com_etapa_1

    previsoes = resultado_validacao.previsoes
    residuo = (previsoes["previsto"] - previsoes["real"]).to_numpy()
    sigma_por_linha = np.where(
        previsoes[COLUNA_CLASSE].to_numpy(), sigma_sem_etapa_1, sigma_com_etapa_1
    )

    cobertura_verificada = {}
    for nivel in NIVEIS_COBERTURA:
        z = norm.ppf((1 + nivel) / 2)
        dentro_do_intervalo = np.abs(residuo) <= z * sigma_por_linha
        cobertura_verificada[f"{nivel:.2f}"] = float(np.mean(dentro_do_intervalo))

    return {
        "forma": "normal",
        "escala": "a3",
        "sigma_por_classe": {
            "com_etapa_1": sigma_com_etapa_1,
            "sem_etapa_1": sigma_sem_etapa_1,
        },
        "sigma_agrupado": resultado_validacao.agrupado.rmse,
        "medido_em": "previsões fora-da-dobra das 5 dobras (ticket 06)",
        "cobertura_verificada": cobertura_verificada,
    }


def _sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _versao_pacote(nome: str) -> str:
    try:
        return pkg_version(nome)
    except PackageNotFoundError:
        return "desconhecida"


def _info_ambiente() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "lightgbm": _versao_pacote("lightgbm"),
        "numpy": _versao_pacote("numpy"),
        "pandas": _versao_pacote("pandas"),
    }


def _info_codigo(raiz_repo: Path) -> dict[str, Any]:
    """Commit e árvore limpa — se houver alteração não commitada, o commit registrado não
    descreve o código que de fato treinou (ticket 03 Decisão 5)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=raiz_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=raiz_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"commit": commit, "arvore_limpa": status.strip() == ""}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "desconhecido", "arvore_limpa": False}


@dataclass(frozen=True)
class ResultadoPipeline:
    """O que uma execução do pipeline produz — devolvido junto do pacote escrito em disco, para
    quem chama decidir o que fazer sem precisar reabrir o `manifest.json`."""

    diretorio_pacote: Path
    caminho_modelo: Path
    caminho_manifesto: Path
    manifesto: dict[str, Any]
    resultado_validacao: ResultadoValidacao
    bateu_portao: bool
    motivo_falha_portao: str


def treinar(
    caminho_csv: Path,
    diretorio_saida: Path,
    *,
    semente: int = SEMENTE_PADRAO,
    portao: dict[str, float] | None = None,
    forcar: bool = False,
    motivo_forca: str | None = None,
    raiz_repo: Path | None = None,
) -> ResultadoPipeline:
    """Do CSV ao pacote — determinístico, com o critério de aceite verificado automaticamente.

    Levanta `PortaoFechadoError` (e não escreve nada em `diretorio_saida`) se o modelo não bater
    o Portão 1, a menos que `forcar=True` — nesse caso publica assim mesmo e grava `motivo_forca`
    no manifesto, nunca em silêncio. `forcar=True` sem `motivo_forca` é erro de uso, não omissão
    silenciosa.

    Nunca lê nem escreve o triênio lacrado (`validation.TRIENIO_LACRADO`) — ele é excluído do
    treino do artefato final abaixo, independente da régua já excluí-lo da avaliação.
    """
    if forcar and not motivo_forca:
        raise ValueError("forcar=True exige motivo_forca — a chave de força é visível, não muda.")

    portao = portao or PORTAO_1
    raiz_repo = raiz_repo or Path(__file__).resolve().parent.parent.parent

    df_canonico = load_and_build(caminho_csv)
    df_features = montar_features(df_canonico)

    fabrica = _fabrica_lgbm(HIPERPARAMETROS_LGBM)
    resultado_validacao = avaliar(
        df_features,
        fabrica,
        nome="pas3-lightgbm",
        features=FEATURES_CANONICAS,
        semente=semente,
        alvo=ALVO_CANONICO,
        janela=None,  # janela expansiva — decisão do ticket 08
    )

    bateu, motivo_falha = _bate_portao(resultado_validacao, portao)
    if not bateu and not forcar:
        raise PortaoFechadoError(
            f"Portão 1 não batido: {motivo_falha}. Publicação recusada — use forcar=True com "
            "motivo_forca se a promoção precisa passar por cima mesmo assim."
        )

    treino_final = df_features[df_features["trienio"] != TRIENIO_LACRADO]
    modelo_final = fabrica(semente)
    modelo_final.fit(treino_final[FEATURES_CANONICAS], treino_final[ALVO_CANONICO])

    diretorio_saida.mkdir(parents=True, exist_ok=True)
    caminho_modelo = diretorio_saida / NOME_ARQUIVO_MODELO
    modelo_final.booster_.save_model(str(caminho_modelo))

    trienios_fonte = sorted(df_canonico["trienio"].unique())
    manifesto: dict[str, Any] = {
        "criado_em": datetime.now(timezone.utc).isoformat(),
        "dado": {
            "arquivo": caminho_csv.name,
            "sha256": _sha256_arquivo(caminho_csv),
            "linhas": int(len(df_canonico)),
            "trienios": trienios_fonte,
            "trienio_lacrado": TRIENIO_LACRADO,
        },
        "codigo": _info_codigo(raiz_repo),
        "ambiente": _info_ambiente(),
        "modelos": [
            {
                "arquivo": NOME_ARQUIVO_MODELO,
                "formato": "lightgbm-texto-nativo",
                "alvo": ALVO_CANONICO,
                "features": list(FEATURES_CANONICAS),
                "hiperparametros": HIPERPARAMETROS_LGBM,
                "semente": semente,
                "treinado_excluindo_trienio": TRIENIO_LACRADO,
            }
        ],
        "avaliacao": {
            "esquema": "validacao deslizante, janela expansiva, 5 dobras (ticket 06/08)",
            "semente": semente,
            "portao": portao,
            "bateu_portao": bateu,
            "motivo_falha_portao": motivo_falha,
            "forcado": forcar,
            "motivo_forca": motivo_forca,
            "metricas": {
                "geral": asdict(resultado_validacao.agrupado),
                "majoritaria": asdict(resultado_validacao.agrupado_majoritaria),
                "minoritaria": (
                    asdict(resultado_validacao.agrupado_minoritaria)
                    if resultado_validacao.agrupado_minoritaria is not None
                    else None
                ),
            },
        },
        "incerteza": _bloco_incerteza(resultado_validacao),
    }

    caminho_manifesto = diretorio_saida / NOME_ARQUIVO_MANIFESTO
    with open(caminho_manifesto, "w", encoding="utf-8") as arquivo:
        json.dump(manifesto, arquivo, indent=2, ensure_ascii=False, sort_keys=True)

    return ResultadoPipeline(
        diretorio_pacote=diretorio_saida,
        caminho_modelo=caminho_modelo,
        caminho_manifesto=caminho_manifesto,
        manifesto=manifesto,
        resultado_validacao=resultado_validacao,
        bateu_portao=bateu,
        motivo_falha_portao=motivo_falha,
    )
