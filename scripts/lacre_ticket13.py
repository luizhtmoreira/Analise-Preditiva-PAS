"""Ticket 13 — abre o lacre uma vez e põe modelo antigo e novo lado a lado.

    .venv/bin/python scripts/lacre_ticket13.py \\
        .scratch/pdf-extraction/saida-nova/resultado_final.csv \\
        --pacote .scratch/treino-modelos-pas3/pacotes/pas3-2026-07-28

Duas coisas, no mesmo script porque as duas leem o mesmo triênio lacrado e abri-lo duas vezes
seria abri-lo duas vezes:

1. **O lacre** (`validation.holdout_final_use_uma_vez`). O ticket 13 é o único lugar do
   repositório autorizado a chamá-lo. Ele **só reporta** — a regra assimétrica do relatório 11
   §4.4 foi escrita antes de o número existir: `σ` acima de `5,5` em `A3` = o app está confiante
   demais, vira ticket com prioridade; abaixo de `4,5` = nota de relatório; entre os dois =
   variação normal entre anos. Nada aqui reajusta modelo, largura ou features.

2. **A comparação lado a lado** que o ticket 13 exige antes da promoção: modelo antigo
   (`modelo_arg_final.joblib`, Argumento Final direto, `σ = 13,49`) contra o novo (`A3` previsto,
   Argumento Final por aritmética, `σ = 3 × σ(A3)`), nos mesmos Alunos, com previsão *e*
   probabilidade de aprovação de cada.

O modelo avaliado é o **artefato do pacote**, carregado do disco — não um gêmeo retreinado aqui.
É o mesmo `fit` que `holdout_final_use_uma_vez` descreve (treina em tudo menos o lacre), e o
script confere isso contra o manifesto antes de medir.

Privacidade: nenhuma linha identificável sai daqui. Os Alunos da comparação individual aparecem
como `Aluno 1..N`, sem `id_pseudonimo`, sem curso e sem campus ([[project_parser_privacy]]).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.stats import norm  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pas_intelligence.dataset_pas3 import FEATURES_CANONICAS  # noqa: E402
from pas_intelligence.training_dataset import load_and_build  # noqa: E402
from pas_intelligence.training_pipeline import montar_features  # noqa: E402
from pas_intelligence.validation import (  # noqa: E402
    COLUNA_CLASSE,
    TRIENIO_LACRADO,
    Erro,
    holdout_final_use_uma_vez,
)

SIGMA_ANTIGO = 13.49
"""O `σ` que `api/services/gestao_service.py:34` usa hoje, em Argumento Final."""

LIMITE_CONFIANTE_DEMAIS = 5.5
LIMITE_NOTA_DE_RODAPE = 4.5
"""A regra assimétrica do relatório 11 §4.4, em `A3`. Constantes aqui para deixar explícito que
elas são anteriores ao número medido — não foram escolhidas depois de ver o resultado."""

NIVEL_CANONICO = 0.80
"""Relatório 11 §4.7: 80% é o nível dos derivados da largura."""

N_ALUNOS_LADO_A_LADO = 12


def _carregar_booster(pacote: Path):
    import lightgbm as lgb  # type: ignore

    return lgb.Booster(model_file=str(pacote / "modelo_pas3.txt"))


def _conferir_pacote(pacote: Path) -> dict:
    manifesto = json.loads((pacote / "manifest.json").read_text(encoding="utf-8"))
    modelo = manifesto["modelos"][0]
    if modelo["treinado_excluindo_trienio"] != TRIENIO_LACRADO:
        raise SystemExit(
            f"O pacote foi treinado excluindo {modelo['treinado_excluindo_trienio']}, não "
            f"{TRIENIO_LACRADO}. Medir o lacre com ele seria medir memória."
        )
    if list(modelo["features"]) != list(FEATURES_CANONICAS):
        raise SystemExit("As features do pacote não são as canônicas — pacote de outra rodada.")
    return manifesto


def _cobertura(residuo: np.ndarray, sigma: np.ndarray, nivel: float) -> float:
    z = norm.ppf((1 + nivel) / 2)
    return float(np.mean(np.abs(residuo) <= z * sigma))


def _veredito_do_lacre(sigma: float) -> str:
    if sigma > LIMITE_CONFIANTE_DEMAIS:
        return (
            f"σ = {sigma:.3f} > {LIMITE_CONFIANTE_DEMAIS} — o app está confiante demais. "
            "Regra do relatório 11 §4.4: vira ticket com prioridade."
        )
    if sigma < LIMITE_NOTA_DE_RODAPE:
        return (
            f"σ = {sigma:.3f} < {LIMITE_NOTA_DE_RODAPE} — melhor que o prometido. "
            "Regra do relatório 11 §4.4: nota de relatório, não machuca ninguém."
        )
    return (
        f"{LIMITE_NOTA_DE_RODAPE} ≤ σ = {sigma:.3f} ≤ {LIMITE_CONFIANTE_DEMAIS} — dentro da "
        "variação normal entre anos (ruído entre dobras ±0,37). Nenhuma ação."
    )


def _cortes_do_lacre(caminho_cortes: Path, af_min: float, af_max: float) -> pd.DataFrame:
    """Menor Nota de Corte Universal por curso+campus+turno no triênio lacrado.

    Mesma limpeza do relatório 11 §6: só `Universal`, só chamada não parcial, e corte dentro da
    faixa de Argumento Final observada — o filtro que remove os defeitos conhecidos dos tickets
    14/15 do mapa `pdf-extraction` (`199.162,872` e `−23.317,084`).
    """
    cortes = pd.read_csv(caminho_cortes)
    cortes = cortes[
        (cortes["trienio"] == TRIENIO_LACRADO)
        & (cortes["sistema_nome"] == "Universal")
        & (cortes["parcial"] == False)  # noqa: E712
        & cortes["nota_corte"].between(af_min, af_max)
    ]
    return (
        cortes.groupby(["campus", "curso", "turno"], as_index=False)["nota_corte"]
        .min()
        .rename(columns={"nota_corte": "corte"})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--pacote", type=Path, required=True)
    parser.add_argument(
        "--cortes",
        type=Path,
        default=RAIZ / ".scratch/pdf-extraction/saida-nova/notas_corte.csv",
    )
    parser.add_argument("--modelos", type=Path, default=RAIZ / "models")
    args = parser.parse_args()

    manifesto = _conferir_pacote(args.pacote)
    booster = _carregar_booster(args.pacote)

    canonico = load_and_build(args.csv)
    df = montar_features(canonico)
    # `montar_features` troca `a1` por `NaN` no Aluno sem Etapa 1 — isso é **codificação de
    # feature** (ticket 10), não o valor do domínio. O `A1` dele existe e é exato: o z de zero
    # (ADR-0008). Recompor o Argumento Final com o `NaN` daria `NaN` para 10% do lacre.
    df["a1_real"] = canonico["a1"].to_numpy()
    dobra = holdout_final_use_uma_vez(df["trienio"].unique())
    print(f"═══ Lacre — {dobra.trienio_teste} ═══")
    print(f"treino do artefato: {', '.join(dobra.trienios_treino)}")

    lacre = df[df["trienio"] == dobra.trienio_teste].copy()
    lacre["previsto_a3"] = booster.predict(lacre[FEATURES_CANONICAS])
    residuo = (lacre["previsto_a3"] - lacre["a3"]).to_numpy()

    erro = Erro.medir(lacre["a3"].to_numpy(), lacre["previsto_a3"].to_numpy())
    com = lacre[~lacre[COLUNA_CLASSE]]
    sem = lacre[lacre[COLUNA_CLASSE]]
    erro_com = Erro.medir(com["a3"].to_numpy(), com["previsto_a3"].to_numpy())
    erro_sem = Erro.medir(sem["a3"].to_numpy(), sem["previsto_a3"].to_numpy())

    incerteza = manifesto["incerteza"]["sigma_por_classe"]
    sigma_por_linha = np.where(
        lacre[COLUNA_CLASSE].to_numpy(), incerteza["sem_etapa_1"], incerteza["com_etapa_1"]
    )

    print(f"\nn = {erro.n}  (com Etapa 1: {erro_com.n} · sem: {erro_sem.n})")
    print(f"σ (RMSE em A3)      geral {erro.rmse:.3f} · com {erro_com.rmse:.3f} · sem {erro_sem.rmse:.3f}")
    print(f"MAE                 geral {erro.mae:.3f}")
    print(f"viés                geral {erro.vies:+.3f} · com {erro_com.vies:+.3f} · sem {erro_sem.vies:+.3f}")
    print(f"RMSE/MAE            {erro.razao_rmse_mae:.4f}  (normal ≈ 1,25)")
    print(f"σ prometido pelo pacote: com {incerteza['com_etapa_1']:.3f} · sem {incerteza['sem_etapa_1']:.3f}")
    for nivel in (0.50, 0.80, 0.90, 0.95):
        print(f"cobertura a {nivel:.0%}: {_cobertura(residuo, sigma_por_linha, nivel):.2%}")
    print(f"\nVEREDITO: {_veredito_do_lacre(erro.rmse)}")

    # ─── Lado a lado ────────────────────────────────────────────────────────────────────────
    print(f"\n═══ Lado a lado — antigo vs. novo, mesmos {erro.n} Alunos ═══")

    import joblib  # type: ignore

    caminho_antigo = args.modelos / "modelo_arg_final.joblib"
    if not caminho_antigo.exists():
        raise SystemExit(f"{caminho_antigo} não existe — sem modelo antigo não há comparação.")
    modelo_antigo = joblib.load(caminho_antigo)

    features_legadas = ["EB_PAS1", "Red_PAS1", "EB_PAS2", "Red_PAS2", "Cresc_EB", "Cresc_Red"]
    # O modelo antigo nunca viu `NaN`: ele foi treinado com o zero literal da Etapa 1 ausente.
    # Alimentá-lo com o `NaN` do ticket 10 mediria o pipeline novo, não o modelo antigo.
    entrada_antiga = lacre[features_legadas].fillna(0.0)
    lacre["af_antigo"] = modelo_antigo.predict(entrada_antiga)
    lacre["af_novo"] = lacre["a1_real"] + 2 * lacre["a2"] + 3 * lacre["previsto_a3"]
    lacre["af_real"] = lacre["argumento_final"]

    erro_antigo = Erro.medir(lacre["af_real"].to_numpy(), lacre["af_antigo"].to_numpy())
    erro_novo = Erro.medir(lacre["af_real"].to_numpy(), lacre["af_novo"].to_numpy())
    print(f"\nArgumento Final — antigo: RMSE {erro_antigo.rmse:.3f} · MAE {erro_antigo.mae:.3f} · viés {erro_antigo.vies:+.3f}")
    print(f"Argumento Final — novo:   RMSE {erro_novo.rmse:.3f} · MAE {erro_novo.mae:.3f} · viés {erro_novo.vies:+.3f}")

    cortes = _cortes_do_lacre(
        args.cortes, float(lacre["af_real"].min()), float(lacre["af_real"].max())
    )
    casado = lacre.merge(cortes, on=["campus", "curso", "turno"], how="left")
    com_corte = casado[casado["corte"].notna()].copy()
    print(f"\nAlunos com corte casado: {len(com_corte)} de {len(casado)}")

    sigma_novo_af = 3 * np.where(
        com_corte[COLUNA_CLASSE].to_numpy(), incerteza["sem_etapa_1"], incerteza["com_etapa_1"]
    )
    com_corte["prob_antiga"] = 1 - norm.cdf(
        com_corte["corte"], loc=com_corte["af_antigo"], scale=SIGMA_ANTIGO
    )
    com_corte["prob_nova"] = 1 - norm.cdf(
        com_corte["corte"], loc=com_corte["af_novo"], scale=sigma_novo_af
    )
    passou = com_corte["af_real"] >= com_corte["corte"]

    for rotulo, coluna in (("antigo", "prob_antiga"), ("novo", "prob_nova")):
        prob = com_corte[coluna]
        errou_decisao = (prob >= 0.5) != passou
        saturada = (prob < 0.01) | (prob > 0.99)
        # Brier: erro quadrático médio da própria probabilidade contra o que aconteceu — a
        # métrica que pune tanto errar quanto acertar sem convicção.
        brier = float(np.mean((prob - passou.astype(float)) ** 2))
        print(
            f"{rotulo:>6}: erro de decisão {errou_decisao.mean():.2%} · "
            f"Brier {brier:.4f} · saturadas {saturada.mean():.1%}"
        )
    print(f"taxa real de aprovação: {passou.mean():.2%}")

    discordam = ((com_corte["prob_antiga"] >= 0.5) != (com_corte["prob_nova"] >= 0.5))
    print(f"antigo e novo discordam sobre passar: {discordam.mean():.2%} ({int(discordam.sum())})")

    print(f"\n─── {N_ALUNOS_LADO_A_LADO} Alunos, sem identificação ───")
    print(
        "Amostra estratificada, não aleatória: sorteio uniforme daria ~10 linhas de `0,0% vs "
        "0,0%` (relatório 11 §8 — 63,6% das probabilidades saturam), que não é o que uma revisão "
        "antes da promoção precisa ver. Os quatro grupos são: perto do corte com Etapa 1, perto "
        "do corte sem Etapa 1, longe por cima e longe por baixo."
    )
    distancia = (com_corte["af_real"] - com_corte["corte"]).abs()
    perto = distancia <= 20
    grupos = [
        ("perto, com E1", com_corte[perto & ~com_corte[COLUNA_CLASSE]]),
        ("perto, sem E1", com_corte[perto & com_corte[COLUNA_CLASSE]]),
        ("longe acima", com_corte[~perto & (com_corte["af_real"] > com_corte["corte"])]),
        ("longe abaixo", com_corte[~perto & (com_corte["af_real"] < com_corte["corte"])]),
    ]
    por_grupo = N_ALUNOS_LADO_A_LADO // len(grupos)
    amostra = pd.concat(
        [
            g.sample(min(por_grupo, len(g)), random_state=20260728).assign(_grupo=rotulo)
            for rotulo, g in grupos
        ]
    ).reset_index(drop=True)
    print(f"{'#':>3} {'grupo':>14} {'AF real':>8} {'AF ant':>8} {'AF novo':>8} "
          f"{'corte':>8} {'P ant':>7} {'P novo':>7} {'passou':>7}")
    for i, linha in amostra.iterrows():
        print(
            f"{i + 1:>3} {linha['_grupo']:>14} "
            f"{linha['af_real']:>8.1f} {linha['af_antigo']:>8.1f} {linha['af_novo']:>8.1f} "
            f"{linha['corte']:>8.1f} {linha['prob_antiga'] * 100:>6.1f}% "
            f"{linha['prob_nova'] * 100:>6.1f}% "
            f"{'sim' if linha['af_real'] >= linha['corte'] else 'não':>7}"
        )

    faixa = norm.ppf((1 + NIVEL_CANONICO) / 2)
    print(
        f"\nFaixa de {NIVEL_CANONICO:.0%} em Argumento Final: "
        f"±{faixa * 3 * incerteza['com_etapa_1']:.1f} (com Etapa 1) · "
        f"±{faixa * 3 * incerteza['sem_etapa_1']:.1f} (sem)"
    )


if __name__ == "__main__":
    main()
