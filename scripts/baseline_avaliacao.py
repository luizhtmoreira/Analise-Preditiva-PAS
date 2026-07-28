"""
Avaliação Baseline — Vetor PAS
==============================

⚠ SUPERADO pelo ticket 07. Use `scripts/baseline_honesto.py`, que mede sobre a régua do ticket
06 (`src/pas_intelligence/validation.py`). **Os números do ADR-0007 são inválidos** por dois
motivos independentes: o vetor de features estava na ordem errada (corrigido abaixo, para o
script não ser uma armadilha), e o método — KFold aleatório sobre `banco_alunos_pas_final.csv` —
mede interpolação dentro de anos conhecidos, coisa que o produto nunca faz, sobre uma base que
os modelos já tinham visto no treino.

Captura métricas densas de todos os modelos existentes para uso como
linha de base na comparação com versões retreinadas.

Saída:
  - Console: tabelas formatadas
  - docs/adr/0007-baseline-modelos-v1.md: ADR com todas as métricas

Uso:
  python scripts/baseline_avaliacao.py

Requerimentos: joblib, scikit-learn, lightgbm, pandas, numpy, scipy
"""

import sys
import os
import warnings
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Caminhos ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "banco_alunos_pas_final.csv"
MODELS_DIR = ROOT / "models"
OUTPUT_ADR = ROOT / "docs" / "adr" / "0007-baseline-modelos-v1.md"

# Adiciona src ao path para importar pas_intelligence
ROOT_SRC = ROOT / "src"
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))
# Também adiciona ROOT para permitir `from src.pas_intelligence...`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Features padrão usadas nos modelos base ─────────────────────────────────
# Ordem lida de `booster.feature_name()` dos próprios artefatos, não da documentação. A lista
# anterior — ["EB_PAS1", "EB_PAS2", "Cresc_EB", "Media_EB", "Std_EB", "CV_EB"] — tinha 5 das 6
# posições trocadas, e é a causa dos `R² = -83` e `MAPE = 1e+19` do ADR-0007: passar features
# como array puro não confere nome nenhum, o modelo lê por posição.
FEATURE_COLS_BASE = ["EB_PAS1", "Red_PAS1", "EB_PAS2", "Red_PAS2", "Cresc_EB", "Cresc_Red"]
TARGET_COL = "EB_PAS3"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula bloco completo de métricas de regressão."""
    residuals = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Métricas adicionais
    medae = np.median(np.abs(residuals))
    max_err = np.max(np.abs(residuals))
    within_1 = np.mean(np.abs(residuals) <= 1.0) * 100
    within_3 = np.mean(np.abs(residuals) <= 3.0) * 100
    within_5 = np.mean(np.abs(residuals) <= 5.0) * 100

    # Teste de normalidade dos resíduos (Shapiro se n <= 5000, senão D'Agostino)
    n = len(residuals)
    if n <= 5000:
        stat_norm, p_norm = stats.shapiro(residuals)
        normality_test = "Shapiro-Wilk"
    else:
        stat_norm, p_norm = stats.normaltest(residuals)
        normality_test = "D'Agostino-Pearson"

    # Viés sistemático (t-test: resíduos != 0?)
    t_stat, p_bias = stats.ttest_1samp(residuals, 0)

    return {
        "n_amostras": n,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MSE": round(mse, 4),
        "R2": round(r2, 4),
        "MAPE_pct": round(mape, 4),
        "MedAE": round(medae, 4),
        "MaxErr": round(max_err, 4),
        "Dentro_1pt_pct": round(within_1, 2),
        "Dentro_3pt_pct": round(within_3, 2),
        "Dentro_5pt_pct": round(within_5, 2),
        "Residuo_media": round(float(np.mean(residuals)), 4),
        "Residuo_std": round(float(np.std(residuals)), 4),
        "Residuo_p5": round(float(np.percentile(residuals, 5)), 4),
        "Residuo_p25": round(float(np.percentile(residuals, 25)), 4),
        "Residuo_p75": round(float(np.percentile(residuals, 75)), 4),
        "Residuo_p95": round(float(np.percentile(residuals, 95)), 4),
        "Normalidade_teste": normality_test,
        "Normalidade_stat": round(float(stat_norm), 4),
        "Normalidade_p": round(float(p_norm), 6),
        "Vies_t": round(float(t_stat), 4),
        "Vies_p": round(float(p_bias), 6),
    }


def prepare_dataset() -> pd.DataFrame:
    """Carrega e enriquece o dataset com features derivadas."""
    print_section("CARREGANDO DATASET")
    df = pd.read_csv(DATA_PATH)
    print(f"  Registros brutos: {len(df)}")

    # Features derivadas de EB
    df["EB_PAS1"] = df["P1_PAS1"] + df["P2_PAS1"]
    df["EB_PAS2"] = df["P1_PAS2"] + df["P2_PAS2"]
    df["EB_PAS3"] = df["P1_PAS3"] + df["P2_PAS3"]

    df["Cresc_EB"] = df["EB_PAS2"] - df["EB_PAS1"]
    df["Media_EB"] = (df["EB_PAS1"] + df["EB_PAS2"]) / 2
    df["Std_EB"] = df[["EB_PAS1", "EB_PAS2"]].std(axis=1)
    df["CV_EB"] = (df["Std_EB"] / df["Media_EB"].replace(0, np.nan)) * 100
    df["CV_EB"] = df["CV_EB"].fillna(0)

    # Remove linhas sem alvo
    df = df.dropna(subset=["EB_PAS3"])

    print(f"  Registros após limpeza: {len(df)}")
    print(f"  Triênios: {sorted(df['Ano_Trienio'].unique())}")
    print(f"\n  Estatísticas do target (EB_PAS3):")
    print(f"    Média:   {df['EB_PAS3'].mean():.3f}")
    print(f"    Std:     {df['EB_PAS3'].std():.3f}")
    print(f"    Min:     {df['EB_PAS3'].min():.3f}")
    print(f"    Max:     {df['EB_PAS3'].max():.3f}")
    print(f"    Mediana: {df['EB_PAS3'].median():.3f}")

    return df


def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray,
                   scaler=None, cv_folds: int = 5) -> dict:
    """Avalia um modelo com hold-out + cross-validation."""
    print(f"\n  → {name}")

    # Predição simples (in-sample — serve como sanity check para baseline)
    X_input = scaler.transform(X) if scaler is not None else X
    try:
        y_pred = model.predict(X_input)
    except Exception as e:
        print(f"    ✗ Erro na predição: {e}")
        return {"erro": str(e)}

    metrics = compute_metrics(y, y_pred)

    # Cross-validation (MAE, R2)
    print(f"    Rodando CV {cv_folds}-fold...")
    try:
        X_cv = scaler.transform(X) if scaler is not None else X
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(model, X_cv, y, cv=cv,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
        cv_r2 = cross_val_score(model, X_cv, y, cv=cv,
                                scoring="r2", n_jobs=1)
        cv_rmse = np.sqrt(-cross_val_score(model, X_cv, y, cv=cv,
                                           scoring="neg_mean_squared_error", n_jobs=1))

        metrics["CV_MAE_media"] = round(float(cv_mae.mean()), 4)
        metrics["CV_MAE_std"] = round(float(cv_mae.std()), 4)
        metrics["CV_R2_media"] = round(float(cv_r2.mean()), 4)
        metrics["CV_R2_std"] = round(float(cv_r2.std()), 4)
        metrics["CV_RMSE_media"] = round(float(cv_rmse.mean()), 4)
        metrics["CV_RMSE_std"] = round(float(cv_rmse.std()), 4)
    except Exception as e:
        print(f"    ⚠ CV falhou: {e}")
        metrics["CV_nota"] = f"Falhou: {e}"

    # Permutation importance (top 3)
    try:
        perm = permutation_importance(
            model, X_input, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        top3_idx = np.argsort(perm.importances_mean)[::-1][:3]
        metrics["PermImportance_top3"] = {
            str(i): round(float(perm.importances_mean[i]), 4) for i in top3_idx
        }
    except Exception:
        pass

    # Print resumo
    print(f"    MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
          f"R²={metrics['R2']:.4f}  Dentro±3pts={metrics['Dentro_3pt_pct']:.1f}%")
    if "CV_MAE_media" in metrics:
        print(f"    CV-MAE={metrics['CV_MAE_media']:.3f}±{metrics['CV_MAE_std']:.3f}  "
              f"CV-R²={metrics['CV_R2_media']:.4f}±{metrics['CV_R2_std']:.4f}")

    return metrics


def evaluate_ensemble(df: pd.DataFrame, model_linear, model_lgbm,
                      scaler) -> dict:
    """Avalia o ensemble dinâmico (Linear + LGBM ponderado por CV)."""
    print(f"\n  → Ensemble Dinâmico (Linear + LGBM via volatilidade)")
    from pas_intelligence.ensemble import predict_with_dynamic_ensemble

    preds = []
    strategies = []
    weights_conservative = []
    cvs = []

    for _, row in df.iterrows():
        try:
            feat = np.array([[
                row["EB_PAS1"], row["EB_PAS2"], row["Cresc_EB"],
                row["Media_EB"], row["Std_EB"], row["CV_EB"]
            ]])
            pred, meta = predict_with_dynamic_ensemble(
                eb_pas1=row["EB_PAS1"],
                eb_pas2=row["EB_PAS2"],
                model_conservative=model_linear,
                model_aggressive=model_lgbm,
                features=feat,
                scaler=scaler,
            )
            preds.append(pred)
            strategies.append(meta["strategy"])
            weights_conservative.append(meta["weight_conservative"])
            cvs.append(meta["volatility_cv"])
        except Exception:
            preds.append(np.nan)
            strategies.append("erro")
            weights_conservative.append(np.nan)
            cvs.append(np.nan)

    y_true = df["EB_PAS3"].values
    y_pred = np.array(preds)
    mask = ~np.isnan(y_pred)

    metrics = compute_metrics(y_true[mask], y_pred[mask])

    # Distribuição de estratégias
    strategy_counts = pd.Series(strategies).value_counts().to_dict()
    metrics["Distribuicao_estrategia"] = strategy_counts
    metrics["CV_medio_alunos"] = round(float(np.nanmean(cvs)), 4)
    metrics["Peso_conservador_medio"] = round(float(np.nanmean(weights_conservative)), 4)

    print(f"    MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
          f"R²={metrics['R2']:.4f}  Dentro±3pts={metrics['Dentro_3pt_pct']:.1f}%")
    print(f"    Estratégias: {strategy_counts}  CV-médio={metrics['CV_medio_alunos']:.2f}%")

    return metrics


def evaluate_by_trienio(df: pd.DataFrame, model, scaler=None,
                        feature_cols=None) -> dict:
    """Avalia métricas de um modelo segmentadas por triênio."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS_BASE

    results = {}
    for trienio, grp in df.groupby("Ano_Trienio"):
        X = grp[feature_cols].values
        y = grp["EB_PAS3"].values
        X_input = scaler.transform(X) if scaler is not None else X
        try:
            y_pred = model.predict(X_input)
            results[trienio] = {
                "n": len(y),
                "MAE": round(mean_absolute_error(y, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y, y_pred)), 4),
                "R2": round(r2_score(y, y_pred), 4),
            }
        except Exception as e:
            results[trienio] = {"erro": str(e)}
    return results


def generate_adr(all_results: dict, df: pd.DataFrame):
    """Gera o ADR em Markdown com todas as métricas."""
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d")

    lines = [
        f"# ADR 0007 — Baseline dos Modelos v1",
        f"",
        f"**Data:** {ts}  ",
        f"**Status:** Registrado (não reversível)  ",
        f"**Autor:** Gerado automaticamente por `scripts/baseline_avaliacao.py`",
        f"",
        f"---",
        f"",
        f"## Contexto",
        f"",
        f"Antes de retreinar os modelos preditivos do Vetor PAS, foi necessário",
        f"capturar o estado de desempenho atual como linha de base. Este ADR",
        f"registra todas as métricas coletadas em **{ts}** para permitir",
        f"comparação objetiva com as versões retreinadas.",
        f"",
        f"## Dataset",
        f"",
        f"| Propriedade | Valor |",
        f"|---|---|",
        f"| Arquivo | `data/banco_alunos_pas_final.csv` |",
        f"| Registros totais | {len(df)} |",
        f"| Target | `EB_PAS3` (Escore Bruto PAS 3) |",
        f"| Target média | {df['EB_PAS3'].mean():.3f} |",
        f"| Target std | {df['EB_PAS3'].std():.3f} |",
        f"| Target min/max | {df['EB_PAS3'].min():.3f} / {df['EB_PAS3'].max():.3f} |",
        f"| Triênios | {', '.join(sorted(df['Ano_Trienio'].unique()))} |",
        f"",
        f"## Features Base",
        f"",
        f"`{', '.join(FEATURE_COLS_BASE)}`",
        f"",
        f"---",
        f"",
        f"## Métricas por Modelo",
        f"",
    ]

    for model_name, data in all_results.items():
        if model_name == "_por_trienio":
            continue

        lines.append(f"### `{model_name}`")
        lines.append(f"")

        if "erro" in data:
            lines.append(f"> ⚠ Erro na avaliação: `{data['erro']}`")
            lines.append(f"")
            continue

        # Tabela principal
        lines.append(f"| Métrica | Valor |")
        lines.append(f"|---|---|")
        skip_keys = {"PermImportance_top3", "Distribuicao_estrategia",
                     "CV_nota", "Normalidade_teste"}
        for k, v in data.items():
            if k in skip_keys or isinstance(v, dict):
                continue
            lines.append(f"| {k} | {v} |")
        lines.append(f"")

        # Distribuição de estratégia (ensemble)
        if "Distribuicao_estrategia" in data:
            lines.append(f"**Distribuição de estratégias:**  ")
            for strat, count in data["Distribuicao_estrategia"].items():
                lines.append(f"- `{strat}`: {count} alunos")
            lines.append(f"")

        # Permutation importance
        if "PermImportance_top3" in data:
            lines.append(f"**Permutation Importance (top 3 features, índice → importância):**  ")
            for idx, imp in data["PermImportance_top3"].items():
                feat_name = FEATURE_COLS_BASE[int(idx)] if int(idx) < len(FEATURE_COLS_BASE) else f"feat_{idx}"
                lines.append(f"- `{feat_name}`: {imp}")
            lines.append(f"")

        # Por triênio
        if "_por_trienio" in all_results and model_name in all_results["_por_trienio"]:
            lines.append(f"**Desempenho por triênio:**")
            lines.append(f"")
            lines.append(f"| Triênio | n | MAE | RMSE | R² |")
            lines.append(f"|---|---|---|---|---|")
            for trienio, m in all_results["_por_trienio"][model_name].items():
                if "erro" not in m:
                    lines.append(
                        f"| {trienio} | {m['n']} | {m['MAE']} | {m['RMSE']} | {m['R2']} |"
                    )
            lines.append(f"")

    # Ranking resumo
    lines += [
        f"---",
        f"",
        f"## Ranking Geral (MAE ascendente)",
        f"",
        f"| Posição | Modelo | MAE | RMSE | R² | Dentro±3pts |",
        f"|---|---|---|---|---|---|",
    ]

    ranking = []
    for name, data in all_results.items():
        if name == "_por_trienio" or "erro" in data or "MAE" not in data:
            continue
        ranking.append((name, data["MAE"], data["RMSE"], data["R2"], data["Dentro_3pt_pct"]))

    ranking.sort(key=lambda x: x[1])
    for i, (name, mae, rmse, r2, w3) in enumerate(ranking, 1):
        lines.append(f"| {i}° | `{name}` | {mae} | {rmse} | {r2} | {w3}% |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Decisão",
        f"",
        f"Este ADR é **somente leitura** — registra o estado dos modelos antes do retreinamento.",
        f"Após o retreinamento, criar `ADR-0008` com as métricas dos novos modelos e",
        f"comparação direta com os valores acima.",
        f"",
        f"## Consequências",
        f"",
        f"- Qualquer métrica nos modelos retreinados deve ser comparada contra este ADR.",
        f"- Se o R² ou MAE piorarem, o retreinamento deve ser revisado antes de substituir os artefatos.",
        f"- Se a calibração do semáforo (% aprovados por cor) piorar, o retreinamento não deve ser colocado em produção.",
        f"",
    ]

    OUTPUT_ADR.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  ✓ ADR salvo em: {OUTPUT_ADR}")


# ─── Semáforo ────────────────────────────────────────────────────────────────────────

def classify_semaforo_prob(
    p_1a: float,
    p_2a: float,
) -> str:
    """
    Classifica o semáforo de risco baseado nas probabilidades de aprovação.

    Regras (conforme UI do produto):
    - Verde:    P(1ª chamada) >= 50%  OU  P(2ª chamada) >= 75%
    - Amarelo:  P(1ª chamada) <  50%  E   P(2ª chamada) >= 30%
    - Vermelho: P(2ª chamada) <  30%
    """
    if p_1a >= 0.50 or p_2a >= 0.75:
        return "verde"
    elif p_2a >= 0.30:
        return "amarelo"
    else:
        return "vermelho"


def evaluate_semaforo_calibration(
    df: pd.DataFrame,
    model_linear,
    model_lgbm,
    scaler,
    notas_corte_path: Path,
    n_cursos_por_aluno: int = 5,
    rmse_modelo: float = 13.49,
    seed: int = 42,
) -> dict:
    """
    Calibração do semáforo de risco.

    Para cada aluno, amostra N cursos. Para cada par (aluno, curso):
    - Prediz o EB_PAS3 via ensemble e estima o Argumento Final previsto
    - Calcula P(aprovação 1ª chamada) e P(aprovação 2ª chamada) via distribuição normal
    - Classifica a cor do semáforo pelas regras de probabilidade
    - Verifica se o aluno REALMENTE passou (Arg_Final_real >= corte_1a ou corte_2a)

    Retorna: para cada cor, a taxa de aprovação real.
    """
    from scipy.stats import norm as sp_norm
    from pas_intelligence.ensemble import predict_with_dynamic_ensemble

    print(f"\n  \u2192 Semáforo de Risco (calibração por probabilidade)")

    if model_linear is None or model_lgbm is None or scaler is None:
        return {"erro": "Ensemble não disponível"}

    # Carrega notas de corte: precisa de 1a e 2a chamada por curso/trienio
    nc = pd.read_csv(notas_corte_path)

    # Usa o trienio mais recente com dados de ambas as chamadas
    nc_1a = nc[nc["Chamada"] == "1ª"].copy()
    nc_2a = nc[nc["Chamada"] == "2ª"].copy()

    # Pega a nota de corte mais recente de cada curso para cada chamada
    nc_1a = (
        nc_1a.sort_values("Trienio")
        .groupby(["Curso_Limpo", "Sistema_Nome"])
        .last()
        .reset_index()[["Curso_Limpo", "Sistema_Nome", "Min"]]
        .rename(columns={"Min": "Corte_1a"})
    )
    nc_2a = (
        nc_2a.sort_values("Trienio")
        .groupby(["Curso_Limpo", "Sistema_Nome"])
        .last()
        .reset_index()[["Curso_Limpo", "Sistema_Nome", "Min"]]
        .rename(columns={"Min": "Corte_2a"})
    )

    # Junta 1a e 2a chamada — só cursos que têm ambas
    nc_merged = nc_1a.merge(nc_2a, on=["Curso_Limpo", "Sistema_Nome"], how="inner")
    nc_merged = nc_merged[
        (nc_merged["Corte_1a"] > 0) & (nc_merged["Corte_2a"] > 0)
    ]
    cursos = nc_merged.to_dict("records")
    n_cursos = len(cursos)
    print(f"    Cursos com 1ª e 2ª chamada: {n_cursos}")

    rng = np.random.default_rng(seed)
    records = []  # (cor, passou_1a, passou_2a)

    for _, row in df.iterrows():
        feat = np.array([[
            row["EB_PAS1"], row["EB_PAS2"], row["Cresc_EB"],
            row["Media_EB"], row["Std_EB"], row["CV_EB"]
        ]])
        try:
            pred_eb3, _ = predict_with_dynamic_ensemble(
                eb_pas1=row["EB_PAS1"],
                eb_pas2=row["EB_PAS2"],
                model_conservative=model_linear,
                model_aggressive=model_lgbm,
                features=feat,
                scaler=scaler,
            )
        except Exception:
            continue

        # Estima o Argumento Final previsto:
        # AF_previsto = AF_real + (EB3_pred - EB3_real) * peso_PAS3
        # Peso do PAS3 no Argumento Final = 3
        delta_eb3 = pred_eb3 - row["EB_PAS3"]
        pred_af = row["Arg_Final"] + delta_eb3 * 3
        af_real = row["Arg_Final"]

        idxs = rng.integers(0, n_cursos, size=n_cursos_por_aluno)
        for idx in idxs:
            curso = cursos[int(idx)]
            corte_1a = curso["Corte_1a"]
            corte_2a = curso["Corte_2a"]

            # Probabilidades via distribuição normal centrada no AF previsto
            p_1a = float(1 - sp_norm.cdf(corte_1a, loc=pred_af, scale=rmse_modelo))
            p_2a = float(1 - sp_norm.cdf(corte_2a, loc=pred_af, scale=rmse_modelo))

            cor = classify_semaforo_prob(p_1a, p_2a)
            passou_1a = bool(af_real >= corte_1a)
            passou_2a = bool(af_real >= corte_2a)

            records.append({
                "cor": cor,
                "passou_1a": passou_1a,
                "passou_2a": passou_2a,
                "p_1a": p_1a,
                "p_2a": p_2a,
            })

    result_df = pd.DataFrame(records)
    total_pares = len(result_df)
    print(f"    Pares (aluno, curso) avaliados: {total_pares:,}")

    calibration = {"rmse_usado": rmse_modelo, "total_pares": total_pares}

    cores = {
        "verde":    "Verde (P(1ª)≥50% ou P(2ª)≥75%)",
        "amarelo":  "Amarelo (P(1ª)<50% e P(2ª)≥30%)",
        "vermelho": "Vermelho (P(2ª)<30%)",
    }

    print(f"\n    {'Cor':<42} {'n pares':>9} {'Aprov 1ª':>10} {'Aprov 2ª':>10}")
    print(f"    {'-'*42} {'-'*9} {'-'*10} {'-'*10}")

    for cor, label in cores.items():
        sub = result_df[result_df["cor"] == cor]
        n = len(sub)
        if n == 0:
            calibration[cor] = {"n_pares": 0}
            print(f"    {label:<42} {0:>9,} {'---':>10} {'---':>10}")
            continue

        pct_1a = round(sub["passou_1a"].mean() * 100, 2)
        pct_2a = round(sub["passou_2a"].mean() * 100, 2)
        pct_repr = round(100 - pct_2a, 2)  # reprovou na 2ª chamada

        calibration[cor] = {
            "label": label,
            "n_pares": n,
            "pct_dist": round(n / total_pares * 100, 2),
            "pct_aprovados_1a_chamada": pct_1a,
            "pct_aprovados_2a_chamada": pct_2a,
            "pct_reprovados_2a_chamada": pct_repr,
        }
        print(f"    {label:<42} {n:>9,} {pct_1a:>9.1f}% {pct_2a:>9.1f}%")

    # Acurácia binária: verde=passou (2ª chamada), amarelo+vermelho=não passou
    result_df["pred_passou"] = result_df["cor"] == "verde"
    acuracia = round((result_df["pred_passou"] == result_df["passou_2a"]).mean() * 100, 2)
    calibration["acuracia_binaria_2a_pct"] = acuracia
    print(f"\n    Acurácia binária (verde=passou / outros=não passou): {acuracia}%")

    return calibration


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print_section("BASELINE DE MODELOS — VETOR PAS")
    print(f"  Data/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Dataset
    df = prepare_dataset()
    X_base = df[FEATURE_COLS_BASE].values
    y = df["EB_PAS3"].values

    # 2. Carrega artefatos
    print_section("CARREGANDO ARTEFATOS")

    def safe_load(path):
        try:
            obj = joblib.load(path)
            print(f"  ✓ {path.name}")
            return obj
        except Exception as e:
            print(f"  ✗ {path.name}: {e}")
            return None

    scaler        = safe_load(MODELS_DIR / "scaler.joblib")
    meta_scaler   = safe_load(MODELS_DIR / "meta_scaler.joblib")
    model_linear  = safe_load(MODELS_DIR / "modelo_linear.joblib")
    model_lgbm    = safe_load(MODELS_DIR / "modelo_lgbm.joblib")
    model_lgbm_v1 = safe_load(MODELS_DIR / "modelo_lgbm_v1.joblib")
    model_arg_final = safe_load(MODELS_DIR / "modelo_arg_final.joblib")
    model_rf      = safe_load(MODELS_DIR / "modelo_rf.joblib")
    model_mlp     = safe_load(MODELS_DIR / "modelo_mlp.joblib")
    model_meta    = safe_load(MODELS_DIR / "meta_model.joblib")
    p1_model      = safe_load(MODELS_DIR / "p1_pas3_model.joblib")
    red_model     = safe_load(MODELS_DIR / "red_pas3_model.joblib")

    # 3. Avaliações individuais
    print_section("AVALIAÇÃO DOS MODELOS BASE")
    all_results = {}
    trienio_results = {}

    def safe_evaluate(name, model, X, y, scaler=None, cv_folds=5):
        if model is None:
            print(f"  → {name}: ✗ não carregado (incompatível)")
            return {"erro": "Modelo não carregado — incompatibilidade de versão do sklearn"}
        return evaluate_model(name, model, X, y, scaler=scaler, cv_folds=cv_folds)

    def safe_by_trienio(model, scaler=None):
        if model is None:
            return {}
        return evaluate_by_trienio(df, model, scaler=scaler)

    # Modelo Linear (com scaler)
    r = safe_evaluate("modelo_linear", model_linear, X_base, y, scaler=scaler)
    all_results["modelo_linear"] = r
    trienio_results["modelo_linear"] = safe_by_trienio(model_linear, scaler=scaler)

    # LGBM (sem scaler)
    r = safe_evaluate("modelo_lgbm", model_lgbm, X_base, y)
    all_results["modelo_lgbm"] = r
    trienio_results["modelo_lgbm"] = safe_by_trienio(model_lgbm)

    # LGBM v1
    r = safe_evaluate("modelo_lgbm_v1", model_lgbm_v1, X_base, y)
    all_results["modelo_lgbm_v1"] = r
    trienio_results["modelo_lgbm_v1"] = safe_by_trienio(model_lgbm_v1)

    # modelo_arg_final (LightGBM)
    r = safe_evaluate("modelo_arg_final", model_arg_final, X_base, y)
    all_results["modelo_arg_final"] = r
    trienio_results["modelo_arg_final"] = safe_by_trienio(model_arg_final)

    # Random Forest
    r = safe_evaluate("modelo_rf", model_rf, X_base, y)
    all_results["modelo_rf"] = r
    trienio_results["modelo_rf"] = safe_by_trienio(model_rf)

    # MLP (com scaler)
    r = safe_evaluate("modelo_mlp", model_mlp, X_base, y, scaler=scaler)
    all_results["modelo_mlp"] = r
    trienio_results["modelo_mlp"] = safe_by_trienio(model_mlp, scaler=scaler)

    # p1_pas3_model — prediz P1 do PAS3
    X_p1_feat = df[["EB_PAS1", "EB_PAS2", "Cresc_EB", "Media_EB", "Std_EB", "CV_EB"]].values
    y_p1 = df["P1_PAS3"].values
    r = safe_evaluate("p1_pas3_model (prediz P1_PAS3)", p1_model, X_p1_feat, y_p1)
    all_results["p1_pas3_model"] = r

    # red_pas3_model — prediz Redação do PAS3
    y_red = df["Red_PAS3"].values
    r = safe_evaluate("red_pas3_model (prediz Red_PAS3)", red_model, X_p1_feat, y_red)
    all_results["red_pas3_model"] = r

    # 4. Ensemble dinâmico
    print_section("AVALIAÇÃO DO ENSEMBLE DINÂMICO")
    if model_linear is not None and model_lgbm is not None and scaler is not None:
        r = evaluate_ensemble(df, model_linear, model_lgbm, scaler)
        all_results["ensemble_dinamico"] = r
    else:
        print("  ⚠ Ensemble dinâmico não avaliado (modelo_linear ou modelo_lgbm não carregado)")
        all_results["ensemble_dinamico"] = {"erro": "Dependências não carregadas"}
    trienio_results["ensemble_dinamico"] = {}

    # 5. Meta-model (classificador — avalia acurácia de seleção)
    print_section("META-MODEL (Classificador de seleção)")
    print("  → meta_model (RandomForestClassifier)")
    try:
        # Features do meta-model: 10 variáveis conforme dossie
        # Tenta com as features disponíveis
        X_meta = df[FEATURE_COLS_BASE].values
        X_meta_scaled = meta_scaler.transform(X_meta) if X_meta.shape[1] == 10 else X_meta
        if meta_scaler.n_features_in_ != X_meta.shape[1]:
            # Cria features extras para satisfazer o meta_scaler
            extra = np.column_stack([
                df["Red_PAS1"].values,
                df["Red_PAS2"].values,
                df["P1_PAS1"].values,
                df["P1_PAS2"].values,
            ])
            X_meta_10 = np.column_stack([X_meta, extra])
            X_meta_scaled = meta_scaler.transform(X_meta_10)

        meta_pred = model_meta.predict(X_meta_scaled)
        from sklearn.metrics import accuracy_score
        # meta_model é classificador — sem y verdadeiro de classe,
        # registramos apenas a distribuição das classes preditas
        pred_counts = pd.Series(meta_pred).value_counts().to_dict()
        all_results["meta_model"] = {
            "nota": "Classificador de seleção de modelo (sem ground-truth de classe)",
            "Distribuicao_classes_preditas": pred_counts,
            "n_amostras": len(meta_pred),
        }
        print(f"    Distribuição de classes: {pred_counts}")
    except Exception as e:
        all_results["meta_model"] = {"erro": str(e)}
        print(f"    ✗ {e}")

    # 6. Semáforo — calibração
    # TODO: Implementar quando a nova extração de dados incluir a coluna
    # (aluno_id, curso_alvo). A avaliação por amostragem aleatória de cursos
    # não reflete a distribuição real de escolhas dos alunos e foi descartada.
    # Ver função evaluate_semaforo_calibration() e classify_semaforo_prob() neste arquivo.
    all_results["semaforo_calibracao"] = {
        "nota": "Pendente — requer dados de curso-alvo por aluno (nova extração)"
    }

    # 7. Salva por triênio
    all_results["_por_trienio"] = trienio_results

    # 7. Ranking no console
    print_section("RANKING FINAL (MAE ↑ melhor)")
    ranking = []
    for name, data in all_results.items():
        if name == "_por_trienio" or "erro" in data or "MAE" not in data:
            continue
        ranking.append((name, data["MAE"], data["RMSE"], data["R2"], data["Dentro_3pt_pct"]))
    ranking.sort(key=lambda x: x[1])

    print(f"\n  {'Modelo':<35} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'±3pts%':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, mae, rmse, r2, w3 in ranking:
        print(f"  {name:<35} {mae:>7.3f} {rmse:>7.3f} {r2:>7.4f} {w3:>6.1f}%")

    # 8. Gera ADR
    print_section("GERANDO ADR 0007")
    generate_adr(all_results, df)

    print(f"\n{'═' * 60}")
    print("  ✓ Baseline completo!")
    print('═' * 60)


if __name__ == "__main__":
    main()
