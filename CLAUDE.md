# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the (locally gitignored) Streamlit app — not part of the public repo, see note below
streamlit run app/streamlit_app.py

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_pas_intelligence.py

# Retrain the PAS 3 model — one command, CSV to package. Refuses to write if the
# acceptance gate (Portão 1) is not met; --forcar records the override in the manifest.
.venv/bin/python scripts/treinar_pipeline.py <resultado_final.csv> --saida <dir>
```

## Architecture

The public backend lives in `src/pas_intelligence/`; it reads from Supabase and serialized ML models under `models/`. The original Streamlit dashboard (`app/streamlit_app.py`) is deprecated and gitignored — it still runs locally via `streamlit run app/streamlit_app.py` if present on disk, but it is not tracked in git or published, since it's being superseded by the Next.js frontend (`landing-page/`) + FastAPI backend (`api/`).

### Key modules

- **`app/streamlit_app.py`** (gitignored, local-only) — legacy single-file Streamlit app. Adds `src/` to `sys.path` at startup. Imports from `pas_intelligence` and `pdf_generator`. Contains the multi-tenant config dict `DOMAINS_CONFIG`.
- **`src/pas_intelligence/model_package.py`** — the only door between `api/` and the trained artifact. Loads `models/pas3/` (LightGBM text + `manifest.json`), computes `A1`/`A2` exactly from the student's notes and declared foreign language, predicts `A3`, and returns `Argumento Final = A1 + 2·A2 + 3·Â3` plus the Largura de Incerteza of the student's class. Reuses the training feature builders so runtime and training cannot drift apart.
- **`src/pas_intelligence/validation.py`** — the project's ruler (ADR-0010): expanding-window sliding validation, 5 folds, sealed 2023/2025 holdout. `holdout_final_use_uma_vez` was opened once, in ticket 13.
- **`src/pas_intelligence/training_pipeline.py`** / **`training_dataset.py`** / **`dataset_pas3.py`** — CSV → canonical dataset → features → trained package, with the acceptance gate enforced in code.
- **`src/pas_intelligence/argument_calculator.py`** — computes the PAS *Argumento Final* using official Cebraspe weights (`PESO_P1=0.72`, `PESO_P2=8.28`, `PESO_REDACAO=1.00`). Also projects historical exam stats via linear regression for normalization.
- **`src/pas_intelligence/target_calculator.py`** — reverse calculator: given a target course cutoff, determines the P2 score the student must achieve in PAS 3. Uses `p1_pas3_model.joblib` and `red_pas3_model.joblib` (both `HistGradientBoostingRegressor`) to estimate P1 and Redação.
- **`src/pas_intelligence/statistics.py`** — probability of approval modeled as `P(X > cutoff)` where `X ~ N(predicted_arg, largura²)`. The width is a **required** parameter with no default: it lives in the model package's manifest, per student class, and changes with every retrain (ADR-0012).
- **`src/pas_intelligence/pas_constants.py`** — `OFFICIAL_STATS` dict keyed by `(year, stage)` with the mean/std published by Cebraspe for P1, P2, and Redação. Its only consumer is `api/services/analytics_service.py` (the public temporal chart) — the Argumento Final, approval probability, and target calculators do **not** read it; they take mean/std as parameters. P1 is stored per foreign language in `parte_1`; `m_p1`/`dp_p1` are derived properties (simple mean of the three) and match no individual student, since each sits a single language.
- **`src/pdf_generator.py`** — `PDFGenerator` class that injects data into whitelabel PDF templates via ReportLab. Templates are in `assets/templates/` (gitignored, local-only — whitelabel product asset, not published; same treatment as `models/`). Currently only consumed by the legacy Streamlit app; not yet ported to `api/`.

### ML models (`models/` — gitignored)

**One model, plus arithmetic.** The eight-artifact ensemble was retired in ADR-0011 (it beat its own best component by 0.10%, inside the fold-to-fold noise) and the dual prediction of Argumento Final and EB was retired in ADR-0009.

| Path | Format | Purpose |
|---|---|---|
| `models/pas3/modelo_pas3.txt` | LightGBM native text | Predict `A3`, the Argumento of Etapa 3 — the **Alvo Canônico** |
| `models/pas3/manifest.json` | JSON | Provenance (CSV hash, git commit, lib versions), feature names, metrics, and the Largura de Incerteza |
| `models/aposentados-2026-07-28/` | joblib | The retired ensemble, kept to revert |
| `p1_pas3_model.joblib`, `red_pas3_model.joblib` | HistGradientBoostingRegressor | Still loaded by `target_calculator.py`; **do not load** under the current sklearn (`ModuleNotFoundError: _loss`) — the reverse path silently answers by weighted mean |

**Canonical feature vector (11):** `[a1, a2, EB_PAS1, Red_PAS1, EB_PAS2, Red_PAS2, Cresc_EB, Cresc_Red, cresc_eb_pct, cresc_red_pct, sinal_cresc_eb]` — order is contract; the manifest carries it and `model_package` refuses to load a package whose order differs. No scaler.

For a student with no Etapa 1 (the three Etapa-1 notes all zero), the eight Etapa-1-derived columns become **native `NaN`**, not literal zero (ADR-0011).

### Domain concepts

- **EB (Escore Bruto):** Raw exam score = P1 + P2
- **Argumento Final:** Weighted cumulative score across all 3 PAS stages used for UnB ranking
- **Argumento de Etapa (`A1`, `A2`, `A3`):** standardized score of one stage. For a student who has sat PAS 1 and 2, `A1` and `A2` are **exact arithmetic**; only `A3` is predicted. `Argumento Final = A1 + 2·A2 + 3·A3`, so `σ(Argumento Final) = 3 × σ(A3)`
- **Volatilidade:** **absolute** dispersion between Argumentos de Etapa (`|A2 − A1|`). No longer a Coefficient of Variation — dividing by the mean is both impossible (the mean is ~0 and negative for 49.3% of the base) and redundant (ADR-0009)
- **Largura de Incerteza:** how much the model typically misses by, used as the σ of the approval-probability normal. One number per student class, living in the package manifest
- **Multi-tenant (whitelabel):** Schools identified by a string key (`"marista"`, `"ideal"`, `"default"`) in `DOMAINS_CONFIG`; each key maps to a logo and a PDF template path

### Environment and secrets

- Set `ENV=PROD` or `ENV=DEV` (defaults to `DEV`)
- Supabase credentials go in `.streamlit/secrets.toml` (gitignored):
  ```toml
  [supabase]
  url = "..."
  key = "..."
  ```

### Landing page

`landing-page/` is an independent Next.js app. See `landing-page/AGENTS.md` before modifying it — the Next.js version has breaking API changes.
