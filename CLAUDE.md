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

# Evaluate model accuracy
python calculate.py
```

## Architecture

The public backend lives in `src/pas_intelligence/`; it reads from Supabase and serialized ML models under `models/`. The original Streamlit dashboard (`app/streamlit_app.py`) is deprecated and gitignored — it still runs locally via `streamlit run app/streamlit_app.py` if present on disk, but it is not tracked in git or published, since it's being superseded by the Next.js frontend (`landing-page/`) + FastAPI backend (`api/`).

### Key modules

- **`app/streamlit_app.py`** (gitignored, local-only) — legacy single-file Streamlit app. Adds `src/` to `sys.path` at startup. Imports from `pas_intelligence` and `pdf_generator`. Contains the multi-tenant config dict `DOMAINS_CONFIG`.
- **`src/pas_intelligence/ensemble.py`** — dynamic ensemble: weights models by student score volatility (Coefficient of Variation). Low CV → linear regression; high CV → LightGBM/RandomForest. Transition via sigmoid.
- **`src/pas_intelligence/argument_calculator.py`** — computes the PAS *Argumento Final* using official Cebraspe weights (`PESO_P1=0.72`, `PESO_P2=8.28`, `PESO_REDACAO=1.00`). Also projects historical exam stats via linear regression for normalization.
- **`src/pas_intelligence/target_calculator.py`** — reverse calculator: given a target course cutoff, determines the P2 score the student must achieve in PAS 3. Uses `p1_pas3_model.joblib` and `red_pas3_model.joblib` (both `HistGradientBoostingRegressor`) to estimate P1 and Redação.
- **`src/pas_intelligence/statistics.py`** — probability of approval modeled as `P(X > cutoff)` where `X ~ N(predicted_arg, RMSE²)`, RMSE=13.49.
- **`src/pas_intelligence/pas_constants.py`** — `OFFICIAL_STATS` dict keyed by `(year, stage)` with historical mean/std for P1, P2, and Redação used in score normalization.
- **`src/pdf_generator.py`** — `PDFGenerator` class that injects data into whitelabel PDF templates via ReportLab. Templates are in `assets/templates/` (gitignored, local-only — whitelabel product asset, not published; same treatment as `models/`). Currently only consumed by the legacy Streamlit app; not yet ported to `api/`.

### ML models (`models/` — gitignored, hosted on Dropbox)

| File | Algorithm | Purpose |
|---|---|---|
| `modelo_lgbm.joblib` | LGBMRegressor | Predict EB PAS 3 (main) |
| `modelo_rf.joblib` | RandomForestRegressor | Predict EB PAS 3 |
| `modelo_linear.joblib` | LinearRegression | Predict EB PAS 3 |
| `modelo_mlp.joblib` | MLPRegressor (100,50) | Predict EB PAS 3 |
| `meta_model.joblib` | RandomForestClassifier | Picks best base model per student |
| `scaler.joblib` | StandardScaler | Scales 6-feature input for linear/MLP |
| `meta_scaler.joblib` | StandardScaler | Scales 10-feature meta-input |
| `modelo_arg_final.joblib` | LGBMRegressor | Predict Argumento Final directly |
| `p1_pas3_model.joblib` | HistGradientBoostingRegressor | Predict P1 in PAS 3 |
| `red_pas3_model.joblib` | HistGradientBoostingRegressor | Predict Redação in PAS 3 |

**Base feature vector (6):** `[eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red]`  
**Meta feature vector (10):** appends `|c_eb|/|eb_p1|`, `|c_red|/|red_p1|`, `(eb_p1+eb_p2)/2`, `sign(c_eb)`

Linear and MLP models require `scaler.transform()`; LightGBM and RF do not.

### Domain concepts

- **EB (Escore Bruto):** Raw exam score = P1 + P2
- **Argumento Final:** Weighted cumulative score across all 3 PAS stages used for UnB ranking
- **Volatilidade (CV):** `std/mean * 100` of `[eb_pas1, eb_pas2]`; drives ensemble model weighting
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
