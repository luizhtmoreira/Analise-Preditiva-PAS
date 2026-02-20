# 📄 Dossiê Técnico: Artefatos de Modelagem Preditiva PAS

## 1. Scalers (Processadores de Escala)

Todos os três scalers foram treinados no formato **StandardScaler** (padronização Z-score), armazenando a média e a variância dos dados de treino.

### `meta_scaler.joblib`
*   **Tipo:** `StandardScaler`
*   **Atributos Principais:** `mean_`, `scale_`, `var_`
*   **Features Esperadas:** 10 variáveis (usadas no metamodelo).

### `scaler.joblib`
*   **Tipo:** `StandardScaler`
*   **Atributos Principais:** `mean_`, `scale_`, `var_`
*   **Features Esperadas:** 6 variáveis (features clássicas dos modelos base).

### `scaler_v1.joblib`
*   **Tipo:** `StandardScaler`
*   **Atributos Principais:** `mean_`, `scale_`, `var_`
*   **Features Esperadas:** 6 variáveis (cópia/backup do treinamento base).

---

## 2. Modelos Base

### 💡 Modelos de Ensemble (Boosting)

#### `modelo_arg_final.joblib`
*   **Algoritmo:** `LGBMRegressor` (LightGBM)
*   **Hiperparâmetros Chave:** `learning_rate`: 0.1, `max_depth`: -1 (sem limite), `n_estimators`: 100, `num_leaves`: 31
*   **Top 3 Features Importantes:** 
    1. `EB_PAS1` (653.0) 
    2. `EB_PAS2` (606.0) 
    3. `Red_PAS2` (592.0)

#### `modelo_lgbm.joblib` & `modelo_lgbm_v1.joblib` *(Versões similares)*
*   **Algoritmo:** `LGBMRegressor`
*   **Hiperparâmetros Chave:** `learning_rate`: 0.1, `max_depth`: -1, `n_estimators`: 100, `num_leaves`: 31
*   **Top 3 Features Importantes:** 
    1. `EB_PAS1` (626.0) 
    2. `EB_PAS2` (589.0) 
    3. `Red_PAS2` (560.0)

#### `p1_pas3_model.joblib` & `red_pas3_model.joblib`
*   **Algoritmo:** `HistGradientBoostingRegressor` (Scikit-Learn nativo)
*   **Hiperparâmetros Chave:** `learning_rate`: 0.1, `max_depth`: 5, `max_iter`: 200, `random_state`: 42
*   **Importância:** Este tipo nativo baseia-se em histogramas e não expõe diretamente o atributo tradicional de `feature_importances_`.

### 🌳 Modelos de Ensemble (Bagging)

#### `meta_model.joblib`
*(Classificador que escolhe o melhor modelo para o aluno)*
*   **Algoritmo:** `RandomForestClassifier`
*   **Hiperparâmetros Chave:** `n_estimators`: 100, `max_depth`: 10, `random_state`: 42
*   **Top 3 Features Importantes (índices):** 
    1. Feature 2 (`EB_PAS2`: 0.2144) 
    2. Feature 8 (`Media_EB`: 0.1872) 
    3. Feature 0 (`EB_PAS1`: 0.1295)

#### `modelo_rf.joblib`
*   **Algoritmo:** `RandomForestRegressor`
*   **Hiperparâmetros Chave:** `n_estimators`: 100, `n_jobs`: -1 (usa todas as CPUs), `random_state`: 42
*   **Top 3 Features Importantes:** 
    1. `EB_PAS2` (0.6620)
    2. `EB_PAS1` (0.0874)
    3. `Red_PAS2` (0.0756)

### 📈 Modelos Matemáticos/Paramétricos

#### `modelo_linear.joblib`
*   **Algoritmo:** `LinearRegression`
*   **Hiperparâmetros Chave:** Usa OLS padrão (`fit_intercept`: True)
*   **Top 3 Variáveis de Maior Coeficiente (Peso Absoluto):** 
    1. Feature 2 (`EB_PAS2`: +6.80)
    2. Feature 0 (`EB_PAS1`: +5.07)
    3. Feature 4 (`Cresc_EB`: +1.45)

#### `modelo_mlp.joblib`
*   **Algoritmo:** `MLPRegressor` (Rede Neural Perceptron de Múltiplas Camadas)
*   **Hiperparâmetros Chave:** `hidden_layer_sizes`: (100, 50) [100 neurônios na 1ª camada oculta, 50 na 2ª], `activation`: relu, `solver`: adam, `max_iter`: 500, `alpha`: 0.0001
*   **Importância:** Redes Neurais densas são inerentemente opacas no nível de variáveis individuais e não exportam score direto de features.
