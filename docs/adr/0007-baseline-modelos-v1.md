# ADR 0007 — Baseline dos Modelos v1

**Data:** 2026-07-20  
**Status:** Registrado (não reversível)  
**Autor:** Gerado automaticamente por `scripts/baseline_avaliacao.py`

---

## Contexto

Antes de retreinar os modelos preditivos do Vetor PAS, foi necessário
capturar o estado de desempenho atual como linha de base. Este ADR
registra todas as métricas coletadas em **2026-07-20** para permitir
comparação objetiva com as versões retreinadas.

## Dataset

| Propriedade | Valor |
|---|---|
| Arquivo | `data/banco_alunos_pas_final.csv` |
| Registros totais | 74465 |
| Target | `EB_PAS3` (Escore Bruto PAS 3) |
| Target média | 24.152 |
| Target std | 18.272 |
| Target min/max | 0.000 / 92.316 |
| Triênios | 2016-2018, 2017-2019, 2018-2020, 2019-2021, 2020-2022, 2021-2023, 2022-2024, 2023-2025, 2024-2026 |

## Features Base

`EB_PAS1, EB_PAS2, Cresc_EB, Media_EB, Std_EB, CV_EB`

---

## Métricas por Modelo

### `modelo_linear`

| Métrica | Valor |
|---|---|
| n_amostras | 74465 |
| MAE | 18.0598 |
| RMSE | 33.5239 |
| MSE | 1123.8518 |
| R2 | -2.3663 |
| MAPE_pct | 4.3191499559703214e+18 |
| MedAE | 11.1745 |
| MaxErr | 5183.8923 |
| Dentro_1pt_pct | 5.35 |
| Dentro_3pt_pct | 15.99 |
| Dentro_5pt_pct | 25.93 |
| Residuo_media | -14.1724 |
| Residuo_std | 30.3808 |
| Residuo_p5 | -54.1901 |
| Residuo_p25 | -26.7103 |
| Residuo_p75 | 0.1875 |
| Residuo_p95 | 11.1643 |
| Normalidade_stat | 264229.9635 |
| Normalidade_p | 0.0 |
| Vies_t | -127.297 |
| Vies_p | 0.0 |

**Permutation Importance (top 3 features, índice → importância):**  
- `Media_EB`: 0.4913
- `EB_PAS1`: 0.3971
- `EB_PAS2`: -0.0004

**Desempenho por triênio:**

| Triênio | n | MAE | RMSE | R² |
|---|---|---|---|---|
| 2016-2018 | 3194 | 10.1047 | 13.6194 | 0.2436 |
| 2017-2019 | 9300 | 10.9848 | 26.9523 | -3.5527 |
| 2018-2020 | 5556 | 10.625 | 70.9519 | -24.322 |
| 2019-2021 | 8105 | 12.7283 | 17.7679 | -0.8193 |
| 2020-2022 | 6854 | 10.6492 | 17.4599 | -0.5398 |
| 2021-2023 | 7629 | 10.2692 | 14.3215 | -0.0514 |
| 2022-2024 | 8116 | 11.4368 | 21.8224 | -1.0138 |
| 2023-2025 | 8718 | 10.3149 | 17.6619 | -0.4225 |
| 2024-2026 | 16993 | 42.0239 | 45.7531 | 0.0 |

### `modelo_lgbm`

| Métrica | Valor |
|---|---|
| n_amostras | 74465 |
| MAE | 13.9354 |
| RMSE | 17.5213 |
| MSE | 306.9971 |
| R2 | 0.0805 |
| MAPE_pct | 2.66788363271591e+18 |
| MedAE | 11.2474 |
| MaxErr | 64.5557 |
| Dentro_1pt_pct | 5.31 |
| Dentro_3pt_pct | 15.49 |
| Dentro_5pt_pct | 25.4 |
| Residuo_media | -1.8309 |
| Residuo_std | 17.4254 |
| Residuo_p5 | -29.3923 |
| Residuo_p25 | -14.2098 |
| Residuo_p75 | 9.574 |
| Residuo_p95 | 26.6175 |
| Normalidade_stat | 416.3129 |
| Normalidade_p | 0.0 |
| Vies_t | -28.672 |
| Vies_p | 0.0 |
| CV_MAE_media | 12.0014 |
| CV_MAE_std | 0.0726 |
| CV_R2_media | 0.2992 |
| CV_R2_std | 0.0052 |
| CV_RMSE_media | 15.2953 |
| CV_RMSE_std | 0.0809 |

**Permutation Importance (top 3 features, índice → importância):**  
- `EB_PAS1`: 0.1725
- `Media_EB`: 0.0192
- `EB_PAS2`: 0.0041

**Desempenho por triênio:**

| Triênio | n | MAE | RMSE | R² |
|---|---|---|---|---|
| 2016-2018 | 3194 | 11.9706 | 15.6794 | -0.0025 |
| 2017-2019 | 9300 | 8.5339 | 11.2389 | 0.2084 |
| 2018-2020 | 5556 | 10.5255 | 13.9815 | 0.0167 |
| 2019-2021 | 8105 | 9.0613 | 11.529 | 0.234 |
| 2020-2022 | 6854 | 10.0265 | 13.1641 | 0.1247 |
| 2021-2023 | 7629 | 10.0831 | 13.3556 | 0.0857 |
| 2022-2024 | 8116 | 12.4715 | 16.3195 | -0.1262 |
| 2023-2025 | 8718 | 11.4943 | 15.3171 | -0.0699 |
| 2024-2026 | 16993 | 25.9581 | 26.5537 | 0.0 |

### `modelo_lgbm_v1`

| Métrica | Valor |
|---|---|
| n_amostras | 74465 |
| MAE | 13.9354 |
| RMSE | 17.5213 |
| MSE | 306.9971 |
| R2 | 0.0805 |
| MAPE_pct | 2.66788363271591e+18 |
| MedAE | 11.2474 |
| MaxErr | 64.5557 |
| Dentro_1pt_pct | 5.31 |
| Dentro_3pt_pct | 15.49 |
| Dentro_5pt_pct | 25.4 |
| Residuo_media | -1.8309 |
| Residuo_std | 17.4254 |
| Residuo_p5 | -29.3923 |
| Residuo_p25 | -14.2098 |
| Residuo_p75 | 9.574 |
| Residuo_p95 | 26.6175 |
| Normalidade_stat | 416.3129 |
| Normalidade_p | 0.0 |
| Vies_t | -28.672 |
| Vies_p | 0.0 |
| CV_MAE_media | 12.0014 |
| CV_MAE_std | 0.0726 |
| CV_R2_media | 0.2992 |
| CV_R2_std | 0.0052 |
| CV_RMSE_media | 15.2953 |
| CV_RMSE_std | 0.0809 |

**Permutation Importance (top 3 features, índice → importância):**  
- `EB_PAS1`: 0.1725
- `Media_EB`: 0.0192
- `EB_PAS2`: 0.0041

**Desempenho por triênio:**

| Triênio | n | MAE | RMSE | R² |
|---|---|---|---|---|
| 2016-2018 | 3194 | 11.9706 | 15.6794 | -0.0025 |
| 2017-2019 | 9300 | 8.5339 | 11.2389 | 0.2084 |
| 2018-2020 | 5556 | 10.5255 | 13.9815 | 0.0167 |
| 2019-2021 | 8105 | 9.0613 | 11.529 | 0.234 |
| 2020-2022 | 6854 | 10.0265 | 13.1641 | 0.1247 |
| 2021-2023 | 7629 | 10.0831 | 13.3556 | 0.0857 |
| 2022-2024 | 8116 | 12.4715 | 16.3195 | -0.1262 |
| 2023-2025 | 8718 | 11.4943 | 15.3171 | -0.0699 |
| 2024-2026 | 16993 | 25.9581 | 26.5537 | 0.0 |

### `modelo_arg_final`

| Métrica | Valor |
|---|---|
| n_amostras | 74465 |
| MAE | 50.2357 |
| RMSE | 54.2591 |
| MSE | 2944.049 |
| R2 | -7.8183 |
| MAPE_pct | 3.053776395193703e+18 |
| MedAE | 52.8267 |
| MaxErr | 129.3432 |
| Dentro_1pt_pct | 0.32 |
| Dentro_3pt_pct | 1.26 |
| Dentro_5pt_pct | 2.5 |
| Residuo_media | 48.794 |
| Residuo_std | 23.7318 |
| Residuo_p5 | 3.971 |
| Residuo_p25 | 38.0582 |
| Residuo_p75 | 63.3671 |
| Residuo_p95 | 80.3929 |
| Normalidade_stat | 13948.2121 |
| Normalidade_p | 0.0 |
| Vies_t | 561.0593 |
| Vies_p | 0.0 |
| CV_MAE_media | 12.0014 |
| CV_MAE_std | 0.0726 |
| CV_R2_media | 0.2992 |
| CV_R2_std | 0.0052 |
| CV_RMSE_media | 15.2953 |
| CV_RMSE_std | 0.0809 |

**Permutation Importance (top 3 features, índice → importância):**  
- `EB_PAS1`: 1.7324
- `Cresc_EB`: 0.7673
- `CV_EB`: 0.1

**Desempenho por triênio:**

| Triênio | n | MAE | RMSE | R² |
|---|---|---|---|---|
| 2016-2018 | 3194 | 62.9802 | 64.5382 | -15.9852 |
| 2017-2019 | 9300 | 54.0785 | 56.0704 | -18.7034 |
| 2018-2020 | 5556 | 58.5407 | 60.417 | -17.3607 |
| 2019-2021 | 8105 | 49.0802 | 51.0866 | -14.04 |
| 2020-2022 | 6854 | 57.0151 | 58.8891 | -16.5165 |
| 2021-2023 | 7629 | 59.4753 | 61.3524 | -18.295 |
| 2022-2024 | 8116 | 56.2205 | 58.6781 | -13.5597 |
| 2023-2025 | 8718 | 58.272 | 60.6826 | -15.7921 |
| 2024-2026 | 16993 | 29.7089 | 36.5658 | 0.0 |

### `modelo_rf`

| Métrica | Valor |
|---|---|
| n_amostras | 74465 |
| MAE | 13.6857 |
| RMSE | 17.3462 |
| MSE | 300.8905 |
| R2 | 0.0987 |
| MAPE_pct | 2.673257555207889e+18 |
| MedAE | 10.8081 |
| MaxErr | 68.3584 |
| Dentro_1pt_pct | 5.57 |
| Dentro_3pt_pct | 16.13 |
| Dentro_5pt_pct | 26.23 |
| Residuo_media | -2.0498 |
| Residuo_std | 17.2247 |
| Residuo_p5 | -30.0775 |
| Residuo_p25 | -14.4024 |
| Residuo_p75 | 9.0188 |
| Residuo_p95 | 25.7911 |
| Normalidade_stat | 189.6678 |
| Normalidade_p | 0.0 |
| Vies_t | -32.4737 |
| Vies_p | 0.0 |
| CV_MAE_media | 9.8336 |
| CV_MAE_std | 0.0967 |
| CV_R2_media | 0.468 |
| CV_R2_std | 0.0074 |
| CV_RMSE_media | 13.3272 |
| CV_RMSE_std | 0.1176 |

**Permutation Importance (top 3 features, índice → importância):**  
- `EB_PAS1`: 0.2177
- `Media_EB`: 0.0136
- `Std_EB`: 0.0085

**Desempenho por triênio:**

| Triênio | n | MAE | RMSE | R² |
|---|---|---|---|---|
| 2016-2018 | 3194 | 11.4351 | 15.0801 | 0.0726 |
| 2017-2019 | 9300 | 8.1577 | 10.7943 | 0.2698 |
| 2018-2020 | 5556 | 10.0574 | 13.4691 | 0.0875 |
| 2019-2021 | 8105 | 8.8051 | 11.2356 | 0.2725 |
| 2020-2022 | 6854 | 9.5356 | 12.6271 | 0.1946 |
| 2021-2023 | 7629 | 9.611 | 12.8258 | 0.1568 |
| 2022-2024 | 8116 | 12.4217 | 16.1656 | -0.1051 |
| 2023-2025 | 8718 | 11.2391 | 14.9717 | -0.0222 |
| 2024-2026 | 16993 | 26.0105 | 26.7305 | 0.0 |

### `modelo_mlp`

| Métrica | Valor |
|---|---|
| n_amostras | 74465 |
| MAE | 85.0401 |
| RMSE | 167.8624 |
| MSE | 28177.7846 |
| R2 | -83.4007 |
| MAPE_pct | 1.2512010251547703e+19 |
| MedAE | 67.8988 |
| MaxErr | 31293.2607 |
| Dentro_1pt_pct | 0.1 |
| Dentro_3pt_pct | 0.3 |
| Dentro_5pt_pct | 0.52 |
| Residuo_media | -85.0201 |
| Residuo_std | 144.7389 |
| Residuo_p5 | -195.7008 |
| Residuo_p25 | -106.093 |
| Residuo_p75 | -45.2452 |
| Residuo_p95 | -21.6603 |
| Normalidade_stat | 332670.8725 |
| Normalidade_p | 0.0 |
| Vies_t | -160.2912 |
| Vies_p | 0.0 |
| CV_MAE_media | 12.0212 |
| CV_MAE_std | 0.0868 |
| CV_R2_media | 0.2938 |
| CV_R2_std | 0.0083 |
| CV_RMSE_media | 15.3548 |
| CV_RMSE_std | 0.0917 |

**Permutation Importance (top 3 features, índice → importância):**  
- `EB_PAS1`: 0.8141
- `Std_EB`: -0.1373
- `EB_PAS2`: -0.1744

**Desempenho por triênio:**

| Triênio | n | MAE | RMSE | R² |
|---|---|---|---|---|
| 2016-2018 | 3194 | 60.2573 | 75.9126 | -22.4998 |
| 2017-2019 | 9300 | 77.8784 | 160.8199 | -161.0892 |
| 2018-2020 | 5556 | 74.1532 | 428.418 | -922.2229 |
| 2019-2021 | 8105 | 79.4986 | 98.8531 | -55.3138 |
| 2020-2022 | 6854 | 67.6681 | 98.678 | -48.1833 |
| 2021-2023 | 7629 | 68.4524 | 84.5765 | -35.6675 |
| 2022-2024 | 8116 | 76.5843 | 125.9075 | -66.0356 |
| 2023-2025 | 8718 | 78.3723 | 111.5884 | -55.7825 |
| 2024-2026 | 16993 | 121.7337 | 149.9008 | 0.0 |

### `p1_pas3_model`

> ⚠ Erro na avaliação: `Modelo não carregado — incompatibilidade de versão do sklearn`

### `red_pas3_model`

> ⚠ Erro na avaliação: `Modelo não carregado — incompatibilidade de versão do sklearn`

### `ensemble_dinamico`

| Métrica | Valor |
|---|---|
| n_amostras | 74465 |
| MAE | 13.4611 |
| RMSE | 24.9298 |
| MSE | 621.4934 |
| R2 | -0.8616 |
| MAPE_pct | 3.27030801311216e+18 |
| MedAE | 9.1032 |
| MaxErr | 4147.2168 |
| Dentro_1pt_pct | 6.28 |
| Dentro_3pt_pct | 18.72 |
| Dentro_5pt_pct | 30.17 |
| Residuo_media | -7.1421 |
| Residuo_std | 23.8848 |
| Residuo_p5 | -37.9349 |
| Residuo_p25 | -17.254 |
| Residuo_p75 | 4.0873 |
| Residuo_p95 | 15.4827 |
| Normalidade_stat | 270551.6752 |
| Normalidade_p | 0.0 |
| Vies_t | -81.5976 |
| Vies_p | 0.0 |
| CV_medio_alunos | 26.1138 |
| Peso_conservador_medio | 0.5076 |

**Distribuição de estratégias:**  
- `conservador`: 38706 alunos
- `arrojado`: 35759 alunos

**Desempenho por triênio:**

| Triênio | n | MAE | RMSE | R² |
|---|---|---|---|---|

### `meta_model`

> ⚠ Erro na avaliação: `could not convert string to float: '6.643.1.1.1ResultadofinaldoscandidatossubjudicenositensdotipoDeresultadofinalnaprovaderedaçãoemLínguaPortuguesa'`

### `semaforo_calibracao`

| Métrica | Valor |
|---|---|
| nota | Pendente — requer dados de curso-alvo por aluno (nova extração) |

---

## Ranking Geral (MAE ascendente)

| Posição | Modelo | MAE | RMSE | R² | Dentro±3pts |
|---|---|---|---|---|---|
| 1° | `ensemble_dinamico` | 13.4611 | 24.9298 | -0.8616 | 18.72% |
| 2° | `modelo_rf` | 13.6857 | 17.3462 | 0.0987 | 16.13% |
| 3° | `modelo_lgbm` | 13.9354 | 17.5213 | 0.0805 | 15.49% |
| 4° | `modelo_lgbm_v1` | 13.9354 | 17.5213 | 0.0805 | 15.49% |
| 5° | `modelo_linear` | 18.0598 | 33.5239 | -2.3663 | 15.99% |
| 6° | `modelo_arg_final` | 50.2357 | 54.2591 | -7.8183 | 1.26% |
| 7° | `modelo_mlp` | 85.0401 | 167.8624 | -83.4007 | 0.3% |

---

## Decisão

Este ADR é **somente leitura** — registra o estado dos modelos antes do retreinamento.
Após o retreinamento, criar `ADR-0008` com as métricas dos novos modelos e
comparação direta com os valores acima.

## Consequências

- Qualquer métrica nos modelos retreinados deve ser comparada contra este ADR.
- Se o R² ou MAE piorarem, o retreinamento deve ser revisado antes de substituir os artefatos.
- Se a calibração do semáforo (% aprovados por cor) piorar, o retreinamento não deve ser colocado em produção.
