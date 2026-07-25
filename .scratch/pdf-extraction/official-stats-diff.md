# Diferenças entre o `OFFICIAL_STATS` atual e os Editais oficiais

Gerado por `python -m pas_extraction.cli stats-diff` (ticket 11). **Nenhum valor foi alterado** — a substituição é o ticket 12.

- Entradas do `OFFICIAL_STATS` comparadas: **21**
- Entradas do `OFFICIAL_STATS` sem cobertura nos Editais: **0**
- Etapas oficiais ausentes do `OFFICIAL_STATS`: **3**
- Divergências entre Editais da mesma Etapa: **0**

## 1. Parte II e Redação — comparação 1-para-1

`atual` = `OFFICIAL_STATS` (estimado do `banco_alunos_pas_final.csv`); `oficial` = tabela publicada no Edital; `Δ` = atual − oficial.

| Ano / Etapa | Campo | Atual | Oficial | Δ | Δ % | Fonte oficial |
|---|---|---:|---:|---:|---:|---|
| 2016 / Etapa 1 | `m_p2` | 24.246 | 23.738 | +0.508 | +2.14% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2016 / Etapa 1 | `dp_p2` | 13.169 | 13.098 | +0.071 | +0.54% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2016 / Etapa 1 | `m_red` | 6.074 | 5.983 | +0.091 | +1.52% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2016 / Etapa 1 | `dp_red` | 2.669 | 2.702 | -0.033 | -1.22% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `m_p2` | 27.408 | 27.045 | +0.363 | +1.34% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `dp_p2` | 13.417 | 13.441 | -0.024 | -0.18% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `m_red` | 6.222 | 6.174 | +0.048 | +0.78% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `dp_red` | 2.639 | 2.664 | -0.025 | -0.94% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `m_p2` | 20.403 | 19.769 | +0.634 | +3.21% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `dp_p2` | 11.959 | 11.666 | +0.293 | +2.51% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `m_red` | 6.100 | 6.016 | +0.084 | +1.40% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `dp_red` | 2.246 | 2.224 | +0.022 | +0.99% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `m_p2` | 25.938 | 25.585 | +0.353 | +1.38% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `dp_p2` | 14.166 | 14.102 | +0.064 | +0.45% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `m_red` | 5.919 | 5.886 | +0.033 | +0.56% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `dp_red` | 2.406 | 2.400 | +0.006 | +0.25% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `m_p2` | 24.410 | 24.055 | +0.355 | +1.48% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `dp_p2` | 12.196 | 12.204 | -0.008 | -0.07% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `m_red` | 7.053 | 7.022 | +0.031 | +0.44% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `dp_red` | 1.641 | 1.639 | +0.002 | +0.12% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `m_p2` | 28.433 | 27.722 | +0.711 | +2.56% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `dp_p2` | 14.280 | 14.027 | +0.253 | +1.80% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `m_red` | 6.848 | 6.782 | +0.066 | +0.97% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `dp_red` | 1.724 | 1.738 | -0.014 | -0.81% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `m_p2` | 27.041 | 26.738 | +0.303 | +1.13% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `dp_p2` | 13.935 | 13.911 | +0.024 | +0.17% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `m_red` | 6.657 | 6.617 | +0.040 | +0.60% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `dp_red` | 2.373 | 2.393 | -0.020 | -0.84% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `m_p2` | 25.439 | 25.080 | +0.359 | +1.43% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `dp_p2` | 12.696 | 12.635 | +0.061 | +0.48% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `m_red` | 6.844 | 6.808 | +0.036 | +0.53% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `dp_red` | 1.752 | 1.758 | -0.006 | -0.34% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `m_p2` | 24.678 | 24.313 | +0.365 | +1.50% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `dp_p2` | 11.531 | 11.511 | +0.020 | +0.17% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `m_red` | 7.013 | 6.984 | +0.029 | +0.42% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `dp_red` | 1.772 | 1.775 | -0.003 | -0.17% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2020 / Etapa 1 | `m_p2` | 24.784 | 24.520 | +0.264 | +1.08% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 1 | `dp_p2` | 13.366 | 13.344 | +0.022 | +0.16% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 1 | `m_red` | 5.743 | 5.720 | +0.023 | +0.40% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 1 | `dp_red` | 2.637 | 2.637 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 2 | `m_p2` | 29.006 | 28.736 | +0.270 | +0.94% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 2 | `dp_p2` | 12.915 | 12.864 | +0.051 | +0.40% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 2 | `m_red` | 7.032 | 7.008 | +0.024 | +0.34% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 2 | `dp_red` | 1.903 | 1.902 | +0.001 | +0.05% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `m_p2` | 28.199 | 27.816 | +0.383 | +1.38% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `dp_p2` | 12.847 | 12.814 | +0.033 | +0.26% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `m_red` | 6.972 | 6.928 | +0.044 | +0.64% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `dp_red` | 1.783 | 1.786 | -0.003 | -0.17% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 1 | `m_p2` | 21.806 | 21.501 | +0.305 | +1.42% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 1 | `dp_p2` | 12.448 | 12.422 | +0.026 | +0.21% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 1 | `m_red` | 5.984 | 5.941 | +0.043 | +0.72% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 1 | `dp_red` | 2.908 | 2.914 | -0.006 | -0.21% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 2 | `m_p2` | 25.349 | 25.083 | +0.266 | +1.06% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 2 | `dp_p2` | 11.911 | 11.897 | +0.014 | +0.12% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 2 | `m_red` | 7.125 | 7.090 | +0.035 | +0.49% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 2 | `dp_red` | 1.839 | 1.848 | -0.009 | -0.49% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 3 | `m_p2` | 23.678 | 23.424 | +0.254 | +1.08% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 3 | `dp_p2` | 12.372 | 12.300 | +0.072 | +0.59% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 3 | `m_red` | 7.009 | 6.988 | +0.021 | +0.30% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 3 | `dp_red` | 1.947 | 1.947 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2022 / Etapa 1 | `m_p2` | 20.709 | 20.406 | +0.303 | +1.48% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 1 | `dp_p2` | 13.581 | 13.533 | +0.048 | +0.35% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 1 | `m_red` | 5.888 | 5.849 | +0.039 | +0.67% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 1 | `dp_red` | 2.779 | 2.793 | -0.014 | -0.50% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 2 | `m_p2` | 22.192 | 21.884 | +0.308 | +1.41% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 2 | `dp_p2` | 11.832 | 11.761 | +0.071 | +0.60% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 2 | `m_red` | 7.505 | 7.477 | +0.028 | +0.37% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 2 | `dp_red` | 1.645 | 1.655 | -0.010 | -0.60% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 3 | `m_p2` | 26.385 | 26.065 | +0.320 | +1.23% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2022 / Etapa 3 | `dp_p2` | 13.146 | 13.126 | +0.020 | +0.15% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2022 / Etapa 3 | `m_red` | 7.482 | 7.456 | +0.026 | +0.35% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2022 / Etapa 3 | `dp_red` | 1.752 | 1.760 | -0.008 | -0.45% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2023 / Etapa 2 | `m_p2` | 30.348 | 29.980 | +0.368 | +1.23% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 2 | `dp_p2` | 13.252 | 13.213 | +0.039 | +0.30% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 2 | `m_red` | 6.937 | 6.909 | +0.028 | +0.41% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 2 | `dp_red` | 1.972 | 1.973 | -0.001 | -0.05% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 3 | `m_p2` | 27.258 | 26.898 | +0.360 | +1.34% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2023 / Etapa 3 | `dp_p2` | 12.923 | 12.861 | +0.062 | +0.48% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2023 / Etapa 3 | `m_red` | 6.893 | 6.864 | +0.029 | +0.42% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2023 / Etapa 3 | `dp_red` | 1.984 | 1.989 | -0.005 | -0.25% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2024 / Etapa 3 | `m_p2` | 32.086 | 31.740 | +0.346 | +1.09% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2024 / Etapa 3 | `dp_p2` | 14.128 | 14.063 | +0.065 | +0.46% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2024 / Etapa 3 | `m_red` | 7.579 | 7.548 | +0.031 | +0.41% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2024 / Etapa 3 | `dp_red` | 1.730 | 1.739 | -0.009 | -0.52% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |

## 2. Parte 1 — um valor atual, três valores oficiais

O `ExamStats` tem um `m_p1`/`dp_p1` único; o Edital publica a Parte 1 separada por língua estrangeira. Cada linha abaixo mostra o valor agregado atual e os três valores oficiais que ele mistura — nenhum dos três é 'o certo': depende da língua que o Aluno fez, informação que o Resultado Final não imprime.

| Ano / Etapa | Campo | Atual (agregado) | Oficial inglesa | Δ inglesa | Oficial francesa | Δ francesa | Oficial espanhola | Δ espanhola | Amplitude oficial |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016 / Etapa 1 | `m_p1` | 4.421 | 3.628 | +0.793 | 4.034 | +0.387 | 5.806 | -1.385 | 2.178 |
| 2016 / Etapa 1 | `dp_p1` | 2.782 | 2.687 | +0.095 | 2.875 | -0.093 | 2.462 | +0.320 | — |
| 2017 / Etapa 1 | `m_p1` | 3.316 | 2.966 | +0.350 | 3.039 | +0.277 | 3.970 | -0.654 | 1.004 |
| 2017 / Etapa 1 | `dp_p1` | 2.859 | 2.959 | -0.100 | 2.756 | +0.103 | 2.419 | +0.440 | — |
| 2017 / Etapa 2 | `m_p1` | 4.516 | 3.871 | +0.645 | 2.124 | +2.392 | 5.115 | -0.599 | 2.991 |
| 2017 / Etapa 2 | `dp_p1` | 2.806 | 3.226 | -0.420 | 2.314 | +0.492 | 2.048 | +0.758 | — |
| 2018 / Etapa 1 | `m_p1` | 3.135 | 2.679 | +0.456 | 4.690 | -1.555 | 3.965 | -0.830 | 2.011 |
| 2018 / Etapa 1 | `dp_p1` | 2.651 | 2.564 | +0.087 | 2.981 | -0.330 | 2.585 | +0.066 | — |
| 2018 / Etapa 2 | `m_p1` | 3.101 | 4.187 | -1.086 | 2.557 | +0.544 | 1.235 | +1.866 | 2.952 |
| 2018 / Etapa 2 | `dp_p1` | 2.907 | 2.724 | +0.183 | 2.848 | +0.059 | 2.214 | +0.693 | — |
| 2018 / Etapa 3 | `m_p1` | 4.550 | 5.035 | -0.485 | 2.432 | +2.118 | 3.671 | +0.879 | 2.603 |
| 2018 / Etapa 3 | `dp_p1` | 2.277 | 2.358 | -0.081 | 2.028 | +0.249 | 1.823 | +0.454 | — |
| 2019 / Etapa 1 | `m_p1` | 4.117 | 3.900 | +0.217 | 5.064 | -0.947 | 4.450 | -0.333 | 1.164 |
| 2019 / Etapa 1 | `dp_p1` | 2.693 | 2.781 | -0.088 | 2.756 | -0.063 | 2.437 | +0.256 | — |
| 2019 / Etapa 2 | `m_p1` | 4.184 | 4.259 | -0.075 | 4.263 | -0.079 | 3.962 | +0.222 | 0.301 |
| 2019 / Etapa 2 | `dp_p1` | 2.291 | 2.407 | -0.116 | 3.136 | -0.845 | 2.133 | +0.158 | — |
| 2019 / Etapa 3 | `m_p1` | 3.268 | 3.581 | -0.313 | 2.445 | +0.823 | 2.700 | +0.568 | 1.136 |
| 2019 / Etapa 3 | `dp_p1` | 2.003 | 2.095 | -0.092 | 1.520 | +0.483 | 1.693 | +0.310 | — |
| 2020 / Etapa 1 | `m_p1` | 2.328 | 1.843 | +0.485 | 4.805 | -2.477 | 3.745 | -1.417 | 2.962 |
| 2020 / Etapa 1 | `dp_p1` | 2.470 | 2.228 | +0.242 | 2.438 | +0.032 | 2.581 | -0.111 | — |
| 2020 / Etapa 2 | `m_p1` | 4.528 | 4.400 | +0.128 | 5.118 | -0.590 | 4.638 | -0.110 | 0.718 |
| 2020 / Etapa 2 | `dp_p1` | 2.456 | 2.644 | -0.188 | 1.969 | +0.487 | 2.098 | +0.358 | — |
| 2020 / Etapa 3 | `m_p1` | 4.018 | 4.598 | -0.580 | 3.546 | +0.472 | 3.115 | +0.903 | 1.483 |
| 2020 / Etapa 3 | `dp_p1` | 2.114 | 2.160 | -0.046 | 2.184 | -0.070 | 1.706 | +0.408 | — |
| 2021 / Etapa 1 | `m_p1` | 4.373 | 3.819 | +0.554 | 3.302 | +1.071 | 6.053 | -1.680 | 2.751 |
| 2021 / Etapa 1 | `dp_p1` | 3.277 | 3.274 | +0.003 | 2.476 | +0.801 | 2.662 | +0.615 | — |
| 2021 / Etapa 2 | `m_p1` | 3.328 | 2.682 | +0.646 | 6.354 | -3.026 | 4.501 | -1.173 | 3.672 |
| 2021 / Etapa 2 | `dp_p1` | 2.176 | 1.871 | +0.305 | 1.894 | +0.282 | 2.223 | -0.047 | — |
| 2021 / Etapa 3 | `m_p1` | 3.284 | 3.198 | +0.086 | 4.060 | -0.776 | 3.356 | -0.072 | 0.862 |
| 2021 / Etapa 3 | `dp_p1` | 1.791 | 1.760 | +0.031 | 1.942 | -0.151 | 1.815 | -0.024 | — |
| 2022 / Etapa 1 | `m_p1` | 3.604 | 3.665 | -0.061 | 3.620 | -0.016 | 3.140 | +0.464 | 0.525 |
| 2022 / Etapa 1 | `dp_p1` | 3.005 | 3.109 | -0.104 | 2.597 | +0.408 | 2.530 | +0.475 | — |
| 2022 / Etapa 2 | `m_p1` | 4.861 | 5.515 | -0.654 | 4.221 | +0.640 | 3.499 | +1.362 | 2.016 |
| 2022 / Etapa 2 | `dp_p1` | 2.655 | 2.536 | +0.119 | 2.397 | +0.258 | 2.432 | +0.223 | — |
| 2022 / Etapa 3 | `m_p1` | 3.361 | 3.589 | -0.228 | 3.928 | -0.567 | 2.832 | +0.529 | 1.096 |
| 2022 / Etapa 3 | `dp_p1` | 1.849 | 1.795 | +0.054 | 1.965 | -0.116 | 1.846 | +0.003 | — |
| 2023 / Etapa 2 | `m_p1` | 3.739 | 3.958 | -0.219 | 3.260 | +0.479 | 3.147 | +0.592 | 0.811 |
| 2023 / Etapa 2 | `dp_p1` | 2.238 | 2.186 | +0.052 | 2.405 | -0.167 | 2.253 | -0.015 | — |
| 2023 / Etapa 3 | `m_p1` | 3.857 | 3.989 | -0.132 | 3.499 | +0.358 | 3.545 | +0.312 | 0.490 |
| 2023 / Etapa 3 | `dp_p1` | 1.947 | 2.049 | -0.102 | 1.877 | +0.070 | 1.712 | +0.235 | — |
| 2024 / Etapa 3 | `m_p1` | 3.768 | 4.297 | -0.529 | 3.556 | +0.212 | 2.537 | +1.231 | 1.760 |
| 2024 / Etapa 3 | `dp_p1` | 2.178 | 2.137 | +0.041 | 1.683 | +0.495 | 1.724 | +0.454 | — |

## 3. Entradas do `OFFICIAL_STATS` sem cobertura nos Editais extraídos

Nenhuma: toda entrada do `OFFICIAL_STATS` tem um Edital oficial correspondente.

## 4. Etapas oficiais ausentes do `OFFICIAL_STATS`

Publicadas em Edital, mas sem entrada no `OFFICIAL_STATS` — dado novo, não correção. Incluí-las é decisão do ticket 12.

| Ano / Etapa | Triênio | Fontes |
|---|---|---|
| 2023 / Etapa 1 | 2023/2025 | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2024 / Etapa 2 | 2023/2025 | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2025 / Etapa 3 | 2023/2025 | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |

## 5. Divergências entre Editais da mesma Etapa

Nenhuma: onde o mesmo triênio aparece em mais de um Edital, os valores batem exatamente.

## 6. Cobertura da varredura

| Ano / Etapa | Triênio | Fontes |
|---|---|---|
| 2016 / Etapa 1 | 2016/2018 | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | 2017/2019 | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | 2016/2018 | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | 2018/2020 | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | 2017/2019 | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | 2016/2018 | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | 2019/2021 | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | 2018/2020 | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | 2017/2019 | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2020 / Etapa 1 | 2020/2022 | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 2 | 2019/2021 | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | 2018/2020 | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 1 | 2021/2023 | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 2 | 2020/2022 | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 3 | 2019/2021 | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2022 / Etapa 1 | 2022/2024 | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 2 | 2021/2023 | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 3 | 2020/2022 | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2023 / Etapa 1 | 2023/2025 | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2023 / Etapa 2 | 2022/2024 | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 3 | 2021/2023 | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2024 / Etapa 2 | 2023/2025 | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2024 / Etapa 3 | 2022/2024 | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2025 / Etapa 3 | 2023/2025 | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |

Editais que não contribuíram com nenhum valor, e por quê (69):

| Motivo | Editais |
|---|---:|
| Família Convocação — não publica tabela de médias | 64 |
| Resultado Final sem tabela de médias (varredura completa, todas as páginas) | 5 |

