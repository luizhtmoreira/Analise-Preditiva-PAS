# Diferenças entre o `OFFICIAL_STATS` atual e os Editais oficiais

Gerado por `python -m pas_extraction.cli stats-diff`. **Este comando não altera o `OFFICIAL_STATS`** — só lê, compara e imprime.

Desde o ticket 12 o `OFFICIAL_STATS` *é* o dado publicado, então a seção 1 deve sair inteiramente zerada: qualquer Δ diferente de zero ali é regressão, não estimativa residual. A seção 2 é a única que continua com Δ por construção (ver o texto dela).

- Entradas do `OFFICIAL_STATS` comparadas: **24**
- Entradas do `OFFICIAL_STATS` sem cobertura nos Editais: **0**
- Etapas oficiais ausentes do `OFFICIAL_STATS`: **0**
- Divergências entre Editais da mesma Etapa: **0**

## 1. Parte II e Redação — comparação 1-para-1

`atual` = valor no `OFFICIAL_STATS`; `oficial` = tabela publicada no Edital; `Δ` = atual − oficial. Antes do ticket 12 o lado `atual` era estimado do `banco_alunos_pas_final.csv` e os Δ chegavam a +0,7; hoje devem ser todos zero.

| Ano / Etapa | Campo | Atual | Oficial | Δ | Δ % | Fonte oficial |
|---|---|---:|---:|---:|---:|---|
| 2016 / Etapa 1 | `m_p2` | 23.738 | 23.738 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2016 / Etapa 1 | `dp_p2` | 13.098 | 13.098 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2016 / Etapa 1 | `m_red` | 5.983 | 5.983 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2016 / Etapa 1 | `dp_red` | 2.702 | 2.702 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `m_p2` | 27.045 | 27.045 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `dp_p2` | 13.441 | 13.441 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `m_red` | 6.174 | 6.174 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 1 | `dp_red` | 2.664 | 2.664 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `m_p2` | 19.769 | 19.769 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `dp_p2` | 11.666 | 11.666 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `m_red` | 6.016 | 6.016 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2017 / Etapa 2 | `dp_red` | 2.224 | 2.224 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `m_p2` | 25.585 | 25.585 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `dp_p2` | 14.102 | 14.102 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `m_red` | 5.886 | 5.886 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 1 | `dp_red` | 2.400 | 2.400 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `m_p2` | 24.055 | 24.055 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `dp_p2` | 12.204 | 12.204 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `m_red` | 7.022 | 7.022 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 2 | `dp_red` | 1.639 | 1.639 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `m_p2` | 27.722 | 27.722 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `dp_p2` | 14.027 | 14.027 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `m_red` | 6.782 | 6.782 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2018 / Etapa 3 | `dp_red` | 1.738 | 1.738 | +0.000 | +0.00% | Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `m_p2` | 26.738 | 26.738 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `dp_p2` | 13.911 | 13.911 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `m_red` | 6.617 | 6.617 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 1 | `dp_red` | 2.393 | 2.393 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `m_p2` | 25.080 | 25.080 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `dp_p2` | 12.635 | 12.635 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `m_red` | 6.808 | 6.808 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 2 | `dp_red` | 1.758 | 1.758 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `m_p2` | 24.313 | 24.313 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `dp_p2` | 11.511 | 11.511 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `m_red` | 6.984 | 6.984 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2019 / Etapa 3 | `dp_red` | 1.775 | 1.775 | +0.000 | +0.00% | Ed_38_2017-2019_PAS_3_media_e_desvio_padrao.pdf |
| 2020 / Etapa 1 | `m_p2` | 24.520 | 24.520 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 1 | `dp_p2` | 13.344 | 13.344 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 1 | `m_red` | 5.720 | 5.720 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 1 | `dp_red` | 2.637 | 2.637 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2020 / Etapa 2 | `m_p2` | 28.736 | 28.736 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 2 | `dp_p2` | 12.864 | 12.864 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 2 | `m_red` | 7.008 | 7.008 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 2 | `dp_red` | 1.902 | 1.902 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `m_p2` | 27.816 | 27.816 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `dp_p2` | 12.814 | 12.814 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `m_red` | 6.928 | 6.928 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2020 / Etapa 3 | `dp_red` | 1.786 | 1.786 | +0.000 | +0.00% | ED_43_PAS_3 _2018 -2020_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 1 | `m_p2` | 21.501 | 21.501 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 1 | `dp_p2` | 12.422 | 12.422 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 1 | `m_red` | 5.941 | 5.941 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 1 | `dp_red` | 2.914 | 2.914 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2021 / Etapa 2 | `m_p2` | 25.083 | 25.083 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 2 | `dp_p2` | 11.897 | 11.897 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 2 | `m_red` | 7.090 | 7.090 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 2 | `dp_red` | 1.848 | 1.848 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2021 / Etapa 3 | `m_p2` | 23.424 | 23.424 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 3 | `dp_p2` | 12.300 | 12.300 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 3 | `m_red` | 6.988 | 6.988 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2021 / Etapa 3 | `dp_red` | 1.947 | 1.947 | +0.000 | +0.00% | ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf |
| 2022 / Etapa 1 | `m_p2` | 20.406 | 20.406 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 1 | `dp_p2` | 13.533 | 13.533 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 1 | `m_red` | 5.849 | 5.849 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 1 | `dp_red` | 2.793 | 2.793 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2022 / Etapa 2 | `m_p2` | 21.884 | 21.884 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 2 | `dp_p2` | 11.761 | 11.761 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 2 | `m_red` | 7.477 | 7.477 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 2 | `dp_red` | 1.655 | 1.655 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2022 / Etapa 3 | `m_p2` | 26.065 | 26.065 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2022 / Etapa 3 | `dp_p2` | 13.126 | 13.126 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2022 / Etapa 3 | `m_red` | 7.456 | 7.456 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2022 / Etapa 3 | `dp_red` | 1.760 | 1.760 | +0.000 | +0.00% | Ed_32_PAS_3_2020_2022_Media_Desvio_Padrão.pdf |
| 2023 / Etapa 1 | `m_p2` | 25.333 | 25.333 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2023 / Etapa 1 | `dp_p2` | 14.686 | 14.686 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2023 / Etapa 1 | `m_red` | 6.076 | 6.076 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2023 / Etapa 1 | `dp_red` | 2.816 | 2.816 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2023 / Etapa 2 | `m_p2` | 29.980 | 29.980 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 2 | `dp_p2` | 13.213 | 13.213 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 2 | `m_red` | 6.909 | 6.909 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 2 | `dp_red` | 1.973 | 1.973 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2023 / Etapa 3 | `m_p2` | 26.898 | 26.898 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2023 / Etapa 3 | `dp_p2` | 12.861 | 12.861 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2023 / Etapa 3 | `m_red` | 6.864 | 6.864 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2023 / Etapa 3 | `dp_red` | 1.989 | 1.989 | +0.000 | +0.00% | Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf |
| 2024 / Etapa 2 | `m_p2` | 29.275 | 29.275 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2024 / Etapa 2 | `dp_p2` | 14.604 | 14.604 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2024 / Etapa 2 | `m_red` | 6.877 | 6.877 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2024 / Etapa 2 | `dp_red` | 2.005 | 2.005 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2024 / Etapa 3 | `m_p2` | 31.740 | 31.740 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2024 / Etapa 3 | `dp_p2` | 14.063 | 14.063 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2024 / Etapa 3 | `m_red` | 7.548 | 7.548 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2024 / Etapa 3 | `dp_red` | 1.739 | 1.739 | +0.000 | +0.00% | Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf |
| 2025 / Etapa 3 | `m_p2` | 30.675 | 30.675 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2025 / Etapa 3 | `dp_p2` | 13.752 | 13.752 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2025 / Etapa 3 | `m_red` | 7.130 | 7.130 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |
| 2025 / Etapa 3 | `dp_red` | 1.789 | 1.789 | +0.000 | +0.00% | 14507028C1AE24C7B05C772373906EFD61A175FA0262FBAFE15DB9B52A29F3FD.pdf |

## 2. Parte 1 — um valor atual, três valores oficiais

O `ExamStats` expõe um `m_p1`/`dp_p1` único — desde o ticket 12, a média simples das três línguas guardadas em `parte_1` — enquanto o Edital publica a Parte 1 separada por língua estrangeira. Cada linha abaixo mostra esse agregado e os três valores oficiais que ele mistura; nenhum dos três é 'o certo': depende da língua que o Aluno fez, informação que o Resultado Final não imprime. Como as três diferem entre si, nenhum agregado possível zera as três colunas ao mesmo tempo — **os Δ desta seção medem a amplitude que a agregação achata, e não uma divergência a corrigir**.

| Ano / Etapa | Campo | Atual (agregado) | Oficial inglesa | Δ inglesa | Oficial francesa | Δ francesa | Oficial espanhola | Δ espanhola | Amplitude oficial |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016 / Etapa 1 | `m_p1` | 4.489 | 3.628 | +0.861 | 4.034 | +0.455 | 5.806 | -1.317 | 2.178 |
| 2016 / Etapa 1 | `dp_p1` | 2.675 | 2.687 | -0.012 | 2.875 | -0.200 | 2.462 | +0.213 | — |
| 2017 / Etapa 1 | `m_p1` | 3.325 | 2.966 | +0.359 | 3.039 | +0.286 | 3.970 | -0.645 | 1.004 |
| 2017 / Etapa 1 | `dp_p1` | 2.711 | 2.959 | -0.248 | 2.756 | -0.045 | 2.419 | +0.292 | — |
| 2017 / Etapa 2 | `m_p1` | 3.703 | 3.871 | -0.168 | 2.124 | +1.579 | 5.115 | -1.412 | 2.991 |
| 2017 / Etapa 2 | `dp_p1` | 2.529 | 3.226 | -0.697 | 2.314 | +0.215 | 2.048 | +0.481 | — |
| 2018 / Etapa 1 | `m_p1` | 3.778 | 2.679 | +1.099 | 4.690 | -0.912 | 3.965 | -0.187 | 2.011 |
| 2018 / Etapa 1 | `dp_p1` | 2.710 | 2.564 | +0.146 | 2.981 | -0.271 | 2.585 | +0.125 | — |
| 2018 / Etapa 2 | `m_p1` | 2.660 | 4.187 | -1.527 | 2.557 | +0.103 | 1.235 | +1.425 | 2.952 |
| 2018 / Etapa 2 | `dp_p1` | 2.595 | 2.724 | -0.129 | 2.848 | -0.253 | 2.214 | +0.381 | — |
| 2018 / Etapa 3 | `m_p1` | 3.713 | 5.035 | -1.322 | 2.432 | +1.281 | 3.671 | +0.042 | 2.603 |
| 2018 / Etapa 3 | `dp_p1` | 2.070 | 2.358 | -0.288 | 2.028 | +0.042 | 1.823 | +0.247 | — |
| 2019 / Etapa 1 | `m_p1` | 4.471 | 3.900 | +0.571 | 5.064 | -0.593 | 4.450 | +0.021 | 1.164 |
| 2019 / Etapa 1 | `dp_p1` | 2.658 | 2.781 | -0.123 | 2.756 | -0.098 | 2.437 | +0.221 | — |
| 2019 / Etapa 2 | `m_p1` | 4.161 | 4.259 | -0.098 | 4.263 | -0.102 | 3.962 | +0.199 | 0.301 |
| 2019 / Etapa 2 | `dp_p1` | 2.559 | 2.407 | +0.152 | 3.136 | -0.577 | 2.133 | +0.426 | — |
| 2019 / Etapa 3 | `m_p1` | 2.909 | 3.581 | -0.672 | 2.445 | +0.464 | 2.700 | +0.209 | 1.136 |
| 2019 / Etapa 3 | `dp_p1` | 1.769 | 2.095 | -0.326 | 1.520 | +0.249 | 1.693 | +0.076 | — |
| 2020 / Etapa 1 | `m_p1` | 3.464 | 1.843 | +1.621 | 4.805 | -1.341 | 3.745 | -0.281 | 2.962 |
| 2020 / Etapa 1 | `dp_p1` | 2.416 | 2.228 | +0.188 | 2.438 | -0.022 | 2.581 | -0.165 | — |
| 2020 / Etapa 2 | `m_p1` | 4.719 | 4.400 | +0.319 | 5.118 | -0.399 | 4.638 | +0.081 | 0.718 |
| 2020 / Etapa 2 | `dp_p1` | 2.237 | 2.644 | -0.407 | 1.969 | +0.268 | 2.098 | +0.139 | — |
| 2020 / Etapa 3 | `m_p1` | 3.753 | 4.598 | -0.845 | 3.546 | +0.207 | 3.115 | +0.638 | 1.483 |
| 2020 / Etapa 3 | `dp_p1` | 2.017 | 2.160 | -0.143 | 2.184 | -0.167 | 1.706 | +0.311 | — |
| 2021 / Etapa 1 | `m_p1` | 4.391 | 3.819 | +0.572 | 3.302 | +1.089 | 6.053 | -1.662 | 2.751 |
| 2021 / Etapa 1 | `dp_p1` | 2.804 | 3.274 | -0.470 | 2.476 | +0.328 | 2.662 | +0.142 | — |
| 2021 / Etapa 2 | `m_p1` | 4.512 | 2.682 | +1.830 | 6.354 | -1.842 | 4.501 | +0.011 | 3.672 |
| 2021 / Etapa 2 | `dp_p1` | 1.996 | 1.871 | +0.125 | 1.894 | +0.102 | 2.223 | -0.227 | — |
| 2021 / Etapa 3 | `m_p1` | 3.538 | 3.198 | +0.340 | 4.060 | -0.522 | 3.356 | +0.182 | 0.862 |
| 2021 / Etapa 3 | `dp_p1` | 1.839 | 1.760 | +0.079 | 1.942 | -0.103 | 1.815 | +0.024 | — |
| 2022 / Etapa 1 | `m_p1` | 3.475 | 3.665 | -0.190 | 3.620 | -0.145 | 3.140 | +0.335 | 0.525 |
| 2022 / Etapa 1 | `dp_p1` | 2.745 | 3.109 | -0.364 | 2.597 | +0.148 | 2.530 | +0.215 | — |
| 2022 / Etapa 2 | `m_p1` | 4.412 | 5.515 | -1.103 | 4.221 | +0.191 | 3.499 | +0.913 | 2.016 |
| 2022 / Etapa 2 | `dp_p1` | 2.455 | 2.536 | -0.081 | 2.397 | +0.058 | 2.432 | +0.023 | — |
| 2022 / Etapa 3 | `m_p1` | 3.450 | 3.589 | -0.139 | 3.928 | -0.478 | 2.832 | +0.618 | 1.096 |
| 2022 / Etapa 3 | `dp_p1` | 1.869 | 1.795 | +0.074 | 1.965 | -0.096 | 1.846 | +0.023 | — |
| 2023 / Etapa 1 | `m_p1` | 2.343 | 2.700 | -0.357 | 2.212 | +0.131 | 2.116 | +0.227 | 0.584 |
| 2023 / Etapa 1 | `dp_p1` | 2.510 | 2.821 | -0.311 | 2.601 | -0.091 | 2.107 | +0.403 | — |
| 2023 / Etapa 2 | `m_p1` | 3.455 | 3.958 | -0.503 | 3.260 | +0.195 | 3.147 | +0.308 | 0.811 |
| 2023 / Etapa 2 | `dp_p1` | 2.281 | 2.186 | +0.095 | 2.405 | -0.124 | 2.253 | +0.028 | — |
| 2023 / Etapa 3 | `m_p1` | 3.678 | 3.989 | -0.311 | 3.499 | +0.179 | 3.545 | +0.133 | 0.490 |
| 2023 / Etapa 3 | `dp_p1` | 1.879 | 2.049 | -0.170 | 1.877 | +0.002 | 1.712 | +0.167 | — |
| 2024 / Etapa 2 | `m_p1` | 3.273 | 5.095 | -1.822 | 3.493 | -0.220 | 1.231 | +2.042 | 3.864 |
| 2024 / Etapa 2 | `dp_p1` | 2.583 | 2.899 | -0.316 | 2.658 | -0.075 | 2.191 | +0.392 | — |
| 2024 / Etapa 3 | `m_p1` | 3.463 | 4.297 | -0.834 | 3.556 | -0.093 | 2.537 | +0.926 | 1.760 |
| 2024 / Etapa 3 | `dp_p1` | 1.848 | 2.137 | -0.289 | 1.683 | +0.165 | 1.724 | +0.124 | — |
| 2025 / Etapa 3 | `m_p1` | 4.317 | 4.104 | +0.213 | 4.630 | -0.313 | 4.218 | +0.099 | 0.526 |
| 2025 / Etapa 3 | `dp_p1` | 2.110 | 2.174 | -0.064 | 2.316 | -0.206 | 1.839 | +0.271 | — |

## 3. Entradas do `OFFICIAL_STATS` sem cobertura nos Editais extraídos

Nenhuma: toda entrada do `OFFICIAL_STATS` tem um Edital oficial correspondente.

## 4. Etapas oficiais ausentes do `OFFICIAL_STATS`

Nenhuma.

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

