"""Roda a calibração do Deslocamento (ticket 06) sobre os Editais isolados de Etapa de
`data/pdfs/editais-de-etapa` e o `resultado_final.csv` da extração, e imprime o resultado.

Sai com código 1 se o portão reprovar — para poder ser usado em CI sem ninguém ler o texto.

    .venv/bin/python scripts/medir_deslocamento.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # type: ignore

from pas_extraction.calibracao_deslocamento import (
    PortaoReprovadoError,
    coletar_stats_empiricos,
    formatar_markdown,
    gerar_relatorio,
    verificar_portao,
)
from pas_intelligence.training_dataset import REQUIRED_SOURCE_COLUMNS, anos_do_trienio  # type: ignore

PDFS = REPO / "data" / "pdfs" / "editais-de-etapa"
CSV_RESULTADO_FINAL = REPO / ".scratch" / "pdf-extraction" / "saida-nova" / "resultado_final.csv"

# Um triênio entra aqui só quando tem Edital isolado de Etapa 1 e/ou 2 baixado e organizado em
# `data/pdfs/editais-de-etapa` (ver `data/pdfs/INDICE.md`). 2022/2024 contribui só a Etapa 1 —
# não há Edital de Etapa 2 em disco para ele — e por isso não entra na tabela por triênio
# (`medir_trienios` exige as duas), só no Deslocamento agregado por Etapa.
MAPEAMENTO = {
    "2018/2020": {
        1: PDFS / "Ed 8 PAS Subprograma 2018 1a etapa Res final tipo D e redação_revisada.pdf",
        2: PDFS / "ED_17_PAS_2_2018-2020_res_fin_itens_tipo_D_e_redação.pdf",
    },
    "2019/2021": {
        1: PDFS / "ED_6_PAS_1_2019-2021_fin_itens_tipo_D_e_redacao.pdf",
        2: PDFS / "ED_20_PAS_2_2019-2021_final_itens_tipo_D_e_redacao.pdf",
    },
    "2020/2022": {
        1: PDFS / "ED_8_PAS_1_2020-2022_final_itens_tipo_D_e_redacao.pdf",
        2: PDFS / "ED_13_PAS_2 _2020-2022_Res_Final_Tipo_D_Redação.pdf",
    },
    "2021/2023": {
        1: PDFS / "ED_5_PAS_1_2021-2023_Res_Final_Tipo_D_e_Redação.pdf",
        2: PDFS / "ED_16_PAS_2_2021-2023_Res_Final_Tipo_D_Redação.pdf",
    },
    "2022/2024": {
        1: PDFS / "ED_8_PAS_1_2022-2024_Retificação_Res_Final_tipo_D_redação.pdf",
    },
    "2023/2025": {
        # Ed_8/2023 é a Retificação PARCIAL (14 páginas, ~827 registros) — não serve como o
        # Edital completo da Etapa. Ed_7 é o Resultado Final inteiro.
        1: PDFS / "Ed_7_PAS_1_2023_2025_Res_final_tipo_D_redação.pdf",
        2: PDFS / "Ed_15_PAS_2_2023-2025_Res_final_tipo_D_redacao.pdf",
    },
}


def main() -> int:
    stats_empiricos = coletar_stats_empiricos(MAPEAMENTO)

    df = pd.read_csv(CSV_RESULTADO_FINAL, usecols=REQUIRED_SOURCE_COLUMNS)
    df = df[df["checksum_fecha"] == True].copy()  # noqa: E712
    anos = df["trienio"].map(anos_do_trienio)
    df["_ano_e1"] = anos.map(lambda t: t[0])
    df["_ano_e2"] = anos.map(lambda t: t[1])

    relatorio = gerar_relatorio(df, stats_empiricos)
    print(formatar_markdown(relatorio))

    try:
        verificar_portao(relatorio)
        print("PORTÃO: APROVADO")
        return 0
    except PortaoReprovadoError as e:
        print(f"PORTÃO: REPROVADO — {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
