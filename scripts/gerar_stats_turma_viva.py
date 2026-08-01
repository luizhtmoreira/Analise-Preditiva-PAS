"""Recomputa as duas entradas `Origem.DERIVADA` de `OFFICIAL_STATS` — `(2024, 1)` e
`(2025, 2)`, a Turma viva (ticket 07) — e imprime bruto/corrigido lado a lado.

É a evidência rastreável por trás dos números escritos à mão em `pas_constants.py`: roda a
mesma porta do ticket 06 (`Correcao.aplicar`) sobre os Editais isolados de Etapa já em disco.
Não escreve nada — só confere que os literais no código continuam batendo com o que a conta
produz.

    .venv/bin/python scripts/gerar_stats_turma_viva.py

**Cuidado ao reusar após o ticket 07 ter mergeado:** por padrão `montar_calibracao` calibra
contra `OFFICIAL_STATS` inteiro, que a esta altura já contém as duas entradas `DERIVADA`. Se
elas entrarem como alvo da calibração, o script mede a correção contra um valor que ela mesma
produziu — resíduo artificialmente perto de zero, calibração contaminada. Por isso o
`stats_oficiais` passado abaixo é filtrado para `origem is Origem.EDITAL` explicitamente.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pas_extraction.calibracao_deslocamento import coletar_stats_empiricos, montar_calibracao
from pas_intelligence.argument_calculator import HistoricalStats
from pas_intelligence.pas_constants import OFFICIAL_STATS, Origem

PDFS = REPO / "data" / "pdfs" / "editais-de-etapa"

# Mesmo mapeamento de `medir_deslocamento.py`, mais os dois arquivos da Turma viva — o Edital
# isolado de Etapa 1/2024 e o de Etapa 2/2025, únicos que existem para o triênio 2024-2026.
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
        1: PDFS / "Ed_7_PAS_1_2023_2025_Res_final_tipo_D_redação.pdf",
        2: PDFS / "Ed_15_PAS_2_2023-2025_Res_final_tipo_D_redacao.pdf",
    },
    "2024/2026": {
        1: PDFS / "Ed_8_2024_PAS_UnB_1_2024-2026_Res_final_tipo_D_redação.pdf",
        2: PDFS / "8BF9D771C58F383321E81D054B720A9E75CF911E7A921DF6E017670779B74EEF.pdf",
    },
}


def _stats_oficiais_apenas_edital() -> dict[tuple[int, int, str], HistoricalStats]:
    """`OFFICIAL_STATS` achatado, excluindo qualquer entrada `DERIVADA` — nunca alvo de
    calibração, só o Edital de médias e desvios de verdade pode ensinar a correção."""
    cache: dict[tuple[int, int, str], HistoricalStats] = {}
    for (ano, etapa), stats in OFFICIAL_STATS.items():
        if stats.origem is not Origem.EDITAL:
            continue
        for lingua, valor in stats.parte_1.items():
            cache[(ano, etapa, lingua)] = HistoricalStats(
                mean_p1=valor.media, std_p1=valor.desvio_padrao,
                mean_p2=stats.m_p2, std_p2=stats.dp_p2,
                mean_red=stats.m_red, std_red=stats.dp_red,
            )
    return cache


def main() -> None:
    stats_empiricos = coletar_stats_empiricos(MAPEAMENTO)
    stats_oficiais = _stats_oficiais_apenas_edital()

    correcao_e1 = montar_calibracao(stats_empiricos, 1, stats_oficiais).correcao()
    correcao_e2 = montar_calibracao(stats_empiricos, 2, stats_oficiais).correcao()

    for ano, etapa, correcao in ((2024, 1, correcao_e1), (2025, 2, correcao_e2)):
        emp = stats_empiricos[(ano, etapa)]
        corrigido = correcao.aplicar(emp)
        print(f"({ano}, {etapa}) bruto:     n={emp.n} m_p1={emp.m_p1:.3f} dp_p1={emp.dp_p1:.3f} "
              f"m_p2={emp.m_p2:.3f} dp_p2={emp.dp_p2:.3f} m_red={emp.m_red:.3f} dp_red={emp.dp_red:.3f}")
        print(f"({ano}, {etapa}) corrigido: m_p2={corrigido.mean_p2:.3f} dp_p2={corrigido.std_p2:.3f} "
              f"m_red={corrigido.mean_red:.3f} dp_red={corrigido.std_red:.3f}")
        print()


if __name__ == "__main__":
    main()
