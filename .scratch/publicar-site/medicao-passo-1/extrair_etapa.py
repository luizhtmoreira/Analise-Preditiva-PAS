"""Extrai as notas dos Editais isolados de Etapa 1 e 2 e compara com o oficial.

O Edital lista, separados por ' / ':
    inscricao, nome, EB parte 1, EB parte 2, somatorio, nota tipo D, nota redacao

O somatorio e um checksum embutido (parte 1 + parte 2), usado aqui para descartar
registro cuja extracao de texto saiu corrompida.

NAO usa o parser de `src/pas_extraction` — e script de medicao descartavel.
"""
import re
import sys
import numpy as np
import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, "src")
from pas_intelligence.pas_constants import OFFICIAL_STATS

PDFS = {
    (2022, 1): "data/pdfs/editais-de-etapa/ED_8_PAS_1_2022-2024_Retificação_Res_Final_tipo_D_redação.pdf",
    (2023, 1): "data/pdfs/editais-de-etapa/Ed_8_PAS_1_2023_2025_Ret_Res_final_tipo_D_redação.pdf",
    (2024, 1): "data/pdfs/editais-de-etapa/Ed_8_2024_PAS_UnB_1_2024-2026_Res_final_tipo_D_redação.pdf",
    (2025, 2): "data/pdfs/editais-de-etapa/8BF9D771C58F383321E81D054B720A9E75CF911E7A921DF6E017670779B74EEF.pdf",
}

LINHA_SO_NUMERO = re.compile(r"^\s*\d{1,4}\s*$")
INSCRICAO = re.compile(r"^\d{6,10}$")


def texto_do_edital(caminho: str) -> str:
    """Concatena as paginas, tirando o numero de pagina que se intercala nos registros."""
    partes = []
    for pagina in PdfReader(caminho).pages:
        linhas = (pagina.extract_text() or "").split("\n")
        while linhas and (not linhas[0].strip() or LINHA_SO_NUMERO.match(linhas[0])):
            linhas.pop(0)
        partes.append("\n".join(linhas))
    return " ".join(partes)


def numero(campo: str):
    """'2. 046' -> 2.046 ; '1 6.005' -> 16.005 ; '0 .220' -> 0.220"""
    limpo = re.sub(r"\s+", "", campo)
    return float(limpo) if re.fullmatch(r"-?\d+\.?\d*", limpo) else None


def registros(caminho: str) -> tuple[pd.DataFrame, dict]:
    bruto = texto_do_edital(caminho)
    diag = {"blocos": 0, "campos_errados": 0, "nao_numerico": 0, "checksum_falhou": 0}
    linhas = []

    for bloco in bruto.split("/"):
        campos = [c.strip() for c in bloco.split(",")]
        if len(campos) < 2 or not INSCRICAO.match(re.sub(r"\s+", "", campos[0])):
            continue
        diag["blocos"] += 1
        if len(campos) != 7:
            diag["campos_errados"] += 1
            continue
        p1, p2, soma, tipo_d, red = (numero(c) for c in campos[2:])
        if None in (p1, p2, soma, tipo_d, red):
            diag["nao_numerico"] += 1
            continue
        if abs((p1 + p2) - soma) > 0.005:  # checksum do proprio Edital
            diag["checksum_falhou"] += 1
            continue
        linhas.append({"inscricao": campos[0], "eb_p1": p1, "eb_p2": p2,
                       "tipo_d": tipo_d, "red": red})

    return pd.DataFrame(linhas), diag


def main():
    resumo = []
    for (ano, etapa), caminho in PDFS.items():
        df, diag = registros(caminho)
        of = OFFICIAL_STATS.get((ano, etapa))
        print(f"\n{'='*72}\n({ano}, Etapa {etapa})  —  {caminho.split('/')[-1][:58]}")
        print(f"{'='*72}")
        print(f"registros aceitos: {len(df)}   | descartes: campos={diag['campos_errados']} "
              f"nao-numerico={diag['nao_numerico']} checksum={diag['checksum_falhou']} "
              f"(de {diag['blocos']} blocos)")

        linha = {"ano": ano, "etapa": etapa, "n": len(df)}
        for rotulo, col in [("m_p2/dp_p2", "eb_p2"), ("m_red/dp_red", "red"),
                            ("m_p1/dp_p1", "eb_p1"), ("tipo_d", "tipo_d")]:
            m, dp = df[col].mean(), df[col].std(ddof=0)
            linha[f"m_{col}"], linha[f"dp_{col}"] = m, dp
            print(f"  {rotulo:14s} media={m:8.3f}  desvio={dp:7.3f}")

        if of is None:
            print("  (sem valor oficial — e justamente a entrada que falta)")
        else:
            print(f"  {'OFICIAL':14s} m_p2={of.m_p2:8.3f}  dp_p2={of.dp_p2:7.3f}"
                  f"   m_red={of.m_red:6.3f}  dp_red={of.dp_red:6.3f}")
            print(f"  {'DIFERENCA':14s} m_p2={df['eb_p2'].mean()-of.m_p2:+8.3f}  "
                  f"dp_p2={df['eb_p2'].std(ddof=0)-of.dp_p2:+7.3f}   "
                  f"m_red={df['red'].mean()-of.m_red:+6.3f}  "
                  f"dp_red={df['red'].std(ddof=0)-of.dp_red:+6.3f}")
            linha.update({"d_m_p2": df["eb_p2"].mean() - of.m_p2,
                          "d_dp_p2": df["eb_p2"].std(ddof=0) - of.dp_p2,
                          "d_m_red": df["red"].mean() - of.m_red,
                          "d_dp_red": df["red"].std(ddof=0) - of.dp_red})
        resumo.append(linha)

    pd.DataFrame(resumo).to_csv(
        ".scratch/publicar-site/medicao-passo-1/editais_isolados.csv", index=False)


if __name__ == "__main__":
    main()
