"""Gera `data/chamadas.csv` — a Nota de Corte de **cada** chamada de cada (curso, Sistema de
Concorrência), não só da última.

`src/pas_extraction/notas_corte.py` deriva a Nota de Corte oficial olhando só a maior chamada
de cada grupo (é a definição do produto: "a nota exigida na última chamada"). A tela "Histórico
de Chamadas" do Preditor precisa da mesma conta repetida para **todas** as chamadas
intermediárias — hoje ela lê `notas_corte.csv` esperando encontrar isso lá, mas esse CSV nunca
teve mais de uma chamada por grupo desde o ticket 10 (achado real: 100% dos 4.971 grupos com
`chamada.nunique() == 1`). Este script preenche essa lacuna sem tocar na derivação oficial.

Mesmo algoritmo de `notas_corte.derivar_notas_corte` (menor Argumento Final entre os convocados
da chamada, preferindo checksum confiável), aplicado grupo a grupo **por chamada** em vez de só
na maior. Lê os dois CSVs já extraídos (`data/convocacao.csv`, `data/resultado_final.csv`— ambos
fora do git, ver `CLAUDE.md`) e escreve `data/chamadas.csv` sem `nome`/`inscricao`, pronto para
entrar em `pas_intelligence.derivado_deploy.COLUNAS_DERIVADO` e ser publicado como o resto do
Derivado de Deploy.

Uso:
    .venv/bin/python scripts/gerar_historico_chamadas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pas_extraction.constants import MAPA_SISTEMAS  # noqa: E402
from pas_extraction.convocacao import LISTA_CONVOCACAO  # noqa: E402

DATA_DIR = RAIZ / "data"
CHAVE_GRUPO = ["trienio", "semestre", "campus", "curso", "turno", "sistema"]


def gerar() -> pd.DataFrame:
    conv = pd.read_csv(DATA_DIR / "convocacao.csv")
    rf = pd.read_csv(
        DATA_DIR / "resultado_final.csv",
        usecols=["inscricao", "trienio", "argumento_final", "checksum_fecha"],
    )

    # Só quem foi efetivamente chamado nesta chamada — `outra_relacao` é ausente de uma
    # chamada anterior, não convocado desta (ver `convocacao.py:117-119`).
    conv = conv[conv["lista"] == LISTA_CONVOCACAO].copy()
    conv["chamada_num"] = pd.to_numeric(conv["chamada"], errors="coerce")
    conv = conv.dropna(subset=["chamada_num"])
    conv["chamada_num"] = conv["chamada_num"].astype(int)

    # (triênio, inscrição) e não só inscrição — a mesma inscrição pode existir em mais de um
    # triênio (ver `notas_corte._indexar_resultado_final`). Mantém a primeira ocorrência: o
    # corpus real não tem (triênio, inscrição) duplicado com Argumento Final diferente.
    rf = rf.drop_duplicates(subset=["trienio", "inscricao"], keep="first")

    fundidos = conv.merge(
        rf, on=["trienio", "inscricao"], how="inner", suffixes=("", "_rf")
    )

    linhas = []
    grupos = fundidos.groupby(CHAVE_GRUPO + ["chamada_num"], dropna=False)
    for chave, grupo in grupos:
        trienio, semestre, campus, curso, turno, sistema, chamada_num = chave

        confiaveis = grupo[grupo["checksum_fecha"] == True]  # noqa: E712
        candidatos = confiaveis if not confiaveis.empty else grupo

        candidatos = candidatos.sort_values(["argumento_final", "inscricao"])
        vencedor = candidatos.iloc[0]

        linhas.append({
            "trienio": trienio,
            "semestre": semestre,
            "campus": campus,
            "curso": curso,
            "turno": turno,
            "sistema": sistema,
            "sistema_nome": MAPA_SISTEMAS.get(int(sistema), ""),
            "chamada": str(chamada_num),
            "nota_corte": vencedor["argumento_final"],
            "convocados_na_chamada": len(grupo),
            "convocados_com_argumento": len(grupo),  # merge "inner": só quem tinha Argumento
            "checksum_fecha": bool(vencedor["checksum_fecha"]),
        })

    return pd.DataFrame(linhas)


def main() -> int:
    df = gerar()
    destino = DATA_DIR / "chamadas.csv"
    df.to_csv(destino, index=False)
    print(f"chamadas: {len(df)} linhas em {destino}")
    print(f"  grupos (curso/sistema/trienio/...) distintos: {df.groupby(CHAVE_GRUPO).ngroups}")
    print(f"  chamadas por grupo — describe:\n{df.groupby(CHAVE_GRUPO).size().describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
