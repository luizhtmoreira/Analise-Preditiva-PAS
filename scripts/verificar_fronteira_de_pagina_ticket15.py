"""Ticket 15 — reprocessa o corpus real e confere a correção de fronteira de página.

Roda `pipeline.extrair_edital` sobre `data/pdfs/resultado-final-pas3/` (8 Editais) e
confirma, de forma reproduzível, as duas afirmações do relatório do ticket
(`.scratch/pdf-extraction/relatorios/15-fronteira-de-pagina-no-parser.md`, §3):

1. Nenhum registro sai com `cota_declarada.padrao_suspeito=True` no corpus inteiro — os 8
   casos que a checagem de fecho pegava (relatório do ticket 06, §3) estão corrigidos.
2. Os 2 casos que ficavam invisíveis à checagem de fecho (padrão `{1,9,10}`, fecho válido
   tanto corrompido quanto correto) têm o padrão certo (`{1,9}`) nas duas inscrições
   identificadas na comparação antes/depois: `18147304` (Ed_37, 2018/2020) e `16125849`
   (Ed_31, 2016/2018).

Não reproduz a comparação "antes x depois" inteira (isso exigiria rodar a versão sem a
correção, que não está preservada em código) — confirma o estado *depois*, que é o que
importa para detectar regressão futura.

Uso: `.venv/bin/python scripts/verificar_fronteira_de_pagina_ticket15.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pas_extraction.pipeline import extrair_edital  # noqa: E402

RAIZ = Path(__file__).parent.parent / "data" / "pdfs" / "resultado-final-pas3"

# As duas inscrições que ficavam invisíveis à checagem de fecho (padrão {1,9,10}, um fecho
# válido tanto com quanto sem a corrupção) — identificadas comparando duas rodadas do
# pipeline, uma antes e uma depois da correção (ver relatório do ticket, §3).
INSCRICOES_ANTES_INVISIVEIS = {
    "18147304": "ED_37_PAS_3 _2018 -2020_Final_Tipo_D_Redacao.pdf",
    "16125849": "Ed_31_2016-2018_PAS_3_Res_final_nao_eliminados.pdf",
}


def main() -> None:
    if not RAIZ.exists():
        print(f"corpus local ausente ({RAIZ}) — nada a verificar aqui.")
        return

    total = 0
    suspeitos = []
    achados_invisiveis = {}

    for pdf in sorted(RAIZ.glob("*.pdf")):
        resultado = extrair_edital(pdf)
        if resultado.familia is None or not resultado.registros:
            continue
        total += len(resultado.registros)
        for r in resultado.registros:
            if r.cota_declarada.padrao_suspeito:
                suspeitos.append((pdf.name, r.inscricao))
            if pdf.name in INSCRICOES_ANTES_INVISIVEIS.values() and r.inscricao in INSCRICOES_ANTES_INVISIVEIS:
                padrao = tuple(i for i in range(1, 11) if r.classificacoes[i] is not None)
                achados_invisiveis[r.inscricao] = padrao

    print(f"total de registros no corpus: {total}")
    print(f"registros com cota_padrao_suspeito=True: {len(suspeitos)}")
    for nome_pdf, inscricao in suspeitos:
        print(f"  suspeito: {nome_pdf} / {inscricao}")

    print("\ncasos antes invisíveis à checagem de fecho:")
    for inscricao in INSCRICOES_ANTES_INVISIVEIS:
        padrao = achados_invisiveis.get(inscricao)
        esperado = (1, 9)
        status = "OK" if padrao == esperado else "DIVERGENTE"
        print(f"  {inscricao}: padrão = {padrao} (esperado {esperado}) [{status}]")

    assert not suspeitos, "regressão: existem registros suspeitos no corpus real"
    assert all(v == (1, 9) for v in achados_invisiveis.values()), (
        "regressão: um dos casos antes invisíveis não tem o padrão corrigido esperado"
    )
    print("\nverificação OK — sem regressão.")


if __name__ == "__main__":
    main()
