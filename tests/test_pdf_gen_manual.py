"""Fumaça do gerador de PDF: o template abre, o overlay entra e sai um PDF válido.

Depende de `assets/templates/`, que é asset whitelabel local (gitignored, ver `CLAUDE.md`) — por
isso o teste **pula** quando os templates não estão na máquina, em vez de falhar. Dados 100%
sintéticos.
"""

import sys
from pathlib import Path

import pytest  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

TEMPLATE = RAIZ / "assets" / "templates" / "MODELO PAS-UNB (ALUNOS) IMPRESSO.pdf"

pytestmark = pytest.mark.skipif(
    not TEMPLATE.exists(),
    reason=f"template whitelabel ausente ({TEMPLATE.relative_to(RAIZ)}) — asset local",
)


def test_pdf_gen(tmp_path):
    from pdf_generator import PDFGenerator  # type: ignore

    dados = {
        "aluno": "Teste Verificacao",
        "curso": "ENGENHARIA DE SOFTWARE - GAMA",
        "pas1_p1": "10.5",
        "pas1_p2": "50.0",
        "pas1_red": "8.5",
        "pas1_arg": "1.200",
        "pas2_p1": "12.0",
        "pas2_p2": "60.0",
        "pas2_red": "9.0",
        "pas2_arg": "1.500",
        "pas3_p1_est": "10.0*",
        "pas3_red_est": "8.0*",
        "arg_acumulado": "2.000",
        "nota_corte": "-1.500",
        "arg_necessario": "-0.500",
    }

    # `PDFGenerator` resolve os assets a partir de `__file__`, então não há diretório de trabalho
    # a ajustar — o `os.chdir` que este teste fazia antes trocava o CWD do processo inteiro do
    # pytest, e o efeito vazava para os testes seguintes.
    pdf_bytes = PDFGenerator().generate_single_pdf(dados)

    assert pdf_bytes, "geração devolveu bytes vazios"
    assert pdf_bytes.startswith(b"%PDF-"), "saída não é um PDF"

    destino = tmp_path / "test_output.pdf"
    destino.write_bytes(pdf_bytes)
    assert destino.stat().st_size > 1000
