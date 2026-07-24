import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction.constants import MAPA_SISTEMAS  # type: ignore
from pas_extraction.convocacao import (  # type: ignore
    ResultadoExtracaoConvocacao,
    extrair_chamada_e_semestre,
    extrair_edital_convocacao,
)
from pas_extraction.models import FamiliaEdital  # type: ignore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_CONVOCACAO = FIXTURES_DIR / "convocacao_registro.pdf"

# Contagem observada ao gerar a fixture com `python -m pas_extraction.cli fixture
# 'data/pdfs/Ed_28_PAS_3_2021_2023_Conv_RA_1ª_Chamada.pdf' 1 5 <destino>`.
# Páginas 1-5 (acima do "3 a 5" sugerido no ticket, no limite de cima) de propósito: é
# o intervalo mais curto que cruza 7 trocas de curso (mesmo espírito da fixture do
# ticket 01 — cobrir de fato o critério de aceite de campus/curso/turno como estado, não
# só "não é None"), preservando a página 1 de onde saem edital/triênio/chamada/semestre.
CONTAGEM_ESPERADA = 167
CURSOS_ESPERADOS = [
    "ADMINISTRAÇÃO (BACHARELADO)",
    "AGRONOMIA (BACHARELADO)",
    "ARQUITETURA E URBANISMO (BACHARELADO)",
    "ARTES CÊNICAS - INTERPRETAÇÃO TEATRAL (BACHARELADO)",
    "ARTES VISUAIS (BACHARELADO)",
    "BIBLIOTECONOMIA (BACHARELADO)",
    "BIOTECNOLOGIA (BACHARELADO)",
    "CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)",
]


def _pular_se_fixture_ausente(caminho: Path) -> None:
    if not caminho.exists():
        pytest.skip(
            f"Fixture {caminho.relative_to(Path(__file__).parent.parent)} não encontrada. "
            "Gere localmente (requer data/pdfs completo) com: "
            "python -m pas_extraction.cli fixture "
            "'data/pdfs/Ed_28_PAS_3_2021_2023_Conv_RA_1ª_Chamada.pdf' "
            f"1 5 {caminho}"
        )


class TestExtrairChamadaESemestre:
    """`extrair_chamada_e_semestre` não depende de fixture — testa só o regex contra
    trechos reais de texto de página 1 (não sintéticos: colados de Editais reais)."""

    def test_chamada_com_redacao_a_partir_de_2018_2020(self):
        # Redação usada de 2018/2020 em diante: "em primeira/segunda/... chamada".
        texto = (
            "A Universidade de Brasília (UnB) torna pública a convocação, em primeira "
            "chamada , para o registro acadêmico on-line dos candidatos selecionados "
            "dentro do quantitativo de vagas para o primeiro semestre , referente ao "
            "Programa de Avaliação Seriada (PAS) – Subprograma 2021 (triênio 2021/2023)"
        )
        semestre, chamada = extrair_chamada_e_semestre(texto)
        assert chamada == "1"
        assert semestre == "1"

    def test_chamada_com_redacao_antiga_2016_2018_sem_a_palavra_chamada(self):
        # Redação do triênio 2016/2018: "[ordinal] convocação", sem a palavra "chamada"
        # em nenhum lugar do Edital — confirmado varrendo o texto inteiro de exemplares
        # reais (Ed_33 a Ed_43, 2016/2018).
        texto = (
            "A Universidade de Brasília torna pública a quarta convocação para o "
            "pré-registro acadêmico referente ao primeiro semestre de 2019, "
            "Subprograma 2016 – Triênio 2016/2018."
        )
        semestre, chamada = extrair_chamada_e_semestre(texto)
        assert chamada == "4"
        assert semestre == "1"

    def test_segundo_semestre(self):
        texto = "convocação, em segunda chamada, para o quantitativo de vagas para o segundo semestre"
        semestre, chamada = extrair_chamada_e_semestre(texto)
        assert chamada == "2"
        assert semestre == "2"

    def test_semestre_ausente_e_desconhecido_sem_quebrar(self):
        # Os Editais do triênio 2018/2020 não mencionam semestre em lugar nenhum
        # (confirmado varrendo o texto inteiro de ED_38/ED_42/ED_46/ED_49 2018/2020) —
        # dado genuinamente ausente do Edital, não um bug do regex. Mesma convenção de
        # `schema.extrair_metadados`: "desconhecido", nunca uma exceção.
        texto = (
            "A Universidade de Brasília (UnB) torna pública a convocação, em segunda "
            "chamada, para o registro acadêmico on-line dos candidatos selecionados "
            "dentro do quantitativo total das vagas, referente ao Subprograma 2018 – "
            "Triênio 2018/2020."
        )
        semestre, chamada = extrair_chamada_e_semestre(texto)
        assert chamada == "2"
        assert semestre == "desconhecido"

    def test_nenhum_marcador_retorna_desconhecido_para_os_dois(self):
        semestre, chamada = extrair_chamada_e_semestre("um texto qualquer sem marcador")
        assert semestre == "desconhecido"
        assert chamada == "desconhecido"


class TestExtrairEditalConvocacao:
    """Exercita a costura `extrair_edital_convocacao`, não a estrutura interna do parser."""

    def test_extrai_a_contagem_esperada_de_registros(self):
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)

        assert isinstance(resultado, ResultadoExtracaoConvocacao)
        assert resultado.familia == FamiliaEdital.CONVOCACAO
        assert len(resultado.registros) == CONTAGEM_ESPERADA

    def test_edital_trienio_semestre_chamada_lidos_do_conteudo(self):
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)

        # Ed_28_PAS_3_2021_2023_Conv_RA_1ª_Chamada.pdf: Edital 28, triênio 2021/2023,
        # primeiro semestre, primeira chamada — tudo lido do texto, não do nome do
        # arquivo (o parser nunca olha para `caminho_pdf.name` além de proveniência).
        assert resultado.edital == "28"
        assert resultado.trienio == "2021/2023"
        assert resultado.semestre == "1"
        assert resultado.chamada == "1"

    def test_campus_curso_turno_vem_dos_cabecalhos_intercalados(self):
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)

        assert all(r.campus for r in resultado.registros)
        assert all(r.curso for r in resultado.registros)
        assert all(r.turno for r in resultado.registros)
        assert {r.campus for r in resultado.registros} == {"DARCY RIBEIRO"}
        assert {r.turno for r in resultado.registros} == {"DIURNO"}

    def test_curso_muda_de_estado_varias_vezes_no_meio_do_fluxo(self):
        # A fixture cruza 7 trocas de curso — prova que curso é estado atualizado a cada
        # cabeçalho encontrado, não um valor lido uma vez só no topo do documento.
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)
        cursos_em_ordem = [r.curso for r in resultado.registros]

        assert set(cursos_em_ordem) == set(CURSOS_ESPERADOS)
        # A ordem em que os cursos aparecem no fluxo bate com a ordem declarada acima
        # (sem repetição fora de ordem — cada curso aparece em um único bloco contíguo).
        cursos_na_primeira_ocorrencia = list(dict.fromkeys(cursos_em_ordem))
        assert cursos_na_primeira_ocorrencia == CURSOS_ESPERADOS

    def test_sistema_e_um_numero_valido_de_mapa_sistemas(self):
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)

        assert all(r.sistema in MAPA_SISTEMAS for r in resultado.registros)

    def test_inscricao_tem_8_digitos_e_nome_nao_tem_digito(self):
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)

        for r in resultado.registros:
            assert r.inscricao.isdigit() and len(r.inscricao) == 8
            assert not any(ch.isdigit() for ch in r.nome)

    def test_cada_registro_carrega_proveniencia_e_semestre_chamada(self):
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)

        for r in resultado.registros:
            assert r.proveniencia.arquivo_origem == FIXTURE_CONVOCACAO.name
            assert r.proveniencia.edital == "28"
            assert r.proveniencia.trienio == "2021/2023"
            assert r.proveniencia.pagina >= 1
            assert r.semestre == "1"
            assert r.chamada == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
