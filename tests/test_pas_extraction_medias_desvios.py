import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore
from pypdf import PdfReader  # type: ignore

from pas_extraction.models import ContextoEdital  # type: ignore
from pas_extraction.medias_desvios import (  # type: ignore
    FamiliaInesperadaError,
    ResultadoExtracaoMediasDesvios,
    extrair_edital_medias_desvios,
    parse_medias_desvios,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Edital avulso dedicado à tabela (ED_34, triênio 2019/2021) — 1 página, a única do PDF.
FIXTURE_AVULSO = FIXTURES_DIR / "medias_desvios_avulso.pdf"

# Página 242 (a cauda) de Ed_38_2024_..._Res_final_não_eliminados.pdf — o mesmo Edital que
# gera a fixture `resultado_final_22_campos.pdf` usada em test_pas_extraction.py, mas aqui é
# a última página em vez das 6 primeiras. Prova que a tabela é achada onde ela de fato mora
# nesse triênio: na cauda do Resultado Final, não num Edital dedicado.
FIXTURE_CAUDA = FIXTURES_DIR / "medias_desvios_cauda.pdf"

_COMANDO_AVULSO = (
    "python -m pas_extraction.cli fixture "
    "'data/pdfs/medias-e-desvios/ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf' 1 1 "
    f"{FIXTURE_AVULSO}"
)
_COMANDO_CAUDA = (
    "python -m pas_extraction.cli fixture "
    "'data/pdfs/resultado-final-pas3/Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf' 242 242 "
    f"{FIXTURE_CAUDA}"
)


def _pular_se_fixture_ausente(caminho: Path, comando: str) -> None:
    if not caminho.exists():
        pytest.skip(
            f"Fixture {caminho.relative_to(Path(__file__).parent.parent)} não encontrada. "
            f"Gere localmente (requer data/pdfs completo) com: {comando}"
        )


# Valores oficiais impressos no Edital avulso (ED_34, triênio 2019/2021), 1ª etapa —
# conferidos direto no PDF com pypdf antes de escrever este teste.
ETAPA_1_AVULSO = {
    ("parte_1", "inglesa"): (3.900, 2.781),
    ("parte_1", "francesa"): (5.064, 2.756),
    ("parte_1", "espanhola"): (4.450, 2.437),
    ("parte_2", None): (26.738, 13.911),
    ("redacao", None): (6.617, 2.393),
}

# Valores oficiais da cauda (Ed_38_2024, triênio 2022/2024), 1ª etapa — os mesmos que
# scripts/NOTES.md, seção 7, cita como divergentes do OFFICIAL_STATS estimado atual
# (m_p2=20.406, dp_p2=13.533, m_red=5.849 contra os 20.709/13.581/5.888 estimados).
ETAPA_1_CAUDA = {
    ("parte_1", "inglesa"): (3.665, 3.109),
    ("parte_1", "francesa"): (3.620, 2.597),
    ("parte_1", "espanhola"): (3.140, 2.530),
    ("parte_2", None): (20.406, 13.533),
    ("redacao", None): (5.849, 2.793),
}


def _por_chave(registros):
    return {(r.etapa, r.prova, r.lingua_estrangeira): r for r in registros}


class TestParseMediasDesviosAvulso:
    def test_extrai_15_registros_5_por_etapa(self):
        _pular_se_fixture_ausente(FIXTURE_AVULSO, _COMANDO_AVULSO)

        reader = PdfReader(str(FIXTURE_AVULSO))
        contexto = ContextoEdital(
            arquivo_origem=FIXTURE_AVULSO.name, edital="34", trienio="2019/2021",
        )
        registros = parse_medias_desvios(reader, contexto)

        assert len(registros) == 15
        assert {r.etapa for r in registros} == {1, 2, 3}

    def test_parte_1_separada_por_lingua_parte_2_e_redacao_nao(self):
        _pular_se_fixture_ausente(FIXTURE_AVULSO, _COMANDO_AVULSO)

        reader = PdfReader(str(FIXTURE_AVULSO))
        contexto = ContextoEdital(
            arquivo_origem=FIXTURE_AVULSO.name, edital="34", trienio="2019/2021",
        )
        registros = parse_medias_desvios(reader, contexto)
        por_chave = _por_chave(registros)

        for prova, lingua in ETAPA_1_AVULSO:
            assert (1, prova, lingua) in por_chave

        # As 3 línguas da Parte 1 têm médias diferentes entre si (prova de que não foram
        # agregadas numa única linha).
        medias_parte_1 = {
            por_chave[(1, "parte_1", lingua)].media
            for lingua in ("inglesa", "francesa", "espanhola")
        }
        assert len(medias_parte_1) == 3

        # Parte II e Redação não têm língua.
        assert por_chave[(1, "parte_2", None)].lingua_estrangeira is None
        assert por_chave[(1, "redacao", None)].lingua_estrangeira is None

    def test_valores_batem_com_o_edital_para_a_1a_etapa(self):
        _pular_se_fixture_ausente(FIXTURE_AVULSO, _COMANDO_AVULSO)

        reader = PdfReader(str(FIXTURE_AVULSO))
        contexto = ContextoEdital(
            arquivo_origem=FIXTURE_AVULSO.name, edital="34", trienio="2019/2021",
        )
        registros = parse_medias_desvios(reader, contexto)
        por_chave = _por_chave(registros)

        for (prova, lingua), (media, desvio) in ETAPA_1_AVULSO.items():
            r = por_chave[(1, prova, lingua)]
            assert r.media == pytest.approx(media)
            assert r.desvio_padrao == pytest.approx(desvio)

    def test_cada_registro_carrega_proveniencia(self):
        _pular_se_fixture_ausente(FIXTURE_AVULSO, _COMANDO_AVULSO)

        reader = PdfReader(str(FIXTURE_AVULSO))
        contexto = ContextoEdital(
            arquivo_origem=FIXTURE_AVULSO.name, edital="34", trienio="2019/2021",
        )
        registros = parse_medias_desvios(reader, contexto)

        for r in registros:
            assert r.proveniencia.arquivo_origem == FIXTURE_AVULSO.name
            assert r.proveniencia.edital == "34"
            assert r.proveniencia.trienio == "2019/2021"
            assert r.proveniencia.pagina == 1


class TestParseMediasDesviosCauda:
    """A mesma tabela, mas encontrada nas páginas finais de um Resultado Final — critério
    de aceite explícito do ticket 03 ('a tabela é encontrada tanto na cauda ... quanto num
    Edital avulso')."""

    def test_extrai_15_registros_mesmo_sem_a_declaracao_de_familia_na_pagina_1(self):
        _pular_se_fixture_ausente(FIXTURE_CAUDA, _COMANDO_CAUDA)

        reader = PdfReader(str(FIXTURE_CAUDA))
        contexto = ContextoEdital(
            arquivo_origem=FIXTURE_CAUDA.name, edital="38", trienio="2022/2024",
        )
        registros = parse_medias_desvios(reader, contexto)

        assert len(registros) == 15
        assert {r.etapa for r in registros} == {1, 2, 3}

    def test_valores_batem_com_o_edital_para_a_1a_etapa(self):
        _pular_se_fixture_ausente(FIXTURE_CAUDA, _COMANDO_CAUDA)

        reader = PdfReader(str(FIXTURE_CAUDA))
        contexto = ContextoEdital(
            arquivo_origem=FIXTURE_CAUDA.name, edital="38", trienio="2022/2024",
        )
        registros = parse_medias_desvios(reader, contexto)
        por_chave = _por_chave(registros)

        for (prova, lingua), (media, desvio) in ETAPA_1_CAUDA.items():
            r = por_chave[(1, prova, lingua)]
            assert r.media == pytest.approx(media)
            assert r.desvio_padrao == pytest.approx(desvio)


class TestExtrairEditalMediasDesviosConveniencia:
    """A costura de conveniência (equivalente a `pipeline.extrair_edital` para esta
    Família) — só serve para o Edital avulso, porque é lá que a página 1 declara a Família
    Médias e Desvios."""

    def test_reconhece_a_familia_e_extrai_do_avulso(self):
        _pular_se_fixture_ausente(FIXTURE_AVULSO, _COMANDO_AVULSO)

        resultado = extrair_edital_medias_desvios(FIXTURE_AVULSO)

        assert isinstance(resultado, ResultadoExtracaoMediasDesvios)
        assert resultado.edital == "34"
        assert resultado.trienio == "2019/2021"
        assert resultado.arquivo_origem == FIXTURE_AVULSO.name
        assert len(resultado.registros) == 15

    def test_levanta_erro_claro_se_a_pagina_1_nao_for_medias_desvios(self):
        # A fixture de Resultado Final do ticket 01 (se existir) tem, na própria página 1,
        # a Família Resultado Final declarada — não Médias e Desvios. Prova que a costura
        # de conveniência não tenta adivinhar; ela exige a declaração explícita.
        fixture_resultado_final = FIXTURES_DIR / "resultado_final_22_campos.pdf"
        if not fixture_resultado_final.exists():
            pytest.skip(
                f"Fixture {fixture_resultado_final.relative_to(Path(__file__).parent.parent)} "
                "não encontrada (gerada pelo ticket 01)."
            )

        with pytest.raises(FamiliaInesperadaError):
            extrair_edital_medias_desvios(fixture_resultado_final)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
