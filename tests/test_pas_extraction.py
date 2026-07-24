import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction import ResultadoExtracao, extrair_edital  # type: ignore
from pas_extraction.models import FamiliaDesconhecidaError, FamiliaEdital  # type: ignore
from pas_extraction.schema import canonizar, classificar_familia  # type: ignore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_RESULTADO_FINAL = FIXTURES_DIR / "resultado_final_22_campos.pdf"

# Contagem observada ao gerar a fixture com `python -m pas_extraction.cli fixture
# 'data/pdfs/Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf' 1 6 <destino>`.
# Páginas 1-6 (acima do "3 a 5" sugerido) de propósito: é o intervalo mais curto que
# inclui uma troca real de curso (ADMINISTRAÇÃO -> AGRONOMIA, na página 6), que é o
# comportamento que este ticket pede para os cabeçalhos intercalados no fluxo. Alguns
# candidatos a registro nessas páginas têm número partido por espaço (ex.: "1 7.539") —
# corrupção real preservada pela fatia, fora do escopo desta costura (ver ticket 02).
CONTAGEM_ESPERADA = 170
CURSO_1 = "ADMINISTRAÇÃO (BACHARELADO)"
CURSO_2 = "AGRONOMIA (BACHARELADO)"


def _pular_se_fixture_ausente(caminho: Path) -> None:
    if not caminho.exists():
        pytest.skip(
            f"Fixture {caminho.relative_to(Path(__file__).parent.parent)} não encontrada. "
            "Gere localmente (requer data/pdfs completo) com: "
            "python -m pas_extraction.cli fixture "
            "'data/pdfs/Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf' "
            f"1 4 {caminho}"
        )


class TestCanonizacao:
    def test_remove_acento_caixa_e_pontuacao(self):
        assert canonizar("Campus/Curso, abaix o") == canonizar("campus curso abaixo")

    def test_espaco_no_fim_nao_afeta(self):
        assert canonizar("nome do candidato ") == canonizar("nome do candidato")


class TestClassificacaoFamilia:
    def test_resultado_final(self):
        texto = (
            "1.1 Resultado final, na seguinte ordem: campus/curso/turno, número de "
            "inscrição, nome do candidato em ordem alfabética, escore bruto da parte 1 "
            "na primeira etapa, ..., argumento final, classificação final no Sistema "
            "Universal."
        )
        assert classificar_familia(texto) == FamiliaEdital.RESULTADO_FINAL

    def test_resultado_final_com_redacao_institucional_nova_a_partir_de_2023(self):
        # "nome da pessoa candidata" substituiu "nome do candidato" a partir de
        # 2023/2025 — o classificador não pode depender da redação exata para acertar.
        texto = (
            "na seguinte ordem: campus/curso/turno, número de inscrição, nome da "
            "pessoa candidata em ordem alfabética, escore bruto da parte 1, argumento "
            "final, classificação final no Sistema Universal."
        )
        assert classificar_familia(texto) == FamiliaEdital.RESULTADO_FINAL

    def test_convocacao(self):
        texto = (
            "na seguinte ordem: campus/turno/curso, número de inscrição, nome do "
            "candidato em ordem alfabética e sistema/subsistema (conforme legenda "
            "abaixo)."
        )
        assert classificar_familia(texto) == FamiliaEdital.CONVOCACAO

    def test_medias_desvios_nao_declara_na_seguinte_ordem(self):
        texto = (
            "A Universidade de Brasília (UnB) torna públicos a média e o "
            "desvio-padrão das provas de cada etapa, para que o candidato possa "
            "calcular o seu argumento final."
        )
        assert classificar_familia(texto) == FamiliaEdital.MEDIAS_DESVIOS

    def test_familia_desconhecida_levanta_erro_claro(self):
        with pytest.raises(FamiliaDesconhecidaError):
            classificar_familia("um texto qualquer sem nenhum marcador conhecido")


class TestExtrairEditalResultadoFinal:
    """Exercita a costura `extrair_edital`, não a estrutura interna do parser."""

    def test_extrai_a_contagem_esperada_de_registros(self):
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        assert isinstance(resultado, ResultadoExtracao)
        assert resultado.familia == FamiliaEdital.RESULTADO_FINAL
        assert len(resultado.registros) == CONTAGEM_ESPERADA

    def test_campus_curso_turno_vem_dos_cabecalhos_intercalados(self):
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        assert all(r.campus for r in resultado.registros)
        assert all(r.curso for r in resultado.registros)
        assert all(r.turno for r in resultado.registros)

    def test_curso_muda_de_estado_no_meio_do_fluxo(self):
        # A fixture cruza uma troca real de curso (ver comentário de CONTAGEM_ESPERADA)
        # — prova que campus/curso/turno são estado atualizado a cada cabeçalho
        # encontrado, não um valor lido uma vez só no topo do documento.
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)
        cursos_em_ordem = [r.curso for r in resultado.registros]

        assert set(cursos_em_ordem) == {CURSO_1, CURSO_2}
        primeiro_indice_curso_2 = cursos_em_ordem.index(CURSO_2)
        assert cursos_em_ordem[:primeiro_indice_curso_2] == [CURSO_1] * primeiro_indice_curso_2
        assert cursos_em_ordem[primeiro_indice_curso_2:] == [CURSO_2] * (
            len(cursos_em_ordem) - primeiro_indice_curso_2
        )
        # campus/turno não mudam junto — só o curso, como no Edital real.
        assert {r.campus for r in resultado.registros} == {"DARCY RIBEIRO"}
        assert {r.turno for r in resultado.registros} == {"DIURNO"}

    def test_traco_e_preservado_como_nao_concorreu(self):
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        # "-" vira None (distinto de um campo ausente: todo registro emitido tem
        # exatamente as 10 posições de classificação, uma por Sistema, preenchidas ou não).
        assert any(v is None for r in resultado.registros for v in r.classificacoes.values())
        assert all(set(r.classificacoes) == set(range(1, 11)) for r in resultado.registros)

    def test_cada_linha_carrega_proveniencia(self):
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        for r in resultado.registros:
            assert r.proveniencia.arquivo_origem == FIXTURE_RESULTADO_FINAL.name
            assert r.proveniencia.edital == "38"
            assert r.proveniencia.trienio == "2022/2024"
            assert r.proveniencia.pagina >= 1

    def test_inscricao_tem_8_digitos_e_nome_nao_tem_digito(self):
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        for r in resultado.registros:
            assert r.inscricao.isdigit() and len(r.inscricao) == 8
            assert not any(ch.isdigit() for ch in r.nome)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
