import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction.etapa import (  # type: ignore
    EditalParcialError,
    LinhaMediasDesviosEtapa,
    MINIMO_REGISTROS,
    ResultadoExtracaoEtapa,
    DiagnosticoDescartes,
    calcular_medias_desvios,
    escrever_csv_medias_desvios_etapa,
    extrair_edital_etapa,
    normaliza_numero,
    parse_registros,
    processar_editais_etapa,
)
from pas_extraction.fixtures import gerar_pdf_texto_sintetico  # type: ignore

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _bloco(inscricao: str, p1: float, p2: float, tipo_d: float, red: float, soma=None) -> str:
    """Um registro sintético, no formato do Edital: inscrição, nome, P1, P2, somatório,
    tipo D, redação. `soma` sobrescreve o checksum embutido, para testar o caso em que ele
    não fecha."""
    if soma is None:
        soma = p1 + p2
    return f"{inscricao}, Fulano de Tal, {p1:.3f}, {p2:.3f}, {soma:.3f}, {tipo_d:.3f}, {red:.3f}"


def _candidatos_sinteticos(n: int, inicio: int = 20100001) -> list:
    return [_bloco(str(inicio + i), 1.0 + i * 0.1, 20.0 + i * 0.2, 0.5, 5.0) for i in range(n)]


class TestNormalizaNumero:
    def test_numero_partido_por_espaco_no_meio(self):
        assert normaliza_numero("2. 046") == pytest.approx(2.046)

    def test_numero_partido_antes_do_separador_de_milhar(self):
        assert normaliza_numero("1 6.005") == pytest.approx(16.005)

    def test_numero_partido_apos_o_sinal_decimal(self):
        assert normaliza_numero("0 .220") == pytest.approx(0.220)

    def test_numero_bem_formado(self):
        assert normaliza_numero("32.302") == pytest.approx(32.302)

    def test_campo_nao_numerico_devolve_none(self):
        assert normaliza_numero("Fulano de Tal") is None


class TestParseRegistros:
    def test_aceita_registro_bem_formado(self):
        texto = " / ".join(_candidatos_sinteticos(3))
        registros, diag = parse_registros(texto)

        assert len(registros) == 3
        assert diag.blocos == 3
        assert diag.aceitos == 3
        assert diag.campos_invalidos == diag.nao_numerico == diag.checksum_falhou == 0

    def test_normaliza_numero_partido_e_o_checksum_prova_que_funcionou(self):
        # O mesmo registro do Edital 7/2023 (23120271, Aaron Caesar...), com o somatório
        # partido por espaço exatamente como sai da extração de texto real.
        bloco = "23120271, Aaron Caesar de Carvalho Poeck, 2.679, 4 6.510, 49. 189, 1.563, 4.150"
        registros, diag = parse_registros(bloco)

        assert diag.checksum_falhou == 0
        assert len(registros) == 1
        assert registros[0].eb_p1 == pytest.approx(2.679)
        assert registros[0].eb_p2 == pytest.approx(46.510)

    def test_descarta_registro_cujo_checksum_nao_fecha(self):
        bloco = _bloco("20100001", 2.0, 10.0, 5.0, 1.0, 4.0)  # soma != p1+p2
        registros, diag = parse_registros(bloco)

        assert registros == []
        assert diag.blocos == 1
        assert diag.checksum_falhou == 1

    def test_descarta_registro_com_numero_de_campos_errado(self):
        bloco = "20100001, Fulano de Tal, 2.000, 10.000, 12.000, 1.000"  # falta um campo
        registros, diag = parse_registros(bloco)

        assert registros == []
        assert diag.blocos == 1
        assert diag.campos_invalidos == 1

    def test_descarta_registro_nao_numerico(self):
        bloco = "20100001, Fulano de Tal, abc, 10.000, 12.000, 1.000, 2.000"
        registros, diag = parse_registros(bloco)

        assert registros == []
        assert diag.nao_numerico == 1

    def test_ignora_ruido_que_nao_comeca_com_inscricao(self):
        texto = "cabeçalho qualquer sem inscrição / " + _bloco("20100001", 1.0, 2.0, 0.5, 3.0)
        registros, diag = parse_registros(texto)

        assert len(registros) == 1
        assert diag.blocos == 1  # o ruído nem chega a virar bloco


# Cabeçalho de página 1 de um Edital isolado de Etapa — o texto real (ver Ed_7 e Ed_15 em
# `data/pdfs`) declara "SUBPROGRAMA <ano> (TRIÊNIO <ini>/<fim>)"; o resto é descartável
# para o parser, que só ancora em "TRIÊNIO aaaa/aaaa".
def _cabecalho(ano_subprograma: int, inicio: int, fim: int) -> str:
    return (
        "UNIVERSIDADE DE BRASILIA (UnB)\n"
        "PROGRAMA DE AVALIACAO SERIADA (PAS)\n"
        f"SUBPROGRAMA {ano_subprograma} (TRIENIO {inicio}/{fim})\n"
        "EDITAL N. 7 - PAS/UnB - SUBPROGRAMA, DE 30 DE JANEIRO\n"
    )


class TestExtrairEditalEtapaSintetico:
    def test_deriva_ano_etapa_e_trienio_do_conteudo(self, tmp_path):
        pdf = gerar_pdf_texto_sintetico(
            [_cabecalho(2023, 2023, 2025) + " / ".join(_candidatos_sinteticos(10))],
            tmp_path / "edital_sintetico.pdf",
        )

        resultado = extrair_edital_etapa(pdf, etapa=1, minimo_registros=5)

        assert isinstance(resultado, ResultadoExtracaoEtapa)
        assert resultado.trienio == "2023/2025"
        assert resultado.ano == 2023  # inicio (2023) + etapa (1) - 1
        assert resultado.etapa == 1
        assert len(resultado.registros) == 10

    def test_etapa_2_soma_um_ano_ao_inicio_do_trienio(self, tmp_path):
        pdf = gerar_pdf_texto_sintetico(
            [_cabecalho(2023, 2023, 2025) + " / ".join(_candidatos_sinteticos(10))],
            tmp_path / "edital_sintetico_etapa2.pdf",
        )

        resultado = extrair_edital_etapa(pdf, etapa=2, minimo_registros=5)

        assert resultado.ano == 2024  # 2023 + 2 - 1

    def test_etapa_invalida_levanta_value_error(self, tmp_path):
        pdf = gerar_pdf_texto_sintetico(
            [_cabecalho(2023, 2023, 2025) + " / ".join(_candidatos_sinteticos(5))],
            tmp_path / "edital_etapa_invalida.pdf",
        )

        with pytest.raises(ValueError):
            extrair_edital_etapa(pdf, etapa=3, minimo_registros=5)

    def test_documento_parcial_e_recusado_com_a_contagem_na_mensagem(self, tmp_path):
        # O caso real que motivou esta verificação: o Edital 8/2023 (retificação) tem 827
        # registros contra os ~19.500 do Edital completo do mesmo triênio — um documento
        # parcial, não um Edital corrompido. Aqui usamos uma fixture pequena (10
        # candidatos) contra o limiar de produção (MINIMO_REGISTROS=5000, bem acima de
        # qualquer documento de teste), que é exatamente o mecanismo que teria recusado o
        # Edital 8/2023 em vez de aceitá-lo silenciosamente.
        pdf = gerar_pdf_texto_sintetico(
            [_cabecalho(2023, 2023, 2025) + " / ".join(_candidatos_sinteticos(10))],
            tmp_path / "edital_parcial.pdf",
        )

        with pytest.raises(EditalParcialError, match=r"10 registros aceitos"):
            extrair_edital_etapa(pdf, etapa=1)  # minimo_registros default = 5000

    def test_documento_completo_passa_quando_acima_do_minimo(self, tmp_path):
        pdf = gerar_pdf_texto_sintetico(
            [_cabecalho(2023, 2023, 2025) + " / ".join(_candidatos_sinteticos(20))],
            tmp_path / "edital_completo.pdf",
        )

        resultado = extrair_edital_etapa(pdf, etapa=1, minimo_registros=20)

        assert len(resultado.registros) == 20


class TestCalcularMediasDesvios:
    def _resultado(self, registros):
        return ResultadoExtracaoEtapa(
            registros=registros,
            diagnostico=DiagnosticoDescartes(len(registros), 0, 0, 0),
            ano=2023, etapa=1, trienio="2023/2025", arquivo_origem="sintetico.pdf",
        )

    def test_media_e_desvio_das_tres_provas(self):
        from pas_extraction.etapa import RegistroEtapa

        registros = [
            RegistroEtapa("20100001", eb_p1=10.0, eb_p2=20.0, tipo_d=1.0, red=5.0),
            RegistroEtapa("20100002", eb_p1=20.0, eb_p2=40.0, tipo_d=1.0, red=7.0),
        ]
        linha = calcular_medias_desvios(self._resultado(registros))

        assert linha.n == 2
        assert linha.m_eb_p1 == pytest.approx(15.0)
        assert linha.m_eb_p2 == pytest.approx(30.0)
        assert linha.m_red == pytest.approx(6.0)
        # Desvio populacional (ddof=0): igual ao que o Cebraspe publica.
        assert linha.dp_eb_p1 == pytest.approx(5.0)


class TestEscreverCsvMediasDesviosEtapa:
    def test_escreve_uma_linha_por_ano_etapa(self, tmp_path):
        linhas = [
            LinhaMediasDesviosEtapa(2023, 1, 100, 20.0, 10.0, 6.0, 2.0, 3.0, 2.5),
            LinhaMediasDesviosEtapa(2024, 2, 200, 25.0, 12.0, 6.5, 2.2, 3.1, 3.0),
        ]
        destino = tmp_path / "medias_desvios_etapa.csv"
        total = escrever_csv_medias_desvios_etapa(linhas, destino)

        assert total == 2
        conteudo = destino.read_text(encoding="utf-8")
        assert "ano,etapa,n,m_eb_p2,dp_eb_p2,m_red,dp_red,m_eb_p1,dp_eb_p1" in conteudo
        assert "2023,1,100" in conteudo
        assert "2024,2,200" in conteudo


class TestProcessarEditaisEtapa:
    def test_aponta_para_editais_e_produz_o_csv(self, tmp_path):
        pdf_e1 = gerar_pdf_texto_sintetico(
            [_cabecalho(2023, 2023, 2025) + " / ".join(_candidatos_sinteticos(10))],
            tmp_path / "e1.pdf",
        )
        pdf_e2 = gerar_pdf_texto_sintetico(
            [_cabecalho(2023, 2023, 2025) + " / ".join(_candidatos_sinteticos(15, inicio=30100001))],
            tmp_path / "e2.pdf",
        )
        destino = tmp_path / "medias_desvios_etapa.csv"

        linhas = processar_editais_etapa(
            {pdf_e1: 1, pdf_e2: 2}, destino, minimo_registros=5,
        )

        assert [(l.ano, l.etapa, l.n) for l in linhas] == [(2023, 1, 10), (2024, 2, 15)]
        assert destino.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
