"""Ticket 08, camada 6 — reconciliação cruzada entre Editais.

Constrói `RegistroResultadoFinal`/`RegistroConvocacao` sintéticos, como
`test_pas_extraction_relatorio_validacao.py` faz para as camadas 1-5: o que está sob teste é
só a lógica de agrupamento por inscrição, que não depende de PDF real nenhum.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pas_extraction.convocacao import RegistroConvocacao  # type: ignore
from pas_extraction.cotas import CotaDeclarada  # type: ignore
from pas_extraction.models import (  # type: ignore
    Proveniencia,
    RegistroResultadoFinal,
    ValidacaoRegistro,
)
from pas_extraction.reconciliacao import reconciliar_nomes  # type: ignore


def _proveniencia(arquivo: str, pagina: int = 1) -> Proveniencia:
    return Proveniencia(arquivo_origem=arquivo, edital="1", trienio="2022/2024", pagina=pagina)


def _resultado_final(inscricao: str, nome: str, arquivo: str = "Ed_rf.pdf") -> RegistroResultadoFinal:
    return RegistroResultadoFinal(
        campus="Darcy Ribeiro", curso="Curso A", turno="Matutino",
        inscricao=inscricao, nome=nome,
        eb_p1_e1=10.0, eb_p2_e1=10.0, red_e1=5.0,
        eb_p1_e2=10.0, eb_p2_e2=10.0, red_e2=5.0,
        eb_p1_e3=10.0, eb_p2_e3=10.0, red_e3=5.0,
        argumento_final=50.0,
        classificacoes={1: 1},
        cota_declarada=CotaDeclarada(
            sistema_negros=False, escola_publica=False, renda_baixa=False,
            ppi=False, pcd=False, perfil="Universal", padrao_suspeito=False,
        ),
        proveniencia=_proveniencia(arquivo),
        validacao=ValidacaoRegistro(
            campos_formato_invalido=(), buracos_classificacao={}, fora_de_ordem_alfabetica=False,
        ),
    )


def _convocacao(inscricao: str, nome: str, arquivo: str = "Ed_conv.pdf") -> RegistroConvocacao:
    return RegistroConvocacao(
        campus="Darcy Ribeiro", curso="Curso A", turno="Matutino",
        inscricao=inscricao, nome=nome, sistema=1, semestre="1", chamada="1",
        proveniencia=_proveniencia(arquivo),
    )


class TestSemDivergencia:
    def test_mesmo_nome_exato_em_duas_familias_nao_diverge(self):
        registros_rf = [_resultado_final("20240001", "Fulano de Tal")]
        registros_conv = [_convocacao("20240001", "Fulano de Tal")]

        assert reconciliar_nomes(registros_rf, registros_conv) == ()

    def test_diferenca_so_de_acento_caixa_ou_espaco_nao_diverge(self):
        # Ruído de extração puro entre dois Editais — a mesma tolerância que
        # `schema.canonizar` já aplica para classificar Família (ver ticket 01).
        registros_rf = [_resultado_final("20240001", "José da Silva ")]
        registros_conv = [_convocacao("20240001", "jose DA SILVA")]

        assert reconciliar_nomes(registros_rf, registros_conv) == ()

    def test_inscricoes_diferentes_nao_se_comparam(self):
        registros_rf = [_resultado_final("20240001", "Fulano de Tal")]
        registros_conv = [_convocacao("20240002", "Outro Nome Completamente")]

        assert reconciliar_nomes(registros_rf, registros_conv) == ()

    def test_lista_vazia_nao_diverge(self):
        assert reconciliar_nomes([], []) == ()


class TestComDivergencia:
    def test_nomes_genuinamente_diferentes_geram_divergencia(self):
        registros_rf = [_resultado_final("20240001", "Fulano de Tal")]
        registros_conv = [_convocacao("20240001", "Beltrano da Silva")]

        divergencias = reconciliar_nomes(registros_rf, registros_conv)

        assert len(divergencias) == 1
        assert divergencias[0].inscricao == "20240001"
        assert set(divergencias[0].nomes) == {"Fulano de Tal", "Beltrano da Silva"}

    def test_proveniencia_de_cada_ocorrencia_e_preservada(self):
        registros_rf = [_resultado_final("20240001", "Fulano de Tal", arquivo="Ed_rf.pdf")]
        registros_conv = [_convocacao("20240001", "Beltrano da Silva", arquivo="Ed_conv.pdf")]

        divergencia = reconciliar_nomes(registros_rf, registros_conv)[0]

        arquivos = {p.arquivo_origem for p in divergencia.proveniencias}
        assert arquivos == {"Ed_rf.pdf", "Ed_conv.pdf"}

    def test_divergencia_entre_duas_convocacoes_do_mesmo_trienio(self):
        # A mesma inscrição aparece em chamadas diferentes da Convocação, não só entre
        # Convocação e Resultado Final.
        primeira_chamada = [_convocacao("20240001", "Fulano de Tal", arquivo="Ed_1a_chamada.pdf")]
        segunda_chamada = [_convocacao("20240001", "Sicrano Nogueira", arquivo="Ed_2a_chamada.pdf")]

        divergencias = reconciliar_nomes(primeira_chamada, segunda_chamada)

        assert len(divergencias) == 1
        assert set(divergencias[0].nomes) == {"Fulano de Tal", "Sicrano Nogueira"}

    def test_grupos_nao_divergentes_nao_aparecem_junto_dos_divergentes(self):
        registros_rf = [
            _resultado_final("20240001", "Fulano de Tal"),
            _resultado_final("20240002", "Outra Pessoa"),
        ]
        registros_conv = [
            _convocacao("20240001", "Beltrano da Silva"),  # diverge
            _convocacao("20240002", "Outra Pessoa"),  # não diverge
        ]

        divergencias = reconciliar_nomes(registros_rf, registros_conv)

        assert [d.inscricao for d in divergencias] == ["20240001"]

    def test_resultado_ordenado_por_inscricao(self):
        registros_rf = [
            _resultado_final("20240009", "Nome A"),
            _resultado_final("20240001", "Nome C"),
        ]
        registros_conv = [
            _convocacao("20240009", "Nome B"),
            _convocacao("20240001", "Nome D"),
        ]

        divergencias = reconciliar_nomes(registros_rf, registros_conv)

        assert [d.inscricao for d in divergencias] == ["20240001", "20240009"]

    def test_determinismo_mesma_entrada_mesma_saida(self):
        registros_rf = [_resultado_final("20240001", "Fulano de Tal")]
        registros_conv = [_convocacao("20240001", "Beltrano da Silva")]

        primeira = reconciliar_nomes(registros_rf, registros_conv)
        segunda = reconciliar_nomes(registros_rf, registros_conv)

        assert primeira == segunda

    def test_nao_muta_os_registros_de_entrada(self):
        registro = _resultado_final("20240001", "Fulano de Tal")
        registros_conv = [_convocacao("20240001", "Beltrano da Silva")]

        reconciliar_nomes([registro], registros_conv)

        assert registro.nome == "Fulano de Tal"
