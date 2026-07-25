"""Ticket 07 — relatório de validação agrupado por padrão.

Constrói `RegistroResultadoFinal` sintéticos em vez de PDFs reais: este ticket só consolida
o que as camadas anteriores (02, 04, 06) já gravam em cada registro, então o que está sob
teste é a lógica de agrupamento e a classificação da distribuição de deltas — nenhuma das
duas depende de um Edital real.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pas_extraction.cotas import CotaDeclarada  # type: ignore
from pas_extraction.models import (  # type: ignore
    ChecksumArgumentoFinal,
    Proveniencia,
    RegistroResultadoFinal,
    ValidacaoRegistro,
)
from pas_extraction.relatorio_validacao import (  # type: ignore
    CAMADA_CHECKSUM,
    CAMADA_COTAS,
    CAMADA_FORMATO_NUMERICO,
    CAMADA_ORDEM_ALFABETICA,
    CAMADA_SEQUENCIA,
    escrever_relatorio,
    formatar_markdown,
    formatar_terminal,
    gerar_relatorio,
)

_CONTADOR = [0]


def _proximo_num_inscricao() -> str:
    _CONTADOR[0] += 1
    return f"{_CONTADOR[0]:08d}"


def _registro(
    *,
    curso: str = "Curso A",
    inscricao: Optional[str] = None,
    campos_formato_invalido: Tuple[str, ...] = (),
    buracos_classificacao: Optional[Dict[int, Tuple[int, ...]]] = None,
    fora_de_ordem_alfabetica: bool = False,
    checksum: Optional[ChecksumArgumentoFinal] = None,
    padrao_suspeito: bool = False,
    perfil: str = "Universal",
) -> RegistroResultadoFinal:
    return RegistroResultadoFinal(
        campus="Darcy Ribeiro",
        curso=curso,
        turno="Matutino",
        inscricao=inscricao or _proximo_num_inscricao(),
        nome="Fulano de Tal",
        eb_p1_e1=10.0, eb_p2_e1=10.0, red_e1=5.0,
        eb_p1_e2=10.0, eb_p2_e2=10.0, red_e2=5.0,
        eb_p1_e3=10.0, eb_p2_e3=10.0, red_e3=5.0,
        argumento_final=50.0,
        classificacoes={1: 1},
        cota_declarada=CotaDeclarada(
            sistema_negros=False, escola_publica=False, renda_baixa=False,
            ppi=False, pcd=False, perfil=perfil, padrao_suspeito=padrao_suspeito,
        ),
        proveniencia=Proveniencia(
            arquivo_origem="Ed_1.pdf", edital="1", trienio="2022/2024", pagina=1,
        ),
        validacao=ValidacaoRegistro(
            campos_formato_invalido=campos_formato_invalido,
            buracos_classificacao=buracos_classificacao or {},
            fora_de_ordem_alfabetica=fora_de_ordem_alfabetica,
        ),
        checksum=checksum,
    )


def _checksum(delta: float) -> ChecksumArgumentoFinal:
    return ChecksumArgumentoFinal(
        argumento_final_recalculado=50.0 + delta,
        delta=delta,
        linguas_por_etapa=("inglesa", "inglesa", "inglesa"),
    )


class TestAgrupamentoPorPadrao:
    def test_falhas_do_mesmo_padrao_viram_um_grupo_so(self):
        registros = [_registro(campos_formato_invalido=("eb_p1_e2",)) for _ in range(5)]
        relatorio = gerar_relatorio(registros)

        grupos_formato = [g for g in relatorio.grupos if g.camada == CAMADA_FORMATO_NUMERICO]
        assert len(grupos_formato) == 1
        assert grupos_formato[0].quantidade == 5
        assert grupos_formato[0].padrao == "eb_p1_e2"

    def test_padroes_diferentes_nao_se_misturam(self):
        registros = (
            [_registro(campos_formato_invalido=("eb_p1_e2",)) for _ in range(3)]
            + [_registro(campos_formato_invalido=("red_e3",)) for _ in range(2)]
        )
        relatorio = gerar_relatorio(registros)

        grupos_formato = {
            g.padrao: g.quantidade
            for g in relatorio.grupos
            if g.camada == CAMADA_FORMATO_NUMERICO
        }
        assert grupos_formato == {"eb_p1_e2": 3, "red_e3": 2}

    def test_um_registro_pode_contribuir_para_mais_de_um_campo(self):
        registros = [_registro(campos_formato_invalido=("eb_p1_e2", "red_e3"))]
        relatorio = gerar_relatorio(registros)

        grupos_formato = {
            g.padrao: g.quantidade
            for g in relatorio.grupos
            if g.camada == CAMADA_FORMATO_NUMERICO
        }
        assert grupos_formato == {"eb_p1_e2": 1, "red_e3": 1}

    def test_grupos_ordenados_do_maior_para_o_menor(self):
        registros = (
            [_registro(campos_formato_invalido=("raro",))]
            + [_registro(campos_formato_invalido=("comum",)) for _ in range(4)]
        )
        relatorio = gerar_relatorio(registros)
        grupos_formato = [g for g in relatorio.grupos if g.camada == CAMADA_FORMATO_NUMERICO]
        assert [g.padrao for g in grupos_formato] == ["comum", "raro"]

    def test_grupo_carrega_exemplos_limitados_de_inscricao(self):
        registros = [
            _registro(inscricao=f"{i:08d}", campos_formato_invalido=("eb_p1_e2",))
            for i in range(20)
        ]
        relatorio = gerar_relatorio(registros)
        grupo = next(g for g in relatorio.grupos if g.camada == CAMADA_FORMATO_NUMERICO)
        assert grupo.quantidade == 20
        assert len(grupo.exemplos) <= 5


class TestNenhumRegistroDescartado:
    def test_total_de_registros_e_preservado_mesmo_com_falhas(self):
        registros = [
            _registro(campos_formato_invalido=("eb_p1_e2",)),
            _registro(fora_de_ordem_alfabetica=True),
            _registro(padrao_suspeito=True),
            _registro(),
        ]
        relatorio = gerar_relatorio(registros)
        assert relatorio.total_registros == 4

    def test_relatorio_nao_muta_os_registros_de_entrada(self):
        registros = [_registro(campos_formato_invalido=("eb_p1_e2",))]
        antes = registros[0].validacao.campos_formato_invalido
        gerar_relatorio(registros)
        assert registros[0].validacao.campos_formato_invalido == antes


class TestCamadaSequencia:
    def test_mesmo_buraco_em_varios_registros_do_curso_forma_um_grupo(self):
        registros = [
            _registro(curso="Medicina", buracos_classificacao={1: (7,)})
            for _ in range(3)
        ]
        relatorio = gerar_relatorio(registros)
        grupos = [g for g in relatorio.grupos if g.camada == CAMADA_SEQUENCIA]
        assert len(grupos) == 1
        assert grupos[0].quantidade == 3
        assert "Medicina" in grupos[0].padrao
        assert "Sistema 1" in grupos[0].padrao

    def test_buracos_de_cursos_diferentes_nao_se_misturam(self):
        registros = [
            _registro(curso="Medicina", buracos_classificacao={1: (7,)}),
            _registro(curso="Direito", buracos_classificacao={1: (7,)}),
        ]
        relatorio = gerar_relatorio(registros)
        grupos = [g for g in relatorio.grupos if g.camada == CAMADA_SEQUENCIA]
        assert len(grupos) == 2

    def test_sem_buraco_nao_gera_grupo(self):
        registros = [_registro(buracos_classificacao={})]
        relatorio = gerar_relatorio(registros)
        assert not [g for g in relatorio.grupos if g.camada == CAMADA_SEQUENCIA]


class TestCamadaOrdemAlfabetica:
    def test_agrupa_por_curso(self):
        registros = [
            _registro(curso="Medicina", fora_de_ordem_alfabetica=True) for _ in range(4)
        ] + [_registro(curso="Direito", fora_de_ordem_alfabetica=True)]
        relatorio = gerar_relatorio(registros)
        grupos = {
            g.padrao: g.quantidade
            for g in relatorio.grupos
            if g.camada == CAMADA_ORDEM_ALFABETICA
        }
        assert grupos == {"Medicina": 4, "Direito": 1}

    def test_em_ordem_nao_gera_grupo(self):
        registros = [_registro(fora_de_ordem_alfabetica=False)]
        relatorio = gerar_relatorio(registros)
        assert not [g for g in relatorio.grupos if g.camada == CAMADA_ORDEM_ALFABETICA]


class TestCamadaCotas:
    def test_agrupa_por_perfil_deduzido(self):
        registros = [
            _registro(padrao_suspeito=True, perfil="EP / Baixa Renda / PPI")
            for _ in range(2)
        ] + [_registro(padrao_suspeito=True, perfil="Universal")]
        relatorio = gerar_relatorio(registros)
        grupos = {
            g.padrao: g.quantidade for g in relatorio.grupos if g.camada == CAMADA_COTAS
        }
        assert grupos == {"EP / Baixa Renda / PPI": 2, "Universal": 1}

    def test_padrao_nao_suspeito_nao_gera_grupo(self):
        registros = [_registro(padrao_suspeito=False)]
        relatorio = gerar_relatorio(registros)
        assert not [g for g in relatorio.grupos if g.camada == CAMADA_COTAS]


class TestDistribuicaoDeChecksum:
    def test_deltas_concentrados_no_mesmo_intervalo_sao_marcados_concentrada(self):
        # 6 dos 10 registros que falham caem no mesmo intervalo de 0,05 em torno de 0,70 —
        # o cenário do protótipo (fórmula incompleta), 60% de share.
        registros = (
            [_registro(checksum=_checksum(0.700 + i * 0.001)) for i in range(6)]
            + [_registro(checksum=_checksum(d)) for d in (0.10, 0.25, 0.40, 0.55)]
        )
        relatorio = gerar_relatorio(registros)
        dist = relatorio.distribuicao_checksum
        assert dist is not None
        assert dist.total_falhas == 10
        assert dist.concentrada is True
        assert dist.faixas[0].quantidade == 6

    def test_deltas_espalhados_por_varios_intervalos_nao_sao_concentrada(self):
        registros = [
            _registro(checksum=_checksum(d)) for d in (0.10, 0.25, 0.40, 0.55, 0.70)
        ]
        relatorio = gerar_relatorio(registros)
        dist = relatorio.distribuicao_checksum
        assert dist is not None
        assert dist.total_falhas == 5
        assert dist.concentrada is False

    def test_deltas_dentro_da_tolerancia_nao_contam_como_falha(self):
        registros = [_registro(checksum=_checksum(0.001)) for _ in range(5)]
        relatorio = gerar_relatorio(registros)
        dist = relatorio.distribuicao_checksum
        assert dist is not None
        assert dist.total_falhas == 0
        assert dist.faixas == ()
        assert dist.concentrada is False

    def test_registros_sem_checksum_nao_contam_como_falha_nem_como_conferidos(self):
        registros = [_registro(checksum=None) for _ in range(3)]
        relatorio = gerar_relatorio(registros)
        assert relatorio.distribuicao_checksum is None
        assert relatorio.total_sem_checksum == 3

    def test_corpus_misto_reporta_sem_checksum_separado_das_falhas(self):
        registros = (
            [_registro(checksum=None) for _ in range(2)]
            + [_registro(checksum=_checksum(0.7)) for _ in range(3)]
        )
        relatorio = gerar_relatorio(registros)
        assert relatorio.total_sem_checksum == 2
        assert relatorio.distribuicao_checksum is not None
        assert relatorio.distribuicao_checksum.total_falhas == 3

    def test_grupos_de_checksum_tambem_aparecem_na_lista_unificada_de_grupos(self):
        registros = [_registro(checksum=_checksum(0.7)) for _ in range(4)]
        relatorio = gerar_relatorio(registros)
        grupos_checksum = [g for g in relatorio.grupos if g.camada == CAMADA_CHECKSUM]
        assert len(grupos_checksum) == 1
        assert grupos_checksum[0].quantidade == 4


class TestFormatacao:
    def test_formatar_terminal_nao_lanca_com_relatorio_vazio(self):
        relatorio = gerar_relatorio([])
        saida = formatar_terminal(relatorio)
        assert "Registros: 0" in saida

    def test_formatar_terminal_relata_forma_da_distribuicao(self):
        registros = [_registro(checksum=_checksum(0.7)) for _ in range(6)] + [
            _registro(checksum=_checksum(d)) for d in (0.1, 0.25, 0.4, 0.55)
        ]
        saida = formatar_terminal(gerar_relatorio(registros))
        assert "concentrada" in saida

    def test_formatar_markdown_cobre_as_cinco_camadas(self):
        registros = [
            _registro(campos_formato_invalido=("eb_p1_e2",)),
            _registro(fora_de_ordem_alfabetica=True),
            _registro(buracos_classificacao={1: (7,)}),
            _registro(padrao_suspeito=True),
            _registro(checksum=_checksum(0.7)),
        ]
        markdown = formatar_markdown(gerar_relatorio(registros))
        for camada in (
            CAMADA_CHECKSUM, CAMADA_SEQUENCIA, CAMADA_ORDEM_ALFABETICA,
            CAMADA_FORMATO_NUMERICO, CAMADA_COTAS,
        ):
            assert camada.capitalize() in markdown or camada.lower() in markdown.lower()

    def test_escrever_relatorio_grava_arquivo(self, tmp_path):
        destino = tmp_path / "sub" / "relatorio.md"
        escrever_relatorio(gerar_relatorio([_registro()]), destino)
        assert destino.exists()
        assert "Relatório de validação" in destino.read_text(encoding="utf-8")
