"""Testes do relatório de diferenças do OFFICIAL_STATS (ticket 11).

O relatório é comparação e formatação — dois comportamentos com custos de teste bem
diferentes. A coleta precisa de um PDF real (fixture, pulada se ausente); a comparação e a
formatação não precisam de PDF nenhum e são exercitadas com `EtapaOficial` montada na mão,
para o teste falhar por causa da regra de comparação e não por causa do parser do ticket 03.

O `OFFICIAL_STATS` de produção é injetado nos testes (`comparar(..., stats_atuais=...)`) em
vez de importado: assim o teste não quebra quando o ticket 12 trocar os valores reais — o que
é exatamente o ponto, já que este relatório existe para ser lido *antes* daquela troca.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction.models import Proveniencia  # type: ignore
from pas_extraction.relatorio_official_stats import (  # type: ignore
    ColetaOficial,
    EtapaOficial,
    TrienioInvalidoError,
    ValorOficial,
    ano_da_prova,
    coletar_valores_oficiais,
    comparar,
    conferir_entre_fontes,
    formatar_markdown,
    formatar_resumo,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Mesma fixture do ticket 03: Edital avulso de médias e desvios (ED_34, triênio 2019/2021).
FIXTURE_AVULSO = FIXTURES_DIR / "medias_desvios_avulso.pdf"

_COMANDO_AVULSO = (
    "python -m pas_extraction.cli fixture "
    "'data/pdfs/ED_34_PAS_3 _2019 -2021_Media_e_desvio_padrao.pdf' 1 1 "
    f"{FIXTURE_AVULSO}"
)


@dataclass
class ExamStatsFalso:
    """Cópia mínima do `ExamStats` de `pas_constants` — só os 6 campos que o relatório lê."""

    m_p1: float
    dp_p1: float
    m_p2: float
    dp_p2: float
    m_red: float
    dp_red: float


def _proveniencia(arquivo="Ed_qualquer.pdf", trienio="2022/2024"):
    return Proveniencia(
        arquivo_origem=arquivo, edital="38", trienio=trienio, pagina=242
    )


def _etapa_oficial(
    chave=(2022, 1),
    trienio="2022/2024",
    p2=(20.406, 13.533),
    red=(5.849, 2.793),
    parte_1=((3.665, 3.109), (3.620, 2.597), (3.140, 2.530)),
    arquivo="Ed_38_2024.pdf",
):
    prov = _proveniencia(arquivo, trienio)
    linguas = ("inglesa", "francesa", "espanhola")
    return EtapaOficial(
        chave=chave,
        trienio=trienio,
        parte_1={
            lingua: ValorOficial(media=m, desvio_padrao=dp, proveniencia=prov)
            for lingua, (m, dp) in zip(linguas, parte_1)
        },
        parte_2=ValorOficial(media=p2[0], desvio_padrao=p2[1], proveniencia=prov),
        redacao=ValorOficial(media=red[0], desvio_padrao=red[1], proveniencia=prov),
        fontes=(arquivo,),
    )


def _coleta(*etapas, divergencias=None, ignorados=None):
    return ColetaOficial(
        etapas={e.chave: e for e in etapas},
        divergencias=list(divergencias or []),
        ignorados=dict(ignorados or {}),
    )


class TestAnoDaProva:
    """A conversão que liga os dois lados: o OFFICIAL_STATS é indexado por ano da prova, o
    Edital é identificado por triênio."""

    def test_etapa_n_do_trienio_cai_no_ano_inicial_mais_n_menos_1(self):
        assert ano_da_prova("2022/2024", 1) == 2022
        assert ano_da_prova("2022/2024", 2) == 2023
        assert ano_da_prova("2022/2024", 3) == 2024

    def test_aceita_trienio_com_hifen(self):
        assert ano_da_prova("2016-2018", 3) == 2018

    def test_recusa_trienio_que_nao_cobre_tres_anos(self):
        with pytest.raises(TrienioInvalidoError):
            ano_da_prova("2022/2023", 1)

    def test_recusa_trienio_desconhecido(self):
        with pytest.raises(TrienioInvalidoError):
            ano_da_prova("desconhecido", 1)


class TestComparacao:
    def test_diferenca_e_atual_menos_oficial_para_parte_2_e_redacao(self):
        # O caso citado no ticket: 2022/2024 Etapa 1, m_p2 estimado 20.709 contra o
        # oficial 20.406.
        atuais = {(2022, 1): ExamStatsFalso(3.604, 3.005, 20.709, 13.581, 5.888, 2.779)}
        relatorio = comparar(_coleta(_etapa_oficial()), stats_atuais=atuais)

        por_campo = {d.campo: d for d in relatorio.diferencas}
        assert set(por_campo) == {"m_p2", "dp_p2", "m_red", "dp_red"}
        assert por_campo["m_p2"].atual == pytest.approx(20.709)
        assert por_campo["m_p2"].oficial == pytest.approx(20.406)
        assert por_campo["m_p2"].diferenca == pytest.approx(0.303)
        assert por_campo["m_p2"].diferenca_relativa == pytest.approx(0.303 / 20.406)

    def test_m_p1_nao_entra_na_comparacao_1_para_1(self):
        """Não existe um `m_p1` oficial — três línguas, três valores. Reportar uma
        diferença única aqui seria inventar um valor que o Edital não publica."""
        atuais = {(2022, 1): ExamStatsFalso(3.604, 3.005, 20.709, 13.581, 5.888, 2.779)}
        relatorio = comparar(_coleta(_etapa_oficial()), stats_atuais=atuais)

        assert not [d for d in relatorio.diferencas if d.campo in ("m_p1", "dp_p1")]

    def test_parte_1_expoe_os_tres_valores_oficiais_que_o_valor_atual_agrega(self):
        atuais = {(2022, 1): ExamStatsFalso(3.604, 3.005, 20.709, 13.581, 5.888, 2.779)}
        relatorio = comparar(_coleta(_etapa_oficial()), stats_atuais=atuais)

        assert len(relatorio.parte_1) == 1
        agregacao = relatorio.parte_1[0]
        assert set(agregacao.por_lingua) == {"inglesa", "francesa", "espanhola"}
        assert agregacao.m_p1_atual == pytest.approx(3.604)
        assert agregacao.diferenca_media("inglesa") == pytest.approx(3.604 - 3.665)
        assert agregacao.diferenca_media("espanhola") == pytest.approx(3.604 - 3.140)
        assert agregacao.amplitude_media == pytest.approx(3.665 - 3.140)

    def test_entrada_do_official_stats_sem_edital_e_listada_e_nao_comparada(self):
        atuais = {
            (2022, 1): ExamStatsFalso(3.604, 3.005, 20.709, 13.581, 5.888, 2.779),
            (2016, 1): ExamStatsFalso(4.421, 2.782, 24.246, 13.169, 6.074, 2.669),
        }
        relatorio = comparar(_coleta(_etapa_oficial()), stats_atuais=atuais)

        assert relatorio.sem_cobertura == [(2016, 1)]
        assert relatorio.chaves_comparadas == [(2022, 1)]

    def test_etapa_oficial_fora_do_official_stats_e_listada_separadamente(self):
        """Cobertura nova (o triênio 2023/2025) é dado a acrescentar, não correção a
        aplicar — as duas listas não podem se confundir."""
        atuais = {(2022, 1): ExamStatsFalso(3.604, 3.005, 20.709, 13.581, 5.888, 2.779)}
        coleta = _coleta(
            _etapa_oficial(),
            _etapa_oficial(chave=(2023, 1), trienio="2023/2025"),
        )
        relatorio = comparar(coleta, stats_atuais=atuais)

        assert relatorio.ausentes_no_official_stats == [(2023, 1)]
        assert relatorio.sem_cobertura == []

    def test_valores_identicos_dao_diferenca_zero(self):
        atuais = {(2022, 1): ExamStatsFalso(3.6, 3.0, 20.406, 13.533, 5.849, 2.793)}
        relatorio = comparar(_coleta(_etapa_oficial()), stats_atuais=atuais)

        assert all(d.diferenca == 0 for d in relatorio.diferencas)


class TestConferenciaEntreFontes:
    """Um triênio pode aparecer em dois Editais (o avulso de médias e a cauda do Resultado
    Final). Quando aparece, os dois têm que dizer a mesma coisa — e quando não dizem, isso
    vira relatório, não desempate silencioso."""

    def test_fontes_iguais_nao_geram_divergencia(self):
        assert conferir_entre_fontes(
            _etapa_oficial(arquivo="avulso.pdf"), _etapa_oficial(arquivo="cauda.pdf")
        ) == []

    def test_diferenca_em_parte_2_e_reportada_com_as_duas_fontes(self):
        divergencias = conferir_entre_fontes(
            _etapa_oficial(arquivo="avulso.pdf"),
            _etapa_oficial(arquivo="cauda.pdf", p2=(20.500, 13.533)),
        )

        assert len(divergencias) == 1
        d = divergencias[0]
        assert (d.prova, d.campo) == ("parte_2", "media")
        assert (d.valor_a, d.fonte_a) == (20.406, "avulso.pdf")
        assert (d.valor_b, d.fonte_b) == (20.500, "cauda.pdf")

    def test_diferenca_em_uma_lingua_da_parte_1_e_reportada_com_a_lingua(self):
        divergencias = conferir_entre_fontes(
            _etapa_oficial(arquivo="avulso.pdf"),
            _etapa_oficial(
                arquivo="cauda.pdf",
                parte_1=((3.665, 3.109), (3.999, 2.597), (3.140, 2.530)),
            ),
        )

        assert [(d.prova, d.lingua_estrangeira, d.campo) for d in divergencias] == [
            ("parte_1", "francesa", "media")
        ]


class TestFormatacao:
    def _relatorio(self):
        atuais = {
            (2022, 1): ExamStatsFalso(3.604, 3.005, 20.709, 13.581, 5.888, 2.779),
            (2016, 1): ExamStatsFalso(4.421, 2.782, 24.246, 13.169, 6.074, 2.669),
        }
        coleta = _coleta(
            _etapa_oficial(),
            _etapa_oficial(chave=(2023, 1), trienio="2023/2025"),
            ignorados={"Ed_28_Conv.pdf": "Família Convocação — não publica tabela de médias"},
        )
        return comparar(coleta, stats_atuais=atuais)

    def test_markdown_mostra_atual_oficial_e_diferenca_por_ano_e_etapa(self):
        texto = formatar_markdown(self._relatorio())

        assert "| 2022 / Etapa 1 | `m_p2` | 20.709 | 20.406 | +0.303 |" in texto

    def test_markdown_lista_as_entradas_sem_cobertura_explicitamente(self):
        texto = formatar_markdown(self._relatorio())

        secao = texto.split("## 3.")[1].split("## 4.")[0]
        assert "2016 / Etapa 1" in secao

    def test_markdown_mostra_o_m_p1_unico_contra_as_tres_linguas(self):
        texto = formatar_markdown(self._relatorio())

        secao = texto.split("## 2.")[1].split("## 3.")[0]
        for lingua in ("inglesa", "francesa", "espanhola"):
            assert lingua in secao
        # o valor agregado e os três oficiais na mesma linha
        assert "| 2022 / Etapa 1 | `m_p1` | 3.604 | 3.665 |" in secao

    def test_resumo_de_terminal_conta_o_que_ficou_de_fora(self):
        texto = formatar_resumo(self._relatorio())

        assert "Entradas comparadas: 1" in texto
        assert "Sem cobertura nos Editais: 1" in texto
        assert "Oficiais fora do OFFICIAL_STATS: 1" in texto


class TestColetaDeEditalReal:
    def test_coleta_do_edital_avulso_indexa_por_ano_da_prova(self):
        if not FIXTURE_AVULSO.exists():
            pytest.skip(
                f"Fixture {FIXTURE_AVULSO.relative_to(Path(__file__).parent.parent)} não "
                f"encontrada. Gere localmente com: {_COMANDO_AVULSO}"
            )

        coleta = coletar_valores_oficiais([FIXTURE_AVULSO])

        # Triênio 2019/2021: Etapa 1 = prova de 2019, Etapa 3 = prova de 2021.
        assert sorted(coleta.etapas) == [(2019, 1), (2020, 2), (2021, 3)]
        etapa_1 = coleta.etapas[(2019, 1)]
        # Valores conferidos no PDF no ticket 03.
        assert etapa_1.parte_2.media == pytest.approx(26.738)
        assert etapa_1.redacao.media == pytest.approx(6.617)
        assert etapa_1.parte_1["francesa"].media == pytest.approx(5.064)
        assert etapa_1.fontes == (FIXTURE_AVULSO.name,)

    def test_official_stats_de_producao_e_o_padrao_quando_nao_injetado(self):
        """A costura real: sem `stats_atuais`, o relatório compara com o que está em
        produção hoje."""
        if not FIXTURE_AVULSO.exists():
            pytest.skip("fixture ausente")

        relatorio = comparar(coletar_valores_oficiais([FIXTURE_AVULSO]))

        assert (2019, 1) in relatorio.chaves_comparadas
        # Todo o resto do OFFICIAL_STATS fica sem cobertura nesta coleta de um Edital só —
        # o que prova que a lista de faltantes é calculada, não presumida vazia.
        assert relatorio.sem_cobertura


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
