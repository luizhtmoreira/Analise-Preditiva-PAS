"""Ticket 08 — rodada completa: dispatch entre Famílias, fallback automático de Médias e
Desvios, relatório do corpus (camadas 1-6) e determinismo.

Exercita `rodada.rodar()`, a costura deste ticket, com as fixtures já commitadas
localmente pelos tickets 01/03/09 — nenhuma fixture nova precisa ser gerada só para este.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction.convocacao import escrever_csv_convocacao  # type: ignore
from pas_extraction.csv_writer import escrever_csv  # type: ignore
from pas_extraction.medias_desvios import escrever_csv_medias_desvios  # type: ignore
from pas_extraction.rodada import (  # type: ignore
    RodadaCompleta,
    derivar_notas_corte_da_rodada,
    formatar_markdown_corpus,
    formatar_terminal_corpus,
    gerar_relatorio_corpus,
    rodar,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIXTURE_RESULTADO_FINAL = FIXTURES_DIR / "resultado_final_22_campos.pdf"
FIXTURE_CONVOCACAO = FIXTURES_DIR / "convocacao_registro.pdf"
FIXTURE_MEDIAS_AVULSO = FIXTURES_DIR / "medias_desvios_avulso.pdf"

# Ed_31 (2016/2018) sem tabela na cauda + o Edital avulso do mesmo triênio — o par real
# usado por `test_pas_extraction_checksum.py` para provar `medias_desvios=<avulso>` na
# chamada manual. Aqui é o mesmo par, mas SEM passar o parâmetro: é o que prova que
# `rodar()` descobre sozinha qual avulso combina com qual Resultado Final.
FIXTURE_SEM_CAUDA = FIXTURES_DIR / "resultado_final_cota_suspeita.pdf"
FIXTURE_AVULSO_MESMO_TRIENIO = FIXTURES_DIR / "medias_desvios_avulso_2016_2018.pdf"
CONTAGEM_SEM_CAUDA = 84
FECHAM_SEM_CAUDA = 82


def _pular_se_ausente(*caminhos: Path) -> None:
    for caminho in caminhos:
        if not caminho.exists():
            pytest.skip(
                f"Fixture {caminho.relative_to(Path(__file__).parent.parent)} não "
                "encontrada (gerada localmente pelos tickets 01/03/06/09)."
            )


class TestDispatchPorFamilia:
    def test_cada_pdf_vai_para_a_lista_da_sua_familia(self):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO)

        rodada = rodar([FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO])

        assert isinstance(rodada, RodadaCompleta)
        assert len(rodada.resultado_final) == 1
        assert len(rodada.convocacao) == 1
        assert len(rodada.medias_desvios) == 1
        assert rodada.ignorados == []

    def test_ordem_de_descoberta_e_preservada_dentro_de_cada_familia(self):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_SEM_CAUDA)

        rodada = rodar([FIXTURE_SEM_CAUDA, FIXTURE_RESULTADO_FINAL])

        assert [r.arquivo_origem for r in rodada.resultado_final] == [
            FIXTURE_SEM_CAUDA.name, FIXTURE_RESULTADO_FINAL.name,
        ]

    def test_um_comando_produz_os_tres_csvs(self, tmp_path):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO)

        rodada = rodar([FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO])

        total_rf = escrever_csv(rodada.resultado_final, tmp_path / "resultado_final.csv")
        total_conv = escrever_csv_convocacao(rodada.convocacao, tmp_path / "convocacao.csv")
        total_md = escrever_csv_medias_desvios(rodada.medias_desvios, tmp_path / "medias_desvios.csv")

        assert total_rf > 0 and total_conv > 0 and total_md > 0
        assert (tmp_path / "resultado_final.csv").exists()
        assert (tmp_path / "convocacao.csv").exists()
        assert (tmp_path / "medias_desvios.csv").exists()


class TestFallbackAutomaticoDeMediasEDesvios:
    """spec, user story 8: o pipeline busca a tabela na cauda e, se não achar, num Edital
    avulso — sem que quem roda o comando precise apontar manualmente qual avulso serve para
    qual Resultado Final (ao contrário de `extract --medias-desvios`, pensado para um
    Edital por vez)."""

    def test_resultado_final_sem_cauda_ganha_checksum_do_avulso_descoberto_junto(self):
        _pular_se_ausente(FIXTURE_SEM_CAUDA, FIXTURE_AVULSO_MESMO_TRIENIO)

        # Sozinho, sem a tabela avulsa na rodada, não há como conferir.
        sozinho = rodar([FIXTURE_SEM_CAUDA])
        assert all(r.checksum is None for r in sozinho.resultado_final[0].registros)

        # Junto do avulso do mesmo triênio, `rodar()` casa os dois sozinha.
        junto = rodar([FIXTURE_SEM_CAUDA, FIXTURE_AVULSO_MESMO_TRIENIO])
        registros = junto.resultado_final[0].registros

        assert len(registros) == CONTAGEM_SEM_CAUDA
        assert all(r.checksum is not None for r in registros)
        assert sum(1 for r in registros if r.checksum.fecha) == FECHAM_SEM_CAUDA

    def test_avulso_de_trienio_diferente_nao_e_usado(self):
        # O avulso de 2016/2018 não deve ser aplicado a um Resultado Final de outro
        # triênio: a tabela em si tem médias/desvios de uma faixa de nota completamente
        # diferente, e um casamento errado produziria deltas na casa das dezenas (ver
        # `pipeline._conferir_trienio`) em vez de simplesmente não conferir.
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_AVULSO_MESMO_TRIENIO)

        # FIXTURE_RESULTADO_FINAL (Ed_38, 2022/2024) já tem tabela na própria cauda? Não —
        # é a fatia de 6 páginas sem a 242, então também depende de avulso para fechar, mas
        # o avulso de 2016/2018 é de outro triênio e não deve ser usado.
        rodada = rodar([FIXTURE_RESULTADO_FINAL, FIXTURE_AVULSO_MESMO_TRIENIO])

        assert all(r.checksum is None for r in rodada.resultado_final[0].registros)


class TestDeterminismo:
    """spec, user story 35: mesma entrada, mesma saída, byte a byte."""

    def test_rodar_duas_vezes_produz_csvs_identicos(self, tmp_path):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO)
        pdfs = [FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO]

        primeira = rodar(pdfs)
        segunda = rodar(pdfs)

        destino1 = tmp_path / "1"
        destino2 = tmp_path / "2"
        for destino, rodada in ((destino1, primeira), (destino2, segunda)):
            escrever_csv(rodada.resultado_final, destino.with_suffix(".rf.csv"))
            escrever_csv_convocacao(rodada.convocacao, destino.with_suffix(".conv.csv"))
            escrever_csv_medias_desvios(rodada.medias_desvios, destino.with_suffix(".md.csv"))

        for sufixo in (".rf.csv", ".conv.csv", ".md.csv"):
            bytes1 = destino1.with_suffix(sufixo).read_bytes()
            bytes2 = destino2.with_suffix(sufixo).read_bytes()
            assert bytes1 == bytes2

    def test_relatorio_do_corpus_e_identico_entre_rodadas(self):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO)
        pdfs = [FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO, FIXTURE_MEDIAS_AVULSO]

        relatorio1 = formatar_markdown_corpus(gerar_relatorio_corpus(rodar(pdfs)))
        relatorio2 = formatar_markdown_corpus(gerar_relatorio_corpus(rodar(pdfs)))

        assert relatorio1 == relatorio2


class TestRelatorioDoCorpus:
    def test_cobre_a_camada_de_reconciliacao_cruzada(self):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO)

        rodada = rodar([FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO])
        relatorio = gerar_relatorio_corpus(rodada)

        assert "Reconciliação cruzada" in formatar_terminal_corpus(relatorio)
        markdown = formatar_markdown_corpus(relatorio)
        assert "## 6. Reconciliação cruzada entre Editais" in markdown

    def test_cobre_as_camadas_1_a_5_herdadas_do_ticket_07(self):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL)

        rodada = rodar([FIXTURE_RESULTADO_FINAL])
        relatorio = gerar_relatorio_corpus(rodada)

        assert relatorio.validacao.total_registros == len(rodada.resultado_final[0].registros)

    def test_relatorio_vazio_nao_lanca(self):
        rodada = RodadaCompleta()
        relatorio = gerar_relatorio_corpus(rodada)

        assert formatar_terminal_corpus(relatorio)
        assert formatar_markdown_corpus(relatorio)
        assert relatorio.divergencias_nome == ()


class TestNotasDeCorteDaRodada:
    """Ticket 10 — a derivação que consome as duas Famílias da rodada de uma vez.

    A regra de derivação em si é testada em `test_pas_extraction_notas_corte.py`, com
    registros sintéticos; aqui só a costura: os registros das duas Famílias chegam
    achatados na derivação, sem releitura de PDF."""

    def test_deriva_a_partir_das_duas_familias_da_rodada(self):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO)

        rodada = rodar([FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO])
        derivacao = derivar_notas_corte_da_rodada(rodada)

        # Cada convocado da fixture ou define/entra num corte, ou é contado como sem
        # Argumento Final — nenhum some pelo caminho. (As duas fixtures são de Editais de
        # triênios diferentes, então na prática nenhum casa: é o caso "Convocação sem o
        # Resultado Final do mesmo triênio no corpus", que precisa sair contado.)
        convocados = sum(len(r.registros) for r in rodada.convocacao)
        sem_argumento = sum(q for _, q in derivacao.convocados_sem_argumento_final)
        assert convocados > 0
        assert sem_argumento > 0
        assert derivacao.grupos_sem_chamada_conhecida == ()

    def test_rodada_vazia_nao_lanca(self):
        derivacao = derivar_notas_corte_da_rodada(RodadaCompleta())

        assert derivacao.notas == ()
        assert derivacao.convocados_sem_argumento_final == ()

    def test_derivar_duas_vezes_produz_o_mesmo_resultado(self):
        _pular_se_ausente(FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO)
        pdfs = [FIXTURE_RESULTADO_FINAL, FIXTURE_CONVOCACAO]

        assert derivar_notas_corte_da_rodada(rodar(pdfs)) == derivar_notas_corte_da_rodada(
            rodar(pdfs)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
