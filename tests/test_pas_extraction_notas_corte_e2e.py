"""Ticket 17 — Fixture sintética ponta a ponta para Nota de Corte.

Fecha a lacuna registrada no relatório do ticket 10 (seção de limitações conhecidas) e no
item 5 de `.scratch/pdf-extraction/relatorios/defeitos-pendentes.md`: até aqui não existia
teste que rodasse o pipeline inteiro — extração de PDF → Resultado Final → Convocação →
derivação de Nota de Corte — sem depender de `data/pdfs` local. As fixtures reais não podem
ser commitadas ([[project_parser_privacy]]: carregam nome de Aluno identificável); a saída
aqui é a mesma que o ticket 02 usou para a mesma restrição — `fixtures.gerar_pdf_texto_
sintetico` monta o PDF a partir de texto **inteiramente inventado** (nenhum nome, inscrição
ou nota vem de um Aluno real) dentro do próprio teste, em `tmp_path`, então nada disso é
gravado no repositório nem exige o corpus local.

A regra de derivação em si já tem 41 testes sintéticos diretos em
`test_pas_extraction_notas_corte.py` (construindo `RegistroResultadoFinal`/`RegistroConvocacao`
à mão); o que faltava era provar que o caminho PDF → registro também produz esses mesmos
objetos corretamente, com os dois extratores reais (`pipeline.extrair_edital` e
`convocacao.extrair_edital_convocacao`) no meio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction.convocacao import extrair_edital_convocacao  # type: ignore
from pas_extraction.fixtures import gerar_pdf_texto_sintetico  # type: ignore
from pas_extraction.notas_corte import derivar_notas_corte  # type: ignore
from pas_extraction.pipeline import extrair_edital  # type: ignore

TRIENIO = "2023/2025"

# Cabeçalho de página 1 de um Resultado Final: institucional + a frase "na seguinte ordem"
# que `schema.classificar_familia` ancora para reconhecer a Família (precisa conter
# "numero de inscricao" canonizado). Nenhum destes nomes/números existe fora deste teste.
_CABECALHO_RESULTADO_FINAL = (
    "UNIVERSIDADE DE BRASILIA (UnB)\n"
    "EDITAL No 17 - PAS/UnB - TRIENIO 2023/2025\n"
    "Os resultados finais serao publicados na seguinte ordem: Numero de inscricao, nome do "
    "candidato, EB da Prova 1 da Etapa 1, EB da Prova 2 da Etapa 1, Nota da Redacao da "
    "Etapa 1, EB da Prova 1 da Etapa 2, EB da Prova 2 da Etapa 2, Nota da Redacao da Etapa "
    "2, EB da Prova 1 da Etapa 3, EB da Prova 2 da Etapa 3, Nota da Redacao da Etapa 3, "
    "Argumento Final, Classificacao em cada um dos Sistemas de Concorrencia.\n"
    "1.1.1 CAMPUS DARCY RIBEIRO - DIURNO\n"
    "1.1.1.1 MEDICINA (BACHARELADO)\n"
)

# Cabeçalho de página 1 de uma Convocação: precisa de "sistema" e "subsistema" na frase de
# ordem (é o que distingue esta Família de Resultado Final em `classificar_familia`), mais a
# chamada e o semestre em algum lugar do texto (`_CHAMADA_RE`/`_SEMESTRE_RE`).
_CABECALHO_CONVOCACAO = (
    "UNIVERSIDADE DE BRASILIA (UnB)\n"
    "EDITAL No 18 - PAS/UnB - TRIENIO 2023/2025\n"
    "Convocacao para matricula, em primeira chamada, do primeiro semestre.\n"
    "Serao publicados na seguinte ordem: numero de inscricao, nome do candidato, "
    "sistema/subsistema de concorrencia.\n"
    "1 DA CONVOCACAO PARA MATRICULA\n"
    "1.1 CAMPUS DARCY RIBEIRO - DIURNO\n"
    "MEDICINA (BACHARELADO)\n"
)


def _registro_resultado_final(inscricao: str, nome: str, argumento_final: float) -> str:
    """Um registro de 22 campos no formato exato do Edital: inscrição, nome, 9 notas de
    3 casas decimais, Argumento Final, 10 classificações. As classificações não importam
    para a derivação de Nota de Corte (que só olha `argumento_final`/`inscricao`), por isso
    saem todas "-" (não concorreu em nenhum Sistema)."""
    notas = "10.000, 20.000, 5.000, 10.000, 20.000, 5.000, 10.000, 20.000, 5.000"
    classificacoes = ", ".join(["-"] * 10)
    return f"{inscricao}, {nome}, {notas}, {argumento_final:.3f}, {classificacoes}"


def _registro_convocacao(inscricao: str, nome: str, sistema: int) -> str:
    """Uma linha de registro de Convocação: inscrição, nome, Sistema — em modo `layout`,
    por isso alinhado com espaço em vez de vírgula (`convocacao._REGISTRO_RE`)."""
    return f"{inscricao} {nome}                    {sistema}"


@pytest.fixture
def notas_corte_derivadas(tmp_path: Path):
    """Roda o pipeline inteiro sobre dois PDFs sintéticos do mesmo triênio (Resultado Final
    + Convocação), com quatro inscrições que se cruzam entre os dois — três delas convocadas
    pelo Sistema 1 (Universal) na mesma chamada, duas delas empatadas no menor Argumento
    Final, para cobrir o mesmo cenário de desempate que
    `test_pas_extraction_notas_corte.py` testa sinteticamente (construção direta de
    registro, sem passar por PDF)."""
    pagina_rf = _CABECALHO_RESULTADO_FINAL + " / ".join([
        _registro_resultado_final("20100001", "Fulano de Tal", 100.000),
        _registro_resultado_final("20100002", "Beltrano da Silva", 95.000),
        _registro_resultado_final("20100003", "Ciclana de Souza", 70.000),
        _registro_resultado_final("20100004", "Sicrano de Almeida", 70.000),
    ])
    pdf_resultado_final = gerar_pdf_texto_sintetico(
        [pagina_rf], tmp_path / "resultado_final_sintetico.pdf"
    )

    pagina_conv = _CABECALHO_CONVOCACAO + "\n".join([
        _registro_convocacao("20100001", "Fulano de Tal", 1),
        _registro_convocacao("20100002", "Beltrano da Silva", 1),
        _registro_convocacao("20100003", "Ciclana de Souza", 1),
        _registro_convocacao("20100004", "Sicrano de Almeida", 1),
    ])
    pdf_convocacao = gerar_pdf_texto_sintetico(
        [pagina_conv], tmp_path / "convocacao_sintetico.pdf"
    )

    resultado_final = extrair_edital(pdf_resultado_final)
    convocacao = extrair_edital_convocacao(pdf_convocacao)

    return resultado_final, convocacao, derivar_notas_corte(
        resultado_final.registros, convocacao.registros
    )


class TestPipelineCompletoNotaDeCorte:
    def test_extratores_leem_as_quatro_inscricoes_dos_dois_pdfs(self, notas_corte_derivadas):
        resultado_final, convocacao, _ = notas_corte_derivadas

        assert resultado_final.trienio == TRIENIO
        assert len(resultado_final.registros) == 4
        assert convocacao.trienio == TRIENIO
        assert len(convocacao.registros) == 4

    def test_deriva_exatamente_um_corte_para_medicina_sistema_1(self, notas_corte_derivadas):
        _, _, derivacao = notas_corte_derivadas

        assert len(derivacao.notas) == 1
        corte = derivacao.notas[0]
        assert corte.trienio == TRIENIO
        assert corte.campus == "DARCY RIBEIRO"
        assert corte.curso == "MEDICINA (BACHARELADO)"
        assert corte.turno == "DIURNO"
        assert corte.sistema == 1
        assert corte.chamada == "1"

    def test_corte_e_o_menor_argumento_final_entre_os_convocados(self, notas_corte_derivadas):
        """20100003 e 20100004 empatam em 70.000 — o menor entre os quatro convocados."""
        _, _, derivacao = notas_corte_derivadas
        corte = derivacao.notas[0]

        assert corte.argumento_final == pytest.approx(70.000)
        assert corte.convocados_na_chamada == 4
        assert corte.convocados_com_argumento == 4
        assert corte.parcial is False

    def test_empate_e_desempatado_pela_menor_inscricao(self, notas_corte_derivadas):
        """Mesmo critério de `derivar_notas_corte`: empate no Argumento Final é resolvido
        pela inscrição mais baixa — escolhe o *representante* da linha, não muda o valor."""
        _, _, derivacao = notas_corte_derivadas
        corte = derivacao.notas[0]

        assert corte.inscricao == "20100003"
        assert corte.nome == "Ciclana de Souza"

    def test_checksum_fecha_e_none_sem_tabela_de_medias_e_desvios(self, notas_corte_derivadas):
        """Nenhum dos dois PDFs sintéticos traz a tabela oficial de Médias e Desvios — o
        checksum do Argumento Final fica `None` (não conferido), nunca `False` (reprovado)."""
        _, _, derivacao = notas_corte_derivadas
        corte = derivacao.notas[0]

        assert corte.checksum_fecha is None

    def test_nao_sobra_nenhum_grupo_sem_argumento_ou_sem_chamada(self, notas_corte_derivadas):
        _, _, derivacao = notas_corte_derivadas

        assert derivacao.grupos_sem_argumento_final == ()
        assert derivacao.grupos_sem_chamada_conhecida == ()
        assert derivacao.convocados_sem_argumento_final == ()
        assert derivacao.inscricoes_com_argumento_ambiguo == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
