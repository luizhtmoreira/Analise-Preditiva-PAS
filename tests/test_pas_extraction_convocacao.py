import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction.constants import MAPA_SISTEMAS  # type: ignore
from pas_extraction.convocacao import (  # type: ignore
    LISTA_CONVOCACAO,
    LISTA_OUTRA_RELACAO,
    ResultadoExtracaoConvocacao,
    extrair_chamada_e_semestre,
    extrair_edital_convocacao,
    parse_convocacao,
)
from pas_extraction.models import ContextoEdital, FamiliaEdital  # type: ignore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_CONVOCACAO = FIXTURES_DIR / "convocacao_registro.pdf"

# Contagem observada ao gerar a fixture com `python -m pas_extraction.cli fixture
# 'data/pdfs/convocacoes/Ed_28_PAS_3_2021_2023_Conv_RA_1ª_Chamada.pdf' 1 5 <destino>`.
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
            "'data/pdfs/convocacoes/Ed_28_PAS_3_2021_2023_Conv_RA_1ª_Chamada.pdf' "
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


class _PaginaFalsa:
    def __init__(self, texto: str) -> None:
        self._texto = texto

    def extract_text(self, extraction_mode: str = "plain") -> str:
        return self._texto


class _ReaderFalso:
    """Reader com texto colado de Editais reais, para exercitar o parse por linha sem
    fatiar uma fixture nova por caso.

    A fixture real (`convocacao_registro.pdf`) é uma fatia de Ed_28, que tem uma seção de
    topo só; os casos abaixo precisam de Editais com **duas** seções (Ed_34/35/37/40-43,
    triênio 2016/2018), e o que está sob teste é o parse de linha, não a leitura do PDF."""

    def __init__(self, *textos: str) -> None:
        self.pages = [_PaginaFalsa(t) for t in textos]


# Colado de Ed_35 _2016-2018_PAS_3_Terceira_Convocacao.pdf (modo layout), com os nomes
# trocados e a lista encurtada. A estrutura — numeração de sumário nos cabeçalhos, e a
# seção 2 repetindo campus/curso com outro prefixo — é a do Edital real.
_TEXTO_DUAS_SECOES = """
1  DA   TERCEIRA      CONVOCAÇÃO         PARA  O  PRÉ-REGISTRO  ACADÊMICO  REFERENTE  AO  PRIMEIRO
SEMESTRE DE 2019
1.1 Convocação para o pré-registro acadêmico referente ao primeiro semestre de 2019, na seguinte
1.1.1 CAMPUS DARCY RIBEIRO           – DIURNO
1.1.1.1 ADMINISTRAÇÃO (BACHARELADO)
16124894       Pedro de Vilhena Moraes Silva                    1
16124895       Marina Alves Ferreira                            2
2 DA RELAÇÃO DOS CANDIDATOS QUE NÃO COMPARECERAM AO PRÉ-REGISTRO ACADÊMICO
2.1 Da relação dos candidatos que  não compareceram ao pré-registro acadêmico, na seguinte ordem:
2.1.1 CAMPUS DARCY RIBEIRO – DIURNO
2.1.1.1 ADMINISTRAÇÃO (BACHARELADO)
16999999       Joana Ribeiro Dourado                            1
3 DO PRÉ-REGISTRO ACADÊMICO
3.1 O pré-registro é composto por duas fases, conforme descrito a seguir.
"""

_CONTEXTO = ContextoEdital(arquivo_origem="Ed_35.pdf", edital="35", trienio="2016/2018")


class TestSecaoDeTopo:
    """Ticket 10: um Edital de Convocação pode publicar **duas** listas de Alunos em
    seções de topo diferentes — a convocação em si (seção 1) e a relação de quem não
    compareceu/esteve ausente (seção 2). São 1.353 registros de outra lista em 7 dos 64
    Editais reais; sem distinguir as duas, quem não compareceu entra no cálculo da Nota de
    Corte como se tivesse sido chamado naquela chamada."""

    def test_registros_da_secao_de_convocacao_sao_marcados_como_convocacao(self):
        registros = parse_convocacao(
            _ReaderFalso(_TEXTO_DUAS_SECOES), _CONTEXTO, semestre="1", chamada="3",
        )

        convocados = [r for r in registros if r.lista == LISTA_CONVOCACAO]
        assert [r.inscricao for r in convocados] == ["16124894", "16124895"]

    def test_registros_da_relacao_de_ausentes_sao_marcados_como_outra_lista(self):
        registros = parse_convocacao(
            _ReaderFalso(_TEXTO_DUAS_SECOES), _CONTEXTO, semestre="1", chamada="3",
        )

        outros = [r for r in registros if r.lista == LISTA_OUTRA_RELACAO]
        assert [r.inscricao for r in outros] == ["16999999"]

    def test_nenhum_registro_e_descartado_ao_separar_as_listas(self):
        # As duas listas continuam saindo — o que muda é o rótulo, não a contagem: o
        # spec proíbe descartar registro em silêncio (user story 23).
        registros = parse_convocacao(
            _ReaderFalso(_TEXTO_DUAS_SECOES), _CONTEXTO, semestre="1", chamada="3",
        )

        assert len(registros) == 3

    def test_titulo_de_ausentes_que_menciona_convocacao_nao_vira_convocacao(self):
        # Os Editais de 2016/2018 — justamente os que têm duas seções — escrevem
        # "convocação" onde os outros escrevem "chamada" (ver `_CHAMADA_RE`). Uma regra
        # que procurasse a palavra "convocação" para marcar convocados classificaria este
        # título ao contrário.
        texto = """
2 DA RELAÇÃO DOS CANDIDATOS AUSENTES NA PRIMEIRA CONVOCAÇÃO
2.1.1 CAMPUS DARCY RIBEIRO – DIURNO
2.1.1.1 ADMINISTRAÇÃO (BACHARELADO)
16999999       Joana Ribeiro Dourado                            1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="3")

        assert [r.lista for r in registros] == [LISTA_OUTRA_RELACAO]

    def test_titulo_de_convocacao_quebrado_em_duas_linhas_continua_convocacao(self):
        # Os títulos são longos e o modo `layout` preserva a quebra de linha do PDF. Se a
        # classificação dependesse de achar "convocação" na linha do cabeçalho, um título
        # cuja palavra caísse na segunda linha apagaria os convocados do Edital inteiro.
        texto = """
1 DO REGISTRO ACADÊMICO ON-LINE DOS CANDIDATOS SELECIONADOS EM PRIMEIRA
CHAMADA PARA CONVOCAÇÃO
1.1.1 CAMPUS DARCY RIBEIRO – DIURNO
1.1.1.1 ADMINISTRAÇÃO (BACHARELADO)
16124894       Pedro de Vilhena Moraes Silva                    1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        assert [r.lista for r in registros] == [LISTA_CONVOCACAO]

    def test_registro_antes_de_qualquer_secao_reconhecida_conta_como_convocacao(self):
        # Se o cabeçalho de seção não for reconhecido (layout inesperado), o parser volta
        # ao comportamento anterior — marcar tudo como convocação — em vez de esvaziar a
        # lista de convocados do Edital inteiro.
        texto = """
1.1.1 CAMPUS DARCY RIBEIRO – DIURNO
1.1.1.1 ADMINISTRAÇÃO (BACHARELADO)
16124894       Pedro de Vilhena Moraes Silva                    1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="3")

        assert [r.lista for r in registros] == [LISTA_CONVOCACAO]


class TestRotuloDeCurso:
    """Ticket 10: o rótulo de curso é o nome do curso, não a posição dele no sumário
    daquele Edital."""

    def test_rotulo_de_curso_nao_carrega_a_numeracao_do_sumario(self):
        registros = parse_convocacao(
            _ReaderFalso(_TEXTO_DUAS_SECOES), _CONTEXTO, semestre="1", chamada="3",
        )

        assert {r.curso for r in registros} == {"ADMINISTRAÇÃO (BACHARELADO)"}

    def test_o_mesmo_curso_em_secoes_diferentes_recebe_o_mesmo_rotulo(self):
        # No Edital real o mesmo curso aparece como "1.1.1.1 ADMINISTRAÇÃO" na seção 1 e
        # "2.1.1.1 ADMINISTRAÇÃO" na seção 2; e entre Editais do mesmo triênio a numeração
        # muda com a lista de cursos daquela chamada (CIÊNCIAS BIOLÓGICAS é 1.1.1.10 num
        # Edital e 1.1.1.11 no seguinte). Agrupar por rótulo numerado partiria o mesmo
        # curso em grupos diferentes, e a "maior chamada" sairia errada.
        registros = parse_convocacao(
            _ReaderFalso(_TEXTO_DUAS_SECOES), _CONTEXTO, semestre="1", chamada="3",
        )

        por_lista = {r.lista: r.curso for r in registros}
        assert por_lista[LISTA_CONVOCACAO] == por_lista[LISTA_OUTRA_RELACAO]

    def test_curso_sem_numeracao_continua_intacto(self):
        # Os Editais de 2021/2023 em diante não numeram os cabeçalhos de curso — o
        # rótulo deles não pode ser alterado por essa limpeza.
        texto = """
1.1.1 CAMPUS DARCY RIBEIRO – DIURNO
ADMINISTRAÇÃO (BACHARELADO)
21180305       Alberto Monteiro Torres                          9
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        assert [r.curso for r in registros] == ["ADMINISTRAÇÃO (BACHARELADO)"]

    def test_cabecalho_de_curso_com_travessao_avanca_o_estado(self):
        # Achado em produção (Ed_70, 2023/2025): nomes de curso compostos usam travessão
        # "–" (U+2013), não hífen "-". _CURSO_RE não incluía "–" na classe de caracteres,
        # então a linha inteira não casava _CURSO_RE.match, o estado "curso" não avançava,
        # e Gabriel Soares de Melo (ENGENHARIAS – AEROESPACIAL/...) saiu gravado como
        # TERAPIA OCUPACIONAL — o curso da seção anterior, herdado por engano.
        texto = """
1.1.3 CAMPUS UNB CEILÂNDIA (FCTS) – DIURNO
TERAPIA OCUPACIONAL (BACHARELADO)
23105303       Raissa Vittoria Magalhaes Ribeiro                 1
1.1.4 CAMPUS UNB GAMA (FCTE) – DIURNO
ENGENHARIAS – AEROESPACIAL / AUTOMOTIVA / ELETRÔNICA / ENERGIA / SOFTWARE
23112791       Gabriel Soares de Melo                            1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        por_inscricao = {r.inscricao: r.curso for r in registros}
        assert por_inscricao["23105303"] == "TERAPIA OCUPACIONAL (BACHARELADO)"
        assert por_inscricao["23112791"] == (
            "ENGENHARIAS – AEROESPACIAL / AUTOMOTIVA / ELETRÔNICA / ENERGIA / SOFTWARE"
        )

    def test_corrida_de_espacos_do_modo_layout_e_colapsada(self):
        # Achado em produção (Ed_35, 2016/2018, pág. 1): o modo `layout` do pypdf às vezes
        # justifica uma linha esticando espaço mesmo dentro de uma palavra —
        # "ADMINISTRAÇÃO (BACHARELAD              O)", 14 espaços entre "D" e "O". A
        # corrida de 2+ espaços colapsa para 1 (fixa também o caso comum, entre campus
        # "DARCY    RIBEIRO"); o espaço espúrio isolado dentro da palavra não é
        # reconstruído — isso exigiria reconhecer o nome do curso, não só normalizar
        # espaço, e fica documentado como limitação conhecida.
        texto = """
1.1.1 CAMPUS DARCY    RIBEIRO – DIURNO
ADMINISTRAÇÃO (BACHARELAD              O)
23115326       Julia Franco Soares                              1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        assert [r.campus for r in registros] == ["DARCY RIBEIRO"]
        assert [r.curso for r in registros] == ["ADMINISTRAÇÃO (BACHARELAD O)"]

    def test_corrida_de_espacos_no_nome_e_colapsada(self):
        # Mesmo artefato do modo `layout`, achado também no nome do candidato
        # ("Gabriel   Costa Rosa", "Isabella Cristina Melo Macedo da          Silva").
        # `_REGISTRO_RE` precisa do espaçamento bruto pra achar onde a coluna Sistema
        # termina, então o colapso só acontece depois do match, no `nome` já extraído —
        # nunca antes, na linha inteira.
        texto = """
1.1.1 CAMPUS DARCY RIBEIRO – DIURNO
ADMINISTRAÇÃO (BACHARELADO)
23115326       Julia Franco    Soares                            1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        assert [r.nome for r in registros] == ["Julia Franco Soares"]


class TestReparoDeNome:
    """Ticket 13: mesma lógica de `nome_repair.reparar_nome` usada em `resultado_final.py`,
    reaplicada aqui (decisão registrada no módulo, junto de `_ESPACOS_RE`) — antes desta
    linha, `parse_convocacao` só colapsava espaço duplicado, nunca reconstruía uma palavra
    quebrada por um único espaço espúrio."""

    def test_palavra_quebrada_por_espaco_e_reparada_e_sinalizada(self):
        texto = """
1.1.1 CAMPUS DARCY RIBEIRO – DIURNO
ADMINISTRAÇÃO (BACHARELADO)
23115326       Isabell a Costa Rosa                              1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        assert [r.nome for r in registros] == ["Isabella Costa Rosa"]
        assert [r.nome_reparado for r in registros] == [True]

    def test_particula_curta_legitima_nao_e_fundida(self):
        texto = """
1.1.1 CAMPUS DARCY RIBEIRO – DIURNO
ADMINISTRAÇÃO (BACHARELADO)
23115326       Maria de Souza                                    1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        assert [r.nome for r in registros] == ["Maria de Souza"]
        assert [r.nome_reparado for r in registros] == [False]

    def test_nome_sem_corrupcao_nao_e_sinalizado(self):
        texto = """
1.1.1 CAMPUS DARCY RIBEIRO – DIURNO
ADMINISTRAÇÃO (BACHARELADO)
23115326       Julia Franco Soares                                1
"""
        registros = parse_convocacao(_ReaderFalso(texto), _CONTEXTO, semestre="1", chamada="1")

        assert [r.nome_reparado for r in registros] == [False]


class TestListaNaFixtureReal:
    def test_a_fixture_de_uma_secao_so_sai_inteira_como_convocacao(self):
        _pular_se_fixture_ausente(FIXTURE_CONVOCACAO)

        resultado = extrair_edital_convocacao(FIXTURE_CONVOCACAO)

        assert {r.lista for r in resultado.registros} == {LISTA_CONVOCACAO}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
