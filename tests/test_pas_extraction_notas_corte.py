"""Ticket 10 — Notas de Corte por curso e por Sistema de Concorrência.

Constrói `RegistroResultadoFinal`/`RegistroConvocacao` sintéticos, como
`test_pas_extraction_reconciliacao.py` (ticket 08) faz para a 6ª camada de validação: o que
está sob teste é a derivação — "menor Argumento Final entre os convocados daquele sistema na
maior chamada que teve convocado nele" — que não depende de PDF real nenhum.

Cada teste nomeia a regra que exercita, não o método que chama: as três decisões que o dono
do produto tomou em 2026-07-24 (maior chamada primeiro, menor nota entre os daquela chamada,
queda para a chamada anterior quando o sistema não teve convocado na última) são regras de
negócio, e é como regra de negócio que elas devem quebrar quando alguém as mudar.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pas_extraction.convocacao import (  # type: ignore
    LISTA_CONVOCACAO,
    LISTA_OUTRA_RELACAO,
    RegistroConvocacao,
)
from pas_extraction.cotas import CotaDeclarada  # type: ignore
from pas_extraction.models import (  # type: ignore
    ChecksumArgumentoFinal,
    Proveniencia,
    RegistroResultadoFinal,
    ValidacaoRegistro,
)
from pas_extraction.notas_corte import (  # type: ignore
    CSV_COLUMNS_NOTAS_CORTE,
    derivar_notas_corte,
    escrever_csv_notas_corte,
)

TRIENIO = "2021/2023"


def _proveniencia(arquivo: str, trienio: str = TRIENIO, pagina: int = 1) -> Proveniencia:
    return Proveniencia(arquivo_origem=arquivo, edital="1", trienio=trienio, pagina=pagina)


def _rf(
    inscricao: str,
    argumento_final: float,
    *,
    trienio: str = TRIENIO,
    nome: str = "Fulano de Tal",
    arquivo: str = "Ed_rf.pdf",
    pagina: int = 1,
    checksum: ChecksumArgumentoFinal = None,
) -> RegistroResultadoFinal:
    return RegistroResultadoFinal(
        campus="Darcy Ribeiro", curso="MEDICINA", turno="DIURNO",
        inscricao=inscricao, nome=nome,
        eb_p1_e1=10.0, eb_p2_e1=10.0, red_e1=5.0,
        eb_p1_e2=10.0, eb_p2_e2=10.0, red_e2=5.0,
        eb_p1_e3=10.0, eb_p2_e3=10.0, red_e3=5.0,
        argumento_final=argumento_final,
        classificacoes={1: 1},
        cota_declarada=CotaDeclarada(
            sistema_negros=False, escola_publica=False, renda_baixa=False,
            ppi=False, pcd=False, perfil="Universal", padrao_suspeito=False,
        ),
        proveniencia=_proveniencia(arquivo, trienio=trienio, pagina=pagina),
        validacao=ValidacaoRegistro(
            campos_formato_invalido=(), buracos_classificacao={}, fora_de_ordem_alfabetica=False,
        ),
        checksum=checksum,
    )


def _conv(
    inscricao: str,
    sistema: int,
    chamada: str,
    *,
    curso: str = "MEDICINA",
    campus: str = "Darcy Ribeiro",
    turno: str = "DIURNO",
    trienio: str = TRIENIO,
    semestre: str = "1",
    nome: str = "Fulano de Tal",
    arquivo: str = "Ed_conv.pdf",
    pagina: int = 1,
    lista: str = LISTA_CONVOCACAO,
) -> RegistroConvocacao:
    return RegistroConvocacao(
        campus=campus, curso=curso, turno=turno,
        inscricao=inscricao, nome=nome, sistema=sistema,
        semestre=semestre, chamada=chamada,
        proveniencia=_proveniencia(arquivo, trienio=trienio, pagina=pagina),
        lista=lista,
    )


def _checksum(delta: float) -> ChecksumArgumentoFinal:
    return ChecksumArgumentoFinal(
        argumento_final_recalculado=0.0, delta=delta, linguas_por_etapa=("inglesa",) * 3,
    )


class TestUmCortePorCursoEPorSistema:
    def test_dois_sistemas_no_mesmo_curso_geram_dois_cortes(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 500.0)]
        conv = [_conv("21180001", sistema=1, chamada="1"), _conv("21180002", sistema=2, chamada="1")]

        notas = derivar_notas_corte(rf, conv).notas

        assert [(n.sistema, n.argumento_final) for n in notas] == [(1, 600.0), (2, 500.0)]

    def test_o_nome_do_sistema_acompanha_o_numero(self):
        rf = [_rf("21180001", 600.0)]
        conv = [_conv("21180001", sistema=2, chamada="1")]

        assert derivar_notas_corte(rf, conv).notas[0].sistema_nome == "Cota para Negros"

    def test_cursos_diferentes_nao_se_misturam(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 400.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="1", curso="MEDICINA"),
            _conv("21180002", sistema=1, chamada="1", curso="DIREITO"),
        ]

        notas = derivar_notas_corte(rf, conv).notas

        assert {(n.curso, n.argumento_final) for n in notas} == {
            ("MEDICINA", 600.0), ("DIREITO", 400.0),
        }

    def test_mesmo_curso_em_campus_diferentes_nao_se_mistura(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 400.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="1", campus="Darcy Ribeiro"),
            _conv("21180002", sistema=1, chamada="1", campus="UNB CEILÂNDIA (FCE)"),
        ]

        notas = derivar_notas_corte(rf, conv).notas

        assert len(notas) == 2
        assert {n.argumento_final for n in notas} == {600.0, 400.0}

    def test_mesmo_curso_em_turnos_diferentes_nao_se_mistura(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 400.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="1", turno="DIURNO"),
            _conv("21180002", sistema=1, chamada="1", turno="NOTURNO"),
        ]

        notas = derivar_notas_corte(rf, conv).notas

        assert {(n.turno, n.argumento_final) for n in notas} == {
            ("DIURNO", 600.0), ("NOTURNO", 400.0),
        }

    def test_trienios_diferentes_nao_se_misturam(self):
        rf = [
            _rf("21180001", 600.0, trienio="2021/2023"),
            _rf("18180001", 400.0, trienio="2018/2020"),
        ]
        conv = [
            _conv("21180001", sistema=1, chamada="1", trienio="2021/2023"),
            _conv("18180001", sistema=1, chamada="1", trienio="2018/2020"),
        ]

        notas = derivar_notas_corte(rf, conv).notas

        assert {(n.trienio, n.argumento_final) for n in notas} == {
            ("2021/2023", 600.0), ("2018/2020", 400.0),
        }

    def test_semestres_diferentes_nao_se_misturam(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 400.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="1", semestre="1"),
            _conv("21180002", sistema=1, chamada="1", semestre="2"),
        ]

        notas = derivar_notas_corte(rf, conv).notas

        assert {(n.semestre, n.argumento_final) for n in notas} == {
            ("1", 600.0), ("2", 400.0),
        }

    def test_a_mesma_inscricao_em_trienios_diferentes_nao_troca_de_argumento(self):
        # Inscrição repetida entre triênios: a busca do Argumento Final é por
        # (triênio, inscrição), não só por inscrição.
        rf = [
            _rf("11110000", 600.0, trienio="2021/2023"),
            _rf("11110000", 400.0, trienio="2018/2020"),
        ]
        conv = [_conv("11110000", sistema=1, chamada="1", trienio="2018/2020")]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert (nota.trienio, nota.argumento_final) == ("2018/2020", 400.0)


class TestMaiorChamadaPrimeiro:
    def test_a_chamada_do_corte_e_a_maior_com_convocado_naquele_sistema(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 550.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="1"),
            _conv("21180002", sistema=1, chamada="3"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert (nota.chamada, nota.argumento_final) == ("3", 550.0)

    def test_convocado_de_chamada_anterior_com_nota_menor_nao_define_o_corte(self):
        # O menor Argumento Final do curso inteiro é 300, mas ele foi convocado na 1ª
        # chamada — o corte é o da 3ª, a maior que teve convocado.
        rf = [_rf("21180001", 300.0), _rf("21180002", 550.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="1"),
            _conv("21180002", sistema=1, chamada="3"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.argumento_final == 550.0

    def test_chamada_e_ordenada_numericamente_e_nao_como_texto(self):
        # "10" > "9" numericamente, mas "10" < "9" como string — o bug que este teste
        # existe para impedir.
        rf = [_rf("21180001", 600.0), _rf("21180002", 550.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="9"),
            _conv("21180002", sistema=1, chamada="10"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert (nota.chamada, nota.argumento_final) == ("10", 550.0)

    def test_sistema_sem_convocado_na_ultima_chamada_cai_para_a_anterior(self):
        # Critério de aceite explícito do ticket: a maior chamada é por sistema, não por
        # curso. O sistema 2 não aparece na 3ª chamada, então o corte dele é o da 2ª.
        rf = [_rf("21180001", 600.0), _rf("21180002", 500.0), _rf("21180003", 450.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="3"),
            _conv("21180002", sistema=2, chamada="2"),
            _conv("21180003", sistema=2, chamada="1"),
        ]

        por_sistema = {n.sistema: n for n in derivar_notas_corte(rf, conv).notas}

        assert (por_sistema[1].chamada, por_sistema[1].argumento_final) == ("3", 600.0)
        assert (por_sistema[2].chamada, por_sistema[2].argumento_final) == ("2", 500.0)


class TestMenorArgumentoFinalEntreOsDaMaiorChamada:
    def test_com_varios_convocados_do_mesmo_sistema_na_maior_chamada_o_corte_e_o_menor(self):
        # Critério de aceite explícito: não a média (566,67), não o maior (620).
        rf = [_rf("21180001", 620.0), _rf("21180002", 480.0), _rf("21180003", 600.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="2"),
            _conv("21180002", sistema=1, chamada="2"),
            _conv("21180003", sistema=1, chamada="2"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.argumento_final == 480.0
        assert nota.convocados_na_chamada == 3

    def test_o_aluno_que_define_o_corte_e_identificado(self):
        rf = [
            _rf("21180001", 620.0, nome="Alberto Monteiro"),
            _rf("21180002", 480.0, nome="Beatriz Nogueira"),
        ]
        conv = [
            _conv("21180001", sistema=1, chamada="2", nome="Alberto Monteiro"),
            _conv("21180002", sistema=1, chamada="2", nome="Beatriz Nogueira"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert (nota.inscricao, nota.nome) == ("21180002", "Beatriz Nogueira")

    def test_empate_no_menor_argumento_desempata_pela_inscricao(self):
        # Dois Alunos com exatamente o mesmo Argumento Final na maior chamada: o corte é o
        # mesmo número de qualquer jeito; quem é registrado como definidor é a menor
        # inscrição, para que duas rodadas devolvam a mesma linha.
        rf = [_rf("21180009", 480.0), _rf("21180002", 480.0)]
        conv = [
            _conv("21180009", sistema=1, chamada="1"),
            _conv("21180002", sistema=1, chamada="1"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.argumento_final == 480.0
        assert nota.inscricao == "21180002"


class TestConvocadoSemArgumentoFinal:
    def test_convocado_ausente_do_resultado_final_nao_derruba_o_corte_mas_marca_parcial(self):
        # O Resultado Final do triênio pode não estar no corpus. O corte sai do que dá para
        # calcular, marcado como parcial — não some, e não finge estar completo.
        rf = [_rf("21180001", 600.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="1"),
            _conv("21180002", sistema=1, chamada="1"),
        ]

        derivacao = derivar_notas_corte(rf, conv)
        nota = derivacao.notas[0]

        assert nota.argumento_final == 600.0
        assert (nota.convocados_na_chamada, nota.convocados_com_argumento) == (2, 1)
        assert nota.parcial is True

    def test_corte_completo_nao_e_parcial(self):
        rf = [_rf("21180001", 600.0)]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        assert derivar_notas_corte(rf, conv).notas[0].parcial is False

    def test_grupo_inteiro_sem_argumento_final_nao_vira_corte_mas_e_reportado(self):
        conv = [_conv("21180002", sistema=1, chamada="1")]

        derivacao = derivar_notas_corte([], conv)

        assert derivacao.notas == ()
        assert derivacao.convocados_sem_argumento_final == (("2021/2023", 1),)

    def test_o_grupo_sem_corte_e_identificado_e_nao_so_contado(self):
        # Sem a chave, a ausência da linha no CSV é indistinguível de "este curso/sistema
        # não existe".
        conv = [_conv("21180002", sistema=2, chamada="1", curso="MEDICINA")]

        derivacao = derivar_notas_corte([], conv)

        assert len(derivacao.grupos_sem_argumento_final) == 1
        chave = derivacao.grupos_sem_argumento_final[0]
        assert (chave.curso, chave.sistema) == ("MEDICINA", 2)

    def test_grupo_com_corte_derivado_nao_aparece_como_sem_argumento(self):
        rf = [_rf("21180001", 600.0)]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        assert derivar_notas_corte(rf, conv).grupos_sem_argumento_final == ()

    def test_convocados_sem_argumento_sao_contados_por_trienio(self):
        conv = [
            _conv("21180002", sistema=1, chamada="1", trienio="2021/2023"),
            _conv("18180003", sistema=1, chamada="1", trienio="2018/2020"),
            _conv("18180004", sistema=2, chamada="1", trienio="2018/2020"),
        ]

        derivacao = derivar_notas_corte([], conv)

        assert derivacao.convocados_sem_argumento_final == (
            ("2018/2020", 2), ("2021/2023", 1),
        )


class TestOutrasListasDoEdital:
    """Um Edital de Convocação pode publicar duas listas de Alunos: a convocação (seção 1)
    e a relação de quem não compareceu / esteve ausente (seção 2, ver
    `convocacao.LISTA_OUTRA_RELACAO`). Só a primeira é convocação naquela chamada."""

    def test_quem_nao_compareceu_nao_define_o_corte(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 300.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="3"),
            # Mesma chamada, mesmo curso, nota muito menor — mas está na relação de quem
            # não compareceu, e não foi convocado nesta chamada.
            _conv("21180002", sistema=1, chamada="3", lista=LISTA_OUTRA_RELACAO),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.argumento_final == 600.0
        assert nota.convocados_na_chamada == 1

    def test_outra_lista_nao_cria_chamada_que_nao_existe(self):
        # A relação de ausentes de um Edital de 3ª chamada lista gente convocada antes; se
        # ela contasse, a "maior chamada" do sistema viraria a 3ª sem que ninguém daquele
        # sistema tivesse sido chamado nela.
        rf = [_rf("21180001", 600.0), _rf("21180002", 300.0)]
        conv = [
            _conv("21180001", sistema=2, chamada="1"),
            _conv("21180002", sistema=2, chamada="3", lista=LISTA_OUTRA_RELACAO),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert (nota.chamada, nota.argumento_final) == ("1", 600.0)

    def test_registros_de_outras_listas_sao_contados_e_nao_somem(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 300.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="3"),
            _conv("21180002", sistema=1, chamada="3", lista=LISTA_OUTRA_RELACAO),
        ]

        derivacao = derivar_notas_corte(rf, conv)

        assert derivacao.registros_de_outras_listas == 1

    def test_grupo_so_com_outra_lista_nao_vira_corte(self):
        rf = [_rf("21180002", 300.0)]
        conv = [_conv("21180002", sistema=1, chamada="3", lista=LISTA_OUTRA_RELACAO)]

        derivacao = derivar_notas_corte(rf, conv)

        assert derivacao.notas == ()
        assert derivacao.registros_de_outras_listas == 1


class TestChamadaDesconhecida:
    def test_convocado_sem_chamada_conhecida_nao_entra_no_maximo(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 400.0)]
        conv = [
            _conv("21180001", sistema=1, chamada="2"),
            _conv("21180002", sistema=1, chamada="desconhecido"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert (nota.chamada, nota.argumento_final) == ("2", 600.0)

    def test_grupo_so_com_chamada_desconhecida_nao_vira_corte_mas_e_reportado(self):
        rf = [_rf("21180002", 400.0)]
        conv = [_conv("21180002", sistema=1, chamada="desconhecido")]

        derivacao = derivar_notas_corte(rf, conv)

        assert derivacao.notas == ()
        assert len(derivacao.grupos_sem_chamada_conhecida) == 1
        assert derivacao.grupos_sem_chamada_conhecida[0].sistema == 1

    def test_semestre_desconhecido_continua_sendo_grupo_valido(self):
        # Ao contrário da chamada, semestre "desconhecido" é caso real (os 4 Editais do
        # triênio 2018/2020 não mencionam semestre) e não impede a derivação: é só mais um
        # valor da chave, constante para o triênio inteiro.
        rf = [_rf("18180001", 400.0, trienio="2018/2020")]
        conv = [
            _conv("18180001", sistema=1, chamada="1", trienio="2018/2020",
                  semestre="desconhecido"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert (nota.semestre, nota.argumento_final) == ("desconhecido", 400.0)


class TestProvenienciaEQualidade:
    def test_cada_corte_carrega_a_proveniencia_das_duas_familias(self):
        rf = [_rf("21180001", 600.0, arquivo="Ed_16_resultado.pdf", pagina=7)]
        conv = [_conv("21180001", sistema=1, chamada="1", arquivo="Ed_28_convoca.pdf", pagina=3)]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.proveniencia_convocacao.arquivo_origem == "Ed_28_convoca.pdf"
        assert nota.proveniencia_convocacao.pagina == 3
        assert nota.proveniencia_resultado_final.arquivo_origem == "Ed_16_resultado.pdf"
        assert nota.proveniencia_resultado_final.pagina == 7

    def test_o_checksum_do_registro_que_define_o_corte_acompanha_a_linha(self):
        rf = [_rf("21180001", 600.0, checksum=_checksum(0.001))]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        assert derivar_notas_corte(rf, conv).notas[0].checksum_fecha is True

    def test_checksum_reprovado_nao_esconde_o_corte_mas_fica_visivel(self):
        rf = [_rf("21180001", 600.0, checksum=_checksum(3.2))]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.argumento_final == 600.0
        assert nota.checksum_fecha is False

    def test_sem_checksum_conferido_o_campo_fica_none_e_nao_false(self):
        rf = [_rf("21180001", 600.0, checksum=None)]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        assert derivar_notas_corte(rf, conv).notas[0].checksum_fecha is None


class TestChecksumSuspeitoNaEscolhaDoCorte:
    """Ticket 14, follow-up ao achado do outlier de Nota de Corte: MEDICINA, Darcy Ribeiro,
    2020/2022, `argumento_final=199162.872` virava a Nota de Corte por ser o único convocado
    da chamada, mesmo com `checksum.fecha=False` (delta de ~199 mil). Aqui há alternativa
    confiável na mesma chamada — o candidato suspeito deixa de vencer só por ter o menor
    número, mas continua saindo quando é o único (`TestProvenienciaEQualidade`, acima).
    """

    def test_candidato_com_checksum_reprovado_perde_para_alternativa_confiavel(self):
        # 30.0 seria o menor Argumento Final e venceria por `min()` puro — mas seu checksum
        # não fecha (delta 199162.872, a mesma ordem de grandeza do achado real), então a
        # escolha cai para o único outro candidato confiável da chamada.
        rf = [
            _rf("21180001", 30.0, checksum=_checksum(199162.872)),
            _rf("21180002", 600.0, checksum=_checksum(0.001)),
        ]
        conv = [
            _conv("21180001", sistema=1, chamada="1"),
            _conv("21180002", sistema=1, chamada="1"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.inscricao == "21180002"
        assert nota.argumento_final == 600.0
        assert nota.checksum_fecha is True

    def test_substituicao_e_contada_por_trienio(self):
        rf = [
            _rf("21180001", 30.0, checksum=_checksum(199162.872)),
            _rf("21180002", 600.0, checksum=_checksum(0.001)),
        ]
        conv = [
            _conv("21180001", sistema=1, chamada="1"),
            _conv("21180002", sistema=1, chamada="1"),
        ]

        derivacao = derivar_notas_corte(rf, conv)

        assert derivacao.convocados_com_checksum_suspeito_substituidos == ((TRIENIO, 1),)

    def test_sem_alternativa_confiavel_nada_e_contado_como_substituido(self):
        # Mesmo cenário de `TestProvenienciaEQualidade.test_checksum_reprovado_...`: o único
        # convocado da chamada continua definindo o corte (linha não é descartada), e por não
        # ter havido substituição real, o contador fica vazio.
        rf = [_rf("21180001", 600.0, checksum=_checksum(3.2))]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        derivacao = derivar_notas_corte(rf, conv)

        assert derivacao.notas[0].checksum_fecha is False
        assert derivacao.convocados_com_checksum_suspeito_substituidos == ()

    def test_candidato_sem_checksum_conferido_nao_e_tratado_como_suspeito(self):
        # `checksum=None` ("não foi possível conferir") não é o mesmo que `fecha=False`
        # ("conferido e reprovado") — não deve perder para um confiável só por não ter sido
        # verificado.
        rf = [
            _rf("21180001", 30.0, checksum=None),
            _rf("21180002", 600.0, checksum=_checksum(0.001)),
        ]
        conv = [
            _conv("21180001", sistema=1, chamada="1"),
            _conv("21180002", sistema=1, chamada="1"),
        ]

        nota = derivar_notas_corte(rf, conv).notas[0]

        assert nota.inscricao == "21180001"
        assert nota.argumento_final == 30.0


class TestInscricaoAmbiguaNoResultadoFinal:
    """A busca do Argumento Final é por (triênio, inscrição). Se a mesma chave aparecer
    duas vezes com notas diferentes, uma delas define o corte e a outra é ignorada — o que
    precisa aparecer, porque é sinal de regressão de parser e não de dado do Edital."""

    def test_duplicata_com_argumento_diferente_e_contada(self):
        rf = [_rf("21180001", 600.0), _rf("21180001", 450.0)]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        assert derivar_notas_corte(rf, conv).inscricoes_com_argumento_ambiguo == 1

    def test_duplicata_com_o_mesmo_argumento_nao_e_ambiguidade(self):
        # Registro repetido com a mesma nota não muda corte nenhum — contá-lo aqui só
        # geraria ruído sobre um problema que é de outra camada (ticket 02).
        rf = [_rf("21180001", 600.0), _rf("21180001", 600.0)]
        conv = [_conv("21180001", sistema=1, chamada="1")]

        assert derivar_notas_corte(rf, conv).inscricoes_com_argumento_ambiguo == 0

    def test_a_mesma_inscricao_em_trienios_diferentes_nao_e_ambiguidade(self):
        rf = [
            _rf("11110000", 600.0, trienio="2021/2023"),
            _rf("11110000", 400.0, trienio="2018/2020"),
        ]
        conv = [_conv("11110000", sistema=1, chamada="1", trienio="2018/2020")]

        assert derivar_notas_corte(rf, conv).inscricoes_com_argumento_ambiguo == 0


class TestDeterminismo:
    def test_mesma_entrada_mesma_saida(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 500.0)]
        conv = [_conv("21180001", sistema=1, chamada="1"), _conv("21180002", sistema=2, chamada="1")]

        assert derivar_notas_corte(rf, conv) == derivar_notas_corte(rf, conv)

    def test_ordem_da_entrada_nao_muda_a_ordem_da_saida(self):
        rf = [_rf("21180001", 600.0), _rf("21180002", 500.0)]
        conv = [_conv("21180001", sistema=2, chamada="1"), _conv("21180002", sistema=1, chamada="1")]

        primeira = derivar_notas_corte(rf, conv).notas
        segunda = derivar_notas_corte(list(reversed(rf)), list(reversed(conv))).notas

        assert primeira == segunda
        assert [n.sistema for n in primeira] == [1, 2]

    def test_nao_muta_os_registros_de_entrada(self):
        registro_rf = _rf("21180001", 600.0)
        registro_conv = _conv("21180001", sistema=1, chamada="1")

        derivar_notas_corte([registro_rf], [registro_conv])

        assert registro_rf.argumento_final == 600.0
        assert registro_conv.chamada == "1"


class TestCsv:
    def test_escreve_cabecalho_e_uma_linha_por_corte(self, tmp_path):
        rf = [_rf("21180001", 600.0), _rf("21180002", 500.0)]
        conv = [_conv("21180001", sistema=1, chamada="1"), _conv("21180002", sistema=2, chamada="1")]
        destino = tmp_path / "notas_corte.csv"

        total = escrever_csv_notas_corte(derivar_notas_corte(rf, conv), destino)

        linhas = list(csv.reader(destino.open(encoding="utf-8")))
        assert total == 2
        assert tuple(linhas[0]) == CSV_COLUMNS_NOTAS_CORTE
        assert len(linhas) == 3

    def test_a_linha_do_csv_carrega_corte_chamada_sistema_e_proveniencia(self, tmp_path):
        rf = [_rf("21180001", 600.0, arquivo="Ed_16_resultado.pdf")]
        conv = [_conv("21180001", sistema=2, chamada="4", arquivo="Ed_28_convoca.pdf")]
        destino = tmp_path / "notas_corte.csv"

        escrever_csv_notas_corte(derivar_notas_corte(rf, conv), destino)

        linha = next(csv.DictReader(destino.open(encoding="utf-8")))
        assert linha["nota_corte"] == "600.0"
        assert linha["chamada"] == "4"
        assert linha["sistema"] == "2"
        assert linha["sistema_nome"] == "Cota para Negros"
        assert linha["arquivo_origem_convocacao"] == "Ed_28_convoca.pdf"
        assert linha["arquivo_origem_resultado_final"] == "Ed_16_resultado.pdf"

    def test_cria_o_diretorio_de_destino_se_nao_existir(self, tmp_path):
        rf = [_rf("21180001", 600.0)]
        conv = [_conv("21180001", sistema=1, chamada="1")]
        destino = tmp_path / "saida" / "notas_corte.csv"

        escrever_csv_notas_corte(derivar_notas_corte(rf, conv), destino)

        assert destino.exists()
