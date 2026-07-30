import csv
import re
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore
from pypdf import PdfReader  # type: ignore

from pas_extraction import ResultadoExtracao, extrair_edital  # type: ignore
from pas_extraction.models import (  # type: ignore
    FamiliaDesconhecidaError,
    FamiliaEdital,
    Proveniencia,
    RegistroResultadoFinal,
    ValidacaoRegistro,
)
from pas_extraction.cotas import CotaDeclarada, deduzir_cota_declarada  # type: ignore
from pas_extraction.csv_writer import CSV_COLUMNS, escrever_csv  # type: ignore
from pas_extraction.schema import canonizar, classificar_familia  # type: ignore
from pas_extraction.validacao import validar_sequencia_e_ordem  # type: ignore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_RESULTADO_FINAL = FIXTURES_DIR / "resultado_final_22_campos.pdf"

# Contagem observada ao gerar a fixture com `python -m pas_extraction.cli fixture
# 'data/pdfs/resultado-final-pas3/Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf' 1 6 <destino>`.
# Páginas 1-6 (acima do "3 a 5" sugerido) de propósito: é o intervalo mais curto que
# inclui uma troca real de curso (ADMINISTRAÇÃO -> AGRONOMIA, na página 6), que é o
# comportamento que este ticket pede para os cabeçalhos intercalados no fluxo.
#
# Ticket 02: 19 dos candidatos a registro nessas páginas têm um campo numérico partido
# por espaço (ex.: "1 7.539") ou sinal negativo separado (ex.: "- 21.683") — antes do
# ticket 02 eram descartados inteiros (170 sobreviviam); agora `_tentar_float` repara o
# valor e `campos_formato_invalido` sinaliza o campo, então os 19 passam a aparecer no
# resultado (189 no total). Só 1 candidato continua descartado de verdade: o último
# registro da página 6, cortado pelo limite da fixture (span incompleto, não corrupção).
CONTAGEM_ESPERADA = 189
CURSO_1 = "ADMINISTRAÇÃO (BACHARELADO)"
CURSO_2 = "AGRONOMIA (BACHARELADO)"

# Segunda fixture: página 1 (schema) + página 186 de Ed_38 — não contíguas, geradas com
# `fatiar_paginas` (ver fixtures.py) em vez de `fatiar_fixture`. A página 186 contém DOIS
# cursos pequenos por inteiro (ARQUIVOLOGIA, 16 Alunos; CIÊNCIAS AMBIENTAIS, 9 Alunos) —
# ao contrário da fixture de 6 páginas, onde todo curso está truncado (só uma fração dele
# cabe na fatia), aqui a classificação 1..N de cada Sistema fecha sem buraco nenhum. É a
# única forma de ter um "curso completo" pequeno o bastante para fixture: um curso grande
# como ADMINISTRAÇÃO nunca caberia inteiro numa fatia pequena.
FIXTURE_CURSO_COMPLETO = FIXTURES_DIR / "resultado_final_curso_completo.pdf"
CURSO_PEQUENO_COMPLETO = "CIÊNCIAS AMBIENTAIS (BACHARELADO)"

# Ticket 06: dois Alunos reais de ARQUIVOLOGIA são classificados nos Sistemas 1, 3, 5, 7 e
# 9 — o fecho para baixo de {Renda, PPI}, o padrão de cota mais rico que existe nas
# fixtures deste projeto (nenhuma delas tem Aluno PcD; ver relatório do ticket 06 para a
# contagem no corpus inteiro). Os testes selecionam por curso + tamanho do padrão, nunca
# por nome ou inscrição — nenhum dos dois precisa ser um literal no código.
CURSO_COM_PADRAO_RICO = "ARQUIVOLOGIA (BACHARELADO)"

# Quarta fixture: Ed_31 (2016/2018), página 1 (schema) + páginas 82-83, onde um registro
# real cai na virada de página. Ela existe para exercitar o único padrão de cota **não
# fecho** encontrado no corpus inteiro (ver relatório do ticket 06): a extração de texto
# emite o número da página no *início* do texto dela, então o último registro da página
# 82, cujo 22º campo só começa na 83, lê o número da página seguinte como se fosse a 10ª
# classificação. Gerada com:
#   python -c "
#   import sys; sys.path.insert(0, 'src')
#   from pas_extraction.fixtures import fatiar_paginas
#   fatiar_paginas(
#       'data/pdfs/resultado-final-pas3/Ed_31_2016-2018_PAS_3_Res_final_nao_eliminados.pdf',
#       [1, 82, 83],
#       'tests/fixtures/resultado_final_cota_suspeita.pdf',
#   )"
# O Aluno afetado é identificado pelo próprio sinal que o ticket 06 pede (o padrão que não
# é fecho), nunca por inscrição — é o único registro suspeito neste recorte.
FIXTURE_COTA_SUSPEITA = FIXTURES_DIR / "resultado_final_cota_suspeita.pdf"

# Terceira fixture: Ed_27 (2021/2023, tipo D + redação), a única família com duas seções
# de schema diferente no mesmo arquivo (ticket 05). Página 1 (schema) + páginas 99-101
# (não contíguas com a 1, geradas com `fatiar_paginas`): 99 é a cauda da seção 1
# (eliminados, registros de 4 campos), 100 tem o cabeçalho "2 DO RESULTADO FINAL DOS
# CANDIDATOS NÃO ELIMINADOS" que marca a transição + o primeiro cabeçalho de
# Campus/Curso/Turno da seção 2, 101 é só registros de 22 campos. Gerada com:
#   python -c "
#   import sys; sys.path.insert(0, 'src')
#   from pas_extraction.fixtures import fatiar_paginas
#   fatiar_paginas(
#       'data/pdfs/resultado-final-pas3/Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf',
#       [1, 99, 100, 101],
#       'tests/fixtures/resultado_final_duas_secoes.pdf',
#   )"
FIXTURE_DUAS_SECOES = FIXTURES_DIR / "resultado_final_duas_secoes.pdf"
# Contagem observada nas páginas 100-101 (registros de 22 campos, seção 2).
CONTAGEM_ESPERADA_DUAS_SECOES = 55

# Mesmo cabeçalho de transição que `resultado_final._SECAO_NAO_ELIMINADOS_RE` varre —
# duplicado aqui de propósito: a prova de que a seção 1 não vaza precisa ser independente
# do parser que está sendo testado, não reaproveitar a mesma regex de produção por atalho.
_SECAO_NAO_ELIMINADOS_RE = re.compile(
    r"\d\s*DO\s+RESULTADO\s+FINAL\s+DOS\s+CANDIDATOS\s+N[ÃA]O\s+ELIMINADOS", re.IGNORECASE,
)
# Registro de 4 campos (inscrição, nome, 2 notas) da seção 1 — formato de candidato
# eliminado, ver módulo docstring de `resultado_final.py`.
_REGISTRO_4_CAMPOS_RE = re.compile(
    r"(?<!\d)(\d{8}),\s*([^,\d][^,]*?),\s*-?\d+\.\d{3},\s*-?\d+\.\d{3}\s*/"
)


def _ultimo_registro_da_secao_1(fixture: Path) -> Tuple[str, str]:
    """(inscrição, nome) do último registro de 4 campos antes do cabeçalho de transição.

    Lido direto do texto bruto da fixture, em tempo de execução — nunca como literal no
    código, porque nome e inscrição são dado real de Aluno (ticket 05). A fixture é
    gitignored, então o valor real nunca entra no git; a força da asserção do teste que
    usa esta função não muda: ela continua provando que um registro específico e
    identificável da seção 1 não vazou para a saída.
    """
    reader = PdfReader(str(fixture))
    blob = " ".join(
        re.sub(r"\s+", " ", p.extract_text(extraction_mode="plain") or "") for p in reader.pages
    )
    fim_secao_1 = _SECAO_NAO_ELIMINADOS_RE.search(blob).start()
    matches = list(_REGISTRO_4_CAMPOS_RE.finditer(blob, 0, fim_secao_1))
    return matches[-1].group(1), matches[-1].group(2).strip()


def _pular_se_fixture_ausente(caminho: Path) -> None:
    if not caminho.exists():
        pytest.skip(
            f"Fixture {caminho.relative_to(Path(__file__).parent.parent)} não encontrada. "
            "Gere localmente (requer data/pdfs completo) com: "
            "python -m pas_extraction.cli fixture "
            "'data/pdfs/resultado-final-pas3/Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf' "
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


class TestValidacaoFormatoNumerico:
    """Ticket 02, verificação 1: `^-?\\d+\\.\\d{3}$` exato contra o texto bruto.

    Número partido por espaço não é mais descartado (era o comportamento do ticket 01) —
    `_tentar_float` repara o valor removendo o espaço interno, e o campo é sinalizado em
    `validacao.campos_formato_invalido` para quem for filtrar por confiança.
    """

    def test_numero_partido_no_meio_e_reparado_e_sinalizado(self):
        # Regressão do caso do protótipo "56.29 1 que deve virar 56.291": aqui o exemplo
        # real da fixture é "1 7.539" -> 17.539 (mesma classe de corrupção, espaço solto
        # dentro do número; ver spec.md e scripts/NOTES.md, achado 5c).
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)
        # Único registro com exatamente este campo sinalizado nesta fixture —
        # identificado pela própria condição sob teste, não por inscrição.
        registro = next(
            r for r in resultado.registros
            if r.validacao.campos_formato_invalido == ("eb_p2_e1",)
        )

        assert registro.eb_p2_e1 == 17.539
        assert registro.validacao.campos_formato_invalido == ("eb_p2_e1",)
        assert not registro.validacao.valido

    def test_sinal_negativo_separado_e_reparado_e_sinalizado(self):
        # Regressão do caso do protótipo "- 58.570 com o sinal negativo separado": aqui o
        # exemplo real da fixture é "- 21.683" -> -21.683, no campo argumento_final, e
        # "4.6 14" -> 4.614 no mesmo registro (dois campos partidos, dois tipos de corte).
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)
        # Único registro com exatamente estes dois campos sinalizados nesta fixture —
        # identificado pela própria condição sob teste, não por inscrição.
        registro = next(
            r for r in resultado.registros
            if set(r.validacao.campos_formato_invalido) == {"eb_p1_e2", "argumento_final"}
        )

        assert registro.argumento_final == -21.683
        assert registro.eb_p1_e2 == 4.614
        assert set(registro.validacao.campos_formato_invalido) == {"eb_p1_e2", "argumento_final"}

    def test_registro_sem_corrupcao_nao_e_sinalizado(self):
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)
        registro = next(r for r in resultado.registros if r.validacao.campos_formato_invalido == ())

        assert registro.validacao.campos_formato_invalido == ()

    def test_total_de_registros_sinalizados_por_formato(self):
        # 19 registros da fixture têm exatamente um campo numérico partido por espaço —
        # fixa a contagem para que uma mudança no reparo não esconda regressão silenciosa.
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        sinalizados = [r for r in resultado.registros if r.validacao.campos_formato_invalido]
        assert len(sinalizados) == 19


class TestCabecalhoDeCursoEngolido:
    """Ticket 02, regressão do caso do protótipo: cabeçalho de curso colado ao fim do
    último registro do curso anterior (achado (a) do NOTES.md, exemplo real "ENGENHARIA
    DE REDES DE COMUNICAÇÃO (BACHARELADO)"). A fixture reproduz a mesma classe de
    corrupção com a troca real ADMINISTRAÇÃO -> AGRONOMIA que ela contém: o cabeçalho do
    novo curso, no PDF de origem, vem colado sem separador ao fim do último registro do
    curso anterior. Regressão: nem o último registro do curso 1 nem o primeiro do curso 2
    devem carregar fragmento do cabeçalho dentro dos seus campos.
    """

    def test_registros_ao_redor_da_troca_de_curso_nao_carregam_o_cabecalho(self):
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)
        cursos = [r.curso for r in resultado.registros]
        indice_transicao = cursos.index(CURSO_2)
        ultimo_curso_1 = resultado.registros[indice_transicao - 1]
        primeiro_curso_2 = resultado.registros[indice_transicao]

        for r in (ultimo_curso_1, primeiro_curso_2):
            assert not any(ch.isdigit() for ch in r.nome)
            assert r.inscricao.isdigit() and len(r.inscricao) == 8


class TestCabecalhoDeCampusSemTurno:
    """Regressão: achado real em Ed_31 (2016/2018), "1.1.4 CAMPUS UnB GAMA (FGA)" — o
    único curso desse campus (Engenharias) nunca tem turno declarado no Edital. Antes,
    `_CAMPUS_RE` exigia o sufixo "- TURNO"; sem ele o cabeçalho inteiro falhava o match e
    nem campus nem turno avançavam, fazendo os 451 registros do curso herdarem por engano
    o campus da seção anterior (UnB Ceilândia). Testado direto contra `_processar_cabecalhos`
    porque é função pura sobre string — não precisa de PDF real nem de `_ReaderFalso`.

    A primeira correção (turno opcional + lookahead pra numeração do próximo cabeçalho de
    curso) tinha uma segunda falha, achada regenerando o corpus inteiro: no Ed_34
    (2023/2025, pág. 230), campus e curso vêm colados na mesma linha, sem numeração
    nenhuma separando os dois ("CAMPUS UNB GAMA (FGA) ENGENHARIAS – ..."). O lookahead não
    tinha onde ancorar o fim do nome do campus e engolia o curso inteiro junto. Fix
    definitivo: casar o nome do campus contra o conjunto fechado dos 4 campi do PAS/UnB
    em vez de tentar adivinhar o fim via lookahead.
    """

    def test_campus_sem_turno_avanca_o_estado(self):
        from pas_extraction.resultado_final import _processar_cabecalhos

        estado = {"campus": "UnB CEILÂNDIA (FCE)", "curso": "TERAPIA OCUPACIONAL", "turno": "DIURNO"}
        ruido = (
            "1.1.4 CAMPUS UnB GAMA (FGA) 1.1.4.1 ENGENHARIAS – AEROESPACIAL/AUTOMOTIVA "
            "(BACHARELADOS) "
        )

        _processar_cabecalhos(ruido, estado)

        assert estado["campus"] == "UnB GAMA (FGA)"
        assert estado["turno"] is None
        assert estado["curso"] == "ENGENHARIAS – AEROESPACIAL/AUTOMOTIVA (BACHARELADOS)"

    def test_campus_com_turno_continua_funcionando(self):
        from pas_extraction.resultado_final import _processar_cabecalhos

        estado = {"campus": None, "curso": None, "turno": None}
        ruido = "1.1.3 CAMPUS UnB CEILÂNDIA (FCE) – DIURNO 1.1.3.1 ENFERMAGEM (BACHARELADO) "

        _processar_cabecalhos(ruido, estado)

        assert estado["campus"] == "UnB CEILÂNDIA (FCE)"
        assert estado["turno"] == "DIURNO"
        assert estado["curso"] == "ENFERMAGEM (BACHARELADO)"

    def test_campus_e_curso_colados_na_mesma_linha_sem_numeracao(self):
        # Achado real Ed_34 (2023/2025) pág. 230: sem turno E sem numeração de curso
        # separando os dois — só o conjunto fechado de nomes de campus resolve.
        from pas_extraction.resultado_final import _processar_cabecalhos

        estado = {"campus": "DARCY RIBEIRO", "curso": "TERAPIA OCUPACIONAL", "turno": "DIURNO"}
        ruido = (
            "1.1.4 CAMPUS UNB GAMA (FGA) ENGENHARIAS – AEROESPACIAL / AUTOMOTIVA "
            "(BACHARELADOS)** "
        )

        _processar_cabecalhos(ruido, estado)

        assert estado["campus"] == "UNB GAMA (FGA)"
        assert estado["turno"] is None
        assert estado["curso"] == "ENGENHARIAS – AEROESPACIAL / AUTOMOTIVA (BACHARELADOS)"


class TestSequenciaDeClassificacao:
    """Ticket 02, verificação 2: classificação como sequência 1..N por curso e Sistema.

    `CURSO_PEQUENO_COMPLETO` é o único curso, entre as fixtures deste projeto, extraído
    por inteiro (as demais são recortes truncados de um curso maior — ver comentário da
    constante) — por isso é a única base em que "sem buraco" é uma afirmação real, não um
    artefato do corte da fixture.
    """

    def test_curso_completo_sem_corrupcao_nao_tem_buraco(self):
        _pular_se_fixture_ausente(FIXTURE_CURSO_COMPLETO)

        resultado = extrair_edital(FIXTURE_CURSO_COMPLETO)
        grupo = [r for r in resultado.registros if r.curso == CURSO_PEQUENO_COMPLETO]

        assert len(grupo) == 9
        assert all(r.validacao.valido for r in grupo)
        assert all(not r.validacao.buracos_classificacao for r in grupo)

    def test_registro_que_o_parser_perdeu_deixa_buraco_detectavel(self):
        # Regressão do caso do protótipo "registros colados, o segundo perde o número de
        # inscrição" (achado (b) do NOTES.md): quando isso acontece, a âncora do segundo
        # registro nunca é encontrada e ele simplesmente não existe na lista extraída —
        # não sobra nada nele mesmo para apontar o problema (spec.md, "Camadas de
        # validação"). Simula exatamente essa perda a partir de dados reais e completos:
        # remove um registro real da lista que `extrair_edital` já extraiu, e confere que
        # o buraco aparece na posição certa para os demais.
        _pular_se_fixture_ausente(FIXTURE_CURSO_COMPLETO)

        resultado = extrair_edital(FIXTURE_CURSO_COMPLETO)
        grupo = [r for r in resultado.registros if r.curso == CURSO_PEQUENO_COMPLETO]
        # Identificado pela própria posição sob teste, não por nome — único registro do
        # curso na posição 4 do Sistema Universal.
        removido = next(r for r in grupo if r.classificacoes[1] == 4)
        assert removido.classificacoes[1] == 4  # posição que vai faltar, Sistema Universal

        restante = [r for r in grupo if r is not removido]
        validar_sequencia_e_ordem(restante)

        assert len(restante) == 8
        for r in restante:
            assert r.validacao.buracos_classificacao == {1: (4,)}
            assert not r.validacao.valido

    def test_registro_de_posicao_maxima_perdido_e_um_ponto_cego_conhecido(self):
        # Limitação inerente à técnica (documentada em validacao.py): N é inferido como
        # o maior valor observado, então perder justo o registro de posição N encolhe o
        # "esperado" junto com ele — nenhum buraco aparece. Fixa esse comportamento como
        # conhecido (não como bug) para que não seja "descoberto" de novo mais tarde; é
        # exatamente o oposto do teste anterior (que perde uma posição do meio).
        _pular_se_fixture_ausente(FIXTURE_CURSO_COMPLETO)

        resultado = extrair_edital(FIXTURE_CURSO_COMPLETO)
        grupo = [r for r in resultado.registros if r.curso == CURSO_PEQUENO_COMPLETO]
        # Identificado pela própria posição sob teste, não por nome — único registro do
        # curso na posição máxima (9) do Sistema Universal.
        removido = next(r for r in grupo if r.classificacoes[1] == 9)
        assert removido.classificacoes[1] == 9  # a maior posição do Sistema Universal aqui

        restante = [r for r in grupo if r is not removido]
        validar_sequencia_e_ordem(restante)

        assert all(not r.validacao.buracos_classificacao for r in restante)


def _registro_minimo(inscricao: str, nome: str, sistema_1: int) -> RegistroResultadoFinal:
    """O menor `RegistroResultadoFinal` que serve para exercitar `validar_sequencia_e_ordem`
    isoladamente — só o Sistema Universal é preenchido, o resto é preenchimento inerte."""
    return RegistroResultadoFinal(
        campus="Darcy Ribeiro", curso="Curso Sintético", turno="Matutino",
        inscricao=inscricao, nome=nome,
        eb_p1_e1=0.0, eb_p2_e1=0.0, red_e1=0.0,
        eb_p1_e2=0.0, eb_p2_e2=0.0, red_e2=0.0,
        eb_p1_e3=0.0, eb_p2_e3=0.0, red_e3=0.0,
        argumento_final=0.0,
        classificacoes={1: sistema_1},
        cota_declarada=CotaDeclarada(
            sistema_negros=False, escola_publica=False, renda_baixa=False,
            ppi=False, pcd=False, perfil="Universal", padrao_suspeito=False,
        ),
        proveniencia=Proveniencia(arquivo_origem="sintetico.pdf", edital="1", trienio="2022/2024", pagina=1),
        validacao=ValidacaoRegistro(
            campos_formato_invalido=(), buracos_classificacao={}, fora_de_ordem_alfabetica=False,
        ),
    )


class TestLimiteDePlausibilidadeDosBuracos:
    """Ticket 08, achado na rodada completa: o campo de classificação não passa pela
    mesma validação de formato dos 9 campos numéricos (ticket 02) — um dígito colado (a
    mesma classe de corrupção do número de página vazando pro último campo, documentada
    em `cotas.py`) pode virar uma posição absurda. Caso real, corpus inteiro: um único
    registro leu uma posição de 6 dígitos no Sistema Universal de um curso com ~900
    classificados — sem limite, `esperado` vira um `range` de centenas de milhares de
    posições, e o único registro corrompido explode o relatório e o CSV do corpus inteiro
    (visto na prática: 6,4 GB em vez dos ~25 MB esperados). Ver `validacao._buracos_por_sistema`.
    """

    def test_posicao_implausivel_nao_gera_buraco_gigante(self):
        registros = [_registro_minimo(f"{i:08d}", f"Aluno {i}", i) for i in range(1, 10)]
        registros.append(_registro_minimo("99999999", "Aluno Corrompido", 280852))

        validar_sequencia_e_ordem(registros)

        # O Sistema com a posição implausível fica de fora do cômputo — não é um buraco
        # contável, e nenhum registro (nem o corrompido, nem os 9 legítimos) carrega um
        # `buracos_classificacao` com centenas de milhares de posições.
        assert all(not r.validacao.buracos_classificacao for r in registros)

    def test_buraco_legitimo_continua_detectado_com_o_limite_ativo(self):
        # O limite não pode apagar o sinal real: um curso de 9 candidatos com a posição 5
        # faltando continua acusando o buraco normalmente.
        posicoes = [1, 2, 3, 4, 6, 7, 8, 9, 10]
        registros = [_registro_minimo(f"{i:08d}", f"Aluno {i}", p) for i, p in enumerate(posicoes, start=1)]

        validar_sequencia_e_ordem(registros)

        assert all(r.validacao.buracos_classificacao == {1: (5,)} for r in registros)


class TestOrdemAlfabetica:
    """Ticket 02, verificação 3: ordem alfabética dentro do curso."""

    def test_curso_completo_sem_corrupcao_esta_em_ordem(self):
        _pular_se_fixture_ausente(FIXTURE_CURSO_COMPLETO)

        resultado = extrair_edital(FIXTURE_CURSO_COMPLETO)
        grupo = [r for r in resultado.registros if r.curso == CURSO_PEQUENO_COMPLETO]

        assert all(not r.validacao.fora_de_ordem_alfabetica for r in grupo)

    def test_registros_colados_fora_de_ordem_sao_sinalizados(self):
        # Regressão da mesma classe de corrupção do teste anterior: registros colados
        # embaralham a ordem em que os nomes aparecem no fluxo. Simula isso trocando dois
        # registros reais e adjacentes de posição, e confere que só o que ficou fora do
        # lugar é sinalizado — os outros sete continuam válidos.
        _pular_se_fixture_ausente(FIXTURE_CURSO_COMPLETO)

        resultado = extrair_edital(FIXTURE_CURSO_COMPLETO)
        grupo = [r for r in resultado.registros if r.curso == CURSO_PEQUENO_COMPLETO]
        # Qualquer par adjacente serve: o grupo já está confirmado em ordem alfabética
        # pelo teste anterior, então trocar dois adjacentes sempre produz exatamente uma
        # quebra — a posição j (o antigo i) fica fora de ordem relativa à posição i (o
        # antigo j). Não depende de nome específico, então não precisa de literal aqui.
        i, j = 3, 4
        nome_esperado_fora_de_ordem = grupo[i].nome

        embaralhado = list(grupo)
        embaralhado[i], embaralhado[j] = embaralhado[j], embaralhado[i]
        validar_sequencia_e_ordem(embaralhado)

        fora_de_ordem = [r.nome for r in embaralhado if r.validacao.fora_de_ordem_alfabetica]
        assert fora_de_ordem == [nome_esperado_fora_de_ordem]


class TestParseDirigidoPorSecao:
    """Ticket 05: Editais de resultado final tipo D + redação têm duas seções com schemas
    diferentes no mesmo arquivo. Só a seção de não eliminados (22 campos) é extraída; a
    transição é detectada pelo cabeçalho numerado "2 DO RESULTADO FINAL DOS CANDIDATOS NÃO
    ELIMINADOS", não por número de página fixo.
    """

    def test_apenas_a_secao_de_nao_eliminados_e_extraida(self):
        _pular_se_fixture_ausente(FIXTURE_DUAS_SECOES)

        resultado = extrair_edital(FIXTURE_DUAS_SECOES)

        assert len(resultado.registros) == CONTAGEM_ESPERADA_DUAS_SECOES
        assert all(len(r.classificacoes) == 10 for r in resultado.registros)

    def test_nenhum_registro_da_secao_1_aparece_na_saida(self):
        # A seção 1 (eliminados) tem registros de 4 campos (inscrição, nome, 2 notas) —
        # sem o vetor de 9 notas, um registro dela nunca deveria compor um
        # RegistroResultadoFinal. Prova isso por nome e inscrição reais da seção 1 (lidos
        # da fixture em tempo de execução, nunca como literal — ver
        # `_ultimo_registro_da_secao_1`), não só pela contagem total.
        _pular_se_fixture_ausente(FIXTURE_DUAS_SECOES)

        inscricao_secao_1, nome_secao_1 = _ultimo_registro_da_secao_1(FIXTURE_DUAS_SECOES)
        resultado = extrair_edital(FIXTURE_DUAS_SECOES)
        nomes = {r.nome for r in resultado.registros}
        inscricoes = {r.inscricao for r in resultado.registros}

        assert nome_secao_1 not in nomes
        assert inscricao_secao_1 not in inscricoes

    def test_edital_de_secao_unica_continua_sem_regressao(self):
        # A fixture de seção única (ticket 01) não tem o cabeçalho de transição — o
        # parser precisa continuar tratando o blob inteiro como uma seção só, e não
        # descartar tudo por procurar um cabeçalho que não existe.
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        assert len(resultado.registros) == CONTAGEM_ESPERADA


def _classificacoes(*sistemas: int) -> dict:
    """Monta o dicionário de 10 classificações a partir dos Sistemas em que o Aluno aparece.

    A posição em si (1º, 2º, ...) é irrelevante para a dedução de cota — o que importa é
    só *ter* ou *não ter* classificação em cada Sistema; qualquer inteiro serve.
    """
    return {i: (1 if i in sistemas else None) for i in range(1, 11)}


class TestCotaDeclarada:
    """Ticket 06: os quatro atributos binários deduzidos do padrão das 10 classificações.

    Os Sistemas de Escola Pública são **aninhados**: quem é ≤1,5 SM concorre também às
    vagas de >1,5 SM, quem é PPI concorre também às não-PPI (cascata da Lei 12.711). Então
    o padrão observado é sempre o *fecho para baixo* do reticulado, e os atributos do Aluno
    são os do subsistema **mais específico** em que ele aparece.
    """

    def test_aluno_so_no_universal_nao_declara_cota_nenhuma(self):
        # 71% dos Alunos. É por isso que o campo se chama *cota declarada* e nunca *cota
        # elegível*: aqui é impossível distinguir quem não tem direito a cota de quem tem
        # e optou por não usar — o dado registra a opção, não a elegibilidade.
        cota = deduzir_cota_declarada(_classificacoes(1))

        assert not cota.sistema_negros
        assert not cota.escola_publica
        assert not cota.renda_baixa
        assert not cota.ppi
        assert not cota.pcd
        assert cota.perfil == "Universal"
        assert not cota.padrao_suspeito

    def test_cota_para_negros_nao_e_escola_publica(self):
        # O Aluno opta por um sistema ou por outro na inscrição: Cota para Negros nunca
        # coocorre com subsistema de Escola Pública.
        cota = deduzir_cota_declarada(_classificacoes(1, 2))

        assert cota.sistema_negros
        assert not cota.escola_publica
        assert cota.perfil == "Cota para Negros"
        assert not cota.padrao_suspeito

    def test_escola_publica_sem_nenhum_outro_atributo(self):
        # Fecho de atributos vazio: só o subsistema menos específico de todos (9), que não
        # exige renda baixa, nem PPI, nem PcD.
        cota = deduzir_cota_declarada(_classificacoes(1, 9))

        assert cota.escola_publica
        assert not cota.renda_baixa and not cota.ppi and not cota.pcd
        assert cota.perfil == "EP / Alta Renda / Não-PPI"
        assert not cota.padrao_suspeito

    def test_os_atributos_vem_do_subsistema_mais_especifico(self):
        # {3, 5, 7, 9} é o fecho para baixo de {Renda, PPI}: o Aluno aparece no subsistema
        # 3 (o mais específico, que exige os dois) e, por cascata, em todos os que exigem
        # menos. Os atributos são os do 3 — não os do 9, onde ele também aparece.
        cota = deduzir_cota_declarada(_classificacoes(1, 3, 5, 7, 9))

        assert cota.escola_publica and cota.renda_baixa and cota.ppi
        assert not cota.pcd
        assert cota.perfil == "EP / Baixa Renda / PPI"
        assert not cota.padrao_suspeito

    def test_fecho_completo_dos_quatro_atributos(self):
        # O Aluno mais específico possível (EP + renda baixa + PPI + PcD) aparece nos 8
        # subsistemas de Escola Pública ao mesmo tempo — o fecho inteiro do reticulado.
        cota = deduzir_cota_declarada(_classificacoes(1, 3, 4, 5, 6, 7, 8, 9, 10))

        assert cota.escola_publica and cota.renda_baixa and cota.ppi and cota.pcd
        assert cota.perfil == "EP / Baixa Renda / PPI / PcD"
        assert not cota.padrao_suspeito

    def test_padrao_que_nao_e_fecho_e_sinalizado_e_nao_descartado(self):
        # Aparecer no subsistema 3 (exige renda baixa + PPI) sem aparecer no 5, 7 e 9
        # (que exigem menos) é impossível pela cascata da Lei 12.711 — é sinal de
        # corrupção de extração. O registro continua com os atributos deduzidos do
        # subsistema mais específico observado; só ganha a marca de suspeito.
        cota = deduzir_cota_declarada(_classificacoes(1, 3))

        assert cota.padrao_suspeito
        assert cota.escola_publica and cota.renda_baixa and cota.ppi
        assert cota.perfil == "EP / Baixa Renda / PPI"

    def test_negros_junto_de_escola_publica_e_padrao_impossivel(self):
        # O Aluno opta por um sistema ou por outro na inscrição, então os dois nunca
        # coocorrem (confirmado no corpus: a única ocorrência em 66.313 registros é um
        # artefato de extração). Sem esta checagem, {1,2,3,5,7,9} passaria limpo — o lado
        # de Escola Pública é um fecho válido — e ainda teria a declaração de Negros
        # apagada do rótulo, que é corrupção silenciosa.
        cota = deduzir_cota_declarada(_classificacoes(1, 2, 3, 5, 7, 9))

        assert cota.padrao_suspeito
        # Nada é descartado: as duas declarações continuam legíveis nos booleanos, mesmo
        # com o rótulo tendo de escolher uma delas.
        assert cota.sistema_negros
        assert cota.escola_publica and cota.renda_baixa and cota.ppi

    def test_todo_aluno_nao_eliminado_recebe_cota_declarada(self):
        # Os campos são ranking, não aprovação: o perfil é registrado para todos os
        # Alunos extraídos, não só para os aprovados.
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        assert len(resultado.registros) == CONTAGEM_ESPERADA
        assert all(r.cota_declarada is not None for r in resultado.registros)
        assert all(isinstance(r.cota_declarada.perfil, str) for r in resultado.registros)

    def test_perfil_de_aluno_real_com_padrao_conhecido(self):
        # Padrão conferido no Edital real (Ed_38, ARQUIVOLOGIA): classificação nos
        # Sistemas 1, 3, 5, 7 e 9, e traço nos demais. Selecionado por curso + tamanho do
        # padrão (não é tautológico: a seleção não afirma QUAIS sistemas, só quantos) —
        # não precisa de inscrição para identificar o registro.
        _pular_se_fixture_ausente(FIXTURE_CURSO_COMPLETO)

        resultado = extrair_edital(FIXTURE_CURSO_COMPLETO)
        aluno = next(
            r for r in resultado.registros
            if r.curso == CURSO_COM_PADRAO_RICO
            and sum(1 for v in r.classificacoes.values() if v is not None) == 5
        )

        assert [i for i in range(1, 11) if aluno.classificacoes[i] is not None] == [1, 3, 5, 7, 9]
        assert aluno.cota_declarada.escola_publica
        assert aluno.cota_declarada.renda_baixa
        assert aluno.cota_declarada.ppi
        assert not aluno.cota_declarada.pcd
        assert not aluno.cota_declarada.sistema_negros
        assert aluno.cota_declarada.perfil == "EP / Baixa Renda / PPI"
        assert not aluno.cota_declarada.padrao_suspeito

    def test_nenhum_padrao_real_da_fixture_viola_o_fecho(self):
        # A validação que sustenta o modelo, no recorte da fixture: todo padrão observado
        # é fecho para baixo válido do reticulado.
        _pular_se_fixture_ausente(FIXTURE_RESULTADO_FINAL)

        resultado = extrair_edital(FIXTURE_RESULTADO_FINAL)

        suspeitos = [r.nome for r in resultado.registros if r.cota_declarada.padrao_suspeito]
        assert suspeitos == []

    def test_padrao_suspeito_real_sobrevive_na_saida_com_a_marca(self):
        # O caso real que a checagem de fecho pegou no corpus (8 ocorrências em 66.313
        # registros, todas desta mesma forma): o Aluno é o último registro da página, seu
        # 22º campo só começa na página seguinte, e o número dessa página — que a extração
        # emite no início do texto dela — entra no lugar da 10ª classificação. O padrão
        # resultante ({1, 10}) é impossível pela cascata da Lei 12.711: aparecer no
        # subsistema que exige PcD sem aparecer no 9, que não exige nada.
        #
        # Este teste fixa o comportamento do ticket 06 (sinalizar, não descartar), não o
        # bug de parse que o originou — esse é de outra camada e está fora deste ticket
        # (ver relatório, seção de escopo). Quando ele for corrigido, o registro passa a
        # ter padrão {1} e este teste falha de propósito: é o lembrete de revisitá-lo.
        _pular_se_fixture_ausente(FIXTURE_COTA_SUSPEITA)

        resultado = extrair_edital(FIXTURE_COTA_SUSPEITA)
        # Identificado pelo próprio sinal sob teste — é o único registro suspeito deste
        # recorte (confirmado logo abaixo) — não precisa de inscrição para selecioná-lo.
        aluno = next(r for r in resultado.registros if r.cota_declarada.padrao_suspeito)

        assert [i for i in range(1, 11) if aluno.classificacoes[i] is not None] == [1, 10]
        assert aluno.cota_declarada.padrao_suspeito
        # Não descartado: o registro continua na saída, com os atributos deduzidos do
        # subsistema mais específico que foi observado.
        assert aluno in resultado.registros
        assert aluno.cota_declarada.pcd and aluno.cota_declarada.escola_publica
        # E é o único do recorte — a marca não é ruído que atinge todo mundo.
        suspeitos = [r for r in resultado.registros if r.cota_declarada.padrao_suspeito]
        assert suspeitos == [aluno]

    def test_as_seis_colunas_derivadas_saem_no_csv_junto_das_classificacoes_cruas(self, tmp_path):
        _pular_se_fixture_ausente(FIXTURE_CURSO_COMPLETO)

        resultado = extrair_edital(FIXTURE_CURSO_COMPLETO)
        destino = tmp_path / "resultado_final.csv"
        escrever_csv([resultado], destino)

        with destino.open(encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))

        derivadas = ("sistema_negros", "escola_publica", "renda_baixa", "ppi", "pcd", "perfil_cota")
        assert all(coluna in CSV_COLUMNS for coluna in derivadas)
        # As 10 classificações cruas continuam lá, junto das derivadas.
        assert all(f"classificacao_sistema_{i}" in CSV_COLUMNS for i in range(1, 11))
        assert len(linhas) == len(resultado.registros)

        # Mesmo critério de seleção do teste de perfil acima: curso + tamanho do padrão,
        # não inscrição.
        linha = next(
            l for l in linhas
            if l["curso"] == CURSO_COM_PADRAO_RICO
            and sum(1 for i in range(1, 11) if l[f"classificacao_sistema_{i}"] != "-") == 5
        )
        assert linha["escola_publica"] == "True"
        assert linha["renda_baixa"] == "True"
        assert linha["ppi"] == "True"
        assert linha["pcd"] == "False"
        assert linha["sistema_negros"] == "False"
        assert linha["perfil_cota"] == "EP / Baixa Renda / PPI"
        assert linha["classificacao_sistema_2"] == "-"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
