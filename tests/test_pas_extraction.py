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
)
from pas_extraction.schema import canonizar, classificar_familia  # type: ignore
from pas_extraction.validacao import validar_sequencia_e_ordem  # type: ignore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_RESULTADO_FINAL = FIXTURES_DIR / "resultado_final_22_campos.pdf"

# Contagem observada ao gerar a fixture com `python -m pas_extraction.cli fixture
# 'data/pdfs/Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf' 1 6 <destino>`.
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
#       'data/pdfs/Ed_27_PAS_3_2021_2023_Res_final_tipo_D_redação.pdf',
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
