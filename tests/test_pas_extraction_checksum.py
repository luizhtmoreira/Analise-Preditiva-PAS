"""Ticket 04 — checksum do Argumento Final + inferência de língua por Etapa.

Exercita a costura `extrair_edital`, não a estrutura interna do parser: o checksum sai
gravado em cada registro que a costura devolve. As funções de `checksum.py` que aparecem
aqui (`montar_tabela_oficial`, `recalcular_argumento_final`, `conferir_argumento_final`)
são usadas apenas para montar os *contrafactuais* que provam as decisões do ticket — a
língua por Etapa contra a língua fixa por Aluno, o registro corrompido contra o íntegro, e
a tabela do triênio errado contra a certa.
"""

import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore
from pypdf import PdfReader  # type: ignore

from pas_extraction import extrair_edital  # type: ignore
from pas_extraction.checksum import (  # type: ignore
    conferir_argumento_final,
    conferir_registros,
    montar_tabela_oficial,
    recalcular_argumento_final,
)
from pas_extraction.constants import (  # type: ignore
    LINGUAS_ESTRANGEIRAS,
    TOLERANCIA_CHECKSUM,
)
from pas_extraction.csv_writer import CSV_COLUMNS, escrever_csv  # type: ignore
from pas_extraction.medias_desvios import (  # type: ignore
    extrair_edital_medias_desvios,
    parse_medias_desvios,
)
from pas_extraction.models import (  # type: ignore
    ChecksumArgumentoFinal,
    ContextoEdital,
    TrienioIncompativelError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixture deste ticket: páginas 1-6 **+ 242** de Ed_38_2024 (triênio 2022/2024). As 6
# primeiras são as mesmas de `resultado_final_22_campos.pdf` (ticket 01), e a 242 é a cauda
# onde mora a tabela oficial de médias e desvios (a mesma de `medias_desvios_cauda.pdf`,
# ticket 03). O checksum precisa das duas coisas no mesmo Edital, e nenhuma fixture
# anterior tinha as duas. Gerada com:
#   python -c "
#   import sys; sys.path.insert(0, 'src')
#   from pas_extraction.fixtures import fatiar_paginas
#   fatiar_paginas(
#       'data/pdfs/Ed_38_2024_PAS_3_2022-2024_Res_final_não_eliminados.pdf',
#       [1, 2, 3, 4, 5, 6, 242],
#       'tests/fixtures/resultado_final_com_checksum.pdf',
#   )"
FIXTURE_COM_CHECKSUM = FIXTURES_DIR / "resultado_final_com_checksum.pdf"

# A mesma fatia de 6 páginas **sem** a cauda (ticket 01) — é o caso "não existe tabela
# oficial neste Edital", em que o checksum não pode ser calculado.
FIXTURE_SEM_TABELA = FIXTURES_DIR / "resultado_final_22_campos.pdf"

# Ed_31 (triênio 2016/2018) + o Edital avulso que publica a tabela desse triênio (Ed_32).
# Este par é o caso em que a tabela **não** está na cauda do Resultado Final: é o que o
# parâmetro `medias_desvios` de `extrair_edital` existe para cobrir. A fixture de
# Resultado Final é a mesma do ticket 06 (páginas 1, 82 e 83 do Ed_31). A avulsa foi gerada
# com:
#   python -m pas_extraction.cli fixture \
#     'data/pdfs/Ed_32_2016-2018_PAS_3_media_e_desvio_padrao.pdf' 1 1 \
#     tests/fixtures/medias_desvios_avulso_2016_2018.pdf
FIXTURE_OUTRO_TRIENIO = FIXTURES_DIR / "resultado_final_cota_suspeita.pdf"
FIXTURE_MEDIAS_AVULSO_2016_2018 = FIXTURES_DIR / "medias_desvios_avulso_2016_2018.pdf"

# Contagens medidas na fixture de Ed_38 (as mesmas 189 do ticket 01/02, agora com o
# checksum aplicado a cada uma):
CONTAGEM_ESPERADA = 189
# Todos os 189 fecham com a tolerância operacional de 0,005 — o resultado do protótipo
# (99,8% em 1.261 registros) reproduzido nesta fatia. O maior delta observado é 0,0030,
# que é o arredondamento oficial de 3 casas e não corrupção.
DELTA_MAXIMO_OBSERVADO = 0.0030
# Registros cuja língua **muda** entre Etapas. É a medida que justifica inferir por Etapa
# em vez de por Aluno (o protótipo mediu 17,4% no corpus inteiro; nesta fatia, 27,5%).
MISTOS_ESPERADOS = 52
# Registros que fecham com língua por Etapa e **não fechariam** com nenhuma das três
# línguas fixas por Aluno — os que uma inferência por Aluno jogaria no lixo.
SALVOS_PELA_INFERENCIA_POR_ETAPA = 47
# Registros em que mais de uma das 27 combinações cabe na tolerância: o delta fecha, mas a
# língua não fica determinada. O caso extremo é a Etapa com as notas zeradas, em que P1 = 0
# dá quase o mesmo resultado nas três línguas.
AMBIGUOS_ESPERADOS = 25
# Ticket 02: os 19 registros com campo numérico partido por espaço nesta fixture.
SINALIZADOS_POR_FORMATO = 19

# Ed_31 conferido contra a tabela avulsa do próprio triênio: 82 dos 84 registros fecham.
CONTAGEM_ESPERADA_OUTRO_TRIENIO = 84
FECHAM_OUTRO_TRIENIO = 82


def _pular_se_fixture_ausente(*caminhos: Path) -> None:
    for caminho in caminhos:
        if not caminho.exists():
            pytest.skip(
                f"Fixture {caminho.relative_to(Path(__file__).parent.parent)} não "
                "encontrada. Gere localmente (requer data/pdfs completo) com o comando "
                "no comentário da constante correspondente."
            )


def _tabela_da_cauda(fixture: Path) -> dict:
    """A tabela oficial lida da própria fixture, para montar contrafactuais nos testes.

    Passa por `parse_medias_desvios` (ticket 03) + `montar_tabela_oficial`, que é o mesmo
    caminho que o pipeline usa — o que este helper *não* faz é escolher a língua, que é
    justamente o que está sob teste.
    """
    reader = PdfReader(str(fixture))
    contexto = ContextoEdital(arquivo_origem=fixture.name, edital="38", trienio="2022/2024")
    return montar_tabela_oficial(parse_medias_desvios(reader, contexto))


def _melhor_delta_com_lingua_fixa(registro, tabela) -> float:
    """O menor delta alcançável tratando a língua como fixa por Aluno (só 3 candidatas).

    É o contrafactual do critério de aceite "a inferência é por Etapa e não por Aluno":
    o que a versão anterior do checksum (83,9% de acerto no protótipo) teria medido.
    """
    return min(
        abs(recalcular_argumento_final(registro.notas, (lingua,) * 3, tabela)
            - registro.argumento_final)
        for lingua in LINGUAS_ESTRANGEIRAS
    )


class TestChecksumPorRegistro:
    """Critério: cada registro tem o Argumento Final recalculado e comparado com o
    impresso, com o delta gravado na linha."""

    def test_todo_registro_recebe_delta_e_linguas(self):
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)

        assert len(resultado.registros) == CONTAGEM_ESPERADA
        for r in resultado.registros:
            assert r.checksum is not None
            assert r.checksum.delta >= 0
            assert len(r.checksum.linguas_por_etapa) == 3
            assert set(r.checksum.linguas_por_etapa) <= set(LINGUAS_ESTRANGEIRAS)

    def test_o_recalculado_bate_com_o_impresso_dentro_da_tolerancia(self):
        # 189/189 na fatia. É o resultado que sustenta o ticket inteiro: um único número
        # verifica os 12 campos (9 notas + as 3 médias/desvios usadas) de cada registro.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)

        assert all(r.checksum.fecha for r in resultado.registros)
        assert max(r.checksum.delta for r in resultado.registros) == pytest.approx(
            DELTA_MAXIMO_OBSERVADO
        )

    def test_o_valor_recalculado_e_gravado_junto_do_delta(self):
        # O delta sozinho não diz *para onde* o registro errou. O valor recalculado é o que
        # torna um delta diagnosticável (ticket 07 vai agrupar falhas por padrão), e ele é
        # o outro lado da mesma subtração: `delta == |recalculado - impresso|`.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)

        for r in resultado.registros:
            diferenca = abs(r.checksum.argumento_final_recalculado - r.argumento_final)
            assert diferenca == pytest.approx(r.checksum.delta, abs=1e-6)

    def test_a_tolerancia_aplicada_e_0_005(self):
        # `fecha` é exatamente `delta <= 0,005`, inclusive na borda. A metade dos
        # registros que fecha em ≤0,001 não é o critério; o arredondamento oficial de 3
        # casas é o que a tolerância acomoda (ver ticket).
        assert TOLERANCIA_CHECKSUM == 0.005

        linguas = ("inglesa", "inglesa", "inglesa")
        na_borda = ChecksumArgumentoFinal(0.0, TOLERANCIA_CHECKSUM, linguas)
        passou_da_borda = ChecksumArgumentoFinal(0.0, TOLERANCIA_CHECKSUM + 1e-9, linguas)

        assert na_borda.fecha
        assert not passou_da_borda.fecha

    def test_corrupcao_de_uma_nota_nao_fecha_o_checksum(self):
        # O que o checksum existe para pegar: um número plausível e errado, que nenhuma
        # outra camada de validação enxerga. Parte de um registro real que fecha, soma
        # 1,000 a uma das 9 notas, e confere que o delta sai da tolerância — a corrupção
        # silenciosa fica visível.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)
        tabela = _tabela_da_cauda(FIXTURE_COM_CHECKSUM)
        registro = resultado.registros[0]
        assert registro.checksum.fecha

        notas_corrompidas = list(registro.notas)
        notas_corrompidas[1] += 1.0  # eb_p2_e1
        delta = min(
            abs(recalcular_argumento_final(notas_corrompidas, combo, tabela)
                - registro.argumento_final)
            for combo in itertools.product(LINGUAS_ESTRANGEIRAS, repeat=3)
        )

        assert delta > TOLERANCIA_CHECKSUM


class TestInferenciaDeLinguaPorEtapa:
    """Critério: a língua é inferida por qual das três faz o checksum fechar, **por Etapa**
    e não por Aluno.

    A língua estrangeira não está impressa em lugar nenhum do Edital — é um dado que o
    pipeline recupera como subproduto do checksum.
    """

    def test_a_lingua_de_cada_etapa_e_gravada_por_aluno(self):
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)

        # As três línguas aparecem no conjunto inferido (não é uma constante disfarçada).
        inferidas = {
            lingua for r in resultado.registros for lingua in r.checksum.linguas_por_etapa
        }
        assert inferidas == set(LINGUAS_ESTRANGEIRAS)

    def test_ha_alunos_que_trocam_de_lingua_entre_etapas(self):
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)

        mistos = [
            r for r in resultado.registros if len(set(r.checksum.linguas_por_etapa)) > 1
        ]
        assert len(mistos) == MISTOS_ESPERADOS

    def test_inferir_por_aluno_descartaria_registros_que_fecham_por_etapa(self):
        # O erro que o ticket manda não cometer: 47 dos 189 registros desta fatia fecham
        # com língua por Etapa e não fechariam com nenhuma das três línguas fixas. Foi
        # essa distinção que, no protótipo, separou 83,9% de 99,8% de acerto — e o que
        # impediria o descarte de 200 registros perfeitamente bons.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)
        tabela = _tabela_da_cauda(FIXTURE_COM_CHECKSUM)

        salvos = [
            r for r in resultado.registros
            if r.checksum.fecha
            and _melhor_delta_com_lingua_fixa(r, tabela) > TOLERANCIA_CHECKSUM
        ]

        assert len(salvos) == SALVOS_PELA_INFERENCIA_POR_ETAPA
        # E, nesses registros, a língua gravada é de fato mista entre Etapas.
        assert all(len(set(r.checksum.linguas_por_etapa)) > 1 for r in salvos)

    def test_lingua_indeterminada_e_marcada_como_ambigua(self):
        # O limite honesto da inferência: em 25 dos 189 registros mais de uma das 27
        # combinações cabe na tolerância, então a língua gravada é *uma* leitura possível e
        # não a certa. Isso é marcado no registro, e não deixado só na documentação —
        # senão o consumidor não teria como filtrar por confiança (spec, user story 32).
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)
        tabela = _tabela_da_cauda(FIXTURE_COM_CHECKSUM)

        ambiguos = [r for r in resultado.registros if r.checksum.lingua_ambigua]
        assert len(ambiguos) == AMBIGUOS_ESPERADOS

        # A marca significa o que diz: mais de uma combinação dentro da tolerância. Confere
        # isso contando as 27 por fora, para os ambíguos e para um não ambíguo.
        def quantas_fecham(registro):
            return sum(
                1 for combo in itertools.product(LINGUAS_ESTRANGEIRAS, repeat=3)
                if abs(recalcular_argumento_final(registro.notas, combo, tabela)
                       - registro.argumento_final) <= TOLERANCIA_CHECKSUM
            )

        assert all(quantas_fecham(r) > 1 for r in ambiguos)
        nao_ambiguo = next(r for r in resultado.registros if not r.checksum.lingua_ambigua)
        assert quantas_fecham(nao_ambiguo) == 1

    def test_a_lingua_gravada_e_a_que_minimiza_o_delta(self):
        # Comportamento, não implementação: nenhuma das 27 combinações produz delta menor
        # que o da combinação gravada. Vale para todo registro da fixture.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)
        tabela = _tabela_da_cauda(FIXTURE_COM_CHECKSUM)

        for r in resultado.registros:
            melhor_possivel = min(
                abs(recalcular_argumento_final(r.notas, combo, tabela) - r.argumento_final)
                for combo in itertools.product(LINGUAS_ESTRANGEIRAS, repeat=3)
            )
            assert r.checksum.delta == pytest.approx(melhor_possivel)


class TestChecksumNaoDescartaNada:
    """Critério: nenhum registro é descartado neste ticket — o checksum só marca."""

    def test_a_contagem_e_a_mesma_com_e_sem_tabela_oficial(self):
        # As duas fixtures são a mesma fatia de 6 páginas; a diferença é só a cauda com a
        # tabela. Se o checksum descartasse alguma coisa, as contagens divergiriam.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM, FIXTURE_SEM_TABELA)

        com_tabela = extrair_edital(FIXTURE_COM_CHECKSUM)
        sem_tabela = extrair_edital(FIXTURE_SEM_TABELA)

        assert len(com_tabela.registros) == len(sem_tabela.registros) == CONTAGEM_ESPERADA

    def test_registro_que_nao_fecha_continua_na_saida_com_a_marca(self):
        # No recorte do Ed_31 há 2 registros que não fecham (ver
        # `FECHAM_OUTRO_TRIENIO`). Eles continuam na saída, marcados — é ticket 07 que
        # agrupa falhas por padrão, e o spec proíbe descarte automático sem padrão
        # explicado.
        _pular_se_fixture_ausente(FIXTURE_OUTRO_TRIENIO, FIXTURE_MEDIAS_AVULSO_2016_2018)

        resultado = extrair_edital(
            FIXTURE_OUTRO_TRIENIO, medias_desvios=FIXTURE_MEDIAS_AVULSO_2016_2018
        )

        assert len(resultado.registros) == CONTAGEM_ESPERADA_OUTRO_TRIENIO
        fecham = [r for r in resultado.registros if r.checksum.fecha]
        assert len(fecham) == FECHAM_OUTRO_TRIENIO
        naos = [r for r in resultado.registros if not r.checksum.fecha]
        assert len(naos) == CONTAGEM_ESPERADA_OUTRO_TRIENIO - FECHAM_OUTRO_TRIENIO
        assert all(r.checksum.delta > TOLERANCIA_CHECKSUM for r in naos)

    def test_campos_reparados_pelo_ticket_02_fecham_o_checksum(self):
        # Verificação cruzada entre camadas, e um resultado que só o checksum podia dar:
        # os 19 registros que tiveram um campo numérico partido por espaço reparado
        # (`"1 7.539"` -> 17.539) fecham todos. O reparo recuperou o valor **certo** — não
        # um valor plausível e errado, que era o risco.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)
        reparados = [
            r for r in resultado.registros if r.validacao.campos_formato_invalido
        ]

        assert len(reparados) == SINALIZADOS_POR_FORMATO
        assert all(r.checksum.fecha for r in reparados)


class TestTabelaOficialAusenteOuExterna:
    def test_sem_tabela_no_edital_o_checksum_fica_indisponivel(self):
        # Nada de fingir verificação: sem a média e o desvio oficiais não existe checksum,
        # e o campo fica `None` em vez de um delta inventado ou de um `fecha=True` gratuito.
        _pular_se_fixture_ausente(FIXTURE_SEM_TABELA)

        resultado = extrair_edital(FIXTURE_SEM_TABELA)

        assert len(resultado.registros) == CONTAGEM_ESPERADA
        assert all(r.checksum is None for r in resultado.registros)

    def test_tabela_de_edital_avulso_pode_ser_fornecida(self):
        # Triênios diferentes publicam a tabela de formas diferentes (spec, user story 8):
        # o Ed_31 (2016/2018) não a traz na cauda, ela sai num Edital avulso. Com a tabela
        # certa em mãos, os registros de 2016/2018 também fecham.
        _pular_se_fixture_ausente(FIXTURE_OUTRO_TRIENIO, FIXTURE_MEDIAS_AVULSO_2016_2018)

        sem = extrair_edital(FIXTURE_OUTRO_TRIENIO)
        com = extrair_edital(
            FIXTURE_OUTRO_TRIENIO, medias_desvios=FIXTURE_MEDIAS_AVULSO_2016_2018
        )

        assert all(r.checksum is None for r in sem.registros)
        assert all(r.checksum is not None for r in com.registros)
        assert len(com.registros) == len(sem.registros)

    def test_tabela_do_trienio_errado_e_recusada_na_entrada(self):
        # Os dois triênios estão impressos na página 1 dos dois Editais, então o
        # casamento errado morre antes de qualquer registro ser conferido — em vez de
        # produzir um CSV inteiro de deltas absurdos para alguém descobrir depois.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM, FIXTURE_MEDIAS_AVULSO_2016_2018)

        with pytest.raises(TrienioIncompativelError):
            extrair_edital(
                FIXTURE_COM_CHECKSUM, medias_desvios=FIXTURE_MEDIAS_AVULSO_2016_2018
            )

    def test_tabela_do_trienio_errado_tambem_grita_no_delta(self):
        # O backstop de quando o triênio de um dos dois Editais não pôde ser lido da
        # página 1 (`METADADO_DESCONHECIDO`), caso em que não há o que comparar na
        # entrada: a magnitude do delta. Conferir Ed_38 (2022/2024) contra a tabela de
        # 2016/2018 não fecha **nenhum** registro, e com deltas na casa das dezenas — não
        # é o tipo de erro que se confunde com arredondamento.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM, FIXTURE_MEDIAS_AVULSO_2016_2018)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)
        tabela_de_outro_trienio = montar_tabela_oficial(
            extrair_edital_medias_desvios(FIXTURE_MEDIAS_AVULSO_2016_2018).registros
        )

        deltas = [
            conferir_argumento_final(r.notas, r.argumento_final, tabela_de_outro_trienio).delta
            for r in resultado.registros
        ]

        assert min(deltas) > 1.0
        assert all(delta > TOLERANCIA_CHECKSUM for delta in deltas)

    def test_tabela_incompleta_nao_produz_checksum(self):
        # `montar_tabela_oficial` só devolve tabela quando as 3 Etapas x 3 línguas estão
        # completas: conferir contra tabela parcial mediria a falta da tabela, não a
        # qualidade do registro.
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        completa = _tabela_da_cauda(FIXTURE_COM_CHECKSUM)
        assert len(completa) == 9

        reader = PdfReader(str(FIXTURE_COM_CHECKSUM))
        contexto = ContextoEdital(
            arquivo_origem=FIXTURE_COM_CHECKSUM.name, edital="38", trienio="2022/2024",
        )
        registros_md = parse_medias_desvios(reader, contexto)
        sem_etapa_3 = [r for r in registros_md if r.etapa != 3]

        assert montar_tabela_oficial(sem_etapa_3) == {}
        assert montar_tabela_oficial([]) == {}

    def test_conferir_registros_recusa_tabela_vazia(self):
        # A regra "tabela parcial não vira checksum" não pode depender só de o pipeline
        # lembrar de checar antes de chamar: quem chamar com `{}` recebe erro em vez de um
        # KeyError obscuro (ou, pior, um delta calculado com metade da tabela).
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        registros = extrair_edital(FIXTURE_COM_CHECKSUM).registros

        with pytest.raises(ValueError):
            conferir_registros(registros, {})


class TestChecksumNoCsv:
    def test_as_seis_colunas_novas_existem(self):
        assert "checksum_delta" in CSV_COLUMNS
        assert "checksum_fecha" in CSV_COLUMNS
        assert "lingua_ambigua" in CSV_COLUMNS
        assert all(f"lingua_e{etapa}" in CSV_COLUMNS for etapa in (1, 2, 3))

    def test_linhas_do_csv_carregam_o_checksum(self, tmp_path):
        _pular_se_fixture_ausente(FIXTURE_COM_CHECKSUM)

        resultado = extrair_edital(FIXTURE_COM_CHECKSUM)
        destino = tmp_path / "resultado_final.csv"
        escrever_csv([resultado], destino)

        with destino.open(encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))

        assert len(linhas) == CONTAGEM_ESPERADA
        assert all(l["checksum_fecha"] == "True" for l in linhas)
        assert all(float(l["checksum_delta"]) <= TOLERANCIA_CHECKSUM for l in linhas)
        assert all(l[f"lingua_e{e}"] in LINGUAS_ESTRANGEIRAS for l in linhas for e in (1, 2, 3))
        assert sum(1 for l in linhas if l["lingua_ambigua"] == "True") == AMBIGUOS_ESPERADOS

    def test_sem_tabela_as_colunas_saem_vazias(self, tmp_path):
        _pular_se_fixture_ausente(FIXTURE_SEM_TABELA)

        resultado = extrair_edital(FIXTURE_SEM_TABELA)
        destino = tmp_path / "resultado_final_sem_tabela.csv"
        escrever_csv([resultado], destino)

        with destino.open(encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))

        assert len(linhas) == CONTAGEM_ESPERADA
        vazias = (
            "checksum_delta", "checksum_fecha", "lingua_ambigua",
            "lingua_e1", "lingua_e2", "lingua_e3",
        )
        for coluna in vazias:
            assert all(l[coluna] == "" for l in linhas)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
