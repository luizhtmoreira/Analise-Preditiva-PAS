import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # type: ignore

from pas_extraction.nome_repair import PARTICULAS, reparar_nome  # type: ignore


class TestPalavraQuebradaPorEspaco:
    """Ticket 13, item 1: palavra partida por um único espaço espúrio."""

    def test_sufixo_de_uma_letra_e_reparado(self):
        # Exemplo real do achado do ticket ("Isabella" -> "Isabell a").
        nome, reparado = reparar_nome("Isabell a")

        assert nome == "Isabella"
        assert reparado

    def test_prefixo_de_uma_letra_maiuscula_e_reparado(self):
        # Mesmo padrão do achado real de scripts/NOTES.md (achado 9) — sobrenome com a
        # primeira letra solta, separada por espaço espúrio do resto minúsculo. Nome
        # sintético aqui: o achado real cita um nome de Aluno real, que não entra em texto
        # versionado (mesma convenção de `test_pas_extraction_convocacao._TEXTO_DUAS_SECOES`,
        # "com os nomes trocados").
        nome, reparado = reparar_nome("Carla V ieira Rocha")

        assert nome == "Carla Vieira Rocha"
        assert reparado

    def test_palavra_partida_em_tres_pedacos_fecha_numa_so_passada(self):
        nome, reparado = reparar_nome("Isa bel la Souza")

        assert nome == "Isabella Souza"
        assert reparado

    def test_quebra_no_primeiro_nome_nao_afeta_o_resto(self):
        nome, reparado = reparar_nome("Ana Be atriz Costa")

        assert nome == "Ana Beatriz Costa"
        assert reparado


class TestEspacoDuplicadoSemQuebra:
    """Ticket 13, item 2: espaço duplicado entre palavras, sem nenhuma palavra partida."""

    def test_espaco_duplo_e_colapsado_para_um_so(self):
        nome, reparado = reparar_nome("Maria  Eduarda  Santos")

        assert nome == "Maria Eduarda Santos"
        assert reparado

    def test_corrida_longa_de_espacos_tambem_colapsa(self):
        nome, reparado = reparar_nome("João          Pedro")

        assert nome == "João Pedro"
        assert reparado


class TestParticulasCurtasLegitimas:
    """Ticket 13, "cuidado central": de/da/do/dos/das/e nunca são fundidas com o vizinho,
    mesmo sendo tokens inteiramente minúsculos como os fragmentos de quebra."""

    @pytest.mark.parametrize("nome", [
        "Maria de Souza",
        "Ana da Silva",
        "João do Nascimento",
        "Carlos dos Santos",
        "Rita das Neves",
        "Sousa e Silva",
    ])
    def test_particula_isolada_nao_e_fundida(self, nome):
        resultado, reparado = reparar_nome(nome)

        assert resultado == nome
        assert not reparado

    def test_todas_as_particulas_estao_cobertas_pela_constante(self):
        assert PARTICULAS == frozenset({"de", "da", "do", "dos", "das", "e"})


class TestNomeSemCorrupcao:
    def test_nome_normal_nao_e_alterado(self):
        nome, reparado = reparar_nome("Isabella Cristina Melo Macedo da Silva")

        assert nome == "Isabella Cristina Melo Macedo da Silva"
        assert not reparado

    def test_espaco_de_borda_e_removido_sem_marcar_como_reparado(self):
        # Espaço só nas pontas não é o defeito do ticket 13 (é higiene padrão de string) —
        # `.strip()` já era aplicado antes deste módulo existir, então não conta como reparo.
        nome, reparado = reparar_nome("  Isabella Costa  ")

        assert nome == "Isabella Costa"
        assert not reparado


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
