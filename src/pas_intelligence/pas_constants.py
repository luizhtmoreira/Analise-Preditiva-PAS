from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Union

# As três línguas estrangeiras que o Cebraspe normaliza separadamente na Parte 1.
LINGUAS_OFICIAIS = ("inglesa", "francesa", "espanhola")


class Origem(Enum):
    """De onde veio a média e o desvio de uma `(Ano, Etapa)`.

    Existe porque, quando o Edital de verdade sair, os números `DERIVADA` serão substituídos
    e as previsões vão mexer — isso precisa estar registrado no dado, não descoberto depois.
    """

    EDITAL = "edital"
    DERIVADA = "derivada"


@dataclass(frozen=True)
class ValorLingua:
    media: float
    desvio_padrao: float


class Parte1(Mapping[str, ValorLingua], ABC):
    """A Parte 1 de uma Etapa, nas duas formas em que as fontes a publicam.

    É um `Mapping` das três línguas oficiais nas duas formas, para que quem só quer o valor de
    uma língua leia igual nos dois casos. O que distingue as formas é o **tipo** (e o
    `misturada`), nunca o número de chaves — as duas têm três.
    """

    misturada: bool

    @abstractmethod
    def para(self, lingua: str) -> ValorLingua:
        """Média e desvio a aplicar a quem fez `lingua`."""

    def __getitem__(self, lingua: str) -> ValorLingua:
        return self.para(lingua)

    def __iter__(self) -> Iterator[str]:
        return iter(LINGUAS_OFICIAIS)

    def __len__(self) -> int:
        return len(LINGUAS_OFICIAIS)


@dataclass(frozen=True)
class Parte1PorLingua(Parte1):
    """A forma do **Edital de médias e desvios**: um par por língua estrangeira.

    Agrupar as três embute viés sistemático contra quem fez espanhol ou francês, então esta
    forma exige as três — uma Parte 1 parcial não é um dado válido deste domínio.
    """

    valores: Dict[str, ValorLingua]
    misturada = False

    def __post_init__(self) -> None:
        if set(self.valores) != set(LINGUAS_OFICIAIS):
            raise ValueError(
                f"Parte 1 por língua exige exatamente {set(LINGUAS_OFICIAIS)}; "
                f"recebido {set(self.valores)}."
            )

    def para(self, lingua: str) -> ValorLingua:
        return self.valores[lingua]


@dataclass(frozen=True)
class Parte1Misturada(Parte1):
    """A forma do **Edital isolado de Etapa**: um único par, as três línguas juntas.

    Essa fonte — o "Resultado final nos itens do tipo D e na prova de redação", a única
    disponível para a Turma viva — lista nota por candidato e não diz a língua de ninguém.
    Preencher as três exigiria inventar valores, então o dado assume o que é.

    O custo está medido: usar a Parte 1 misturada em vez da língua declarada custa 0,46 ponto
    de Argumento Final em média (máximo 3,21) com viés zero — é ruído, não erro sistemático.
    """

    valor: ValorLingua
    misturada = True

    def para(self, lingua: str) -> ValorLingua:
        if lingua not in LINGUAS_OFICIAIS:
            raise KeyError(lingua)
        return self.valor


@dataclass
class ExamStats:
    m_p2: float; dp_p2: float
    m_red: float; dp_red: float
    # Parte 1 nas duas formas acima — o Edital nunca publica um valor único por língua
    # (ver `m_p1`/`dp_p1`). Sem default porque um ExamStats sem Parte 1 não é um dado válido.
    # Um `dict` cru é aceito como atalho para `Parte1PorLingua`, que é a forma das entradas
    # vindas de Edital de médias e desvios.
    parte_1: Union[Parte1, Dict[str, ValorLingua]]
    # `EDITAL` por default porque é o que toda entrada escrita à mão aqui é; a forma derivada
    # nasce de um único ponto de construção programática, que declara.
    origem: Origem = field(default=Origem.EDITAL)

    def __post_init__(self) -> None:
        if not isinstance(self.parte_1, Parte1):
            self.parte_1 = Parte1PorLingua(dict(self.parte_1))

    def _media_parte_1(self, atributo: str) -> float:
        valores = self.parte_1.values()
        return sum(getattr(v, atributo) for v in valores) / len(valores)

    @property
    def m_p1(self) -> float:
        """Média das três línguas oficiais de Parte 1.

        Não existe um `m_p1` oficial: cada Aluno fez uma única língua estrangeira e o
        Resultado Final não imprime qual. Esta média simples das três existe só para manter
        `api/services/analytics_service.py` funcionando sem mudança de interface — para
        precisão por língua, leia `parte_1` diretamente. Na forma misturada as três línguas
        são o mesmo valor, então a média é o próprio valor.
        """
        return self._media_parte_1("media")

    @property
    def dp_p1(self) -> float:
        return self._media_parte_1("desvio_padrao")


# Valores oficiais de média e desvio-padrão publicados pelo Cebraspe nos Editais do PAS/UnB
# (Parte 1 por língua estrangeira, Parte II e Redação). Chave: (Ano da Prova, Etapa).
OFFICIAL_STATS = {
    (2016, 1): ExamStats(
        m_p2=23.738, dp_p2=13.098, m_red=5.983, dp_red=2.702,
        parte_1={
            "inglesa": ValorLingua(3.628, 2.687),
            "francesa": ValorLingua(4.034, 2.875),
            "espanhola": ValorLingua(5.806, 2.462),
        },
    ),
    (2017, 1): ExamStats(
        m_p2=27.045, dp_p2=13.441, m_red=6.174, dp_red=2.664,
        parte_1={
            "inglesa": ValorLingua(2.966, 2.959),
            "francesa": ValorLingua(3.039, 2.756),
            "espanhola": ValorLingua(3.970, 2.419),
        },
    ),
    (2017, 2): ExamStats(
        m_p2=19.769, dp_p2=11.666, m_red=6.016, dp_red=2.224,
        parte_1={
            "inglesa": ValorLingua(3.871, 3.226),
            "francesa": ValorLingua(2.124, 2.314),
            "espanhola": ValorLingua(5.115, 2.048),
        },
    ),
    (2018, 1): ExamStats(
        m_p2=25.585, dp_p2=14.102, m_red=5.886, dp_red=2.400,
        parte_1={
            "inglesa": ValorLingua(2.679, 2.564),
            "francesa": ValorLingua(4.690, 2.981),
            "espanhola": ValorLingua(3.965, 2.585),
        },
    ),
    (2018, 2): ExamStats(
        m_p2=24.055, dp_p2=12.204, m_red=7.022, dp_red=1.639,
        parte_1={
            "inglesa": ValorLingua(4.187, 2.724),
            "francesa": ValorLingua(2.557, 2.848),
            "espanhola": ValorLingua(1.235, 2.214),
        },
    ),
    (2018, 3): ExamStats(
        m_p2=27.722, dp_p2=14.027, m_red=6.782, dp_red=1.738,
        parte_1={
            "inglesa": ValorLingua(5.035, 2.358),
            "francesa": ValorLingua(2.432, 2.028),
            "espanhola": ValorLingua(3.671, 1.823),
        },
    ),
    (2019, 1): ExamStats(
        m_p2=26.738, dp_p2=13.911, m_red=6.617, dp_red=2.393,
        parte_1={
            "inglesa": ValorLingua(3.900, 2.781),
            "francesa": ValorLingua(5.064, 2.756),
            "espanhola": ValorLingua(4.450, 2.437),
        },
    ),
    (2019, 2): ExamStats(
        m_p2=25.080, dp_p2=12.635, m_red=6.808, dp_red=1.758,
        parte_1={
            "inglesa": ValorLingua(4.259, 2.407),
            "francesa": ValorLingua(4.263, 3.136),
            "espanhola": ValorLingua(3.962, 2.133),
        },
    ),
    (2019, 3): ExamStats(
        m_p2=24.313, dp_p2=11.511, m_red=6.984, dp_red=1.775,
        parte_1={
            "inglesa": ValorLingua(3.581, 2.095),
            "francesa": ValorLingua(2.445, 1.520),
            "espanhola": ValorLingua(2.700, 1.693),
        },
    ),
    (2020, 1): ExamStats(
        m_p2=24.520, dp_p2=13.344, m_red=5.720, dp_red=2.637,
        parte_1={
            "inglesa": ValorLingua(1.843, 2.228),
            "francesa": ValorLingua(4.805, 2.438),
            "espanhola": ValorLingua(3.745, 2.581),
        },
    ),
    (2020, 2): ExamStats(
        m_p2=28.736, dp_p2=12.864, m_red=7.008, dp_red=1.902,
        parte_1={
            "inglesa": ValorLingua(4.400, 2.644),
            "francesa": ValorLingua(5.118, 1.969),
            "espanhola": ValorLingua(4.638, 2.098),
        },
    ),
    (2020, 3): ExamStats(
        m_p2=27.816, dp_p2=12.814, m_red=6.928, dp_red=1.786,
        parte_1={
            "inglesa": ValorLingua(4.598, 2.160),
            "francesa": ValorLingua(3.546, 2.184),
            "espanhola": ValorLingua(3.115, 1.706),
        },
    ),
    (2021, 1): ExamStats(
        m_p2=21.501, dp_p2=12.422, m_red=5.941, dp_red=2.914,
        parte_1={
            "inglesa": ValorLingua(3.819, 3.274),
            "francesa": ValorLingua(3.302, 2.476),
            "espanhola": ValorLingua(6.053, 2.662),
        },
    ),
    (2021, 2): ExamStats(
        m_p2=25.083, dp_p2=11.897, m_red=7.090, dp_red=1.848,
        parte_1={
            "inglesa": ValorLingua(2.682, 1.871),
            "francesa": ValorLingua(6.354, 1.894),
            "espanhola": ValorLingua(4.501, 2.223),
        },
    ),
    (2021, 3): ExamStats(
        m_p2=23.424, dp_p2=12.300, m_red=6.988, dp_red=1.947,
        parte_1={
            "inglesa": ValorLingua(3.198, 1.760),
            "francesa": ValorLingua(4.060, 1.942),
            "espanhola": ValorLingua(3.356, 1.815),
        },
    ),
    (2022, 1): ExamStats(
        m_p2=20.406, dp_p2=13.533, m_red=5.849, dp_red=2.793,
        parte_1={
            "inglesa": ValorLingua(3.665, 3.109),
            "francesa": ValorLingua(3.620, 2.597),
            "espanhola": ValorLingua(3.140, 2.530),
        },
    ),
    (2022, 2): ExamStats(
        m_p2=21.884, dp_p2=11.761, m_red=7.477, dp_red=1.655,
        parte_1={
            "inglesa": ValorLingua(5.515, 2.536),
            "francesa": ValorLingua(4.221, 2.397),
            "espanhola": ValorLingua(3.499, 2.432),
        },
    ),
    (2022, 3): ExamStats(
        m_p2=26.065, dp_p2=13.126, m_red=7.456, dp_red=1.760,
        parte_1={
            "inglesa": ValorLingua(3.589, 1.795),
            "francesa": ValorLingua(3.928, 1.965),
            "espanhola": ValorLingua(2.832, 1.846),
        },
    ),
    (2023, 1): ExamStats(
        m_p2=25.333, dp_p2=14.686, m_red=6.076, dp_red=2.816,
        parte_1={
            "inglesa": ValorLingua(2.700, 2.821),
            "francesa": ValorLingua(2.212, 2.601),
            "espanhola": ValorLingua(2.116, 2.107),
        },
    ),
    (2023, 2): ExamStats(
        m_p2=29.980, dp_p2=13.213, m_red=6.909, dp_red=1.973,
        parte_1={
            "inglesa": ValorLingua(3.958, 2.186),
            "francesa": ValorLingua(3.260, 2.405),
            "espanhola": ValorLingua(3.147, 2.253),
        },
    ),
    (2023, 3): ExamStats(
        m_p2=26.898, dp_p2=12.861, m_red=6.864, dp_red=1.989,
        parte_1={
            "inglesa": ValorLingua(3.989, 2.049),
            "francesa": ValorLingua(3.499, 1.877),
            "espanhola": ValorLingua(3.545, 1.712),
        },
    ),
    (2024, 2): ExamStats(
        m_p2=29.275, dp_p2=14.604, m_red=6.877, dp_red=2.005,
        parte_1={
            "inglesa": ValorLingua(5.095, 2.899),
            "francesa": ValorLingua(3.493, 2.658),
            "espanhola": ValorLingua(1.231, 2.191),
        },
    ),
    (2024, 3): ExamStats(
        m_p2=31.740, dp_p2=14.063, m_red=7.548, dp_red=1.739,
        parte_1={
            "inglesa": ValorLingua(4.297, 2.137),
            "francesa": ValorLingua(3.556, 1.683),
            "espanhola": ValorLingua(2.537, 1.724),
        },
    ),
    (2025, 3): ExamStats(
        m_p2=30.675, dp_p2=13.752, m_red=7.130, dp_red=1.789,
        parte_1={
            "inglesa": ValorLingua(4.104, 2.174),
            "francesa": ValorLingua(4.630, 2.316),
            "espanhola": ValorLingua(4.218, 1.839),
        },
    ),
}
