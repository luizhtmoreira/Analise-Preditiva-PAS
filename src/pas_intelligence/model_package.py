"""O pacote de modelo promovido, servindo uma previsão em produção (ticket 13).

Este módulo é a única porta entre `api/` e o artefato treinado. Ele existe para que a resposta da
API seja construída pela **mesma** aritmética que o treino usou — o desencontro entre como uma
feature é montada no treino e como é montada no request (*train/serve skew*) não quebra nada, só
produz número errado, e por isso as funções de `dataset_pas3` são reaproveitadas aqui em vez de
reescritas para uma linha só.

A cadeia, fixada pelo ADR-0009 (ticket 04):

    Â3               ← a única previsão do modelo
    A1, A2           ← aritmética exata sobre as notas que o Aluno digitou
    Argumento Final  = A1 + 2·A2 + 3·Â3
    σ(Arg. Final)    = 3 × σ(A3)          — exato: A1 e A2 têm variância zero

Duas condições fazem este módulo **recusar** em vez de responder por aproximação, e as duas são
propositais (ADR-0012 §6): pacote ausente e Edital de Etapa ainda não extraído. O estado
"previsão sim, largura não" não é representável, e `A1`/`A2` aproximados destruiriam justamente a
parte exata da conta. Quem chama traduz a recusa em `modelo_disponivel: False`.

Privacidade: nada aqui lê ou escreve arquivo de Aluno; a entrada são as seis notas do request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore

from .argument_calculator import calculate_argument_etapa
from .dataset_pas3 import FEATURES_CANONICAS, montar_features
from .pas_constants import LINGUAS_OFICIAIS, Origem
from .training_dataset import (
    EstatisticaOficialAusenteError,
    anos_do_trienio,
    etapa_1_ausente as _detectar_etapa_1_ausente,
    origem_da_prova,
    stats_da_prova,
)

DIRETORIO_PADRAO = Path(__file__).resolve().parent.parent.parent / "models" / "pas3"
"""Onde a promoção deixa o pacote. `models/` fica fora do git (ticket 03).

⚠ **O domicílio do ticket 03 ainda não existe.** A Decisão 3 daquele ticket põe o pacote num
repositório privado no Hugging Face, assado na imagem no build; hoje ele mora só neste diretório
local, e reverter é copiar de volta à mão. Enquanto for assim, uma máquina nova sobe **sem
pacote** até alguém copiar o diretório — ver relatório 13 §8."""

ESCALA_DA_LARGURA = "a3"
"""A Largura de Incerteza é medida em `A3` e vale `3×` em Argumento Final (ADR-0009). Conferida
no carregamento, para que ninguém multiplique duas vezes."""

# `LINGUAS_OFICIAIS` é reexportado de `pas_constants` — a lista mora junto do dado que ela
# indexa, e quem já importava daqui continua importando daqui.


class PacoteIndisponivelError(RuntimeError):
    """Não há pacote carregável — sem previsão e sem largura, nunca uma das duas."""


class EstatisticasIndisponiveisError(RuntimeError):
    """Falta o Edital de média e desvio de uma das Etapas já feitas pelo Aluno.

    Condição esperada no triênio vivo, não defeito: enquanto `(2024, Etapa 1)` e `(2025, Etapa 2)`
    não forem extraídos, o Aluno de 2024-2026 não tem `A1` e `A2` exatos — e a decisão do ticket
    04 é recusar em vez de aproximar.
    """


@dataclass(frozen=True)
class NotasDeEtapa:
    """As três notas que o Aluno tirou numa Etapa, na escala de escore bruto do Edital."""

    p1: float
    p2: float
    redacao: float


@dataclass(frozen=True)
class EntradaDePrevisao:
    """Tudo o que o modelo precisa saber sobre um Aluno — e nada além disso.

    `lingua_e1`/`lingua_e2` estão aqui, e não num default do módulo, porque a Parte 1 é
    normalizada por língua e um default silencioso é exatamente o viés que o ticket 04 §5 mediu.

    São dois campos, não um, porque o Cebraspe registra a língua **por Etapa**: 13,9% da base
    trocam de língua entre a Etapa 1 e a Etapa 2 (majoritariamente inglês → espanhol), e aplicar
    a língua de uma Etapa na estatística da outra normaliza a Parte 1 com a média e o desvio
    errados (defeito 11 de `defeitos-pendentes.md`).
    """

    etapa_1: NotasDeEtapa
    etapa_2: NotasDeEtapa
    lingua_e1: str
    lingua_e2: str
    trienio: str

    @property
    def etapa_1_ausente(self) -> bool:
        return bool(
            _detectar_etapa_1_ausente(self.etapa_1.p1, self.etapa_1.p2, self.etapa_1.redacao)
        )


@dataclass(frozen=True)
class Previsao:
    """O resultado inteiro de um Aluno, com as partes exatas separadas da prevista.

    `a1` e `a2` viajam junto de propósito: são o que permite a quem chama reconstruir qualquer
    número da tela sem chamar o modelo de novo, e são a evidência de que o Argumento Final
    mostrado não é um segundo palpite.
    """

    a1: float
    a2: float
    a3: float
    argumento_final: float
    largura_a3: float
    largura_argumento_final: float
    etapa_1_ausente: bool
    # Ticket 07: `A1` e/ou `A2` deste Aluno vieram de estatística `DERIVADA` (Edital isolado de
    # Etapa corrigido, não o Edital de médias e desvios do Cebraspe). Quando o Edital de verdade
    # sair, `OFFICIAL_STATS` troca de valor e esta previsão muda — por isso a tela precisa saber.
    usa_estatistica_derivada: bool


class PacoteDeModelo:
    """Um pacote (`modelo_pas3.txt` + `manifest.json`) carregado e pronto para responder."""

    def __init__(self, diretorio: Path, booster: Any, manifesto: dict[str, Any]) -> None:
        self.diretorio = diretorio
        self._booster = booster
        self.manifesto = manifesto
        self._incerteza = manifesto["incerteza"]["sigma_por_classe"]

    # ─── Carregamento ──────────────────────────────────────────────────────────────────────

    @classmethod
    def carregar(cls, diretorio: Path = DIRETORIO_PADRAO) -> "PacoteDeModelo":
        """Lê o pacote do disco, conferindo o que precisa ser conferido antes do primeiro request.

        As conferências acontecem no carregamento e não na previsão porque um pacote errado deve
        derrubar o *startup*, não uma resposta de Aluno no meio da tarde.
        """
        caminho_manifesto = diretorio / "manifest.json"
        if not caminho_manifesto.exists():
            raise PacoteIndisponivelError(
                f"Sem manifesto em {diretorio} — nenhum pacote promovido. Rode "
                "`scripts/treinar_pipeline.py` e promova a saída."
            )
        manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
        descricao = manifesto["modelos"][0]

        if list(descricao["features"]) != list(FEATURES_CANONICAS):
            raise PacoteIndisponivelError(
                "As features do manifesto não são as canônicas, na ordem canônica. Ordem trocada "
                "não levanta erro em tempo de previsão — devolve número errado, que foi o que "
                "invalidou o ADR-0007."
            )

        escala = manifesto["incerteza"]["escala"]
        if escala != ESCALA_DA_LARGURA:
            raise PacoteIndisponivelError(
                f"A Largura de Incerteza do manifesto está em {escala!r}, não em "
                f"{ESCALA_DA_LARGURA!r}. Este módulo multiplica por 3 para chegar ao Argumento "
                "Final; com a largura já convertida ele multiplicaria duas vezes — o convite que "
                "o relatório 11 §7.1 levantou ao guardar as duas unidades no mesmo arquivo."
            )

        caminho_modelo = diretorio / descricao["arquivo"]
        if not caminho_modelo.exists():
            raise PacoteIndisponivelError(f"Manifesto aponta para {caminho_modelo}, que não existe.")

        import lightgbm as lgb  # type: ignore

        return cls(diretorio, lgb.Booster(model_file=str(caminho_modelo)), manifesto)

    # ─── Previsão ──────────────────────────────────────────────────────────────────────────

    def _argumentos_exatos(self, entrada: EntradaDePrevisao) -> tuple[float, float, bool]:
        for lingua in (entrada.lingua_e1, entrada.lingua_e2):
            if lingua not in LINGUAS_OFICIAIS:
                raise ValueError(
                    f"língua {lingua!r} não é uma das três oficiais: "
                    f"{', '.join(LINGUAS_OFICIAIS)}."
                )
        ano_e1, ano_e2, _ = anos_do_trienio(entrada.trienio)
        try:
            stats_e1 = stats_da_prova(ano_e1, 1, entrada.lingua_e1)
            stats_e2 = stats_da_prova(ano_e2, 2, entrada.lingua_e2)
            # `stats_da_prova` já confere que a chave existe — `origem_da_prova` só lê a mesma
            # entrada de novo, então não pode levantar aqui.
            usa_estatistica_derivada = (
                origem_da_prova(ano_e1, 1) is Origem.DERIVADA
                or origem_da_prova(ano_e2, 2) is Origem.DERIVADA
            )
        except EstatisticaOficialAusenteError as erro:
            raise EstatisticasIndisponiveisError(str(erro)) from erro

        a1 = calculate_argument_etapa(
            entrada.etapa_1.p1, entrada.etapa_1.p2, entrada.etapa_1.redacao, stats_e1
        )
        a2 = calculate_argument_etapa(
            entrada.etapa_2.p1, entrada.etapa_2.p2, entrada.etapa_2.redacao, stats_e2
        )
        return a1, a2, usa_estatistica_derivada

    def montar_features(self, entrada: EntradaDePrevisao) -> pd.DataFrame:
        """A linha única que vai ao modelo, montada pelas funções do treino.

        Público porque é o que se inspeciona quando uma previsão parece absurda — e porque o
        teste que garante a paridade com o treino precisa vê-la.
        """
        a1, a2, _ = self._argumentos_exatos(entrada)
        return self._linha_de_features(entrada, a1, a2)

    def _linha_de_features(
        self, entrada: EntradaDePrevisao, a1: float, a2: float
    ) -> pd.DataFrame:
        linha = pd.DataFrame(
            [
                {
                    "a1": a1,
                    "a2": a2,
                    "eb_pas1": entrada.etapa_1.p1 + entrada.etapa_1.p2,
                    "red_e1": entrada.etapa_1.redacao,
                    "eb_pas2": entrada.etapa_2.p1 + entrada.etapa_2.p2,
                    "red_e2": entrada.etapa_2.redacao,
                    "etapa_1_ausente": entrada.etapa_1_ausente,
                }
            ]
        )
        return montar_features(linha)[list(FEATURES_CANONICAS)]

    def largura_de_incerteza(self, etapa_1_ausente: bool) -> float:
        """A Largura de Incerteza em `A3`, na classe do Aluno (ADR-0012).

        Vem do manifesto, nunca de constante de código: é o que impede a próxima troca de modelo
        de deixar a probabilidade descrevendo um modelo morto em silêncio, como o `13,49` fazia.
        """
        chave = "sem_etapa_1" if etapa_1_ausente else "com_etapa_1"
        return float(self._incerteza[chave])

    def prever(self, entrada: EntradaDePrevisao) -> Previsao:
        a1, a2, usa_estatistica_derivada = self._argumentos_exatos(entrada)
        a3 = float(self._booster.predict(self._linha_de_features(entrada, a1, a2))[0])
        largura_a3 = self.largura_de_incerteza(entrada.etapa_1_ausente)

        return Previsao(
            a1=a1,
            a2=a2,
            a3=a3,
            argumento_final=a1 + 2 * a2 + 3 * a3,
            largura_a3=largura_a3,
            largura_argumento_final=3 * largura_a3,
            etapa_1_ausente=entrada.etapa_1_ausente,
            usa_estatistica_derivada=usa_estatistica_derivada,
        )
