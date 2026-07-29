"""Materializa o dataset de treino canônico do PAS 3 a partir do `resultado_final.csv`.

Regras herdadas dos tickets 01, 02, 04 e 14 do mapa `treino-modelos-pas3`
(`.scratch/treino-modelos-pas3/relatorios/`):

- inclusão de linha: `checksum_fecha == True` (ticket 01) — filtro único, sem excluir por
  `campos_formato_invalido`;
- alvo canônico: `A3`, o Argumento da Etapa 3 (ticket 04); `A1` e `A2` são aritmética exata e
  entram como features;
- `etapa_1_ausente` é uma classe do produto, nunca um descarte (ticket 14);
- `nome` e `inscricao` são PII e não saem desta função — só um hash de `inscricao` sobrevive,
  para permitir agrupar aluno repetido entre triênios sem carregar o número original.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd  # type: ignore

from .argument_calculator import HistoricalStats, calculate_argument_etapa
from .pas_constants import OFFICIAL_STATS

REQUIRED_SOURCE_COLUMNS = [
    "campus",
    "curso",
    "turno",
    "inscricao",
    "eb_p1_e1",
    "eb_p2_e1",
    "red_e1",
    "eb_p1_e2",
    "eb_p2_e2",
    "red_e2",
    "eb_p1_e3",
    "eb_p2_e3",
    "red_e3",
    "argumento_final",
    "lingua_e1",
    "lingua_e2",
    "lingua_e3",
    "lingua_ambigua",
    "trienio",
    "checksum_fecha",
]

# Tolerância do checksum (ticket 01) mais a folga de 3 arredondamentos independentes
# (A1, A2, A3 a 3 casas, pesados por 1/2/3) somada ao original. Ver relatório 05, seção de
# verificação de escala.
_TOLERANCIA_RECOMPOSICAO = 0.01


def anos_do_trienio(trienio: str) -> tuple[int, int, int]:
    """`"2023/2025"` (ou `"2023-2025"`) → `(2023, 2024, 2025)`, um ano por Etapa.

    Os dois separadores porque a API fala com hífen e o `resultado_final.csv` com barra, e o
    triênio é o mesmo objeto de domínio nos dois lados — duas funções de parse seriam o começo
    de dois formatos.
    """
    partes = trienio.replace("/", "-").split("-")
    if len(partes) != 2 or not all(p.strip().isdigit() for p in partes):
        raise ValueError(
            f"Triênio {trienio!r} malformado — esperado 'AAAA-AAAA' ou 'AAAA/AAAA'."
        )
    ano_e1 = int(partes[0])
    return ano_e1, ano_e1 + 1, int(partes[1])


def etapa_1_ausente(nota_p1, nota_p2, nota_red):
    """A regra do ADR-0008: as três notas da Etapa 1 iguais a zero.

    Função e não expressão solta porque o **runtime** precisa dela tanto quanto o treino
    (`model_package`): sem a detecção, o modelo lê *"fez a prova e tirou zero em tudo"* e erra a
    previsão, não só a largura. Duas cópias da regra é o começo de duas regras.

    Serve escalar e `Series` — o treino chama vetorizado, o runtime chama com um Aluno.
    """
    return (nota_p1 == 0) & (nota_p2 == 0) & (nota_red == 0)


def _stats_por_ano_etapa_lingua() -> dict[tuple[int, int, str], HistoricalStats]:
    """Achata `OFFICIAL_STATS` para `(ano, etapa, lingua) -> HistoricalStats`.

    Só reformata o que já existe em `pas_constants`; o cálculo em si continua sendo
    `calculate_argument_etapa`, para não duplicar a fórmula do Argumento de Etapa.
    """
    cache: dict[tuple[int, int, str], HistoricalStats] = {}
    for (ano, etapa), stats in OFFICIAL_STATS.items():
        for lingua, valor in stats.parte_1.items():
            cache[(ano, etapa, lingua)] = HistoricalStats(
                mean_p1=valor.media,
                std_p1=valor.desvio_padrao,
                mean_p2=stats.m_p2,
                std_p2=stats.dp_p2,
                mean_red=stats.m_red,
                std_red=stats.dp_red,
            )
    return cache


_STATS_POR_ANO_ETAPA_LINGUA = _stats_por_ano_etapa_lingua()


class EstatisticaOficialAusenteError(KeyError):
    """O Edital daquela `(Ano, Etapa)` ainda não foi extraído para o `OFFICIAL_STATS`.

    Erro próprio, e não `KeyError` cru, porque em produção esta é uma condição **esperada**: o
    triênio vivo sempre tem Etapas cujo Edital de média e desvio ainda não saiu, e quem chama
    precisa distinguir "falta dado deste ano" de "chave escrita errado".
    """


def stats_da_prova(ano: int, etapa: int, lingua: str) -> HistoricalStats:
    """Média e desvio oficiais de uma `(Ano, Etapa, língua estrangeira)`.

    Ponto único de leitura do `OFFICIAL_STATS` para quem vai calcular um Argumento de Etapa —
    treino e runtime pela mesma porta, senão o `A1` que a API mostra deixa de ser o `A1` com que
    o modelo foi treinado.
    """
    try:
        return _STATS_POR_ANO_ETAPA_LINGUA[(ano, etapa, lingua)]
    except KeyError:
        raise EstatisticaOficialAusenteError(
            f"OFFICIAL_STATS não cobre (ano={ano}, etapa={etapa}, língua={lingua!r})."
        ) from None


def _pseudonimizar(inscricao: pd.Series) -> pd.Series:
    """Hash de mão única de `inscricao`, sem sal, para agrupar aluno repetido sem guardar PII.

    Determinístico de propósito (mesma inscrição -> mesmo id em qualquer rodada), o que é o que
    o dataset precisa para o split do ticket 06. Sem sal ele é teoricamente reversível por força
    bruta (inscrição é um número de ~8 dígitos), mas nada no dataset publica o hash junto de
    campus+curso+data que permitiria a um atacante montar essa força bruta com alvo — o mesmo
    nível de exposição de um índice de linha. Ver relatório 05 para a decisão completa.
    """
    return inscricao.astype(str).map(
        lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest()[:16]
    )


def _calcular_argumentos_etapa(df: pd.DataFrame, etapa: int, ano_col: str) -> pd.Series:
    """Argumento de uma Etapa por linha, via `calculate_argument_etapa` — a mesma função que
    `argument_calculator.calculate_argument_final` usa, não uma reimplementação vetorizada."""
    anos = df[ano_col]
    linguas = df[f"lingua_e{etapa}"]

    faltando = sorted(
        {
            (ano, lingua)
            for ano, lingua in zip(anos, linguas)
            if (ano, etapa, lingua) not in _STATS_POR_ANO_ETAPA_LINGUA
        }
    )
    if faltando:
        raise ValueError(
            f"OFFICIAL_STATS não cobre (ano, língua) para a etapa {etapa}: {faltando}"
        )

    valores = [
        calculate_argument_etapa(
            nota_p1, nota_p2, nota_red, _STATS_POR_ANO_ETAPA_LINGUA[(ano, etapa, lingua)]
        )
        for nota_p1, nota_p2, nota_red, ano, lingua in zip(
            df[f"eb_p1_e{etapa}"], df[f"eb_p2_e{etapa}"], df[f"red_e{etapa}"], anos, linguas
        )
    ]
    return pd.Series(valores, index=df.index).round(3)


def build_training_dataset(source: pd.DataFrame) -> pd.DataFrame:
    """Aplica a regra de inclusão, calcula A1/A2/A3 e remove PII.

    `source` já deve ter passado por `REQUIRED_SOURCE_COLUMNS`. Não lê nem escreve arquivo —
    quem faz isso é `load_and_build` / `scripts/build_training_dataset.py`.
    """
    df = source[source["checksum_fecha"] == True].copy()  # noqa: E712

    anos = df["trienio"].map(anos_do_trienio)
    df["_ano_e1"] = anos.map(lambda t: t[0])
    df["_ano_e2"] = anos.map(lambda t: t[1])
    df["_ano_e3"] = anos.map(lambda t: t[2])

    df["etapa_1_ausente"] = etapa_1_ausente(
        df["eb_p1_e1"], df["eb_p2_e1"], df["red_e1"]
    )

    repetidas = df["inscricao"].value_counts()
    df["inscricao_repetida_entre_trienios"] = df["inscricao"].map(
        lambda insc: repetidas[insc] > 1
    )

    df["id_pseudonimo"] = _pseudonimizar(df["inscricao"])

    df["a1"] = _calcular_argumentos_etapa(df, 1, "_ano_e1")
    df["a2"] = _calcular_argumentos_etapa(df, 2, "_ano_e2")
    df["a3"] = _calcular_argumentos_etapa(df, 3, "_ano_e3")

    af_derivado = (1 * df["a1"] + 2 * df["a2"] + 3 * df["a3"]).round(3)
    delta = (af_derivado - df["argumento_final"]).abs()
    inconsistentes = delta > _TOLERANCIA_RECOMPOSICAO
    if inconsistentes.any():
        raise ValueError(
            f"{int(inconsistentes.sum())} linha(s) com Argumento Final recomposto fora da "
            f"tolerância de {_TOLERANCIA_RECOMPOSICAO} — a língua gravada não bate com o "
            "checksum. Investigar antes de treinar."
        )

    df["eb_pas1"] = (df["eb_p1_e1"] + df["eb_p2_e1"]).round(3)
    df["eb_pas2"] = (df["eb_p1_e2"] + df["eb_p2_e2"]).round(3)
    df["eb_pas3"] = (df["eb_p1_e3"] + df["eb_p2_e3"]).round(3)

    colunas_finais = [
        "id_pseudonimo",
        "campus",
        "curso",
        "turno",
        "trienio",
        "etapa_1_ausente",
        "inscricao_repetida_entre_trienios",
        "eb_p1_e1",
        "eb_p2_e1",
        "red_e1",
        "lingua_e1",
        "eb_p1_e2",
        "eb_p2_e2",
        "red_e2",
        "lingua_e2",
        "eb_p1_e3",
        "eb_p2_e3",
        "red_e3",
        "lingua_e3",
        "lingua_ambigua",
        "eb_pas1",
        "eb_pas2",
        "eb_pas3",
        "a1",
        "a2",
        "a3",
        "argumento_final",
    ]
    return df[colunas_finais].reset_index(drop=True)


def load_and_build(source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv, usecols=REQUIRED_SOURCE_COLUMNS)
    return build_training_dataset(df)
