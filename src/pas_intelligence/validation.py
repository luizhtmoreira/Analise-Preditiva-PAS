"""A régua deste projeto: validação deslizante com holdout lacrado.

Especificação: ticket 06 do mapa `treino-modelos-pas3`
(`.scratch/treino-modelos-pas3/relatorios/06-esquema-de-validacao.md`, §§2.2, 4, 5, 7 e 8) e
[ADR-0010](../../docs/adr/0010-validacao-deslizante-com-holdout-lacrado.md).

**O módulo dirige o laço.** Ele recebe uma fábrica de modelo e faz tudo — percorre as dobras,
treina, prevê, mede e monta o resultado. A alternativa (entregar as dobras e deixar cada ticket
escrever o próprio laço) foi descartada porque cinco tickets escrevendo o mesmo laço escrevem
cinco laços ligeiramente diferentes: um esquece de segurar a janela, outro compara dobras
cruzadas, outro emite o número da classe minoritária numa dobra que não o sustenta. As travas da
§2.2 só são travas se existirem em um lugar só.

Consequência de graça: **o lacre fica mecânico**. `gerar_dobras` nunca produz o triênio
2023/2025; não existe caminho acidental até ele, só `holdout_final_use_uma_vez`, que o ticket 13
é o único autorizado a chamar. Fiscalização é `git grep` do nome, não honra.

Privacidade: nada aqui lê arquivo nem escreve arquivo. `ResultadoValidacao.previsoes` é uma
tabela em memória com `id_pseudonimo` (nunca `nome` nem `inscricao`), e relatórios citam
agregado, nunca linha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

TRIENIO_LACRADO = "2023/2025"
"""O holdout lacrado. Não entra em treino nenhum e é lido uma única vez, no ticket 13."""

ALVO_CANONICO = "a3"
"""Ticket 04: o modelo prevê `A3`, e todo o resto sai dele por aritmética."""

COLUNA_CLASSE = "etapa_1_ausente"
"""Ticket 14: a classe é do produto, não um descarte — todo candidato reporta dois números."""

_MINIMO_TRIENIOS_DE_TREINO = 2
"""A dobra 1 treina em dois triênios (relatório 06 §1). Abaixo disso não há dobra."""

# Colunas que *são* a resposta, ou aritmética exata dela: `argumento_final = a1 + 2·a2 + 3·a3`.
# `lingua_e3` fica de fora desta lista de propósito — o Aluno informa a própria língua (ticket
# 04), então ela é conhecida antes da prova acontecer, ao contrário das notas.
_COLUNAS_DA_ETAPA_3 = frozenset(
    {"a3", "argumento_final", "eb_pas3", "eb_p1_e3", "eb_p2_e3", "red_e3"}
)


class Modelo(Protocol):
    """O contrato mínimo — `fit`/`predict` de estimador scikit-learn.

    `X` é um `DataFrame` **com nomes de coluna**, não um array de posições. Isso não é
    formalidade: passar as features na ordem errada para um array não gera erro, gera número
    errado, e foi exatamente o que invalidou o ADR-0007 (`scripts/baseline_avaliacao.py:55`).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


FabricaDeModelo = Callable[[int], Modelo]
"""Função que devolve um modelo **novo, ainda não treinado**, a cada chamada, recebendo a semente.

Fábrica e não modelo pronto porque, com um modelo já criado, a dobra 2 treinaria em cima do que a
dobra 1 já treinou — o erro clássico e silencioso desse tipo de laço, que não quebra nada e só
produz números bons demais.
"""


# ─── Dobras ─────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Dobra:
    """Um par *"treinei com estes triênios, prevejo aquele"*."""

    indice: int
    trienios_treino: tuple[str, ...]
    trienio_teste: str


def _trienios_ordenados(trienios: Iterable[str]) -> tuple[str, ...]:
    ordenados = tuple(sorted(set(trienios)))
    posteriores = [t for t in ordenados if t > TRIENIO_LACRADO]
    if posteriores:
        raise ValueError(
            f"Triênio posterior ao lacre encontrado: {posteriores}. A régua foi escrita para "
            f"{TRIENIO_LACRADO} como último triênio; um triênio novo muda qual ano é o holdout "
            "lacrado, e essa é uma decisão de produto (ver ADR-0010), não algo que este módulo "
            "possa adivinhar."
        )
    return ordenados


def gerar_dobras(trienios: Iterable[str], *, janela: int | None = None) -> tuple[Dobra, ...]:
    """As dobras da validação deslizante — treina no passado, prevê o triênio seguinte.

    `janela` é o número máximo de triênios de treino, contados do mais recente para trás; `None`
    é a janela expansiva (treina em tudo o que houver antes do teste), que é o padrão porque
    imita a produção. A janela é **parâmetro**, não constante da régua: é exatamente o que o
    ticket 08 varre. Ela trunca só o treino — os triênios de teste são os mesmos para qualquer
    janela, senão o ticket 08 estaria comparando dobras diferentes entre si em vez de janelas.

    O triênio lacrado nunca aparece, nem como treino nem como teste.
    """
    disponiveis = tuple(t for t in _trienios_ordenados(trienios) if t != TRIENIO_LACRADO)
    if len(disponiveis) <= _MINIMO_TRIENIOS_DE_TREINO:
        raise ValueError(
            f"São necessários pelo menos {_MINIMO_TRIENIOS_DE_TREINO + 1} triênios fora do "
            f"lacre para formar uma dobra; recebi {len(disponiveis)}."
        )

    dobras = []
    for posicao in range(_MINIMO_TRIENIOS_DE_TREINO, len(disponiveis)):
        treino = disponiveis[:posicao]
        if janela is not None:
            treino = treino[-janela:]
        dobras.append(
            Dobra(
                indice=len(dobras) + 1,
                trienios_treino=treino,
                trienio_teste=disponiveis[posicao],
            )
        )
    return tuple(dobras)


def holdout_final_use_uma_vez(trienios: Iterable[str]) -> Dobra:
    """⚠ O lacre. Treina em tudo até 2022/2024 e prevê 2023/2025.

    **O ticket 13 é o único lugar do repositório autorizado a chamar esta função**, e uma única
    vez. Abrir o lacre produz **um número, não uma decisão**: ou promove com ele, ou desiste da
    rodada. Reajustar o modelo depois de ver o resultado transforma o triênio em mais uma dobra,
    e aí não sobra nenhum ano limpo até o Edital de 2027.

    O nome é constrangedor de propósito — ele existe para aparecer num `git grep`.
    """
    ordenados = _trienios_ordenados(trienios)
    if TRIENIO_LACRADO not in ordenados:
        raise ValueError(f"O triênio lacrado {TRIENIO_LACRADO} não está nos dados.")
    return Dobra(
        indice=len(ordenados) - _MINIMO_TRIENIOS_DE_TREINO,
        trienios_treino=tuple(t for t in ordenados if t != TRIENIO_LACRADO),
        trienio_teste=TRIENIO_LACRADO,
    )


# ─── Erro ───────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Erro:
    """As três métricas de regressão da §4, medidas na escala de `A3`.

    Papéis, que importam mais que as métricas (relatório 06 §4): o **RMSE decide** o ranking
    entre modelos, porque é ele que o `statistics.py` enfia direto em `N(previsão, σ²)`; o
    **MAE** é o número que se fala com um humano; o **viés** existe para validar que o RMSE é um
    σ honesto, já que `RMSE² = viés² + variância` e só com viés zero o RMSE é dispersão pura.
    """

    n: int
    rmse: float
    mae: float
    vies: float

    @staticmethod
    def medir(real: np.ndarray, previsto: np.ndarray) -> "Erro":
        residuo = np.asarray(previsto, dtype=float) - np.asarray(real, dtype=float)
        return Erro(
            n=int(residuo.size),
            rmse=float(np.sqrt(np.mean(np.square(residuo)))),
            mae=float(np.mean(np.abs(residuo))),
            vies=float(np.mean(residuo)),
        )

    @property
    def razao_rmse_mae(self) -> float:
        """Teste de normalidade de graça: para erro bem-comportado dá ~1,25.

        Bem acima disso significa que existe uma minoria sendo massacrada e a média esconde —
        pergunta que o ticket 11 precisa fazer antes de escolher a forma da distribuição.
        """
        return self.rmse / self.mae if self.mae else float("nan")

    @property
    def em_argumento_final(self) -> "Erro":
        """O mesmo erro na escala do Argumento Final, que é exatamente `3×` (ticket 04).

        Existe para que relatórios escrevam os dois lados sem ninguém multiplicar por 3 na mão.
        """
        return Erro(n=self.n, rmse=self.rmse * 3, mae=self.mae * 3, vies=self.vies * 3)


@dataclass(frozen=True)
class ErroDeClasse:
    """O erro de uma classe numa dobra, junto do contexto que a **trava 3** da §2.2 exige.

    `n_treino_classe` e `taxa_treino_classe` são campos obrigatórios ao lado do erro, não rodapé:
    o número da classe minoritária melhora ao longo das dobras porque o treino fica mais rico na
    classe, e sem esses dois campos isso parece deriva do mundo.
    """

    n_treino_classe: int
    taxa_treino_classe: float
    n_teste_classe: int
    taxa_teste_classe: float
    erro: Erro | None
    motivo_ausencia: str | None = None


@dataclass(frozen=True)
class ResultadoDobra:
    dobra: Dobra
    n_treino: int
    n_teste: int
    geral: Erro
    majoritaria: ErroDeClasse
    minoritaria: ErroDeClasse


@dataclass(frozen=True)
class ResultadoValidacao:
    """O resultado de uma rodada da régua — estrutura tipada, não dicionário.

    Tipada para que `resultado.rmse_agrupado` escrito errado vire `AttributeError` na hora, em
    vez de devolver vazio e alguém reportar um número que não existe.

    **Trava 2 da §2.2:** existe `serie_majoritaria` e não existe `serie_minoritaria`. Para a
    classe minoritária a saída canônica é **um número só** (`agrupado_minoritaria`), porque um
    número não tem tendência para alguém ler errado. As cinco dobras continuam alcançáveis por
    `dobras[i].minoritaria`, para depuração — mas quem quiser plotar a série da minoria tem que
    montá-la à mão, e nesse momento já sabe o que está fazendo.
    """

    nome: str
    semente: int
    features: tuple[str, ...]
    alvo: str
    janela: int | None
    dobras: tuple[ResultadoDobra, ...]
    previsoes: pd.DataFrame = field(repr=False)

    @property
    def agrupado(self) -> Erro:
        """O número único do modelo: erro sobre **todas** as linhas de teste das dobras.

        Agrupado e não média dos cinco, porque a média daria à dobra de 5.804 Alunos o mesmo peso
        que à de 8.499.
        """
        return Erro.medir(self.previsoes["real"].to_numpy(), self.previsoes["previsto"].to_numpy())

    @property
    def agrupado_majoritaria(self) -> Erro:
        return self._agrupado_da_classe(minoritaria=False)

    @property
    def agrupado_minoritaria(self) -> Erro | None:
        """Agrupado só sobre as dobras que qualificam na **trava 1** — pode não existir."""
        return self._agrupado_da_classe(minoritaria=True)

    @property
    def serie_majoritaria(self) -> tuple[Erro, ...]:
        """A série por dobra da classe majoritária — legítima, e é o insumo do ticket 08."""
        return tuple(d.majoritaria.erro for d in self.dobras if d.majoritaria.erro is not None)

    def _agrupado_da_classe(self, *, minoritaria: bool) -> Erro | None:
        indices_validos = [
            d.dobra.indice
            for d in self.dobras
            if (d.minoritaria if minoritaria else d.majoritaria).erro is not None
        ]
        linhas = self.previsoes[
            (self.previsoes[COLUNA_CLASSE] == minoritaria)
            & (self.previsoes["dobra"].isin(indices_validos))
        ]
        if linhas.empty:
            return None
        return Erro.medir(linhas["real"].to_numpy(), linhas["previsto"].to_numpy())


# ─── O laço ─────────────────────────────────────────────────────────────────────────────────


def _conferir_features(dataset: pd.DataFrame, features: Sequence[str], alvo: str) -> None:
    faltando = [c for c in (*features, alvo, "trienio", COLUNA_CLASSE) if c not in dataset.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no dataset: {faltando}")

    resposta = sorted(set(features) & (_COLUNAS_DA_ETAPA_3 | {alvo}))
    if resposta:
        raise ValueError(
            f"Feature da Etapa 3 recusada: {resposta}. Essas colunas são a resposta, ou "
            "aritmética exata dela — usá-las como entrada mede memória, não previsão."
        )


def _medir_classe(
    treino: pd.DataFrame, teste: pd.DataFrame, residuos: pd.DataFrame, *, minoritaria: bool
) -> ErroDeClasse:
    treino_classe = treino[treino[COLUNA_CLASSE] == minoritaria]
    teste_classe = teste[teste[COLUNA_CLASSE] == minoritaria]
    linhas = residuos[residuos[COLUNA_CLASSE] == minoritaria]

    n_treino = len(treino_classe)
    n_teste = len(teste_classe)
    contexto = dict(
        n_treino_classe=n_treino,
        taxa_treino_classe=n_treino / len(treino) if len(treino) else 0.0,
        n_teste_classe=n_teste,
        taxa_teste_classe=n_teste / len(teste) if len(teste) else 0.0,
    )

    # Trava 1 da §2.2: o número não existe onde não pode existir. Pedir generalização para mais
    # casos da classe do que o modelo jamais viu não é medição — quem plotar a série recebe um
    # buraco, não um ponto baixo. Vale para as duas classes; hoje só a minoritária dispara.
    if n_teste == 0:
        return ErroDeClasse(**contexto, erro=None, motivo_ausencia="classe ausente no teste")
    if n_treino < n_teste:
        return ErroDeClasse(
            **contexto,
            erro=None,
            motivo_ausencia=(
                f"treino mais pobre na classe que o teste ({n_treino} contra {n_teste})"
            ),
        )
    return ErroDeClasse(
        **contexto,
        erro=Erro.medir(linhas["real"].to_numpy(), linhas["previsto"].to_numpy()),
    )


def avaliar(
    dataset: pd.DataFrame,
    fabrica: FabricaDeModelo,
    *,
    nome: str,
    features: Sequence[str],
    semente: int,
    alvo: str = ALVO_CANONICO,
    janela: int | None = None,
) -> ResultadoValidacao:
    """Roda a régua inteira sobre `dataset` e devolve o resultado tipado.

    `semente` é obrigatória e fica registrada no resultado: ao contrário do ticket 05, aqui
    existe passo aleatório — o modelo. Nenhum número deste módulo entra num relatório sem o
    recorte que o produziu (quais linhas, qual dobra, qual semente).

    O lacre não é alcançável por aqui em hipótese nenhuma; ver `holdout_final_use_uma_vez`.
    """
    features = tuple(features)
    _conferir_features(dataset, features, alvo)

    dobras = gerar_dobras(dataset["trienio"].unique(), janela=janela)
    resultados: list[ResultadoDobra] = []
    previsoes: list[pd.DataFrame] = []
    ja_criados: list[Modelo] = []

    colunas_de_identidade = [
        c for c in ("id_pseudonimo", "a1", "a2") if c in dataset.columns
    ]

    for dobra in dobras:
        treino = dataset[dataset["trienio"].isin(dobra.trienios_treino)]
        teste = dataset[dataset["trienio"] == dobra.trienio_teste]

        modelo = fabrica(semente)
        if any(modelo is anterior for anterior in ja_criados):
            raise ValueError(
                "A fábrica devolveu um modelo já usado numa dobra anterior. Ela precisa "
                "construir um modelo novo a cada chamada — senão a dobra seguinte treina em "
                "cima do que a anterior já treinou, e o erro sai bom demais sem nada quebrar."
            )
        ja_criados.append(modelo)

        modelo.fit(treino[list(features)], treino[alvo])
        previsto = np.asarray(modelo.predict(teste[list(features)]), dtype=float)

        residuos = teste[[*colunas_de_identidade, "trienio", COLUNA_CLASSE]].copy()
        residuos["dobra"] = dobra.indice
        residuos["real"] = teste[alvo].to_numpy(dtype=float)
        residuos["previsto"] = previsto
        previsoes.append(residuos)

        resultados.append(
            ResultadoDobra(
                dobra=dobra,
                n_treino=len(treino),
                n_teste=len(teste),
                geral=Erro.medir(residuos["real"].to_numpy(), previsto),
                majoritaria=_medir_classe(treino, teste, residuos, minoritaria=False),
                minoritaria=_medir_classe(treino, teste, residuos, minoritaria=True),
            )
        )

    return ResultadoValidacao(
        nome=nome,
        semente=semente,
        features=features,
        alvo=alvo,
        janela=janela,
        dobras=tuple(resultados),
        previsoes=pd.concat(previsoes, ignore_index=True),
    )


# ─── Erro de decisão ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResultadoDecisao:
    """A quarta métrica: *"em X% dos Alunos o sistema teria dito a coisa errada sobre passar"*.

    **Veto conversado, não ranking** (§4.4): depende de uma tabela de corte que ainda vai mudar
    (tickets 14/15 do `pdf-extraction`), e um sim/não joga fora quase toda a informação das
    previsões. Um modelo que ganha no RMSE e piora feio aqui não troca automaticamente — abre
    discussão.
    """

    n: int
    n_sem_corte: int
    taxa_erro: float
    largura_faixa: float
    n_na_faixa: int
    taxa_erro_na_faixa: float
    erro_na_faixa: Erro


def faixa_de_decisao(baseline: ResultadoValidacao) -> float:
    """A largura da faixa de decisão: **um RMSE de Argumento Final do baseline do ticket 07**.

    Medida uma vez e **congelada**. Assim a faixa é exatamente "onde o erro do modelo é capaz de
    virar a resposta", tem motivo em vez de gosto, e é a mesma para todos os modelos comparados.
    Faixa por modelo tornaria a métrica auto-referente e incapaz de melhorar por construção.
    """
    return baseline.agrupado.em_argumento_final.rmse


def erro_de_decisao(previsoes: pd.DataFrame, *, largura_faixa: float) -> ResultadoDecisao:
    """Quantos Alunos mudam de lado do corte entre o Argumento Final real e o previsto.

    `previsoes` é a tabela de `ResultadoValidacao.previsoes` acrescida de uma coluna
    `nota_corte` — o **menor corte entre os sistemas em que o Aluno concorre** (§4.5: quem tem
    cota concorre no Universal *e* nos sistemas da cota dele ao mesmo tempo, e passa se limpar
    qualquer um). Linhas sem corte casado ficam de fora e são contadas em `n_sem_corte`, nunca
    silenciadas: a métrica não vale medindo um terço da base.

    O casamento com a tabela de cortes fica com quem chama, de propósito — ele depende de
    `perfil_cota`, das exclusões da §4.5 e do casamento de nome de curso, e nada disso é a régua.

    `taxa_erro` sozinha mente (um Aluno errado por 0,5 ponto e um errado por 30 contam igual), e
    por isso ela nunca sai sem a companheira: o erro medido **dentro da faixa de decisão**, entre
    os Alunos cujo Argumento Final real está a menos de `largura_faixa` do corte deles.
    """
    if "nota_corte" not in previsoes.columns:
        raise ValueError(
            "Falta a coluna `nota_corte` — o menor corte entre os sistemas do Aluno (§4.5)."
        )

    n_sem_corte = int(previsoes["nota_corte"].isna().sum())
    casados = previsoes[previsoes["nota_corte"].notna()]
    if casados.empty:
        raise ValueError("Nenhuma linha casou com um corte; não há erro de decisão a medir.")

    af_real = casados["a1"] + 2 * casados["a2"] + 3 * casados["real"]
    af_previsto = casados["a1"] + 2 * casados["a2"] + 3 * casados["previsto"]
    errou = (af_real >= casados["nota_corte"]) != (af_previsto >= casados["nota_corte"])

    na_faixa = (af_real - casados["nota_corte"]).abs() <= largura_faixa
    linhas_na_faixa = casados[na_faixa]

    return ResultadoDecisao(
        n=len(casados),
        n_sem_corte=n_sem_corte,
        taxa_erro=float(errou.mean()),
        largura_faixa=largura_faixa,
        n_na_faixa=int(na_faixa.sum()),
        taxa_erro_na_faixa=float(errou[na_faixa].mean()) if na_faixa.any() else float("nan"),
        erro_na_faixa=Erro.medir(
            linhas_na_faixa["real"].to_numpy(), linhas_na_faixa["previsto"].to_numpy()
        ),
    )
