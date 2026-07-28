"""Ticket 07 — mede os baselines triviais e os `.joblib` atuais sobre a régua do ticket 06.

Produz a tabela de referência única que os tickets 08 a 13 citam. Roda com:

    .venv/bin/python scripts/baseline_honesto.py

Substitui `scripts/baseline_avaliacao.py` (ADR-0007), cuja metodologia — KFold aleatório sobre
`banco_alunos_pas_final.csv`, com o vetor de features na ordem errada — foi invalidada pelo
ticket 06 e pelo relatório 03.

Privacidade: lê `resultado_final.csv` (que tem PII) apenas para o `perfil_cota`, e nada além de
agregado sai daqui.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pas_intelligence.validation import (  # noqa: E402
    Erro,
    ResultadoValidacao,
    avaliar,
    erro_de_decisao,
    gerar_dobras,
)

warnings.filterwarnings("ignore")

SEMENTE = 20260728

DATASET = RAIZ / "data" / "training" / "pas3_dataset.parquet"
BANCO_ANTIGO = RAIZ / "data" / "banco_alunos_pas_final.csv"
MODELOS = RAIZ / "models"

# A ordem real do vetor de features, lida de `booster.feature_name()` dos próprios artefatos —
# não da documentação. `scripts/baseline_avaliacao.py:55` declara outra, e é a causa dos
# `R² = -83` do ADR-0007.
FEATURES_LEGADAS = ["EB_PAS1", "Red_PAS1", "EB_PAS2", "Red_PAS2", "Cresc_EB", "Cresc_Red"]


# ─── Preparo ────────────────────────────────────────────────────────────────────────────────


def carregar_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    # As 6 features com que os `.joblib` atuais foram treinados, reconstruídas com os mesmos
    # nomes que os artefatos carregam dentro de si.
    df["EB_PAS1"] = df["eb_pas1"]
    df["Red_PAS1"] = df["red_e1"]
    df["EB_PAS2"] = df["eb_pas2"]
    df["Red_PAS2"] = df["red_e2"]
    df["Cresc_EB"] = df["eb_pas2"] - df["eb_pas1"]
    df["Cresc_Red"] = df["red_e2"] - df["red_e1"]
    return df


def linhas_de_teste(df: pd.DataFrame) -> pd.DataFrame:
    """As 37.844 linhas de teste das 5 dobras — o mesmo recorte que a régua usa."""
    testes = [d.trienio_teste for d in gerar_dobras(df["trienio"].unique())]
    return df[df["trienio"].isin(testes)].copy()


# ─── Baselines triviais ─────────────────────────────────────────────────────────────────────


class RepeteEtapa2:
    """`A3 = A2`. A Etapa mais recente é frequentemente a melhor previsora."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["a2"].to_numpy(dtype=float)


class MediaDasEtapas:
    """`A3 = (A1 + A2) / 2`."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return ((X["a1"] + X["a2"]) / 2).to_numpy(dtype=float)


class MediaDoTreino:
    """Responde sempre a média do treino — o preditor contra o qual o R² compara."""

    def fit(self, X, y):
        self.media = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.media)


class MediaDoCurso:
    """Média de `A3` do curso no treino, ignorando o aluno. Cai na média global se o curso é novo."""

    def fit(self, X, y):
        tabela = pd.DataFrame({"curso": X["curso"].to_numpy(), "a3": np.asarray(y, dtype=float)})
        self.por_curso = tabela.groupby("curso")["a3"].mean()
        self.media = float(tabela["a3"].mean())
        return self

    def predict(self, X):
        return X["curso"].map(self.por_curso).fillna(self.media).to_numpy(dtype=float)


def regressao_linear(semente: int):
    from sklearn.linear_model import LinearRegression  # type: ignore

    return LinearRegression()


# ─── Modelos congelados (os `.joblib` de hoje) ──────────────────────────────────────────────


def medir_congelado(
    teste: pd.DataFrame, real: np.ndarray, previsto: np.ndarray
) -> dict[str, Erro]:
    """Erro geral e por classe de um artefato já treinado.

    Artefato congelado não passa por `avaliar`: sem treino por dobra, a estrutura de dobras só
    fatiaria as mesmas linhas, e a trava 1 da §2.2 (que compara o treino da classe com o teste)
    não tem sobre o que se apoiar. O recorte é o mesmo — as 37.844 linhas de teste das 5 dobras —,
    então o número é comparável ao `agrupado` da régua.
    """
    minoria = teste["etapa_1_ausente"].to_numpy()
    return {
        "geral": Erro.medir(real, previsto),
        "majoritaria": Erro.medir(real[~minoria], previsto[~minoria]),
        "minoritaria": Erro.medir(real[minoria], previsto[minoria]),
    }


def sobreposicao_com_o_banco_antigo(teste: pd.DataFrame) -> pd.Series:
    """Quais linhas de teste os `.joblib` atuais já viram no treino.

    `banco_alunos_pas_final.csv` guarda `Inscricao` em texto puro; o dataset guarda o
    `id_pseudonimo` (SHA-256 truncado). Reaplicar o mesmo hash é o que permite cruzar os dois
    sem trazer a inscrição de volta.
    """
    antigo = pd.read_csv(BANCO_ANTIGO, usecols=["Inscricao"], low_memory=False)
    vistos = set(
        antigo["Inscricao"]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .map(lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest()[:16])
    )
    return teste["id_pseudonimo"].isin(vistos)


# ─── Nota de Corte por Aluno ────────────────────────────────────────────────────────────────

CORTES = Path(".scratch/pdf-extraction/saida-nova/notas_corte.csv")
RESULTADO_FINAL = Path(".scratch/pdf-extraction/saida-nova/resultado_final.csv")


def menor_corte_por_aluno(df: pd.DataFrame) -> pd.DataFrame:
    """O menor corte entre tudo em que o Aluno concorre — sistemas **e** semestres.

    **Sistemas** (§4.5 do ticket 06): quem tem cota concorre no Universal *e* no sistema da cota
    dele ao mesmo tempo, e passa se limpar qualquer um.

    **Semestres** (regra do PAS, confirmada pelo dono do produto nesta sessão): não existe "em
    qual semestre o Aluno concorreu". Todos fazem a mesma prova e concorrem de uma vez; há um
    limite de vagas para o 1º semestre, e quem não se classifica nele continua vivo disputando as
    vagas do 2º. Logo o corte do 2º é sempre o mais baixo — conferido, **1.317 de 1.317 chaves
    curso+sistema, sem uma exceção** — e "o Aluno entrou" é `Argumento Final ≥ o menor dos dois`.

    Exclusões declaradas: linhas de corte que falham `checksum_fecha` ou estão `parcial`, e
    Alunos com `cota_padrao_suspeito`. A regra dos 3 convocados do §4.5 foi relaxada para 1 por
    decisão do dono do produto nesta sessão: `convocados_com_argumento` vale 1 ou 2 em 68% das
    linhas porque na última chamada de um sistema frequentemente só uma pessoa é convocada — o
    que é a definição literal da Nota de Corte, não ruído. Custo evitado: 34 pontos de cobertura.
    """
    c = pd.read_csv(RAIZ / CORTES)
    c = c[c["checksum_fecha"] & ~c["parcial"] & (c["convocados_com_argumento"] >= 1)]
    menor = (
        c.groupby(["trienio", "campus", "curso", "turno", "sistema_nome"], dropna=False)[
            "nota_corte"
        ]
        .min()
        .reset_index()
    )

    r = pd.read_csv(
        RAIZ / RESULTADO_FINAL,
        usecols=["inscricao", "trienio", "perfil_cota", "cota_padrao_suspeito"],
    )
    r["id_pseudonimo"] = (
        r["inscricao"].astype(str).map(lambda v: hashlib.sha256(v.encode()).hexdigest()[:16])
    )
    r = r[~r["cota_padrao_suspeito"].fillna(False)]

    alunos = df[["id_pseudonimo", "trienio", "campus", "curso", "turno"]].merge(
        r[["id_pseudonimo", "trienio", "perfil_cota"]], on=["id_pseudonimo", "trienio"]
    )

    universal = menor[menor["sistema_nome"] == "Universal"].drop(columns="sistema_nome")
    alunos = alunos.merge(
        universal.rename(columns={"nota_corte": "corte_universal"}),
        on=["trienio", "campus", "curso", "turno"],
        how="left",
    ).merge(
        menor.rename(columns={"sistema_nome": "perfil_cota", "nota_corte": "corte_cota"}),
        on=["trienio", "campus", "curso", "turno", "perfil_cota"],
        how="left",
    )
    alunos["nota_corte"] = alunos[["corte_universal", "corte_cota"]].min(axis=1)
    return alunos[["id_pseudonimo", "trienio", "nota_corte"]]


def detalhar_erro_de_decisao(
    resultado: ResultadoValidacao, cortes: pd.DataFrame, largura: float
) -> None:
    """Onde os erros de decisão moram, e se a forma normal do erro se sustenta.

    Duas coisas que a taxa agregada esconde:

    1. **`7,4%` e `34,3%` são populações diferentes.** O segundo é o recorte dentro da faixa, e
       é difícil de propósito — quem está longe do corte quase nunca troca de lado, porque o erro
       do modelo não tem tamanho para virar o veredito.
    2. **A probabilidade de erro depende só da distância até o corte**, medida em erros do
       modelo. Para erro normal e sem viés, `P(erro | d) = Φ(−|d|/σ)`, com o sinal de `d`
       sumindo pela simetria da normal. Comparar o observado com essa curva é uma verificação da
       forma normal **independente** da razão RMSE/MAE — insumo do ticket 11.
    """
    from scipy.stats import norm  # type: ignore

    p = resultado.previsoes.merge(cortes, on=["id_pseudonimo", "trienio"], how="left")
    p = p[p["nota_corte"].notna()]

    af_real = p["a1"] + 2 * p["a2"] + 3 * p["real"]
    af_previsto = p["a1"] + 2 * p["a2"] + 3 * p["previsto"]
    errou = (af_real >= p["nota_corte"]) != (af_previsto >= p["nota_corte"])
    distancia = (af_real - p["nota_corte"]).abs()
    dentro = distancia <= largura

    print(f"\n— Onde moram os erros de decisão ({resultado.nome}) —")
    for rotulo, mascara in (
        ("todos os casados", pd.Series(True, index=p.index)),
        (f"dentro da faixa (<={largura:.3f})", dentro),
        ("fora da faixa", ~dentro),
    ):
        print(f"    {rotulo:<32} n={int(mascara.sum()):>6}  erros={int(errou[mascara].sum()):>5}"
              f"  {errou[mascara].mean() * 100:>5.1f}%")
    print(f"    → {errou[dentro].sum() / errou.sum() * 100:.0f}% dos erros vêm de dentro da "
          f"faixa, que é {dentro.mean() * 100:.0f}% dos Alunos")

    print("\n— Erro por distância do corte, contra Φ(−|d|/σ) —")
    print(f"    {'|d|/σ':>12} {'n':>6} {'observado':>10} {'previsto':>9}")
    for inicio, fim in ((0, .25), (.25, .5), (.5, .75), (.75, 1.), (1., 1.5), (1.5, 2.)):
        faixa = (distancia / largura >= inicio) & (distancia / largura < fim)
        if not faixa.any():
            continue
        previsto = norm.cdf(-(inicio + fim) / 2)
        print(f"    {inicio:.2f}–{fim:.2f}{'':>3} {int(faixa.sum()):>6} "
              f"{errou[faixa].mean() * 100:>9.1f}% {previsto * 100:>8.1f}%")

    piso = norm.cdf(-1) + norm.pdf(0) - norm.pdf(1)  # ∫₀¹ Φ(−u) du, resolvida por partes
    print(f"\n    piso teórico ∫₀¹Φ(−u)du = Φ(−1) + φ(0) − φ(1) = {piso:.4f} "
          f"({piso * 100:.1f}%), supondo |d| uniforme na faixa")
    primeiro_quarto = ((distancia / largura) < 0.25).sum()
    print(f"    e |d| não é uniforme: {int(primeiro_quarto)} Alunos no primeiro quarto contra "
          f"~{int(dentro.sum() - primeiro_quarto) // 3} em cada um dos outros três")


# ─── Saída ──────────────────────────────────────────────────────────────────────────────────


def linha(nome: str, erro: Erro | None, sufixo: str = "") -> str:
    if erro is None:
        return f"{nome:<38} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {sufixo}"
    return (
        f"{nome:<38} {erro.rmse:>8.3f} {erro.mae:>8.3f} {erro.vies:>+8.3f} "
        f"{erro.razao_rmse_mae:>8.2f} {sufixo}"
    )


CABECALHO = f"{'preditor':<38} {'RMSE':>8} {'MAE':>8} {'viés':>8} {'R/M':>8}"


def main() -> None:
    df = carregar_dataset()
    teste = linhas_de_teste(df)
    print(f"dataset: {len(df)} linhas, {df['trienio'].nunique()} triênios")
    print(f"recorte de teste: {len(teste)} linhas (5 dobras), semente {SEMENTE}\n")

    resultados: dict[str, ResultadoValidacao] = {}

    def rodar(nome: str, fabrica, features: list[str]) -> None:
        resultados[nome] = avaliar(
            df, fabrica, nome=nome, features=features, semente=SEMENTE
        )

    rodar("repete a Etapa 2 (A3 = A2)", lambda s: RepeteEtapa2(), ["a1", "a2"])
    rodar("média das Etapas ((A1+A2)/2)", lambda s: MediaDasEtapas(), ["a1", "a2"])
    rodar("média do treino", lambda s: MediaDoTreino(), ["a1", "a2"])
    rodar("média do curso", lambda s: MediaDoCurso(), ["a1", "a2", "curso"])
    rodar("linear em (A1, A2)", regressao_linear, ["a1", "a2"])
    rodar("linear nas 6 features legadas", regressao_linear, FEATURES_LEGADAS)
    rodar("linear em (A1, A2) + as 6 legadas", regressao_linear, ["a1", "a2", *FEATURES_LEGADAS])

    print("═══ Baselines triviais — erro em A3, agrupado sobre as 5 dobras ═══")
    print(CABECALHO)
    for nome, r in resultados.items():
        print(linha(nome, r.agrupado))

    print("\n═══ Os mesmos, na escala do Argumento Final (3× A3) ═══")
    print(CABECALHO)
    for nome, r in resultados.items():
        print(linha(nome, r.agrupado.em_argumento_final))

    print("\n═══ Por classe (ticket 14: dois números, sempre) ═══")
    print(f"{'preditor':<38} {'RMSE maj':>9} {'RMSE min':>9}")
    for nome, r in resultados.items():
        minoria = r.agrupado_minoritaria
        print(
            f"{nome:<38} {r.agrupado_majoritaria.rmse:>9.3f} "
            f"{(f'{minoria.rmse:.3f}' if minoria else '—'):>9}"
        )

    print("\n═══ Série por dobra, classe majoritária (insumo do ticket 08) ═══")
    dobras = [d.dobra.trienio_teste for d in resultados["linear em (A1, A2)"].dobras]
    print(f"{'preditor':<38} " + " ".join(f"{t:>10}" for t in dobras))
    for nome, r in resultados.items():
        print(f"{nome:<38} " + " ".join(f"{e.rmse:>10.3f}" for e in r.serie_majoritaria))

    # ─── Os artefatos de hoje ───────────────────────────────────────────────────────────
    print("\n═══ Os `.joblib` atuais, sobre as mesmas 37.844 linhas ═══")
    ja_visto = sobreposicao_com_o_banco_antigo(teste)
    print(
        f"sobreposição com `banco_alunos_pas_final.csv`: "
        f"{ja_visto.sum()} de {len(teste)} ({ja_visto.mean() * 100:.1f}%) — "
        f"{(~ja_visto).sum()} linhas limpas"
    )

    X = teste[FEATURES_LEGADAS]
    a3_real = teste["a3"].to_numpy(dtype=float)
    eb3_real = teste["eb_pas3"].to_numpy(dtype=float)
    base_af = (teste["a1"] + 2 * teste["a2"]).to_numpy(dtype=float)

    modelos = {n: joblib.load(MODELOS / f"{n}.joblib") for n in
               ("modelo_arg_final", "modelo_lgbm", "modelo_rf", "modelo_linear", "modelo_mlp")}
    escalador = joblib.load(MODELOS / "scaler.joblib")

    print("\n— `modelo_arg_final`, em A3 (prevê o Argumento Final; A3 = (AF − A1 − 2·A2)/3) —")
    print(CABECALHO)
    af_previsto = modelos["modelo_arg_final"].predict(X)
    a3_previsto = (af_previsto - base_af) / 3
    for recorte, mascara in (("todas as linhas", np.ones(len(teste), bool)),
                             ("só as linhas limpas", (~ja_visto).to_numpy())):
        blocos = medir_congelado(teste[mascara], a3_real[mascara], a3_previsto[mascara])
        for classe, erro in blocos.items():
            print(linha(f"modelo_arg_final · {classe}", erro, f"({recorte}, n={erro.n})"))

    print("\n— Os modelos de EB, na escala de EB PAS 3 (não comparável com A3) —")
    print(CABECALHO)
    previsoes_eb = {}
    for nome in ("modelo_lgbm", "modelo_rf", "modelo_linear", "modelo_mlp"):
        entrada = escalador.transform(X) if nome in ("modelo_linear", "modelo_mlp") else X
        previsoes_eb[nome] = np.asarray(modelos[nome].predict(entrada), dtype=float)
        for recorte, mascara in (("todas", np.ones(len(teste), bool)),
                                 ("limpas", (~ja_visto).to_numpy())):
            erro = Erro.medir(eb3_real[mascara], previsoes_eb[nome][mascara])
            print(linha(nome, erro, f"({recorte}, n={erro.n})"))
    print(linha("repete o EB da Etapa 2", Erro.medir(eb3_real, teste["eb_pas2"].to_numpy(float))))

    # O ensemble por volatilidade — `std/mean × 100` sobre [EB1, EB2] com média **com sinal**,
    # exatamente como `ensemble.calculate_volatility`, e a mesma sigmoide.
    from pas_intelligence.ensemble import _sigmoid_weight  # noqa: E402

    eb1 = teste["eb_pas1"].to_numpy(float)
    eb2 = teste["eb_pas2"].to_numpy(float)
    media = (eb1 + eb2) / 2
    cv = np.where(media != 0, np.abs(eb2 - eb1) / 2 / media * 100, np.nan)
    peso = np.array([_sigmoid_weight(v) if np.isfinite(v) else 0.5 for v in cv])
    ensemble = (1 - peso) * previsoes_eb["modelo_linear"] + peso * previsoes_eb["modelo_lgbm"]
    for recorte, mascara in (("todas", np.ones(len(teste), bool)),
                             ("limpas", (~ja_visto).to_numpy())):
        erro = Erro.medir(eb3_real[mascara], ensemble[mascara])
        print(linha("ensemble por volatilidade (linear×lgbm)", erro, f"({recorte}, n={erro.n})"))

    # O arranjo que a tela usa de verdade: o meta-modelo escolhe **um** dos quatro por aluno.
    meta = joblib.load(MODELOS / "meta_model.joblib")
    meta_escalador = joblib.load(MODELOS / "meta_scaler.joblib")
    entrada_meta = np.column_stack([
        X["EB_PAS1"], X["Red_PAS1"], X["EB_PAS2"], X["Red_PAS2"], X["Cresc_EB"], X["Cresc_Red"],
        np.abs(X["Cresc_EB"]) / (np.abs(X["EB_PAS1"]) + 0.01),
        np.abs(X["Cresc_Red"]) / (np.abs(X["Red_PAS1"]) + 0.01),
        (X["EB_PAS1"] + X["EB_PAS2"]) / 2,
        np.sign(X["Cresc_EB"]),
    ])
    rotulo = meta.predict(meta_escalador.transform(entrada_meta))
    nomes = {0: "modelo_lgbm", 1: "modelo_rf", 2: "modelo_linear", 3: "modelo_mlp"}
    roteado = np.array([previsoes_eb[nomes[int(r)]][i] for i, r in enumerate(rotulo)])
    escolhas = pd.Series(rotulo).map(nomes).value_counts().to_dict()
    for recorte, mascara in (("todas", np.ones(len(teste), bool)),
                             ("limpas", (~ja_visto).to_numpy())):
        erro = Erro.medir(eb3_real[mascara], roteado[mascara])
        print(linha("meta-modelo roteador (o que a tela usa)", erro, f"({recorte}, n={erro.n})"))
    print(f"    escolhas do roteador: {escolhas}")

    print("\n— De onde vêm as 1.810 linhas limpas —")
    print(teste[~ja_visto]["trienio"].value_counts().sort_index().to_string())

    print("\n— A trava 1 da §2.2 na régua real —")
    r = resultados["linear em (A1, A2)"]
    for d in r.dobras:
        m = d.minoritaria
        print(f"    dobra {d.dobra.indice} ({d.dobra.trienio_teste}): "
              f"treino {m.n_treino_classe} ({m.taxa_treino_classe * 100:.2f}%) vs "
              f"teste {m.n_teste_classe} ({m.taxa_teste_classe * 100:.2f}%) → "
              f"{'—  ' + (m.motivo_ausencia or '') if m.erro is None else f'RMSE {m.erro.rmse:.3f}'}")
    print(f"    agrupado da minoria: n={r.agrupado_minoritaria.n} "
          f"(as 5 dobras somam {sum(d.minoritaria.n_teste_classe for d in r.dobras)})")

    # ─── A faixa de decisão, congelada aqui ─────────────────────────────────────────────
    melhor = min(resultados.values(), key=lambda r: r.agrupado.rmse)
    largura = melhor.agrupado.em_argumento_final.rmse
    print(
        f"\n═══ Faixa de decisão, congelada neste ticket ═══\n"
        f"melhor baseline trivial: {melhor.nome} — RMSE {melhor.agrupado.rmse:.3f} em A3\n"
        f"largura da faixa = 1 RMSE de Argumento Final = {largura:.3f}"
    )

    # ─── Erro de decisão ────────────────────────────────────────────────────────────────
    cortes = menor_corte_por_aluno(df)
    print(
        f"\n═══ Erro de decisão (faixa congelada de ±{largura:.3f}) ═══\n"
        f"{'preditor':<38} {'n':>7} {'erra':>7} {'na faixa':>9} {'erra lá':>8} {'RMSE faixa':>11}"
    )
    for nome, r in resultados.items():
        d = erro_de_decisao(r.previsoes.merge(cortes, on=["id_pseudonimo", "trienio"], how="left"),
                            largura_faixa=largura)
        print(f"{nome:<38} {d.n:>7} {d.taxa_erro * 100:>6.1f}% {d.n_na_faixa:>9} "
              f"{d.taxa_erro_na_faixa * 100:>7.1f}% {d.erro_na_faixa.rmse:>11.3f}")

    # Montada a partir de `teste`, não enxertada na tabela de outro modelo: o parquet não está
    # ordenado por triênio, então `previsoes` (concatenada dobra a dobra) e `teste` (na ordem do
    # arquivo) não são a mesma sequência de linhas.
    previsoes_arg = teste[["id_pseudonimo", "trienio", "a1", "a2"]].copy()
    previsoes_arg["real"] = a3_real
    previsoes_arg["previsto"] = a3_previsto
    d = erro_de_decisao(previsoes_arg.merge(cortes, on=["id_pseudonimo", "trienio"], how="left"),
                        largura_faixa=largura)
    print(f"{'modelo_arg_final (o de hoje)':<38} {d.n:>7} {d.taxa_erro * 100:>6.1f}% "
          f"{d.n_na_faixa:>9} {d.taxa_erro_na_faixa * 100:>7.1f}% {d.erro_na_faixa.rmse:>11.3f}")
    print(f"\nsem corte casado: {d.n_sem_corte} de {d.n + d.n_sem_corte} "
          f"({d.n_sem_corte / (d.n + d.n_sem_corte) * 100:.1f}%)")

    detalhar_erro_de_decisao(melhor, cortes, largura)


if __name__ == "__main__":
    main()
