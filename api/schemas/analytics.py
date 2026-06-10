from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Análise Temporal (pública)
# ---------------------------------------------------------------------------


class EtapaStat(BaseModel):
    ano: int
    etapa: int
    p1_medio: float
    p2_medio: float
    eb_medio: float
    red_media: float


class TemporalResponse(BaseModel):
    etapas: list[EtapaStat]
    cursos: list[str]


class CorteEvolucao(BaseModel):
    trienio: str
    corte_1sem: float | None
    corte_2sem: float | None


# ---------------------------------------------------------------------------
# Escola vs. População
# ---------------------------------------------------------------------------


class EscolaRequest(BaseModel):
    inscricoes: list[str]
    trienio: str = ""


class EtapaComparativo(BaseModel):
    etapa: str
    escola: float
    geral: float


class HistBin(BaseModel):
    x0: float
    x1: float
    count: int


class EscolaResponse(BaseModel):
    n_enviados: int
    n_encontrados: int
    taxa_match: float
    trienio: str
    trienios_disponiveis: list[str]
    media_escola: float
    media_geral: float
    std_escola: float
    std_geral: float
    diff: float
    percentil: float
    p_value: float
    significativo: bool
    etapas: list[EtapaComparativo]
    hist_arg: list[HistBin]


# ---------------------------------------------------------------------------
# Comparação entre grupos (A/B)
# ---------------------------------------------------------------------------


class ComparacaoRequest(BaseModel):
    group_a: list[float]
    group_b: list[float]
    name_a: str = "Grupo A"
    name_b: str = "Grupo B"
    metric: str = ""


class BoxStats(BaseModel):
    min: float
    q1: float
    med: float
    q3: float
    max: float
    mean: float
    n: int


class ComparacaoResponse(BaseModel):
    mean_a: float
    mean_b: float
    diff: float
    p_value: float
    t_statistic: float
    effect_size: float
    significativo: bool
    box_a: BoxStats
    box_b: BoxStats
