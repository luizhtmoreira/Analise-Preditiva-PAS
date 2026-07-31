from typing import Literal, Optional

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    nome: str
    turma: str = ""
    unidade: str = ""
    curso_alvo: str = ""
    cota: str = "Sistema Universal"
    ano_trienio: str = "2024-2026"
    # A Parte 1 é normalizada por língua estrangeira, **por Etapa** — o Cebraspe registra uma
    # língua por Etapa, não uma por Aluno (defeito 11 de `defeitos-pendentes.md`, ticket 13).
    # Aqui — e **só** aqui — o default sobrevive: a planilha que a escola envia não tem a coluna,
    # e exigi-la deixaria o lote inteiro sem resposta. O custo está medido e é conhecido: o Aluno
    # de espanhol ou francês tem a Parte 1 calculada com a estatística de inglês, o que move 7,2%
    # do peso do Argumento de Etapa sempre na mesma direção. Dívida declarada no relatório 13 §6,
    # não descuido; some quando a planilha ganhar a coluna. O default virou por Etapa junto com o
    # campo, sem mudar a natureza da dívida.
    lingua_e1: Literal["inglesa", "francesa", "espanhola"] = "inglesa"
    lingua_e2: Literal["inglesa", "francesa", "espanhola"] = "inglesa"
    # As seis notas são obrigatórias e não têm default. `red_pas1`/`red_pas2` já tiveram
    # default 6.0 — uma Redação mediana inventada onde a nota não foi informada, que inflava
    # o Argumento da Etapa em ~3,3 pontos na direção do otimismo, sem deixar rastro.
    # Nota faltante agora é 422 nomeando o campo, nunca um número plausível. Ver ADR-0008.
    p1_pas1: float
    p2_pas1: float
    red_pas1: float
    p1_pas2: float
    p2_pas2: float
    red_pas2: float


class StudentResult(BaseModel):
    nome: str
    turma: str
    unidade: str
    curso_alvo: str
    sistema_concorrencia: str
    arg_previsto: float
    gap: float
    chance_display: str
    historico_pct: float
    sugestao: str
    status: str          # "green" | "yellow" | "red" | "grey"
    status_label: str    # "Baixo Risco" | "Oportunidade (2º Sem)" | "Alto Risco" | "Sem previsão"
    prob_1_sem: float
    prob_2_sem: float


class GestaoKpis(BaseModel):
    total: int
    n_red: int
    n_yellow: int
    n_green: int
    # Alunos cujo triênio ainda não tem Edital de média e desvio extraído para todas as Etapas
    # já feitas — sem isso `A1` e `A2` não são exatos e a previsão é recusada (ticket 04 §9).
    n_sem_previsao: int = 0


class GestaoRequest(BaseModel):
    students: list[StudentInput]
    trienio: str = "2024-2026"
    cenario: str = "padrao"   # "padrao" | "tendencia"


class GestaoResponse(BaseModel):
    results: list[StudentResult]
    kpis: GestaoKpis
    trienio_ref: str
    # "O pacote de modelo carregou" — nada além disso. Aluno sem previsão é `kpis.n_sem_previsao`
    # mais o `motivo_sem_previsao` abaixo; juntar os dois num só booleano esconderia qual dos dois
    # problemas a coordenação está vendo.
    modelo_disponivel: bool
    motivo_sem_previsao: Optional[str] = None
