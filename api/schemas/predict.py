from typing import Literal, Optional
from pydantic import BaseModel


class PredictInput(BaseModel):
    p1_pas1: float
    p2_pas1: float
    red_pas1: float
    p1_pas2: float
    p2_pas2: float
    red_pas2: float
    # O Cebraspe normaliza a Parte 1 **por língua estrangeira**, e o Edital não diz qual língua
    # cada candidato fez. Agrupar as três embute viés sistemático contra quem fez espanhol ou
    # francês — em (2024, Etapa 2) a diferença entre a maior e a menor média foi de 3,86 pontos,
    # mais de um desvio-padrão inteiro. Por isso o formulário pergunta (ticket 04 §5.3).
    #
    # **Obrigatória, sem default.** Um default silencioso é exatamente o viés que o ticket 04
    # mediu: o Aluno de espanhol receberia um número plausível e errado, sempre na mesma direção.
    # Cliente que não enviar recebe 422 nomeando o campo — o mesmo tratamento que as seis notas.
    lingua: Literal["inglesa", "francesa", "espanhola"]
    cota: str = "Sistema Universal"
    trienio: str = "2024-2026"
    curso_alvo: Optional[str] = None

    # As três notas da Etapa 1 iguais a zero significam **Aluno sem Etapa 1** (ADR-0008), não
    # "fez a prova e tirou zero". O formulário tem um botão que preenche os três campos com zero,
    # para o Aluno declarar em vez de adivinhar a codificação.


class CourseResult(BaseModel):
    curso: str
    turno: str
    campus: str
    nota_corte: float
    prob: float
    semestre: str


class PredictResponse(BaseModel):
    arg_previsto: float
    # `a1` e `a2` são aritmética exata sobre as notas que o Aluno digitou, não previsão — vão na
    # resposta para que a tela possa mostrar de onde vem o Argumento Final sem chamar a API de
    # novo, e para que ninguém precise reimplementar a fórmula do Argumento de Etapa no frontend.
    a1: float
    a2: float
    a3_previsto: float
    # A Largura de Incerteza da classe deste Aluno, em pontos de Argumento Final (`3 × σ(A3)`).
    # Sai do manifesto do pacote de modelo, nunca de constante de código. A tela **não** a mostra
    # como `±` ao lado do número (ADR-0012 §7: a faixa sem rótulo acertava 63% e respondia uma
    # pergunta que o Aluno não fez); ela alimenta a probabilidade por curso.
    largura_incerteza: float
    etapa_1_ausente: bool
    curso_alvo_result: Optional[CourseResult]
    top_cursos: list[CourseResult]
    trienio_ref: str
    modelo_disponivel: bool
    # Por que a previsão não saiu, quando `modelo_disponivel` é falso — em vez de zeros mudos.
    motivo_indisponivel: Optional[str] = None
