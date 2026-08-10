# O que ainda não fazemos

Esta página existe por uma razão prática: todo limite listado aqui seria descoberto por você mais
cedo ou mais tarde. Descobrir depois de assinar é o pior momento possível — para você e para nós.

## Alguns números da turma atual ainda são derivados, não publicados

A turma que está no ensino médio agora — o triênio **2024–2026** — é atendida normalmente: o aluno
digita as notas que já tirou e recebe a previsão.

Mas há uma ressalva que preferimos escrever do que esconder. Para calcular a nota padronizada de
uma etapa, precisamos da média e do desvio daquela prova, que o Cebraspe publica em edital. Para
duas das provas dessa turma — a primeira etapa de 2024 e a segunda de 2025 — **o edital com esses
números ainda não saiu**. Nós os derivamos a partir dos editais de resultado por etapa, usando uma
calibração medida sobre os seis triênios que já têm edital oficial.

O que isso implica:

- Toda entrada derivada é **marcada como derivada** dentro do próprio dado, e a tela **avisa** que
  aquela previsão usa estimativa.
- Quando o edital oficial sair, esses números serão **substituídos pelos verdadeiros — e as
  previsões desses alunos vão mudar um pouco.** Preferimos avisar que isso acontece a apresentar o
  número como definitivo.
- A calibração passou por um critério de aceitação aplicado em código, que **reprovou na primeira
  rodada** e só passou na segunda. Ela não é um chute; e o erro que ela deixa é menor do que o erro
  típico do próprio modelo.
- Um triênio sem nenhuma cobertura de dados continua recebendo "sem previsão", em vez de um número
  inventado.

## Onde a previsão ajuda menos

Já detalhado em [Como sabemos que acerta](confianca/como-sabemos-que-acerta.md), mas repetido aqui
porque pertence a esta lista: quanto mais perto da nota de corte o aluno está, menor a nossa
vantagem sobre um palpite. Para o aluno exatamente em cima da linha, é cara ou coroa — e nenhum
modelo muda isso.

Uma consequência disso é que, para cerca de dois terços dos alunos, a chance de aprovação sai
muito próxima de 0% ou de 100%. O número está correto, mas é pouco informativo: ele confirma o
que já era evidente. **O valor do produto para esses alunos não está na probabilidade** — está na
distância até a meta e no quanto falta.

## O teto de acurácia, e por que ele existe

Não vamos prometer melhorias grandes de acurácia, porque medimos o teto antes de gastar tempo
perseguindo-o.

As notas anteriores de um aluno explicam cerca de **74%** da variação da nota final dele. Os
outros 26% são o ano que ele ainda vai viver: o dia bom, o dia ruim, a doença na semana da prova,
o mês em que ele destravou. Isso não está em nenhum dado que exista, e nenhum algoritmo o inventa.

Dizemos isso porque a alternativa — prometer que o próximo modelo vai ser muito melhor — seria uma
promessa que não temos como cumprir.

## A previsão assume que o aluno não reage a ela

Uma limitação conceitual, e não um defeito: a probabilidade é calculada supondo que o aluno se
comporte como se comportaram os alunos históricos. Mas o aluno **vê** a previsão, e pode mudar de
comportamento por causa dela — estudar mais por ter levado um susto, ou relaxar por ter se sentido
seguro.

Se a ferramenta funcionar como esperamos, ela deliberadamente torna suas próprias previsões
pessimistas para quem reage bem a elas. Não sabemos medir isso hoje.

## A previsão não se atualiza sozinha ao longo do ano

A chance de aprovação de um aluno não muda entre o dia em que ele abre a ferramenta e a véspera da
prova, a menos que ele mesmo atualize as notas. Não existe recalibração contínua ao longo do ano
letivo.

## Qualidade de dados: o que ainda não está perfeito

Somos transparentes sobre o acervo porque ele é a base de tudo:

- **Cerca de 3% dos registros históricos não passam na conferência aritmética** e ficam marcados
  como tais. Eles não entram no treinamento do modelo.
- **Alunos eliminados não fazem parte da base.** Eles têm apenas duas notas e não formam o
  histórico necessário. É uma exclusão consciente, não uma falha.
- **Um punhado de notas de corte históricas tem valor implausível**, herdado de um defeito de
  leitura já corrigido na origem. Elas estão sinalizadas, mas ainda não foram removidas do
  histórico.
- **Cerca de 2,7% dos nomes** saem da leitura dos PDFs com um espaço no meio de uma palavra. Não
  afeta nenhum número — só a grafia do nome quando ele é impresso em relatório.

## Limites de infraestrutura

O servidor de previsão roda hoje em plano gratuito. Depois de um período sem uso, a primeira
requisição leva cerca de **32 segundos** para responder, enquanto o serviço sobe; as seguintes são
imediatas. A interface avisa e espera. Não é uma falha — é o comportamento normal desse tipo de
hospedagem, e é o item mais direto de resolver quando houver contrato que o justifique.

---

Se algum item desta página for decisivo para a sua escola, fale com a gente antes de decidir.
Vários deles têm prazo, e alguns podem ser priorizados.
