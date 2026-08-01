# 04 — Alvo canônico: prever o Argumento Final direto ou as 3 notas e derivar?

**Type:** grilling
**Status:** concluído — 2026-07-27
**Blocked by:** 02

> **Resposta: nenhuma das duas. O alvo canônico é `A3`, o Argumento da Etapa 3.** O Argumento
> Final, o EB e o escore necessário saem dele por aritmética, com `A1` e `A2` calculados exatos.
> → [relatório](../relatorios/04-alvo-canonico-argumento-ou-tres-notas.md) ·
> [ADR-0009](../../../docs/adr/0009-alvo-canonico-argumento-da-etapa-3.md)

## Question

O que o modelo prevê — o **Argumento Final** de uma vez, ou **P1, P2 e Redação da Etapa 3**
separadamente, com o Argumento saindo da fórmula oficial do Cebraspe?

Hoje o projeto faz **as duas coisas ao mesmo tempo**, sem que nenhuma seja declarada canônica:

- `modelo_arg_final.joblib` (LGBM) prevê o Argumento Final direto;
- `p1_pas3_model.joblib` e `red_pas3_model.joblib` (HistGradientBoosting) preveem P1 e Redação
  da Etapa 3, e o `target_calculator.py` usa isso para o caminho reverso (dado um corte, qual P2
  o aluno precisa tirar);
- o ensemble (`modelo_lgbm`/`rf`/`linear`/`mlp` + `meta_model`) prevê o **EB** da Etapa 3.

Três rotas para números que deveriam ser consistentes entre si, e nada garante que sejam. Um
aluno pode receber um Argumento previsto pela rota A que é incompatível com as notas previstas
pela rota B — e a tela mostra as duas.

**O que pende da resposta:** a camada de probabilidade (`P(X > corte)`) precisa do Argumento; o
`target_calculator` precisa das notas por componente; o produto mostra ambos. A decisão não é
"qual é mais preciso" — é **qual é a fonte de verdade e como a outra é derivada dela sem
divergir**.

**Argumentos de cada lado, para a conversa não começar do zero:**

- *Prever o Argumento direto* otimiza exatamente a métrica que ranqueia o aluno, sem acumular
  erro de três modelos. Mas perde a decomposição que o produto usa para dizer *onde* o aluno
  precisa melhorar, e deixa o `target_calculator` sem base.
- *Prever as 3 notas e derivar* preserva a decomposição e mantém uma única fonte, com o
  Argumento saindo de uma fórmula determinística. Mas os erros dos três se compõem através
  dos pesos (`PESO_P2=8.28` amplifica o erro de P2 em ~11× o de P1), então uma previsão de P2
  medíocre estraga o Argumento.

**O que o ticket 02 já entregou, e que pesa nesta decisão:**

1. **A fórmula NÃO mudou.** Os pesos `0,72 / 8,28 / 1,00` e os multiplicadores de Etapa
   `1 / 2 / 3` foram recuperados por regressão dos próprios Editais de 2016/2018 e 2017/2019,
   com resíduo máximo de 0,005. Derivar-por-fórmula é seguro em toda a série — o risco de
   dependência de versão que este ticket temia **não existe**.
2. **O Argumento Final é mais estável que o EB.** O EB da Etapa 3 varia ~35% entre triênios,
   enquanto o Argumento Final se mantém estável — porque a normalização por média e desvio do
   ano absorve a diferença de dificuldade da prova. Isso é um argumento direto a favor do
   Argumento Final como alvo: prever EB obriga o modelo a adivinhar a dificuldade de uma prova
   que ainda não aconteceu, e prever o Argumento não.

Note que (2) empurra para "Argumento direto" e (1) remove a objeção contra "3 notas + fórmula".
Não se anulam: a rota das 3 notas continua viável, mas herda o problema de dificuldade de prova
que o Argumento não tem. Isso precisa ser encarado, não contornado.

- [x] Escolhido o alvo canônico, com o motivo — **`A3`**, terceira opção que o ticket não previa
- [x] Definido como a outra rota é derivada da canônica sem poder divergir dela — P2 é
      **resolvido** pela fórmula, não previsto; EB e Argumento Final são álgebra sobre o mesmo `A3`
- [x] Verificado, com número, o quanto as rotas divergem **hoje** — mediana **15,29**, acima do
      RMSE declarado em **57%** dos Alunos, e **11%** discordam sobre passar (n = 7.838)
- [x] Confirmado o que o `target_calculator.py` e a camada de probabilidade passam a consumir —
      §7 do relatório; `σ(Argumento Final) = 3 × σ(A3)`, exato
- [x] Relatório em `relatorios/04-alvo-canonico-argumento-ou-tres-notas.md`
