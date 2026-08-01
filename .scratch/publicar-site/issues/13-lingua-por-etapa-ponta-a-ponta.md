# 13 — Língua estrangeira por Etapa, ponta a ponta

**What to build:** o Aluno que fez inglês na Etapa 1 e espanhol na Etapa 2 informa as duas, e o
Argumento dele é calculado com a estatística certa em cada Etapa.

## O defeito

O `resultado_final.csv` grava `lingua_e1`, `lingua_e2` e `lingua_e3` — **uma por Etapa**, porque é
assim que o Cebraspe registra. O treino respeita isso: normaliza a Parte 1 de cada Etapa com a
estatística da língua **daquela** Etapa.

O runtime não. `EntradaDePrevisao` tem **um** campo `lingua`, aplicado às duas Etapas, e os schemas
fecham o contrato no mesmo formato. O produto inteiro — formulário, request, cálculo — presume que a
língua é atributo do **Aluno**. Ela é atributo do par **(Aluno, Etapa)**.

É o defeito 11 de `defeitos-pendentes.md`.

## Quantos trocam, e quanto custa

Medido sobre as 64.298 linhas limpas: **8.950 (13,9%)** têm `lingua_e1 ≠ lingua_e2`. E a troca não é
uniforme — **6.462 das 8.950 (72%) são inglesa → espanhola**. Não é ruído de extração; é um
movimento real e de mão única da coorte.

| Ele declara | Quem erra | \|erro\| médio | máx | Peso | Máx no Arg. Final |
|---|---|---:|---:|---|---:|
| a língua da Etapa 2 | `A1` | 0,353 | 1,039 | ×1 | 1,04 |
| a língua da Etapa 1 | `A2` | 0,499 | 1,896 | ×2 | **3,79** |

**O que torna isto diferente do default de língua da Gestão de Ativos** (dívida já declarada e
aceita no relatório 13 §6.2): aquele atinge quem **não informa** a língua. Este atinge o Aluno que
informa **corretamente**, no Preditor público, onde o campo é obrigatório e sem default justamente
para não embutir viés de língua. O produto pergunta a coisa certa e não consegue registrar a
resposta certa, porque o campo tem a cardinalidade errada.

## Por que nada pegou, e o que impede a volta

`tests/test_model_package.py::test_o_runtime_monta_as_mesmas_features_que_o_treino` existe
exatamente para prender desencontro entre treino e runtime, e o docstring dele diz que o
desencontro *"devolve previsão errada com cara de certa, para sempre"*. Ele **passa mesmo assim**: o
fixture crava `"inglesa"` nas três Etapas, e com a língua constante as duas portas concordam por
construção. **O teste não falhou; ele é cego a esta dimensão.**

Estender esse teste com um caso de língua trocada é o item que impede a correção de voltar atrás na
próxima refatoração, e é o único critério desta lista que **tem** que entrar junto.

## As decisões de forma

- `EntradaDePrevisao` carrega `lingua_e1` e `lingua_e2`; o cálculo consome cada uma na sua Etapa.
- O schema do Preditor troca `lingua` por `lingua_e1` e `lingua_e2`, **ambas obrigatórias, sem
  default**. Não há alias de compatibilidade: o único cliente é o nosso próprio frontend, que está
  sendo reescrito nesta mesma rodada, e um default silencioso é exatamente o viés que o ticket 04
  §5.3 se propôs a eliminar.
- **O formulário pré-preenche o segundo campo com o primeiro**, visivelmente e editável. O
  pré-preenchimento é de interface, não de contrato — a API continua exigindo os dois. Isso atende
  os 86% que não trocaram sem penalizar os 13,9% que trocaram.
- O schema da Gestão tem o default por Etapa, sem mudar a natureza da dívida já aceita.

## Por que depois do merge

`PreditorPage.tsx` é um dos dois conflitos reais do ticket 10. Fazer este ticket antes significa
escrever o formulário duas vezes.

**Blocked by:** 10 (Merge do portal para cima do modelo).

**Status:** ready-for-agent

- [x] `EntradaDePrevisao` carrega a língua por Etapa e o cálculo usa cada uma na sua Etapa
- [x] O schema do Preditor exige `lingua_e1` e `lingua_e2`, sem default; faltar qualquer uma devolve
      422 nomeando o campo
- [x] O formulário tem dois campos, com o segundo pré-preenchido pelo primeiro e editável
- [x] O schema da Gestão tem o default por Etapa
- [x] `test_o_runtime_monta_as_mesmas_features_que_o_treino` ganha um caso de **língua trocada** e
      falharia sem a correção
- [x] Um Aluno com línguas diferentes produz o mesmo `A1`/`A2` pelo caminho do treino e pelo caminho
      do runtime
- [x] O defeito 11 de `defeitos-pendentes.md` é marcado como corrigido
- [x] `pytest tests/`, `eslint` e `tsc --noEmit` verdes (432 passam, 0 falham — ver relatório 13 §6)
