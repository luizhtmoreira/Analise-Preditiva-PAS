# Como sabemos que acerta

Qualquer fornecedor pode dizer que seu modelo é bom. O que separa uma afirmação verificável de
uma propaganda é **como** o número de acerto foi obtido. Esta página descreve o método antes de
dar o resultado, porque o método é o que dá valor ao resultado.

## A regra que evita enganar a si mesmo

O erro clássico em previsão é medir o modelo nos mesmos dados em que ele aprendeu. Isso mede
memória, não previsão, e produz números excelentes que não se repetem na vida real.

Nós medimos de outro jeito, que imita exatamente o que o produto faz de verdade: **treinar no
passado e prever o futuro.**

```
treina 2016/2018 e 2017/2019  →  prevê 2018/2020
  + 2018/2020                 →  prevê 2019/2021
  + 2019/2021                 →  prevê 2020/2022
  + 2020/2022                 →  prevê 2021/2023
  + 2021/2023                 →  prevê 2022/2024
────────────────── LACRE ──────────────────
  tudo até 2022/2024          →  prevê 2023/2025   (uma única vez)
```

Cada linha é uma medição independente sobre alunos que o modelo nunca viu, do mesmo jeito que um
aluno de hoje é alguém que o modelo nunca viu.

## O lacre

O triênio **2023/2025** foi separado no início do projeto e mantido fora de tudo: fora do treino,
fora de toda escolha de método, fora de toda comparação. E a regra de uso foi escrita **antes** de
o número existir: abrir o lacre produz *um número, não uma decisão* — ou o modelo é promovido com
aquele resultado, ou a rodada inteira é descartada. Reajustar o modelo depois de ver o resultado
é proibido.

O lacre foi aberto **uma única vez**, em 28 de julho de 2026. O que está abaixo é o que saiu de lá.

!!! note "Por que isso não é só boa vontade"
    O lacre é mecânico, não disciplinar. O código que separa os dados nunca produz o triênio
    lacrado; ele só sai por uma função com nome constrangedor, `holdout_final_use_uma_vez()`, e
    existe um único lugar autorizado a chamá-la em todo o projeto. Conferir isso é uma busca no
    código, não uma questão de confiança.

## O resultado

**Em 7.449 alunos do triênio lacrado, o veredito de aprovação foi acertado em 94,6% dos casos.**

Esse número precisa das letras miúdas, e elas são estas:

| Distância da nota de corte | Acerto | Quantos alunos |
|---|---:|---:|
| Mais de 2 larguras | **99,9%** | 72,4% |
| Entre 1 e 2 larguras | **91,5%** | 13,2% |
| Entre 0,5 e 1 largura | **80,5%** | 6,5% |
| Menos de meia largura | **63,0%** | 7,9% |
| **Todos** | **94,6%** | 100% |

!!! info "Largura"
    A **Largura de Incerteza** é o quanto o modelo costuma errar — hoje, cerca de 15 pontos de
    Argumento Final. "Estar a duas larguras do corte" significa estar a uns 30 pontos dele: longe
    o bastante para que o erro típico do modelo não mude o resultado.

Lendo a tabela com honestidade: **a maior parte do 94,6% vem de alunos cujo caso já era claro.**
Quem está muito acima ou muito abaixo do corte é fácil de acertar — e é a maioria. Quanto mais
perto da linha o aluno está, menor a nossa vantagem, até chegar ao aluno exatamente em cima do
corte, onde nenhum modelo do mundo decide: é cara ou coroa, e isso é matemática, não limitação
nossa.

**Por isso o produto não foi desenhado para dar um veredito.** Para o aluno na faixa decisiva, o
que entregamos não é "você passa" ou "você não passa" — é a **distância honesta até a meta** e
quanto ele precisa tirar para cobri-la. Essa é uma informação que continua verdadeira e acionável
mesmo onde a previsão não é confiante.

## A probabilidade é calibrada — e isso é conferível

Quando o sistema diz "70% de chance", ele está afirmando algo que pode ser cobrado depois: de
cada 100 alunos que receberem 70%, cerca de 70 deveriam passar. Um sistema mal calibrado infla
esse número e ninguém percebe.

Nós medimos. Estes são os intervalos que o produto promete contra o que de fato aconteceu:

| Quando prometemos | Aconteceu |
|---|---|
| 50% | **50,4%** |
| 80% | **80,4%** |
| 90% | **90,1%** |
| 95% | **94,9%** |

As quatro linhas batem dentro de meio ponto percentual. Essa medida viaja dentro do próprio
arquivo do modelo e é refeita a cada retreinamento — não é uma constante escrita no código que
poderia continuar descrevendo um modelo já substituído.

## O que a previsão vale, em termos práticos

Sem nenhuma informação, chutar o Argumento Final de um aluno do PAS dá uma margem de cerca de
±64 pontos — é a dispersão natural da população. Com as notas que o aluno já tirou nas etapas
anteriores, essa margem cai para cerca de **±19 pontos**.

A incerteza fica, portanto, **3,3 vezes mais estreita** do que não saber nada. É esse
estreitamento, e não uma bola de cristal, o que o produto vende.

## Só uma coisa é prevista

Um detalhe de desenho que evita uma classe inteira de erro: para um aluno que já fez PAS 1 e
PAS 2, **as notas dessas duas etapas não são previstas — elas são conta exata**. As notas estão
na mão e as estatísticas oficiais daqueles anos são públicas.

O único número desconhecido é o da terceira etapa. Tudo o que o produto mostra — Argumento Final
estimado, chance de aprovação, quanto falta — deriva desse único número por aritmética. A
consequência prática é que **os números na tela não podem se contradizer entre si**, porque todos
vêm da mesma fonte.

## O que fica de fora deste número

- O acerto de 94,6% vale para o **veredito de aprovação**, com a nota de corte do curso pretendido
  já conhecida. Ele não é uma promessa sobre a nota exata de nenhum aluno.
- 10% dos alunos da medição não tinham corte comparável e ficaram fora do cálculo.
- Os limites conhecidos do produto estão reunidos e explicados em
  [O que ainda não fazemos](../limites.md).

---

**A seguir:** [O que ainda não fazemos](../limites.md) — os limites, escritos por nós antes de
você descobrir sozinho.
