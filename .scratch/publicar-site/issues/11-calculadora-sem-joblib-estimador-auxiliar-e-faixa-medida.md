# 11 — Calculadora sem `.joblib`: Estimador Auxiliar e a faixa medida da P2

**What to build:** o Aluno escolhe um curso e a Calculadora de Estratégia diz quanto ele precisa
tirar na Parte 2 do PAS 3 — por aritmética, sem carregar nenhum modelo serializado, e com os
limites de "impossível" e "garantido" vindos de número medido em vez de chute.

A Calculadora está no ar em nenhum lugar e degradada em toda parte. Este ticket a torna publicável
**deixando o código menor do que está**.

## Bloqueador 1: os dois modelos não carregam

`p1_pas3_model.joblib` e `red_pas3_model.joblib` falham com `ModuleNotFoundError: No module named
'_loss'` no ambiente atual — foram serializados quando o `sklearn` tinha esse módulo interno em
outro lugar. Os números seguem íntegros no arquivo; a receita de remontagem é que aponta para o
vazio. É o defeito 3 de `defeitos-pendentes.md`, e a Calculadora responde por média ponderada há
meses.

**A saída não é consertar, é remover.** O **Estimador Auxiliar** do relatório 04 (§2.1 e §7.1)
prevê P1 e Redação por **média ponderada de z-scores** — aritmética sobre notas que já temos:

```
Â3                ← única previsão do modelo
P1̂, R̂ed           ← Estimador Auxiliar (média ponderada dos z das Etapas 1 e 2) + override do Aluno
P2                = resolvido:  z_p2 = (A3 − 0,72·z_p1 − 1,00·z_red) / 8,28
```

Um **z-score** é a nota expressa em desvios-padrão em relação à média daquela prova; é o que
permite comparar uma nota de 2024 com uma de 2025 sem herdar a dificuldade de cada uma. O peso é 1
para a Etapa 1 e 2 para a Etapa 2 — o mesmo da média ponderada que o código já usa como fallback. O
que muda é que a média passa a ser feita na escala padronizada e depois reconvertida para a escala
da Etapa 3, em vez de somar notas de provas com dificuldades diferentes.

**Erro medido nos três triênios recentes:** 1,47 ponto em P1 e 1,36 na Redação. Com `A3` fixo, o
erro de P1 é amortecido em 60% (move o P2 necessário em ~0,59); o da Redação passa quase inteiro
(~1,29) — e é exatamente por isso que a caixa de override da Redação é a que mais importa.

**Sai do módulo, por remoção:** o carregamento de `.joblib`, o registro de degradação, o erro de
carga, o modo estrito por variável de ambiente e o import de `joblib`. Isso elimina deste módulo a
classe inteira de defeito "artefato serializado com outra versão de biblioteca".

**A língua da Etapa 3.** O caminho reverso precisa da estatística da Parte 1 da Etapa 3, que o Aluno
ainda não fez. Usa-se a língua da **Etapa 2** como a língua provável da Etapa 3 — não a da Etapa 1 —
porque a troca de língua é de mão única (72% das trocas são inglesa → espanhola) e a última
declarada é a melhor evidência disponível. Assunção explícita, para ser revisitada se algum dia
medirmos `lingua_e2 → lingua_e3`.

## Bloqueador 2: a faixa da P2 é um chute

`P2_MAXIMO = 100.0` e `P2_MINIMO = -100.0` decidem sozinhas quando o produto diz "impossível" e
quando diz "garantido", e não têm procedência em Edital nenhum. Agora estão medidas — Etapa 3, 8
triênios, ~64 mil Alunos:

| | Chute atual | Medido |
|---|---:|---:|
| Piso de P2 | −100 | **0,24** (0% negativo em 8 triênios) |
| Teto de P2 | +100 | **85,6** (o maior de 64 mil Alunos) |
| P2 no percentil 99,9 | — | ~78 |
| Teto de `EB = P1 + P2` | — | 92,3 |

O teto teórico continua 100, porque o fator de normalização existe para que acertar tudo dê 100 —
mas ele é de `P1 + P2` **juntos**, e a P1 sozinha já come até 8,5.

**A faixa é por Etapa, e essa distinção fica no código:** nos Editais de Etapa 2, 2,3% dos
candidatos ficaram abaixo de zero (o pior em −19,6); na Etapa 3, **zero em 64 mil**. A Calculadora
resolve para a Etapa 3, então usa a faixa da Etapa 3.

**Os quatro status passam a significar algo medido:**

- **impossível** — a nota necessária passa de `100 − P1̂`. Aritmética, não opinião. (Com a faixa
  antiga, uma nota necessária de 95 era classificada como "possível".)
- **improvável** — passa de 85,6, o recorde histórico em 64 mil Alunos. Existe no papel, nunca
  aconteceu.
- **garantido** — a nota necessária fica abaixo do piso de 0,24. Com a faixa antiga esse ramo era
  praticamente código morto; com a medida ele volta a significar algo verdadeiro.
- **possível** — o resto.

A mensagem do ramo `garantido` deixa de exibir o valor truncado (*"Meta alcançável! Você precisa de
-99.4 pts na Parte 2"*) e passa a dizer em português o que aconteceu — resolve a nota de comunicação
do defeito 1.

## O que a Calculadora *não* precisa

O modelo de correção item a item (110 itens na Parte 2, tipos A/B/C/D com pesos 1/2/2/3, desconto
por erro, fator de normalização) alimenta o **Simulador de Itens**, que é outra tela e depende de
saber quantos itens de cada tipo tinha cada prova — dado que não sai em Edital, só no caderno de
questões. **Confundir as duas coisas foi o que manteve a Calculadora bloqueada por engano.**

## Teste que não pode regredir

`test_override_parcial_e_respeitado` verifica que mexer só na Redação não descarta também o override
de P1 (defeito 7, corrigido em 2026-07-27). Ele tem que continuar passando **depois** da troca do
estimador — é o que prova que a remoção não mexeu no contrato.

E `test_guaranteed_scenario` / `test_alvo_baixo_mas_dentro_da_faixa_ainda_e_possivel_nao_garantido`
fixam a fronteira entre `garantido` e `possível` sob a faixa antiga. Eles **mudam de valor
esperado** aqui, e a mudança tem que ser deliberada e escrita — não um ajuste até ficar verde.

**Blocked by:** 05 (`TRIENNIUM_STATS` sai) e 10 (Merge do portal — a Calculadora chega nele).

**Status:** ready-for-agent

- [ ] Nenhum `joblib.load` no caminho da Calculadora; o carregamento de modelo, o registro de
      degradação, o erro de carga e o modo estrito saíram do módulo
- [ ] O Estimador Auxiliar prevê P1 e Redação por média ponderada de z-scores (peso 1 : 2) e
      reconverte para a escala da Etapa 3
- [ ] A estatística da Parte 1 da Etapa 3 usa a língua da Etapa 2, com a assunção documentada
- [ ] `P2_MAXIMO` / `P2_MINIMO` viram a faixa medida da Etapa 3, e a origem do número está na
      docstring
- [ ] Os quatro status disparam conforme a faixa medida, incluindo `impossível` a partir de
      `100 − P1̂` e não de uma constante
- [ ] A mensagem de `garantido` não exibe valor negativo truncado
- [ ] `test_override_parcial_e_respeitado` continua passando sem alteração
- [ ] Os dois testes de fronteira `garantido`/`possível` são atualizados com valor esperado novo e
      justificativa escrita
- [ ] O defeito 3 de `defeitos-pendentes.md` é marcado como resolvido **por remoção**, no que toca à
      calculadora reversa
- [ ] `pytest tests/` continua verde
