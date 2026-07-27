# 06 — Esquema de validação: como medir sem enganar a si mesmo

**Type:** grilling
**Status:** open
**Blocked by:** 05

## Question

Qual split de treino/validação/teste, e por quê? Esta é a **régua** — todo número dos tickets
07 a 13 sai dela, e uma régua errada faz o mapa inteiro chegar a uma conclusão errada com
confiança alta.

**A questão central: split aleatório ou temporal?**

O uso real é **extrapolação no tempo**: um aluno do triênio 2024/2026 fez PAS 1 e 2, e o
modelo prevê a Etapa 3 dele — que ainda não aconteceu, num ano que não está na base. Um split
aleatório mistura triênios entre treino e teste e mede uma coisa que o produto nunca faz:
interpolar dentro de anos conhecidos. Ele vai reportar um erro otimista, e o otimismo é maior
justamente se houver deriva entre anos — que é exatamente o que o ticket 08 quer detectar.

Um split temporal (treina nos triênios antigos, testa no mais recente) mede o que o produto faz
de verdade, mas gasta o triênio mais informativo como teste e reduz o tamanho do treino.

**Por que este ticket vem antes do 08 e não depois:** a pergunta "vale usar dado de 2018?" só
tem resposta *através* de uma régua. Escolher a régua depois de ver os resultados é escolher a
conclusão. O esquema fecha primeiro, e o 08 roda dentro dele.

**Sub-decisões que vêm junto:**

- **Quantos triênios de teste.** Um só (2023/2025) é ~8.700 alunos mas uma única realização do
  ano; validação temporal deslizante (treina até T, testa em T+1, repete) dá várias medições ao
  custo de mais computação.
- **Agrupamento.** Se o ticket 05 achar alunos em mais de um triênio, o split precisa agrupar
  por aluno, não por linha.
- **A métrica.** RMSE é o herdado (`13.49` em `statistics.py`) e é o que a camada de
  probabilidade consome, mas penaliza outlier pesado. Precisa de pelo menos uma métrica que
  reflita o que o produto *decide*: errar 5 pontos perto da Nota de Corte importa muito mais do
  que errar 5 pontos longe dela.
- **Critério de aceite.** O número que, se batido, encerra o mapa. Sem isso, "melhorar o modelo"
  não tem fim.

- [ ] Escolhido o split (temporal, aleatório ou deslizante), com o motivo e o que se perde
- [ ] Definido o agrupamento, se o ticket 05 indicar necessidade
- [ ] Definido o conjunto de métricas, incluindo pelo menos uma sensível à proximidade do corte
- [ ] Escrito o critério de aceite que encerra o mapa
- [ ] O esquema está implementado como código reutilizável, não descrito em prosa — todos os
      tickets seguintes chamam a mesma função
- [ ] Relatório em `relatorios/06-esquema-de-validacao.md`
