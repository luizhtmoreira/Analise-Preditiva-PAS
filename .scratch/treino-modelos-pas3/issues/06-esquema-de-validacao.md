# 06 — Esquema de validação: como medir sem enganar a si mesmo

**Type:** grilling
**Status:** decidido — implementação pendente
**Blocked by:** 05
**Decisão:** [relatório](../relatorios/06-esquema-de-validacao.md) ·
[ADR-0010](../../../docs/adr/0010-validacao-deslizante-com-holdout-lacrado.md)

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

- [x] Escolhido o split — **deslizante, 5 dobras, com 2023/2025 lacrado**, mais a regra de uso do
      lacre escrita antes de o número ser conhecido (§1)
- [x] Definido o agrupamento — **não agrupa e não remove**, por motivo estrutural e não por
      tamanho (§3)
- [x] Definido o conjunto de métricas — **RMSE decide, MAE se fala, viés valida o RMSE como σ,
      erro de decisão + faixa vetam conversando**; corte pelo menor sistema do Aluno (§4)
- [x] Escrito o critério de aceite — **não-regressão + coerência + incerteza honesta + regra de
      parada**, porque a §6 mediu que não há acurácia a ganhar (§7)
- [ ] O esquema está implementado como código reutilizável, não descrito em prosa — todos os
      tickets seguintes chamam a mesma função → **`src/pas_intelligence/validation.py`,
      especificado na §8. Sessão nova.**
- [x] Relatório em `relatorios/06-esquema-de-validacao.md`

**Achado que atravessa o mapa (§6):** o teto de acurácia foi medido e um LightGBM de 400 árvores
empata com uma regressão linear de duas variáveis (0,2%). Os tickets 08, 09 e 10 passam a ser
timeboxados — ver `map.md`.
