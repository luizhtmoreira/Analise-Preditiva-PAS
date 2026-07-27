# 11 — Incerteza calibrada para a camada de probabilidade

**Type:** grilling
**Status:** open
**Blocked by:** 10

## Question

De onde a camada de probabilidade tira a incerteza do aluno, agora que o modelo mudou?

**A camada continua sendo uma conta separada** — decisão do dono do produto, e correta: o
modelo prevê nota, e a probabilidade de aprovação é `P(X > Nota de Corte)` calculada por cima.
Este ticket não transforma isso num modelo. Ele troca **um parâmetro** dessa conta.

**O parâmetro em questão:** hoje `statistics.py` faz `X ~ N(previsão, RMSE²)` com
`RMSE = 13.49` — uma constante, igual para todo aluno. Dois problemas:

1. **Fica errada por construção quando o modelo muda.** O 13,49 é o resíduo do modelo antigo,
   medido sobre dado que ninguém registrou. Trocar o modelo sem trocar esse número faz a
   probabilidade descrever a incerteza de um modelo que não está mais rodando.
2. **Trata alunos diferentes como iguais.** O aluno com trajetória regular e o aluno errático
   recebem a mesma largura de distribuição — apesar de a volatilidade ser justamente o sinal
   que o ensemble atual usa como eixo central. O modelo sabe que sabe menos sobre o segundo, e
   a conta joga essa informação fora.

**O caminho padrão de mercado:** **conformal prediction**. Ela produz intervalos com cobertura
garantida sem exigir nada do modelo — funciona por cima de qualquer estimador escolhido no
ticket 10, com um conjunto de calibração separado. A variante normalizada (*Mondrian* ou
conformal com escore normalizado) dá largura **por aluno**, resolvendo (2). Alternativas:
regressão quantílica (o modelo prevê quantis diretamente) ou NGBoost. Todas mantêm a conta
`P(X > corte)` intacta — só mudam de onde vem a dispersão.

**A pergunta que fecha o ticket:** a normal continua sendo a forma certa? Se os resíduos forem
assimétricos — plausível, já que a nota tem teto e piso, e o Argumento Final observado varia de
`-74` a `+39` nas amostras — a normal atribui massa de probabilidade a regiões impossíveis, e a
probabilidade de aprovação sai enviesada perto dos extremos, que é onde a decisão do aluno é
mais sensível.

**Ressalva de avaliação:** medir "probabilidade de aprovação" ponta a ponta exige a tabela de
Notas de Corte, que ainda carrega os defeitos dos tickets 14 e 15 do mapa `pdf-extraction`
(ex. `MEDICINA/Darcy/Universal/2020-2022 = 199.162,872`). Calibrar contra corte contaminado
calibra contra ruído. Excluir os cursos afetados da avaliação, ou coordenar com aqueles tickets.

- [ ] Resíduos do modelo do ticket 10 examinados: simétricos? homocedásticos? dependem da
      volatilidade do aluno?
- [ ] Decidido o mecanismo de incerteza, com o motivo
- [ ] Decidido se a normal continua, ou o que a substitui
- [ ] Cobertura empírica verificada — um intervalo de 80% contém a nota real em ~80% dos casos
      do holdout?
- [ ] `statistics.py` deixa de ter constante cravada; a incerteza vem do artefato de modelo
- [ ] Cursos com Nota de Corte contaminada excluídos da calibração, e isso registrado
- [ ] Relatório em `relatorios/11-incerteza-calibrada.md`
