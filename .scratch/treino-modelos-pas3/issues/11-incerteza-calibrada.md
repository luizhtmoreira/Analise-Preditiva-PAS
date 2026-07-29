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

- [x] Resíduos do modelo do ticket 10 examinados: **simétricos** (assimetria `−0,045`, RMSE/MAE
      `1,260`); **levemente heterocedásticos** pelo **nível**, não pela volatilidade (`σ` de
      `4,39` a `5,24` por decil de previsão); **não dependem da Volatilidade** (correlação
      `+0,024`)
- [x] Decidido o mecanismo de incerteza, com o motivo — **duas larguras fixas por classe de
      `etapa_1_ausente`** (`4,9884` / `5,2174` em `A3`), no manifesto do pacote. ADR-0012
- [x] Decidido se a normal continua — **continua**, e sem refinamento: largura por Aluno desloca
      a probabilidade em `0,21 p.p.` na média e `3,07` no máximo
- [x] Cobertura empírica verificada — 80% prometido → **80,41%** real, sobre as previsões
      fora-da-dobra. **Não no holdout**: como escrito, o item contradizia o lacre do ADR-0010; o
      ticket 13 reporta o número do lacre sob a regra assimétrica, sem poder alterá-lo
- [x] `statistics.py` deixa de ter constante cravada; a incerteza vem do artefato de modelo
- [x] Bloco `incerteza` gravado no `manifest.json` por `training_pipeline.py`
- [x] ~~Cursos com Nota de Corte contaminada excluídos da calibração~~ — **sem objeto**: a largura
      é medida sobre resíduos e nunca encosta numa Nota de Corte. O corte entrou só nas
      evidências (folga por curso, erro de decisão), ali filtrado para `Universal` não-parcial
      dentro da faixa de Argumento Final observada
- [ ] Relatório em `relatorios/11-incerteza-calibrada.md`

**Achado fora do escopo, registrado em `map.md` § Not yet specified:** a probabilidade satura
(0% ou 100%) para **63,6%** dos Alunos, e a saturação piora com a concorrência do curso.
