# Relatório — Ticket 11: Incerteza calibrada para a camada de probabilidade

**Ticket:** `.scratch/treino-modelos-pas3/issues/11-incerteza-calibrada.md`
**Status:** decisões fechadas; **código pendente** (duas mudanças pequenas, §7)
**Tipo:** grilling (Opus, alto — HITL, dono do produto na sala)
**ADR:** `docs/adr/0012-largura-fixa-por-classe-em-vez-de-incerteza-por-aluno.md`
**Glossário do projeto:** `CONTEXT.md` ganhou o termo **Largura de Incerteza**
**Medições:** 37.844 previsões fora-da-dobra, 5 dobras do ADR-0010, semente `20260728`, receita do
ADR-0011. **O triênio lacrado (2023/2025) não foi tocado.**

---

## 1. O que foi pedido

A camada de probabilidade calcula `P(X > Nota de Corte)` sobre `X ~ N(previsão, σ²)`. O `σ` era
`13,49`, cravado em `statistics.py:5` e replicado em `api/services/gestao_service.py:34` como
`ARG_FINAL_MAE`. O ticket apontava dois defeitos: o número é resíduo de um modelo aposentado, e é
igual para todo Aluno. Pedia **incerteza por Aluno**, sugerindo *conformal prediction*, e listava
isso — junto da coerência do ticket 04 — como um dos dois valores centrais do mapa inteiro.

**O ticket entrega a correção do primeiro defeito e o descarte fundamentado do segundo.** A
incerteza por Aluno foi medida e não paga o próprio custo. O que muda de verdade não é a
precisão da largura: é ela deixar de ser uma constante de código e passar a viajar dentro do
pacote de modelo, para que a próxima troca de modelo não deixe a probabilidade descrevendo um
modelo morto em silêncio.

---

## 2. O caminho, na ordem em que as coisas caíram

Registrado nesta ordem de propósito: cada queda mudou a pergunta seguinte, e uma sessão futura que
só ler as conclusões vai reabrir a discussão pelo lado errado.

**Primeiro caiu a hipótese do ticket.** A premissa era "o Aluno errático merece largura maior".
A Volatilidade correlaciona `+0,024` com o tamanho do erro — o **pior** preditor de largura entre
todos os testados. Segunda morte da Volatilidade neste mapa; o ADR-0011 já a havia matado como
mecanismo de arquitetura e como feature.

**Depois apareceu o que de fato move a largura, e não era o esperado.** É o **nível** do Aluno:
`σ` sobe de `4,39` no decil de menor previsão para `5,24` no maior. O modelo é menos certo sobre
o Aluno **forte**. Junto veio um viés por nível que nenhum ticket havia listado: `+0,523` em `A3`
(`+1,57` de Argumento Final) no decil mais fraco contra `−0,022` no mais forte — o modelo encolhe
todo mundo para a média, e isso é **falsa esperança sistemática para quem está por baixo**.

**Aí o dono do produto derrubou o viés com um argumento de domínio, antes de qualquer medição.**
A hipótese dele: os cursos que o Aluno fraco alcança têm folga grande entre o primeiro colocado e
a Nota de Corte, então um erro de mira de 1,5 ponto não vira erro de "passa ou não passa".
Medido, e confirmado nos dois níveis:

- **A premissa:** nos 20% de cursos de corte mais baixo a folga mediana entre o primeiro colocado
  e o corte é **148,9 pontos**; nos 20% mais concorridos, **67,4**.
- **A consequência:** corrigir o viés integralmente move o erro de decisão de **4,7637% para
  4,7700%**. Nada. O Aluno do decil mais fraco está a **82,4 pontos** do corte dele na mediana —
  um viés de 1,5 não atravessa 82.

**Por último caiu a largura, e ela caiu sozinha.** Se corrigir a mira não muda nada, corrigir a
largura muda? Trocar `13,49` pelo número honesto desloca a probabilidade mostrada em `0,53 p.p.`
na média e `2,61` no máximo; usar uma **tabela por decil** em vez de um número desloca `0,21 p.p.`
na média e `3,07` no máximo. Pelo mesmo motivo: o Aluno mediano está a **3,9 larguras** do corte
dele, e nessa distância a conta já saturou.

**Restou um achado que ninguém procurava** e que não é escopo deste ticket: a saturação (§8).

---

## 3. Os números

### 3.1 A forma é normal, e não precisa de ajuda

| medida | valor | leitura |
|---|---:|---|
| assimetria | `−0,0448` | simétrico |
| curtose em excesso | `+0,2187` | cauda quase normal |
| RMSE / MAE | `1,2601` | teórico para normal: `1,25` |
| viés geral | `+0,1789` | `RMSE² = viés² + variância` — o RMSE é dispersão quase pura |

Quantis do resíduo contra os quantis normais equivalentes — que é, na prática, a comparação
"conformal contra normal":

| percentil | resíduo real enfileirado | normal equivalente |
|---|---:|---:|
| 0,5% | `−13,204` | `−12,716` |
| 5% | `−8,130` | `−8,055` |
| 50% | `+0,231` | `+0,179` |
| 95% | `+8,260` | `+8,413` |
| 99,5% | `+13,147` | `+13,074` |

### 3.2 Cobertura empírica — o único teste que importa

| nível prometido | cobertura com os dois números | com número único |
|---|---:|---:|
| 50% | **50,40%** | 50,47% |
| 80% | **80,41%** | 80,44% |
| 90% | **90,16%** | 90,16% |
| 95% | **94,97%** | 94,98% |

Empate no agregado. A diferença está onde os dois números existem para estar:

| classe | n | cobertura a 80% com `σ` próprio | com `σ` emprestado |
|---|---:|---:|---:|
| com Etapa 1 | 34.488 | 80,50% | 80,69% |
| **sem Etapa 1** | 3.356 | **79,53%** | **77,83%** |

### 3.3 Heterocedasticidade — existe, e é pequena

Correlação de `|resíduo|` com cada candidato a driver: previsão `+0,046`, `a2` `+0,052`, nível
`+0,050`, `a1` `+0,041`, **Volatilidade `+0,024`**, Momentum `+0,023`.

`σ` por decil da previsão: `4,386 · 4,726 · 4,801 · 4,970 · 5,044 · 5,160 · 5,244 · 5,236 · 5,244
· 5,207`. Por decil de Volatilidade: `4,929` a `5,183` — praticamente plano.

Cobertura de um intervalo de 80% com `σ` único, por decil da previsão: `86,0%` embaixo, `77,6%`
em cima. É o defeito real que uma tabela por decil consertaria — e vale `0,21 p.p.` na tela.

### 3.4 Estabilidade no tempo — o ano manda mais que o Aluno

| dobra | triênio testado | `σ` em `A3` | viés |
|---|---|---:|---:|
| 1 | 2018/2020 | 5,431 | +0,219 |
| 2 | 2019/2021 | 4,967 | +0,426 |
| 3 | 2020/2022 | 4,997 | −0,004 |
| 4 | 2021/2023 | 5,048 | +0,188 |
| 5 | 2022/2024 | **4,715** | +0,052 |

15% de amplitude entre anos — mais que a diferença entre classes (4,6%) e do tamanho de toda a
variação por Aluno. E a tendência é clara: **quanto mais triênios treinam, menor o erro** (a dobra
1 treina com 2 triênios, a 5 com 6). O artefato final treina com 7, então o agrupado `5,0091` é
**conservador de propósito**.

### 3.5 Erro de decisão — o teste do "passa ou não passa"

31.635 Alunos com Nota de Corte casada (6.209 sem casar), corte `Universal`, menor entre os
semestres, curso que o Aluno de fato disputou. Taxa real de aprovação: 33,55%.

| decil da previsão | viés (Arg. Final) | erro de decisão | após corrigir o viés | distância mediana ao corte |
|---|---:|---:|---:|---:|
| 0 (mais fraco) | `+1,569` | 4,58% | 4,61% | **82,4** |
| 4 | `+0,301` | 3,26% | 3,22% | 66,5 |
| 8 | `+0,074` | **8,95%** | 8,95% | 39,4 |
| 9 (mais forte) | `−0,065` | 8,57% | 8,63% | 39,8 |

**Geral: `4,7637%` → `4,7700%` corrigindo o viés.** Os erros de decisão não estão onde o viés
está.

---

## 4. As decisões, e o porquê de cada uma

1. **Largura fixa, não por Aluno.** Custo real (segundo modelo, ou tabela, ou conformal
   normalizada) contra `0,21 p.p.` de efeito médio na tela. Mesma barra de 1% relativo que
   aposentou o ensemble (ADR-0011).
2. **A forma normal continua.** Sustenta-se sozinha na §3.1 e §3.2. O teto e o piso do alvo, que
   o ticket temia, não mordem: previsão `+2σ` passa do máximo observado em `0,11%` dos casos.
3. **O número é o agrupado das 5 dobras (`5,0091` em `A3`), não a dobra mais recente (`4,715`).**
   Errar para largo é o lado seguro — largura pequena demais faz o app prometer mais do que
   entrega. E "escolher a dobra que dá o número que eu quero" é precedente ruim num mapa cuja
   régua existe para impedir exatamente isso.
4. **O triênio lacrado só reporta, e a regra é assimétrica — escrita antes de o número existir.**
   `σ` medido acima de `5,5` em `A3` = o app está confiante demais, vira ticket com prioridade.
   Abaixo de `4,5` = nota de relatório, não machuca ninguém. Entre os dois = variação normal entre
   anos (ruído entre dobras `±0,37`). A alternativa honesta — substituir com compromisso cego
   escrito antes de olhar — foi considerada e descartada: ela custa a frase *"avaliado num ano que
   nenhuma decisão deste mapa tocou"*, que é a resposta à desconfiança que originou o mapa, e
   compra `14,16` em vez de `15,03`.
5. **Dois números, por classe de `etapa_1_ausente`.** Não por acurácia média (empate: 80,41 vs
   80,44), mas para não mentir com a minoria (`77,83% → 79,53%`). A linha que separa isso de uma
   tabela por decil: a classe existe porque é uma **população que o produto nomeia**, com ADR
   próprio (ADR-0008), não porque minimiza erro.
6. **Sem largura de reserva.** A largura viaja dentro do pacote de modelo. Sem pacote, não há
   previsão *nem* probabilidade — o app usa o caminho degradado que já existe
   (`modelo_disponivel: False`). O estado "previsão sim, largura não" não é representável. É
   resposta direta ao defeito de `target_calculator.py:66`, que engole falha de carregamento com
   um `print` e segue respondendo em produção.
7. **O `±` sai da tela.** `predict_service.py:100-101` mostra `previsto ± 13,49`, sem rótulo, e
   acerta **63%** — 1 em cada 3 Alunos termina fora da faixa que o app lhe mostrou. Com número
   honesto daria para rotulá-la a 80% (`±19,3`), mas uma faixa ao lado de um número responde a uma
   pergunta que o Aluno não fez: ele quer saber em que entra, não entre que valores a nota cai.
   A largura passa a alimentar a probabilidade por curso e o leque de cursos na faixa dele. **80%
   fica como o nível canônico desses derivados.** Execução é do ticket 13 ou de frontend; o que
   fica fixado é que a faixa não volta sem rótulo.
8. **A classe precisa ser detectada em tempo de execução — e já era obrigatório.** A receita do
   ADR-0011 troca as 8 colunas derivadas da Etapa 1 por `NaN`; sem a detecção, o modelo lê "fez a
   prova e tirou zero em tudo", errando a **previsão**, não só a largura. Regra:
   `training_dataset.py:140` (as três notas da Etapa 1 iguais a zero). O formulário
   (`api/schemas/predict.py`) tem seis campos numéricos obrigatórios e nenhuma forma de declarar
   ausência — **ganha um botão que preenche os três campos da Etapa 1 com zero**, para o Aluno
   *declarar* em vez de adivinhar a codificação. Contrato da API não muda. Dependência do ticket
   13.

---

## 5. Por que *conformal prediction* caiu

O caminho que o próprio ticket recomendava. Caiu por três motivos de pesos bem diferentes, e vale
guardar separados — o terceiro é o único que é objeção técnica, os dois primeiros são "resolve um
problema que não temos".

**O que ela é, sem jargão:** enfileire os erros passados do menor para o maior e use o que está na
posição 80% como margem. Não supõe formato de distribuição nenhum, e por isso vem com garantia
matemática de cobertura.

1. **A garantia é sobre um problema inexistente.** Ela protege de o formato dos erros ser
   esquisito; o nosso é sino quase perfeito. A tabela de quantis da §3.1 **é** a comparação
   conformal-contra-normal, e as duas colunas são a mesma coluna. Conformal daria as mesmas
   respostas com mais maquinário.
2. **A versão que interessava precisa de um sinal que não existe.** A conformal simples dá uma
   margem só, igual para todos — que é o que já temos. A **normalizada**, que dá margem por
   pessoa, precisa de um `σ̂(x)`; todos os candidatos foram medidos (§3.3) e o melhor vale
   `3 p.p.` na tela no pior caso.
3. **O formato da saída é incompatível com a conta.** Conformal responde *"me dê a faixa que cobre
   80%"* — nível fixo, faixa como saída. A camada de probabilidade pergunta *"qual a chance de ele
   passar de 87,3"*, com o 87,3 variando por curso, cota e semestre — isso exige a **curva
   inteira**, não uma faixa. Dá para arrancar a curva varrendo muitos níveis, mas é reconstruir
   caro o que o sino dá em uma linha; e o atalho para evitar a varredura é supor um formato, que é
   voltar ao sino por um caminho pior.

**Regressão quantílica** caiu junto com (1): ela resolve assimetria de resíduo, e não há
assimetria (`−0,045`). **NGBoost** caiu por (2) mais o custo de reabrir a família já fixada pelo
ADR-0011.

---

## 6. Limpeza de dado nas evidências que usaram Nota de Corte

A largura **nunca encosta numa Nota de Corte** — é medida sobre resíduos do modelo. O corte entrou
só nas evidências da §3.5 e do §8, e ali foi filtrado: apenas `Universal`, `parcial == False`, e
`nota_corte` dentro da faixa de Argumento Final observada. O filtro removeu **10 de 5.225** linhas,
incluindo os defeitos conhecidos dos tickets 14/15 do mapa `pdf-extraction` (`199.162,872` e
`−23.317,084`). Sobraram **1.223** linhas → **779** chaves curso+triênio com limiar.

---

## 7. O que falta em código

Duas mudanças, ambas testáveis sem o `resultado_final.csv` na mão. **Nada em `api/`** — a fiação
é do ticket 13.

**7.1 — `src/pas_intelligence/training_pipeline.py`:** o manifesto ganha o bloco `incerteza`,
preenchido a partir do `ResultadoValidacao` que o pipeline já tem em mãos. Escala `A3`, porque é
a escala do Alvo Canônico e o ADR-0009 já fixa que Argumento Final é `3×` — guardar as duas
unidades no mesmo arquivo é convite a alguém multiplicar duas vezes.

```json
"incerteza": {
  "forma": "normal",
  "escala": "a3",
  "sigma_por_classe": { "com_etapa_1": 4.9884, "sem_etapa_1": 5.2174 },
  "sigma_agrupado": 5.0091,
  "medido_em": "previsões fora-da-dobra das 5 dobras (ticket 06)",
  "cobertura_verificada": { "0.50": 0.5040, "0.80": 0.8041, "0.90": 0.9016, "0.95": 0.9497 }
}
```

**7.2 — `src/pas_intelligence/statistics.py:5`:** apagar o valor padrão de
`rmse: float = 13.49`, tornando o parâmetro obrigatório. Verificado: **nenhum teste** chama a
função, e os dois pontos de chamada da API já passam o valor explicitamente — a mudança não
quebra nada hoje, e impede que alguém chame a conta sem largura amanhã. Renomear o parâmetro para
o vocabulário do `CONTEXT.md` (**Largura de Incerteza**) é oportuno aqui, mas toca os dois
chamadores em `api/` — se isso empurrar a mudança para dentro do escopo do 13, deixar o nome como
está e trocá-lo lá.

---

## 8. Achado fora de escopo — a probabilidade satura

Não é sobre incerteza; apareceu ao medir uma. **Não era esperado pelo dono do produto**, e por
isso está registrado em `map.md` § *Not yet specified*, não como ticket — não há escopo definido.

- **63,6%** dos Alunos recebem probabilidade `<1%` ou `>99%`. Só **19,5%** caem entre 5% e 95%.
- **Piora com a concorrência do curso**, ao contrário do esperado:

| curso | n | saturada | informativa | aprovação real |
|---|---:|---:|---:|---:|
| os demais | 13.872 | 54,5% | 25,4% | 63,2% |
| top 20% concorridos | 9.954 | 64,6% | 18,8% | 12,9% |
| **top 5% concorridos** | 7.809 | **78,3%** | **9,7%** | **7,2%** |

- **Não é defeito de cálculo.** A taxa real de aprovação no grupo mais concorrido é 7,2%: o app
  dizer "menos de 1%" para a maioria é a verdade, dita de forma dura.
- **O produto que existe nesse número é pequeno e concentrado.** Entre os Alunos a menos de duas
  larguras do corte de um curso concorrido — **685 pessoas, 9,5% do triênio** — 63% recebem
  probabilidade informativa, e um veredito baseado só na previsão pontual erraria **14%** deles.
- **A faixa do modelo é 3,3× mais estreita que não saber nada** (`±19,3` contra `±64,5`, sendo o
  desvio da população de Argumento Final `~50`). Então há informação; ela é só menos dramática do
  que a tela sugere.

A pergunta que fica: **um app que responde "0%" ou "100%" para 2 em cada 3 Alunos está entregando
informação ou obviedade?** Se o valor está no leque de cursos, no *Quanto Falta* e nos cursos onde
o Aluno está *perto* da linha, quem precisa mudar é a tela do Preditor — grande demais para um
mapa de treino de modelo.

---

## 9. Glossário

- **Resíduo** — `previsto − real`, em pontos de `A3`. O erro de um Aluno específico, com sinal.
  Positivo = o modelo chutou alto demais.
- **Viés** — a média dos resíduos. Erro de **mira**: se é `+1,5`, o modelo erra para cima
  sistematicamente. Diferente da largura, que é erro de **espalhamento**.
- **Cobertura empírica** — se o sistema promete "intervalo de 80%", em quantos % dos Alunos a nota
  real caiu mesmo dentro dele. O teste final de uma incerteza honesta.
- **Heterocedasticidade** — quando o tamanho do erro muda conforme quem é o Aluno. O contrário
  (erro do mesmo tamanho para todos) é *homocedasticidade*, que é o que um número único assume.
- **Encolhimento para a média** (*shrinkage*) — a tendência de um modelo de regressão puxar toda
  previsão em direção ao centro. Produz viés positivo embaixo e negativo em cima, que é
  exatamente o padrão da §3.5.
- **Saturação** — a probabilidade colar em 0% ou 100% porque o Aluno está longe demais do corte
  para a largura importar. §8.
- **Conformal prediction** — "enfileire seus erros passados e use o percentil que você quer como
  margem". Cobertura garantida sem supor formato de distribuição. §5.
- **Largura de Incerteza** — o termo canônico do projeto para o `σ`. Ver `CONTEXT.md`.

---

## 10. Onde continuar

- **Ticket 13 (treinar, avaliar e promover) — é o único bloqueado por este.** Herda: (a) ler a
  largura do bloco `incerteza` do manifesto e passá-la em `api/services/`, apagando
  `ARG_FINAL_MAE` de `gestao_service.py:34`; (b) tirar `arg_min`/`arg_max` da resposta
  (`predict_service.py:100-101`) ou rotulá-los a 80%; (c) construir a detecção de
  `etapa_1_ausente` em tempo de execução — **obrigatória para a previsão, não só para a
  largura** — e o botão de declaração no formulário; (d) reportar o `σ` do lacre sob a regra
  assimétrica da §4.4, **sem poder alterá-lo**.
- **Régua de parada do mapa (ticket 06):** este ticket **não** move acurácia e não entra na
  contagem de "<1% relativo em dois tickets seguidos" — ele não é um candidato de modelo. A régua
  segue onde o ticket 10 a deixou: um ticket de distância do gatilho.
- **Ticket de frontend / produto, ainda sem escopo:** a saturação da §8 e a tela do Preditor.
- **Se uma sessão futura propuser conformal prediction, regressão quantílica ou NGBoost:** a
  resposta está medida na §3 e §5, e o ADR-0012 existe exatamente para essa conversa.
