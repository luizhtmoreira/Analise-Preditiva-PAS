# Relatório — Ticket 06: esquema de validação

**Ticket:** `.scratch/treino-modelos-pas3/issues/06-esquema-de-validacao.md`
**Status:** decisões concluídas — implementação de `validation.py` pendente (sessão nova)
**ADR:** [0010 — Validação deslizante com holdout lacrado](../../../docs/adr/0010-validacao-deslizante-com-holdout-lacrado.md)
**Tipo:** decisão (grilling) + código reutilizável
**Dado analisado:** `data/training/pas3_dataset.parquet` (64.298 linhas, 8 triênios),
`.scratch/pdf-extraction/saida-nova/resultado_final.csv`,
`.scratch/pdf-extraction/saida-nova/notas_corte.csv`
**Privacidade:** só agregados e contagens.

> **Este relatório é a especificação de `src/pas_intelligence/validation.py`.** A sessão que
> implementar não precisa do contexto da sessão de grilling — precisa das §§2.2, 4, 5, 7 e 8.

---

## 1. O split — validação deslizante com holdout lacrado

**Decisão: validação deslizante (treina no passado, prevê o futuro), com o triênio 2023/2025
lacrado e aberto uma única vez, no ticket 13.**

```
dobra 1 · treina 2016/2018, 2017/2019       → prevê 2018/2020
dobra 2 · treina os anteriores + 2018/2020  → prevê 2019/2021
dobra 3 · treina os anteriores + 2019/2021  → prevê 2020/2022
dobra 4 · treina os anteriores + 2020/2022  → prevê 2021/2023
dobra 5 · treina os anteriores + 2021/2023  → prevê 2022/2024
──────────────────── LACRE ────────────────────
        · treina tudo até 2022/2024         → prevê 2023/2025   (ticket 13, uma vez)
```

**Por que deslizante e não um único split temporal.** Com um teste só você tem 8.703 alunos mas
**uma realização de ano**. O erro padrão *dentro* do ano é minúsculo; a variação *entre* anos é a
incerteza que interessa, e ela é invisível com uma realização. Cinco dobras dão o desvio entre
anos — o número que o ticket 08 precisa para separar deriva de ruído de ano, e que o ticket 11
vai querer para a incerteza. Custo: computação, que a esta escala é desprezível.

**Por que não split aleatório.** O ticket original argumentava pela escala do alvo; esse
argumento **caiu** com o ticket 04. O relatório 05 §4 mostra `A3` com média ~0 e desvio ~9,1 nos
oito triênios, sem degrau — a escala não deriva. O argumento que sobrevive é outro: o que pode
derivar é a **relação** entre `(A1, A2)` e `A3` — a força do Momentum, o quanto a Etapa 2 prevê a
Etapa 3 —, e o split aleatório continua cego a ela porque mede interpolação dentro de anos
conhecidos, coisa que o produto nunca faz.

**O que se perde:** o triênio mais recente e mais informativo não entra em nenhum treino de
decisão. Aceito de propósito — é o preço de ter uma medição não contaminada no fim.

### 1.1 Regra de uso do lacre — escrita antes de o número ser conhecido

**(a) O que conta como tocar.** Nenhum código dos tickets 07 a 12 lê 2023/2025 — nem para
treinar, nem para olhar distribuição. A função de split **não devolve** o holdout; ele sai só por
uma função de nome constrangedor (`holdout_final_use_uma_vez()`), e o ticket 13 é o único lugar
do repositório que a chama. Fiscalização é `git grep` do nome, não honra.

**(b) O modelo embarcado treina nos 8 triênios**, incluindo 2023/2025 — e o número registrado é o
do **modelo medido** (mesma receita, treinada até 2022/2024). O manifesto do ticket 03 escreve a
frase inteira: *"erro X — medido sobre a mesma receita treinada até 2022/2024; o arquivo embarcado
foi treinado até 2023/2025 e não foi medido"*.

Motivo: o que se mede é a **receita**, não o arquivo. O triênio mais recente é o mais parecido com
o Aluno vivo do triênio 2024/2026; deixá-lo fora do artefato que roda, para preservar uma medição
já tirada, é pagar duas vezes pelo mesmo dado. A alternativa (embarcar o modelo medido) troca
fidelidade do dado por literalidade do número.

**(c) Abrir o lacre produz um número, não uma decisão.** Se o erro em 2023/2025 vier bem pior do
que as dobras faziam esperar, há duas saídas — promover com o número feio, ou desistir da rodada.
**Nenhuma delas é reajustar.** Reajustar depois de ver transforma o triênio em mais uma dobra, e
aí não sobra nenhum ano limpo até o Edital de 2027.

Consolo real: o número feio **vira a largura da incerteza** no ticket 11. Promover com erro 9,4 em
vez de 6,9 significa que o Aluno recebe uma faixa mais larga e uma probabilidade mais cautelosa —
o que é *mais* honesto, não menos. O modelo continua sendo o melhor disponível; ele só passa a
admitir o quanto não sabe.

---

## 2. O desequilíbrio da classe minoritária entre triênios — declarado, não corrigido

**Achado desta sessão.** A taxa de `etapa_1_ausente` no dataset **não é estável** entre triênios:

| triênio | n | `etapa_1_ausente` | % |
|---|---:|---:|---:|
| 2016/2018 | 8.877 | 34 | **0,38%** |
| 2017/2019 | 8.874 | 30 | **0,34%** |
| 2018/2020 | 5.804 | 420 | 7,24% |
| 2019/2021 | 8.392 | 477 | 5,68% |
| 2020/2022 | 7.130 | 577 | 8,09% |
| 2021/2023 | 8.019 | 897 | 11,19% |
| 2022/2024 | 8.499 | 985 | 11,59% |
| 2023/2025 | 8.703 | 865 | 9,94% |

**Não é o mundo mudando — é o filtro do ticket 01 vazando para dentro da régua.** No regime antigo
o Aluno sem Etapa 1 recebia uma regra mais generosa (ticket 02), falha o `checksum_fecha` e some no
filtro. Os descartes dos dois triênios antigos (734 e 978) são 7,6% e 9,9% da base bruta —
exatamente a faixa dos triênios recentes. A classe existe lá; ela foi filtrada.

Consequência para a régua: **em toda dobra o treino é mais pobre na classe do que o teste.**

| dobra | testa em | n treino | % aus. treino | n teste | % aus. teste |
|---|---|---:|---:|---:|---:|
| 1 | 2018/2020 | 17.751 | 0,36% | 5.804 | 7,24% |
| 2 | 2019/2021 | 23.555 | 2,05% | 8.392 | 5,68% |
| 3 | 2020/2022 | 31.947 | 3,01% | 7.130 | 8,09% |
| 4 | 2021/2023 | 39.077 | 3,94% | 8.019 | 11,19% |
| 5 | 2022/2024 | 47.096 | 5,17% | 8.499 | 11,59% |
| LACRE | 2023/2025 | 55.595 | 6,15% | 8.703 | 9,94% |

**O risco concreto:** o número da classe minoritária vai *melhorar ao longo das dobras*, porque o
treino fica mais rico na classe — e isso vai **parecer** que dado recente ajuda mais, que é
exatamente a conclusão que o ticket 08 quer tirar. Ele leria como deriva do mundo o que é o
filtro do ticket 01.

### 2.1 Tentativa de recuperar as linhas — falhou, e por quê

1.465 linhas descartadas nos dois triênios antigos têm Etapa 1 zerada **e Etapa 3 completa**
(611 em 2016/2018, 854 em 2017/2019). Se recuperadas, a taxa da classe naqueles triênios viraria
6,8% e 9,1% — em linha com os recentes.

**Teste aplicado:** se a regra antiga fosse "z de zero mais uma constante", recompor com a
constante faria o Argumento Final impresso voltar a bater. Isso identificaria a regra **e**
validaria as notas das Etapas 2 e 3 de quebra — uma nota mal extraída espalharia o resíduo.

**Resultado: não fecha.** O resíduo `(Argumento Final impresso) − (recomposição com z-de-zero na
Etapa 1)` não é constante:

| triênio | n | mediana | IQR | dentro de ±0,01 da mediana | ±0,05 |
|---|---:|---:|---:|---:|---:|
| 2016/2018 | 611 | 0,035 | −0,065 a 0,179 | 5,24% | 23,73% |
| 2017/2019 | 854 | 1,095 | 0,065 a 1,995 | 0,47% | 2,34% |

Trazer essas linhas de volta seria admitir linhas cuja única verificação de qualidade está
quebrada. **Ficam fora.**

*Anotação para quem reabrir:* estas medianas (0,035 e 1,095) **não batem** com os `+2,704` e
`+3,549` do relatório 02. A explicação mais provável é que a língua gravada nessas linhas não é
confiável — coisa que o próprio relatório 02 registra. Não foi investigado aqui; não é escopo do
ticket 06.

### 2.2 As quatro travas contra a leitura errada

Decisão: **aceitar e declarar** (não reponderar, não cortar dobras, não cortar triênios — todas
essas alternativas decidem algo que pertence ao ticket 08). Mas a declaração é **mecânica**, não
prosa, porque aviso em prosa não sobrevive a um copiar-e-colar de tabela:

1. **O número não existe onde não pode existir.** A avaliação devolve **vazio**, não um número,
   para a classe minoritária em qualquer dobra cujo treino tenha menos exemplos da classe do que o
   teste. Motivo com razão, não redondo: pedir generalização para mais casos da classe do que o
   modelo jamais viu não é medição. Hoje isso barra a dobra 1 (64 no treino contra 420 no teste).
   Quem plotar a série recebe um buraco, não um ponto baixo.
2. **Não existe série para ler tendência.** Para a classe minoritária a saída canônica é **um
   número só** — erro agrupado sobre as dobras que qualificam, ponderado pelo tamanho de cada
   teste. Um número não tem tendência. Os cinco continuam dentro da estrutura para depuração, mas
   o que entra em relatório e o que o ticket 08 lê é um número. A classe majoritária **mantém** os
   cinco: lá a série é legítima e é o que o 08 precisa.
3. **O número nunca viaja sozinho.** `n_treino_classe` e `taxa_treino_classe` são campos
   obrigatórios da estrutura de resultado, ao lado do erro — não rodapé.
4. **A régua nega antecipadamente a pergunta errada.** Escrito também no ticket 08: a curva de
   erro contra número de triênios é medida **só na classe majoritária**. Para o Aluno sem Etapa 1,
   "2018 ajuda?" é inrespondível com este dataset, e a régua declara isso.

A trava que carrega o peso é a **2** — ela remove o objeto que gera a confusão. As outras três são
cinto de segurança.

---

## 3. Agrupamento por aluno — não agrupa

**Decisão: o split não agrupa por `id_pseudonimo`, e os alunos repetidos não são removidos.**

**Medido:** 144 alunos em 296 linhas, 161 pares. Todos nos **quatro triênios recentes** — nenhuma
repetição de 2016/2018 a 2019/2021. 134 dos 161 pares são em triênios **vizinhos**, e **79 dos 144
escolheram o mesmo curso nas duas pernas**. Isso descarta a hipótese de reciclagem de número de
inscrição: número reciclado cairia em curso aleatório, não no mesmo em 55% das vezes. **83 alunos
têm uma perna dentro do lacre 2023/2025 e outra antes** — 0,95% das 8.703 linhas de teste.

**O motivo não é "é pouco" — é estrutural.** Para haver vazamento, uma linha de treino teria que
carregar a **resposta** de uma linha de teste. Sob split temporal isso é impossível por
construção. Caso conferido: aluno no triênio 2022/2024 (treino) e 2023/2025 (lacre) — a resposta
do teste é a Etapa 3 dele em **2025**, prova que não aparece em lugar nenhum do treino, que
termina em 2024. O que acontece é o inverso: a *resposta* da linha de treino (Etapa 3 de 2024)
reaparece como *feature* da linha de teste (Etapa 2 do triênio novo). Resposta virando pergunta
não é vazamento.

**E isso é mais um argumento pelo split temporal:** num split aleatório o problema seria real,
porque uma linha de triênio posterior poderia entrar no treino carregando provas do mesmo ano da
resposta testada.

Sobra uma dependência estatística fraca — 83 pessoas cuja habilidade o modelo conheceu por outro
caminho. Isso mexe na variância da estimativa, não no valor.

**Argumento decisivo contra remover, do dono do produto:** são casos que acontecem, e o Aluno vivo
do triênio 2024/2026 pode ser um deles. Tirar os repetentes do teste torna o teste **menos**
parecido com a produção, não mais — mesmo princípio do ADR-0008, que recusou descartar o Aluno sem
Etapa 1. Não se remove quem existe.

**Item fechado:** sai do *Not yet specified* do `map.md`, onde estava pendurado desde o ticket 01.

---

## 4. As métricas

**Escala: tudo medido em `A3`.** O ticket 04 cravou que o erro do Argumento Final é exatamente
`3×` o erro de `A3` — não existe régua separada. Relatórios escrevem os dois lados.

**Quatro métricas, e o papel importa mais que a métrica:**

| # | métrica | papel |
|---|---|---|
| 1 | **RMSE** em `A3` | **decide** o ranking entre modelos |
| 2 | **MAE** em `A3` | o número que se fala com um humano; diagnóstico via razão com o RMSE |
| 3 | **Viés** (erro médio com sinal) | valida se o RMSE é um desvio padrão honesto |
| 4 | **Erro de decisão** + RMSE na faixa | veto conversado, não ranking |

### 4.1 RMSE decide — e a sensibilidade a outlier é virtude

O ticket suspeitava do RMSE por "penalizar outlier pesado". Aqui isso é **virtude**: um aluno com
previsão muito errada é um aluno a quem o sistema disse uma bobagem grande na cara, e punir isso
desproporcionalmente está alinhado com o produto.

Razão mais forte: o `statistics.py` enfia esse número direto em `N(previsão, σ²)` — o **RMSE é o
σ**. Ranquear por outra métrica seria otimizar um número que depois teria que ser convertido em
RMSE de qualquer jeito.

### 4.2 O viés — métrica que existe porque o RMSE só é σ sob condição

```
RMSE² = viés² + variância do erro
```

`RMSE = desvio padrão do erro` **só quando o viés é zero**. Com viés, o RMSE engorda com ele e
deixa de ser dispersão pura. A direção é segura — RMSE é sempre ≥ o desvio padrão verdadeiro,
então usá-lo como σ erra para o lado **largo**, produzindo probabilidade mais cautelosa e nunca
mais confiante do que deveria. Mas isso precisa ser sabido, não presumido: se o viés der +3, três
pontos do RMSE são teimosia do modelo, não incerteza.

**Veredito sobre a herança:** "RMSE como σ" **não foi decisão errada e fica**. O que está errado no
`13,49` é outra coisa — (a) não tem procedência registrada; (b) está na escala errada desde o
ticket 04; (c) é o mesmo número para todo Aluno (ticket 11); (d) a forma normal pode não servir
(ticket 11).

### 4.3 MAE não é desvio padrão

Para erro bem-comportado, `MAE ≈ 0,8 × σ` — sistematicamente **menor**. Jogar o MAE na fórmula da
probabilidade estreitaria a distribuição e deixaria **toda probabilidade confiante demais**: o
Aluno que merece ouvir 70% ouviria 85%, justamente perto do corte.

**MAE é o número que se fala; RMSE é o número com que se calcula.**

Bônus: a razão esperada `1/0,8 ≈ 1,25` faz de `RMSE/MAE` um teste de normalidade de graça. Se a
razão vier em 1,6, existe uma minoria sendo massacrada e a média esconde. A régua entrega isso
pronto ao ticket 11, que precisa perguntar exatamente "os resíduos são normais?".

### 4.4 Erro de decisão — a métrica que o produto decide

O dataset já carrega curso, campus e turno reais de cada Aluno. Então:

```
Argumento Final real     = a1 + 2·a2 + 3·a3_real
Argumento Final previsto = a1 + 2·a2 + 3·a3_previsto
erro de decisão = % de Alunos em que (real ≥ corte) ≠ (previsto ≥ corte)
```

Frase que se fala em reunião: *"em 4,1% dos Alunos o sistema teria dito a coisa errada sobre
passar"*.

**Companheira obrigatória, porque a taxa sozinha mente:** um Aluno errado por 0,5 ponto e um
errado por 30 contam igual na taxa. Junto vai o **RMSE dentro da faixa de decisão** — o erro
medido só nos Alunos cujo Argumento Final real está perto do corte do curso deles.

**Largura da faixa:** `±1 RMSE do Argumento Final do baseline do ticket 07`, medido uma vez e
**congelado**. Assim a faixa é exatamente "onde o erro do modelo é capaz de virar a resposta", tem
motivo em vez de gosto, e é a mesma para todos os modelos comparados. Faixa por modelo
invalidaria a comparação.

**Por que 3 e 4 não decidem o ranking:** dependem de uma tabela de corte que ainda vai mudar
(tickets 14/15 do `pdf-extraction`); um sim/não joga fora quase toda a informação de 8.700
previsões; e não é ela que alimenta a probabilidade. Papel: **veto conversado** — um modelo que
ganha no RMSE e piora feio no erro de decisão não troca automaticamente, abre discussão.

### 4.5 Qual Nota de Corte — o sistema real do Aluno, não Universal para todos

**Correção de rumo desta sessão.** A recomendação inicial era "Universal para todos, declarado
como limitação", com base na leitura de que os tickets 14/15 do `pdf-extraction` tornavam o campo
de sistema não confiável. **Isso estava errado**, e a verificação mostrou por quê:

- **Ticket 14** afeta o *número da colocação* dentro de um sistema, não a identificação do
  sistema. Caso real conhecido: **um** registro em 66.313.
- **Ticket 15** afeta a 10ª coluna de classificação em registros na borda de página: **10 casos
  conhecidos**, 8 já marcados.
- O **ticket 06 do `pdf-extraction` já resolveu** a dedução da cota: existem `perfil_cota`,
  `sistema_negros`, `escola_publica`, `renda_baixa`, `ppi`, `pcd`, com `cota_padrao_suspeito`
  marcando padrão impossível.

**Cobertura medida nas 64.298 linhas do dataset:** apenas **8 registros suspeitos (0,01%)**, todos
nos triênios antigos. E **32,3% dos Alunos não são Universal**:

| perfil_cota | n |
|---|---:|
| Universal | 43.519 |
| EP / Alta Renda / Não-PPI | 7.928 |
| EP / Alta Renda / PPI | 4.522 |
| Cota para Negros | 3.802 |
| EP / Baixa Renda / PPI | 2.247 |
| EP / Baixa Renda / Não-PPI | 2.184 |
| demais (PcD) | 96 |

Assumir Universal para todos jogaria um terço da base contra um limiar errado — e como esses
Alunos seriam chamados de reprovados tanto pelo modelo quanto pela realidade, eles concordariam
trivialmente e **diluiriam a métrica para o lado otimista**.

**Decisão — a semântica correta do PAS:** quem tem cota concorre no Universal **e** nos sistemas
da cota dele ao mesmo tempo, e passa se limpar qualquer um:

```
aprovado = Argumento Final ≥ menor corte entre os sistemas em que o Aluno concorre
```

Exclusões declaradas: os 8 registros com `cota_padrao_suspeito`; linhas de corte que falham
`checksum_fecha` ou estão `parcial`; e cortes apoiados em menos de 3 convocados
(`convocados_com_argumento`, para não calibrar contra ruído). O relatório final registra quantos
Alunos caíram em cada exclusão.

---

## 5. Janela de treino, comparação e número único

- **Janela expansiva por padrão** (dobra N treina em tudo até T), porque imita a produção — você
  usaria todo dado que tem. Mas a janela é **parâmetro da função de split, não constante da
  régua**: é exatamente o que o ticket 08 varre. Se a régua cravasse a janela, o 08 não teria o
  que fazer.
- **Comparação entre modelos é sempre pareada dentro da dobra**, com a janela segurada igual.
  Nunca modelo A na dobra 2 contra modelo B na dobra 4. Isso resolve o problema de a dobra 1
  treinar com 2 triênios e a dobra 5 com 6: o que muda entre dobras é idêntico para os dois lados
  comparados e se cancela.
- **O número único de um modelo é o erro agrupado** sobre as 37.844 linhas de teste das 5 dobras,
  não a média dos 5 números. Média simples daria à dobra 1 (5.804 Alunos) o mesmo peso que à dobra
  5 (8.499). **A série dos 5 continua publicada ao lado** — é ela que mostra se o erro é estável
  entre anos, e é o insumo do ticket 08. O agrupado é quem entra no critério de aceite.

---

## 6. O teto de acurácia — medido antes de o mapa procurar por ele

Medição indicativa na **dobra 5** (treina até 2021/2023, testa em 2022/2024), classe majoritária,
n_treino = 44.661, n_teste = 7.514, desvio de `A3` no teste 9,17. **O lacre não foi tocado** —
dobras existem para decidir.

| preditor | RMSE em `A3` | vs. burro |
|---|---:|---:|
| repete a Etapa 2 (`A3 = A2`) | 5,187 | — |
| **regressão linear em (A1, A2)** | **4,690** | −9,6% |
| linear em (A1, A2) + os 6 EBs crus das Etapas 1/2 | 4,666 | −10,0% |
| LightGBM em (A1, A2) | 4,707 | −9,3% |
| **LightGBM, 400 árvores, + os 6 EBs crus** | **4,681** | −9,8% |

**Um LightGBM de 400 árvores empata com uma regressão linear de duas variáveis — 0,2%.** Adicionar
seis features cruas move 0,5%. `A1` e `A2` já contêm quase toda a informação existente sobre a
Etapa 3 (R² = 0,739); os ~26% restantes da variância são o ano do próprio Aluno — o dia bom, o dia
ruim, a doença, o estirão de estudo. Não está no dado, e nenhuma família de modelo o inventa.

**Diagnóstico de forma, de graça para o ticket 11:** o baseline burro deu **viés `+0,00`** e razão
**RMSE/MAE `1,26`**, contra o `1,25` teórico do erro normal bem-comportado. A forma normal se
sustenta no baseline — o trabalho do ticket 11 é largura **por Aluno**, não trocar a forma.

**Correção de leitura sobre o `13,49`.** A primeira leitura desta sessão foi "está quase igual ao
baseline burro (`3 × 5,187 = 15,72`), logo o ensemble não se justifica". Com o número melhor,
`3 × 4,690 = 14,07` é o Argumento Final de um modelo praticamente no teto, e o `13,49` fica logo
abaixo. A leitura correta não é que o ensemble erra mais: é que **todo mundo está no teto**. Ele é
injustificado por ser complicado sem ganhar nada. O defeito do `13,49` continua sendo outro — ser
**o mesmo número para todo Aluno**.

**Ressalva.** Uma dobra, classe majoritária, LightGBM sem ajuste de hiperparâmetro, sem testar
`curso`/`campus`/`turno` nem língua. **O teto não está declarado sobre feature não medida** — é o
ticket 09 que mede. Mas quando 400 árvores empatam com uma reta, a chance de o resto virar o jogo
é pequena. Os números definitivos são do ticket 07; estes são âncora para escrever o critério de
aceite sem chutar.

---

## 7. O critério de aceite

O ticket pedia "o número que, se batido, encerra o mapa". A §6 mostrou que **esse número não pode
ser sobre acurácia** — ela já estava no teto antes de o mapa começar. O critério muda de natureza:
de *melhorar* para *não estragar, e ser honesto*.

**Portão 1 — não-regressão.** RMSE agrupado em `A3` **não pior** que o melhor baseline trivial do
ticket 07 (dentro do ruído entre dobras), **nas duas classes**, com **|viés agrupado| ≤ 0,5** ponto
de `A3`. O viés entra como condição porque sem ele o RMSE não é um σ honesto e a probabilidade sai
torta.

**Portão 2 — coerência.** O ticket 04 mediu o problema real: as duas rotas que a tela mostra hoje
divergem **15,29** na mediana e **11% dos Alunos discordam sobre passar** — contradição interna
**maior que o erro do modelo**. O mapa fecha quando todo número exibido sai do mesmo `A3` por
aritmética e essa divergência é **zero por construção**, não pequena.

**Portão 3 — incerteza honesta.** `statistics.py` sem constante cravada; a largura vem de medição
e é **por classe**, no mínimo duas (ticket 14).

**Portão 4 — regra de parada.** A busca encerra quando o ganho seca: **menos de 1% relativo no
RMSE agrupado em dois tickets consecutivos** entre 08, 09 e 10. Pela §6 isso provavelmente dispara
já no 08 — e por isso os três entram **timeboxados**, registrado no `map.md`.

**Cláusula anti-renegociação.** O critério está escrito antes de o baseline definitivo existir.
Baixar a barra depois de ver o resultado é permitido, mas exige **decisão explícita e
registrada** — mesma disciplina da chave de força do manifesto (ticket 03), e a mesma doença de
reajustar depois de abrir o lacre.

**Descartado, e por que vale registrar:** a proposta inicial era "reduzir 20% relativo o erro de
decisão dentro da faixa" (33,5% → ≤26,8%). Morreu na medição. A tentação de exigir esse tipo de
número vai voltar, então fica a tabela:

| acerto dentro da faixa | erro | queda de RMSE necessária |
|---|---:|---:|
| 66,5% (baseline medido) | 33,5% | — |
| 72% | 27,9% | −20% |
| 75% ("errar 1 em 4") | 25% | **−32%** |
| 80% | 19,5% | −50% |

E os 33,5% não são defeito do baseline: para modelo sem viés e erro normal, o erro esperado numa
faixa de ±1 RMSE em torno do corte é `∫₀¹Φ(−u)du ≈ 31,6%`. O baseline está praticamente no limite
matemático, enquanto o espaço inteiro entre a regressão linear e o melhor modelo testado é 0,2%.

Isso também é o motivo de a faixa ficar **congelada** no RMSE do baseline do ticket 07: se cada
modelo usasse a própria faixa, a métrica seria auto-referente e **não poderia melhorar por
construção**.

---

## 8. A forma do código

**Mora em `src/pas_intelligence/validation.py`** — mesmo tratamento do `training_dataset.py` do
ticket 05. É código de produção, não script: os tickets 12 (pipeline) e 13 (promoção) importam.
Nada em `.scratch/`.

**O módulo dirige o laço.** Ele recebe o modelo e faz tudo: percorre as dobras, treina, prevê,
mede, monta o resultado. A alternativa — o módulo entrega as dobras e cada ticket escreve o
próprio laço — foi descartada porque cinco tickets escrevendo o mesmo laço escrevem cinco laços
ligeiramente diferentes: um esquece de segurar a janela, outro compara dobras cruzadas, outro
emite o número da classe minoritária na dobra 1. **As travas da §2.2 só são travas se existirem em
um lugar só** — senão a régua volta a ser prosa.

Benefício de graça: o lacre fica mecânico. O gerador de dobras **nunca produz** 2023/2025; não
existe caminho acidental até ele, só `holdout_final_use_uma_vez()`.

**Recebe uma fábrica, não um modelo pronto.** Fábrica = função que devolve um modelo novo, ainda
não treinado, a cada chamada, de modo que cada dobra treina do zero. Com um modelo já criado, a
dobra 2 treinaria em cima do que a dobra 1 já treinou — o erro clássico e silencioso desse tipo de
laço, que não quebra nada e só produz números bons demais.

**Resultado como estrutura tipada, não dicionário**, para que `resultado.rmse_agrupado` escrito
errado vire erro na hora em vez de devolver vazio e alguém reportar um número que não existe.
Campos obrigatórios por dobra e por classe: erro, `n_treino_classe`, `taxa_treino_classe`
(trava 3 da §2.2).

**Semente:** exigida explicitamente por rodada e registrada no resultado. Ao contrário do ticket
05, aqui existe passo aleatório — o modelo.

---

## 9. Onde continuar

- **Implementar `validation.py`** — sessão nova, com este relatório como especificação. Não precisa
  do contexto da sessão de decisão.
- **Ticket 07 (baseline):** congela a faixa de decisão e preenche o número do Portão 1. A §6 indica
  onde o baseline deve cair (~4,69 em `A3`).
- **Ticket 08 (janela):** varre a janela como parâmetro; curva de erro **só na classe majoritária**.
- **Tickets 09 e 10:** timeboxados, expectativa de empate.
- **Ticket 11:** viés `+0,00` e RMSE/MAE `1,26` indicam que a forma normal serve; o trabalho é
  largura por Aluno.
- **Ticket 13:** único autorizado a chamar `holdout_final_use_uma_vez()`.

---

## 10. Limitações

- **A medição da §6 é indicativa.** Uma dobra, classe majoritária, sem ajuste de hiperparâmetro,
  sem `curso`/`campus`/`turno`/língua. O número que vale é o do ticket 07.
- **A cobertura do erro de decisão não foi verificada.** No teste rápido só **34%** dos Alunos
  casaram com um corte, mas isso foi com filtro `sistema == 1` apenas. Com todos os sistemas deve
  subir; se não subir, é defeito de casamento de nome de curso, e precisa ser reportado antes de a
  métrica ser usada.
- **O resíduo das 1.465 linhas (§2.1) não bate com o relatório 02** — medianas 0,035 e 1,095 aqui
  contra +2,704 e +3,549 lá. Não investigado.
- **O 1% do Portão 4 é uma escolha, não uma medição.** Desenhado para ser pequeno perto do teto
  medido, não derivado de intervalo de confiança.

---

## 11. Glossário — termos novos deste relatório

Gravados em [`glossario.md`](../glossario.md), Parte 4: **Dobra**, **Validação deslizante**,
**Holdout lacrado**, **Receita**, **Modelo medido × modelo embarcado**.
