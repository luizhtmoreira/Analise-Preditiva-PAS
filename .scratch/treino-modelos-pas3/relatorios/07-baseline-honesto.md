# Relatório — Ticket 07: baseline honesto

**Ticket:** `.scratch/treino-modelos-pas3/issues/07-baseline-honesto.md`
**Status:** concluído
**Tipo:** medição + código reutilizável
**Régua:** `src/pas_intelligence/validation.py` (ticket 06), escrita nesta mesma sessão
**Script:** `scripts/baseline_honesto.py` — reproduz tudo abaixo com um comando
**Recorte de toda medição:** as **37.844** linhas de teste das 5 dobras
(2018/2020 a 2022/2024), semente **20260728**, dataset
`data/training/pas3_dataset.parquet` (64.298 linhas, 8 triênios)
**Lacre:** o triênio 2023/2025 **não foi tocado**. Nenhum número deste relatório vem dele.
**Privacidade:** só agregados e contagens.

> **Esta é a tabela de referência única que os tickets 08 a 13 citam.** Nenhum número aqui foi
> medido sobre dado que o preditor viu no treino — com **uma exceção declarada e quantificada**,
> a dos `.joblib` de hoje (§4), que não podem ser retreinados por dobra porque são artefatos
> congelados.

---

## 1. A tabela de referência

Erro em `A3` (o Argumento da Etapa 3, o Alvo Canônico do ticket 04), agrupado sobre as 37.844
linhas. O erro do Argumento Final é exatamente `3×` este número, e vai na coluna ao lado.

| preditor | RMSE `A3` | MAE `A3` | viés | RMSE/MAE | RMSE Arg. Final |
|---|---:|---:|---:|---:|---:|
| média do treino (responde sempre a média) | 9,094 | 7,260 | +0,264 | 1,25 | 27,282 |
| média do curso, ignorando o Aluno | 8,464 | 6,767 | +0,587 | 1,25 | 25,392 |
| repete a Etapa 2 (`A3 = A2`) | 5,661 | 4,477 | −0,005 | 1,26 | 16,984 |
| média das Etapas (`(A1+A2)/2`) | 5,481 | 4,304 | −0,002 | 1,27 | 16,443 |
| linear nas 6 features legadas | 5,408 | 4,290 | +0,663 | 1,26 | 16,223 |
| linear em (`A1`, `A2`) | 5,185 | 4,099 | −0,188 | 1,27 | 15,556 |
| **linear em (`A1`, `A2`) + as 6 legadas** | **5,167** | **4,088** | **+0,177** | **1,26** | **15,500** |
| — | | | | | |
| `modelo_arg_final` (o que roda hoje) ⚠ | 5,420 | 4,311 | +0,520 | 1,26 | 16,260 |

⚠ **medido sobre dado que ele já viu** — 95,2% de sobreposição. Ver §4.

**Por classe** (ticket 14: todo candidato reporta dois números):

| preditor | RMSE majoritária | RMSE minoritária (`etapa_1_ausente`) |
|---|---:|---:|
| repete a Etapa 2 | 5,632 | 5,921 |
| média das Etapas | 5,107 | 8,381 |
| **linear em (`A1`, `A2`)** | **5,038** | 6,353 |
| **linear em (`A1`, `A2`) + as 6 legadas** | 5,053 | **6,028** |
| `modelo_arg_final` ⚠ | 5,421 | 5,419 |

O agrupado da minoritária sai sobre **2.936** linhas, não 3.356: a **trava 1** do §2.2 do ticket
06 barrou a dobra 1, onde o treino tinha 64 exemplos da classe contra 420 no teste. A trava
disparou exatamente onde o relatório 06 previu.

**Série por dobra, classe majoritária** — o insumo do ticket 08. A classe minoritária **não tem
série** (trava 2), e é por isso que ela não aparece aqui.

| preditor | 2018/2020 | 2019/2021 | 2020/2022 | 2021/2023 | 2022/2024 |
|---|---:|---:|---:|---:|---:|
| repete a Etapa 2 | 6,066 | 5,515 | 5,791 | 5,721 | 5,187 |
| média das Etapas | 5,514 | 5,093 | 5,054 | 5,161 | 4,806 |
| linear em (`A1`, `A2`) | 5,438 | 4,978 | 5,075 | 5,094 | **4,705** |
| linear + as 6 legadas | 5,428 | 5,005 | 5,077 | 5,106 | 4,742 |

**Reconciliação com o §6 do relatório 06.** Lá a medição indicativa deu **4,690** para a linear
em (`A1`, `A2`); aqui a mesma célula (dobra 5, classe majoritária) dá **4,705**. A diferença é de
recorte de treino, não de resultado: o §6 treinou **só na classe majoritária** (n = 44.661),
enquanto a régua treina nas duas e reporta separado (n = 47.096). Treinar em todos e reportar por
classe é a leitura correta — o ticket 14 fechou que atender a classe não é o mesmo que treinar
nela, e *se* vale treinar separado é medição do ticket 10.

**Por que o agrupado (5,17) é pior que a dobra 5 (4,71).** As dobras iniciais treinam com menos
dado e são piores; o agrupado pondera as cinco pelo tamanho. A expectativa registrada no handoff
("RMSE ≈ 4,69") era o número da dobra 5, e ele bateu. **Não há sinal de vazamento** — o resultado
não veio melhor que o esperado, veio na direção certa pelo motivo certo.

---

## 2. Duas premissas que caíram nesta sessão

### 2.1 Não existe "em qual semestre o Aluno concorreu"

A tabela de Notas de Corte traz corte separado por semestre de ingresso, e o dataset não registra
o semestre do Aluno. Eu tratei isso como lacuna da especificação e perguntei. **A premissa estava
errada:** todos os Alunos fazem a mesma prova e concorrem de uma vez; existe um limite de vagas
para o 1º semestre, e quem não se classifica nele **continua vivo** disputando as vagas do 2º.

Consequência: o corte do 2º semestre é sempre o mais baixo, e `Aluno entrou` é
`Argumento Final ≥ o menor dos dois`. **Conferido nos dados: 1.317 de 1.317 chaves curso+sistema
que têm os dois semestres, sem uma exceção**, com o corte do 1º acima do 2º por 31,5 pontos de
Argumento Final na mediana.

Isso não é uma extensão da regra do §4.5 — é a mesma regra. O Aluno concorre em tudo ao mesmo
tempo (Universal, o sistema da cota dele, 1º e 2º semestre) e passa se limpar qualquer um.

### 2.2 A regra dos 3 convocados custava 34 pontos de cobertura

O §4.5 excluiu cortes apoiados em menos de 3 convocados, "para não calibrar contra ruído", e o
§10 deixou a cobertura como limitação em aberto. Medida agora: `convocados_com_argumento` vale 1
ou 2 em **68%** das linhas de corte, porque na última chamada de um sistema frequentemente só uma
pessoa é convocada — o que é a **definição literal** da Nota de Corte, não ruído. A regra derrubava
a cobertura de 91,9% para 55,8%.

**Decisão do dono do produto nesta sessão: relaxar para ≥ 1 convocado.** Cobertura final: **34.050
de 37.844 (90,0%)**; 3.794 Alunos (10,0%) ficam sem corte casado.

### 2.3 O "34% de cobertura" do §10 não era defeito

O §10 do relatório 06 registrou que só 34% dos Alunos casavam com um corte, e pediu que, se não
subisse, fosse reportado como defeito de casamento de nome de curso. **Subiu, e não é defeito.**
Sem exclusão nenhuma a cobertura é **91,9%**; o 34% era artefato do filtro `sistema == 1`
(Universal), e 32,3% da base não é Universal. **A limitação do §10 está fechada.**

---

## 3. Os baselines triviais — e o teto, confirmado

**O melhor baseline trivial é uma regressão linear em (`A1`, `A2`) mais as 6 features legadas:
RMSE 5,167 em `A3`.** A regressão linear em duas variáveis sozinha dá 5,185 — **0,3% pior**.
Somar seis features cruas move menos de meio por cento.

Repetir a Etapa 2 sem pensar (`A3 = A2`) dá 5,661, só **8,7%** pior que o melhor. E responder
sempre a média do treino — o preditor mais burro que existe — dá 9,094, que é essencialmente o
desvio padrão de `A3` (~9,1). Ou seja: **toda a informação que existe sobre a Etapa 3 comprime a
dispersão de 9,1 para 5,2, e nada do que foi testado passa disso.**

Isto confirma o teto medido no §6 do ticket 06 sobre uma base cinco vezes maior, e confirma o
timebox dos tickets 08, 09 e 10.

**Diagnóstico de forma para o ticket 11:** a razão RMSE/MAE fica entre **1,25 e 1,27** em todos os
baselines, contra 1,25 do erro normal bem-comportado. **A forma normal se sustenta** — o trabalho
do ticket 11 é largura **por Aluno**, não trocar a forma. O viés dos dois melhores baselines é
−0,188 e +0,177, bem dentro do ±0,5 do Portão 1.

---

## 4. Os `.joblib` de hoje — e o vazamento, quantificado

**Sobreposição medida:** dos 37.844 Alunos do recorte de teste, **36.034 (95,2%)** estão em
`data/banco_alunos_pas_final.csv`, a base em que os modelos atuais foram treinados. Sobram
**1.810 linhas limpas**, e elas estão bem distribuídas entre os cinco triênios (307, 369, 347,
396, 391) — a comparação limpo × sujo não está confundida com ano.

Medi tudo nos dois recortes. O contraste é a evidência.

### 4.1 `modelo_arg_final` — o único que a tela usa de verdade

| recorte | n | RMSE `A3` | MAE | viés |
|---|---:|---:|---:|---:|
| todas as linhas | 37.844 | 5,420 | 4,311 | +0,520 |
| só as linhas limpas | 1.810 | 5,288 | 4,182 | +1,277 |

Praticamente igual nos dois — **este modelo não decorou**. Mas o número que importa é outro:
**5,420 é pior que os 5,167 do melhor baseline trivial**, e isso *com* 95% de vantagem de ter
visto o dado. Um LightGBM em produção perde para uma regressão linear.

### 4.2 Os modelos de EB — onde o vazamento aparece

Escala de EB da Etapa 3, **não comparável** com a coluna de `A3` acima (é outra grandeza).

| preditor | RMSE (todas, n=37.844) | RMSE (limpas, n=1.810) | degradação |
|---|---:|---:|---:|
| **`modelo_rf`** | **5,198** | **8,422** | **+62%** |
| `modelo_mlp` | 8,668 | 8,042 | −7% |
| `modelo_lgbm` | 8,359 | 8,105 | −3% |
| `modelo_linear` | 8,795 | 8,370 | −5% |
| repete o EB da Etapa 2 | 9,632 | — | — |

**O `modelo_rf` é o retrato do vazamento.** Nas linhas que ele viu, RMSE 5,198 e razão RMSE/MAE de
**1,41** — muito acima do 1,26 de todo o resto, que é a assinatura de uma distribuição de erro
deformada por linhas reproduzidas de cor. Nas linhas limpas ele vira o **pior** dos quatro. Os
5,198 nunca foram qualidade.

---

## 5. Veredito: o arranjo de vários modelos se justifica?

**Não. Nenhum dos dois arranjos.** E há um achado que antecede a pergunta.

**O ensemble por volatilidade não está em uso em lugar nenhum.**
`ensemble.predict_with_dynamic_ensemble` não é chamado nem pela API (`api/services/`) nem pelo
Streamlit legado. O que a tela usa é o **meta-modelo roteador**, que escolhe *um* dos quatro
modelos por Aluno. O `CLAUDE.md` e o mapa descrevem o arranjo por volatilidade como se ele
rodasse; ele é código morto.

Medi os dois, nas 1.810 linhas limpas — as únicas em que o número quer dizer alguma coisa:

| arranjo ou modelo | RMSE EB (limpas) |
|---|---:|
| `modelo_mlp` sozinho | **8,042** |
| `modelo_lgbm` sozinho | 8,105 |
| ensemble por volatilidade (linear × lgbm) | 8,181 |
| **meta-modelo roteador (o que a tela usa)** | **8,296** |
| `modelo_linear` sozinho | 8,370 |
| `modelo_rf` sozinho | 8,422 |

**Os dois arranjos caem no meio dos próprios componentes.** Nenhum dos dois bate simplesmente usar
o MLP. O roteador é o pior dos dois, e o motivo é visível: ele manda **28.300 dos 37.844 Alunos
(75%) para o `modelo_rf`** — exatamente o modelo que decorou. Nas linhas que ele viu, o roteador
parece brilhante (RMSE 6,186); nas limpas ele desaba para 8,296. **O meta-modelo aprendeu a
escolher o modelo que memoriza.**

Resposta ao ticket: **o ensemble por CV de volatilidade não se justifica — nem ele, nem o
roteador que o substituiu na prática.** Ambos são indireção que custa e não paga. Isso alimenta o
ticket 10, que já entrava com o ensemble "sem o roteador" por decisão do ticket 04.

---

## 6. A proveniência do `13,49`

Encontrada, e ela é completa.

**O que o número é.** `13,49` é o **MAE** (erro absoluto médio, não RMSE) do `modelo_arg_final`,
na escala do Argumento Final, calculado em `calculate.py:81`:

```python
mae_arg = np.mean(np.abs(final_preds_arg - df_valid_arg['arg_final_real'].values))
```

**Sobre qual dado.** `calculate.py:10` filtra `banco_alunos_pas_final.csv` para
`Ano_Trienio == '2023-2025'` — ou seja, o número foi medido **no triênio 2023/2025, sobre a base
em que o próprio modelo foi treinado**. Medição no dado de treino, com a métrica errada.

**Onde ele vira RMSE.** A constante nasce como `ARG_FINAL_MAE = 13.49` (`app/streamlit_app.py:112`,
com o comentário "Erro médio do modelo") e é passada como o parâmetro `rmse` de
`calculate_approval_probability` em 6 lugares. O docstring de `statistics.py` afirma que é RMSE.

**Isso já estava documentado e o registro sumiu do repositório.** O commit o commit-raiz da PII (SHA não citado aqui de propósito — ver ticket 15)
("docs: record MAE-vs-RMSE calibration finding") criou
`docs/notas/calibracao-modelo-arg-final.md` com exatamente este achado, mais um recálculo
(MAE 14,32 · RMSE 18,01 · viés +8,82) e a decisão registrada de **não aplicar a correção**. O
arquivo não existe na árvore de trabalho atual e o commit não é ancestral do `HEAD`.

**O quanto difere do medido agora.** Na régua, `modelo_arg_final` dá **MAE 12,93 e RMSE 16,26** em
Argumento Final. O MAE medido bate com o `13,49` declarado — o que **confirma** que a constante é
um MAE. Mas o número que a fórmula da probabilidade precisa é o RMSE, e ele é **16,26**:

> **O RMSE real é 20,5% maior que a constante declarada** — dito do outro lado, a incerteza que o
> Aluno vê é 17,0% mais estreita do que deveria. Como o §4.3 do ticket 06 previu, isso deixa toda
> probabilidade confiante demais: o Aluno que merece ouvir 70% ouve mais.

O defeito **não é** que 13,49 seja grande demais (a leitura corrigida no §6 do ticket 06). É que
ele é (a) a métrica errada, (b) medido no próprio treino, (c) o mesmo para todo Aluno — que é o
trabalho do ticket 11.

---

## 7. A faixa de decisão, congelada aqui

**Largura = 15,500 pontos de Argumento Final** = 1 RMSE de Argumento Final do melhor baseline
trivial (linear em `A1`,`A2` + as 6 legadas, RMSE 5,167 em `A3`, `× 3`).

**Este número está congelado.** Ele não se recalcula por modelo — se cada modelo usasse a própria
faixa, a métrica seria auto-referente e incapaz de melhorar por construção.

### Erro de decisão

Cobertura: **34.050 Alunos** (90,0%); 3.794 sem corte casado. Corte = o menor entre os sistemas
*e* os semestres em que o Aluno concorre (§2.1). Exclusões: linhas de corte com `checksum_fecha`
falso ou `parcial`; Alunos com `cota_padrao_suspeito` (2 no recorte).

| preditor | erra sobre passar | na faixa | erra na faixa | RMSE `A3` na faixa |
|---|---:|---:|---:|---:|
| repete a Etapa 2 | 8,2% | 6.154 | 36,4% | 5,548 |
| média das Etapas | 7,9% | 6.154 | 35,8% | 5,286 |
| linear em (`A1`, `A2`) | 7,6% | 6.154 | 35,2% | 5,084 |
| **linear + as 6 legadas** | **7,4%** | 6.154 | 34,3% | **5,047** |
| `modelo_arg_final` ⚠ | 7,4% | 6.154 | 32,9% | 5,305 |
| média do treino | 10,6% | 6.154 | 34,7% | 9,773 |

Frase que se fala em reunião: **"em 7,4% dos Alunos o sistema teria dito a coisa errada sobre
passar"** — e, entre os que estão perto do corte, em **34,3%**.

**O 34,3% não é defeito do baseline.** O §7 do ticket 06 calculou que, para modelo sem viés e erro
normal, o erro esperado numa faixa de ±1 RMSE em torno do corte é `∫₀¹Φ(−u)du ≈ 31,6%`. Todos os
preditores medidos caem entre 32,9% e 36,4%. **Estão no limite matemático**, e a distância entre o
melhor e o pior é de 3,5 pontos. Confirma a tabela do §7 e a decisão de não fazer disso um
critério de aceite.

---

## 8. O Portão 1, preenchido

O Portão 1 do critério de aceite exige RMSE agrupado em `A3` **não pior** que o melhor baseline
trivial, **nas duas classes**, com `|viés agrupado| ≤ 0,5`. Os números que ele referencia agora
existem:

| o que o modelo novo precisa bater | valor |
|---|---:|
| RMSE `A3` agrupado, geral | **≤ 5,167** |
| RMSE `A3` agrupado, classe majoritária | **≤ 5,038** |
| RMSE `A3` agrupado, classe minoritária | **≤ 6,028** |
| \|viés agrupado\| | **≤ 0,500** |
| largura da faixa de decisão (congelada) | **15,500** |

O ruído entre dobras, que o Portão 1 admite como folga, é de **±0,37** no melhor baseline (5,428
na dobra 1 contra 4,742 na dobra 5, classe majoritária).

---

## 9. Decisões desta sessão que a especificação não cobria

Três, todas de convenção e nenhuma redecidindo o ticket 06. Ficam registradas porque código não
declara convenção sozinho.

1. **Sinal do viés.** `viés = média(previsto − real)`. Positivo significa **modelo otimista** —
   disse ao Aluno mais do que ele tirou. Escolhido porque é a leitura do produto, não a da
   estatística.
2. **A trava 1 vale para as duas classes.** O §2.2 escreveu a regra para a minoritária; implementei
   genérica ("qualquer classe cujo treino tenha menos exemplos que o teste"). Hoje ela nunca
   dispara na majoritária, e nas janelas que o ticket 08 vai varrer também não — conferido. É a
   mesma regra, num lugar só.
3. **O casamento com a tabela de cortes fica com quem chama a régua**, não dentro dela. A
   `erro_de_decisao` recebe a coluna `nota_corte` já pronta. Motivo: o casamento depende de
   `perfil_cota`, das exclusões do §4.5 e de nome de curso — nada disso é a régua, e enfiar tudo
   lá dentro amarraria `validation.py` ao formato do `notas_corte.csv`.

Uma quarta, mecânica: **`validation.py` recusa como feature qualquer coluna da Etapa 3**
(`a3`, `argumento_final`, `eb_pas3`, `eb_p1_e3`, `eb_p2_e3`, `red_e3`), porque são a resposta ou
aritmética exata dela. `lingua_e3` **não** entra na lista — o Aluno informa a própria língua
(ticket 04), então ela é conhecida antes da prova.

---

## 10. Defeitos encontrados

1. **`scripts/baseline_avaliacao.py:55` tinha o vetor de features errado — corrigido.** A ordem
   real, lida de `booster.feature_name()` dos próprios artefatos, é
   `[EB_PAS1, Red_PAS1, EB_PAS2, Red_PAS2, Cresc_EB, Cresc_Red]`; o script declarava
   `[EB_PAS1, EB_PAS2, Cresc_EB, Media_EB, Std_EB, CV_EB]`, com 5 das 6 posições trocadas. É a
   causa dos `R² = -83` e `MAPE = 1e+19` do ADR-0007. O script foi marcado como **superado** por
   `scripts/baseline_honesto.py`, porque o método dele (KFold aleatório sobre a base de treino)
   também caiu com o ticket 06 — mas o vetor foi consertado para o arquivo não ficar como
   armadilha.
2. **`p1_pas3_model.joblib` e `red_pas3_model.joblib` seguem não carregando**
   (`ModuleNotFoundError: No module named '_loss'`). A falha silenciosa em
   `target_calculator.py:66` já está corrigida (commit `a7573ff`). O ticket 04 remove os dois de
   qualquer forma.
3. **`ensemble.py` é código morto** e o `CLAUDE.md` o descreve como se estivesse em produção. Ver
   §5.
4. **⚠ PII no repositório remoto.** O commit o commit-raiz da PII (SHA não citado aqui de propósito — ver ticket 15) traz
   `docs/notas/calibracao-modelo-arg-final.md` com **nome completo de 6 Alunos reais e a chance de
   aprovação de cada um**. O commit não é ancestral do `HEAD`, mas é alcançável pela branch
   `feat/proof-section` — **e por `origin/feat/proof-section`, ou seja, está publicado**. Isso
   viola a restrição dura de privacidade do mapa ([[project_parser_privacy]]) e sobreviveu às 4
   rodadas de expurgo de 2026-07-25. **Não mexi**: reescrever uma branch já publicada é
   destrutivo e é decisão sua.

---

## 11. Limitações

- **Os `.joblib` de hoje só têm 1.810 linhas limpas** (4,8% do recorte). O número limpo é honesto,
  mas o intervalo de confiança dele é largo — serve para diagnosticar vazamento (onde a diferença
  é de 62%, inequívoca), não para ranquear modelos que empatam.
- **O erro de decisão cobre 90,0%, não 100%.** Os 3.794 Alunos sem corte são cursos que não
  aparecem na tabela de Notas de Corte naquele triênio; não foi investigado se há padrão neles.
- **A tabela de cortes ainda vai mudar** (tickets 14/15 do `pdf-extraction`). Por isso o erro de
  decisão é veto conversado e não ranking, e por isso a faixa foi congelada no RMSE do baseline —
  que não depende da tabela.
- **Os modelos de EB são medidos em EB, não em `A3`.** Não há conversão exata: `A3` precisa de
  P1, P2 e Redação separados, e eles preveem só a soma P1+P2. A comparação entre eles é interna à
  escala de EB, e é suficiente para o veredito do §5, que é sobre o arranjo e não sobre a escala.
- **Nenhum hiperparâmetro foi ajustado.** Os baselines são triviais de propósito — é o que
  "baseline" quer dizer. O ticket 10 é quem ajusta.

---

## 12. Glossário — termos novos deste relatório

Para gravar na [`glossario.md`](../glossario.md), Parte 4:

**Vazamento (*leakage*)** — medir um modelo em linhas que ele já viu no treino. O número sai bom
por memória, não por qualidade, e some quando o modelo encontra alguém de verdade. Assinatura
neste projeto: o `modelo_rf` com RMSE 5,198 nas linhas vistas e 8,422 nas limpas.

**Linha limpa** — linha do recorte de teste que **não** está na base em que o modelo foi treinado.
É a única sobre a qual o número de um artefato congelado quer dizer alguma coisa.

**Artefato congelado × receita** — um `.joblib` que já existe não pode ser retreinado por dobra;
ele é medido sobre as mesmas linhas de teste, mas o que se mede é *aquele arquivo*, não a receita.
Por isso ele não passa por `avaliar()` e carrega sempre a ressalva do vazamento.

**Meta-modelo roteador** — o arranjo que a tela usa: um modelo que, para cada Aluno, escolhe qual
dos quatro modelos de EB responde. Diferente do **ensemble**, que mistura as respostas com pesos.
Medido no §5: manda 75% dos Alunos para o modelo que memorizou.

**Erro de decisão** — a fração de Alunos em que o sistema erra o **sim/não** sobre passar, mesmo
quando erra pouco no número. Vem sempre acompanhado do **RMSE dentro da faixa de decisão**,
porque errar por 0,5 ponto e errar por 30 contam igual na fração.

**Faixa de decisão** — a janela de ±1 RMSE de Argumento Final em torno do corte do Aluno, onde o
erro do modelo é capaz de virar a resposta. Congelada em **15,500** neste ticket, e a mesma para
todos os modelos comparados daqui em diante.

**Portão** (do critério de aceite) — cada uma das quatro condições que o modelo novo precisa
cumprir para ser promovido. O Portão 1 é não-regressão, e os números dele estão no §8.

---

## 13. Onde continuar

- **Ticket 08 (janela):** varre a janela como parâmetro de `gerar_dobras`. A série por dobra do §1
  é o insumo. Curva de erro **só na classe majoritária** — para a minoritária a régua devolve um
  número só, de propósito.
- **Ticket 09 (features):** o teto do §3 **não** está declarado sobre `curso`, `campus`, `turno`
  nem língua. É o 09 que mede. `média do curso` sozinha dá 8,464, então curso carrega pouca
  informação isolado — mas isso não é o mesmo que carregar pouca *a mais*.
- **Ticket 10 (família):** o §5 já entrega o veredito de que o arranjo de vários modelos não paga.
- **Ticket 11 (incerteza):** RMSE/MAE entre 1,25 e 1,27 em tudo — a forma normal serve. A largura
  honesta hoje é **16,26** em Argumento Final, não 13,49. E precisa virar por Aluno.
- **Ticket 13:** único autorizado a chamar `holdout_final_use_uma_vez()`.
