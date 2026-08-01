# Um LightGBM único, com valor faltante nativo, substitui o ensemble por volatilidade

O ensemble por volatilidade (quatro modelos de EB mais um meta-modelo roteador escolhido por CV) foi o desenho herdado, nunca medido contra a alternativa óbvia — um único GBM que modela interação e não-linearidade sozinho. O ticket 07 já havia mostrado que nem o ensemble nem o roteador batem o melhor componente sozinho nas linhas limpas (RMSE de EB 8,181 e 8,296, contra 8,042 do MLP isolado); o ticket 10 repete a pergunta sobre o alvo e a régua novos — `A3`, validação deslizante — e o veredito se confirma.

**Medido sobre as mesmas 5 dobras do ticket 06 (37.844 linhas, semente 20260728), com o conjunto de features do ticket 09** (as 6 legadas + `A1`/`A2` + as 3 derivadas de trajetória, RMSE de referência 5,057):

| candidato | RMSE geral | RMSE majoritária | RMSE minoritária | viés |
|---|---:|---:|---:|---:|
| Ridge (regularizado, `alpha=0,01`) | 5,057 | 5,014 | 5,357 | +0,215 |
| **LightGBM único** (`n_estimators=400, learning_rate=0,01, num_leaves=15`) | **5,014** | **4,992** | 5,187 | **+0,129** |
| Ensemble por volatilidade (linear × LightGBM, sem roteador) | 5,009 | 4,988 | 5,182 | +0,188 |

O ensemble reimplementado — mesmo mecanismo de `ensemble.py` (sigmoide sobre o CV de `[EB_PAS1, EB_PAS2]`, sem o meta-modelo roteador), treinado sobre `A3` e a régua nova — dá o menor número dos três, mas por **0,10%** sobre o LightGBM sozinho. Abaixo da barra de "ganho material" que o mapa já usa em outros pontos (1% relativo): **o ensemble é aposentado**. Reimplementá-lo custaria dois artefatos, uma sigmoide com dois limiares e uma dependência a mais no manifesto (ticket 03), para um ganho que a régua não consegue distinguir de ruído entre dobras (±0,37, ticket 07 §1).

**A volatilidade não desaparece — foi testada como feature**, per ADR-0009 (§ Consequences): CV de `[EB_PAS1, EB_PAS2]` somado às features do candidato vencedor não move o RMSE (5,014 → 5,014, −0,01%). A hipótese de que a volatilidade carrega sinal próprio, além do que já está nos EBs crus e nas derivadas de trajetória, **não se sustenta** — nem como mecanismo de arquitetura, nem como coluna de entrada.

**LightGBM sozinho ganha do Ridge por 0,85% relativo** (5,057 → 5,014) — abaixo da mesma barra de 1%, o que colocaria o desempate em "o mais simples vence" (Ridge). A escolha por LightGBM não se apoia nesse número isolado, mas em três coisas que se somam:

1. LightGBM é **estritamente melhor em todo eixo medido** — geral, majoritária, minoritária, viés (0,129 contra 0,215) e erro de decisão (7,0% contra 7,2% Ridge) — nunca perde em nenhum. Não é uma troca; é vantagem sem contrapartida, ainda que pequena.
2. **Valor faltante nativo**, testado como o mapa (ticket 14, restrição sobre o ticket 10) pediu: um único LightGBM enxergando `NaN` nas colunas derivadas da Etapa 1 do Aluno sem Etapa 1 (em vez de tratar o zero estrutural como desempenho) reduz o RMSE minoritário de **5,187 para 5,158** (0,56%) sem custar nada na majoritária. Ridge/linear não têm essa porta — teriam que imputar, inventando um valor.
3. **A alternativa que teria justificado o custo — dois modelos treinados por classe — foi medida e perde**: RMSE minoritário **5,379**, pior que o modelo único (5,187) e pior que o único-com-NaN (5,158). A dobra 1 treina o submodelo da minoria com **64 exemplos**; um modelo dedicado não tem dado suficiente cedo na série para superar o que o modelo conjunto aprende por generalização cruzada entre classes.

**Este é um critério com peso, não um desempate isolado** (redação do próprio ticket 14): a decisão não é "LightGBM ganha porque aceita faltante"; é que LightGBM já vence (por margem pequena) em todos os eixos, e o valor faltante nativo é o desempate que fecha a única lacuna onde a vantagem seria discutível.

## Consequences

- **A receita final:** LightGBM (`n_estimators=400, learning_rate=0,01, num_leaves=15, random_state` = a semente da rodada), sobre `FEATURES_CANONICAS` (`a1, a2, EB_PAS1, Red_PAS1, EB_PAS2, Red_PAS2, Cresc_EB, Cresc_Red, cresc_eb_pct, cresc_red_pct, sinal_cresc_eb`), com as 8 colunas derivadas da Etapa 1 (`a1`, as 6 legadas relacionadas à Etapa 1, as 3 derivadas de trajetória) trocadas por `NaN` nas linhas `etapa_1_ausente`. Hiperparâmetros escolhidos numa dobra de ajuste dedicada (treina 2016/2018, valida 2017/2019) — disjunta das 5 dobras de medição e do lacre — não sobre o número que este relatório reporta.
- **Os quatro `.joblib` de EB, o `meta_model`/`meta_scaler` e `ensemble.py` saem da rota de produção.** O ticket 13 é quem promove o artefato de fato; este ADR fixa a receita, não o arquivo. `ensemble.py` deixa de ser descrito como "em produção" em qualquer documentação nova.
- **`target_calculator.py`/`p1_pas3_model`/`red_pas3_model` já saem por decisão do ADR-0009** (alvo canônico); este ADR não muda essa consequência, só confirma que o substituto de fato é um LightGBM único, não outro ensemble.
- **A tabela de manifesto do ticket 03 ganha um bloco real para preencher**: um artefato só, features nomeadas, o tratamento de faltante como parte da receita (não um pré-processamento externo).
- **Risco aceito, registrado:** o ganho de LightGBM sobre Ridge (0,85%) não bateria a barra de "ganho material" (1%) usada em outros pontos do mapa se medido isoladamente. A escolha depende da soma dos três argumentos acima, não de um número só — decisão de julgamento, não mecânica, e reversível: se uma sessão futura quiser o piso mais simples (Ridge), o números para essa troca já estão nesta tabela.

## Considered Options

- **Ridge (regularização L2)**: descartado por margem pequena — perde em todo eixo medido (nunca por muito), e fecha a porta do valor faltante nativo. Seria a escolha se o critério fosse só "o mais simples, sem números fortes o suficiente para justificar mais".
- **Ensemble por volatilidade, reimplementado sem roteador**: descartado — 0,10% acima de LightGBM sozinho, dentro do ruído entre dobras (±0,37). Testado com fidelidade ao mecanismo original (mesma sigmoide, mesmos limiares 10/20), não uma versão redesenhada.
- **Meta-modelo roteador**: não testado de novo neste ticket — o ticket 07 já mediu que ele é pior que qualquer componente sozinho (RMSE de EB 8,296 nas linhas limpas, roteando 75% dos Alunos para o modelo que memorizou). Reconfirmar custaria uma medição sem hipótese nova.
- **Modelo multi-saída (prever P1, P2 e Redação da Etapa 3 juntos)**: não se aplica — o ADR-0009 já fixou que o alvo é só `A3`, não as três notas.
- **Dois modelos por classe (`etapa_1_ausente`)**: medido e descartado — RMSE minoritário pior que o modelo único (5,379 contra 5,187), porque a dobra 1 treina o submodelo da minoria com só 64 exemplos. Sem dado suficiente cedo na série, especializar por classe custa mais do que ganha.
