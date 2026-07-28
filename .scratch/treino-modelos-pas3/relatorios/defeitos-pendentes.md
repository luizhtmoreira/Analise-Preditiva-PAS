# Defeitos pendentes conhecidos — `src/pas_intelligence/` e avaliação de modelos

Registro consolidado dos defeitos **documentados e ainda não corrigidos** na camada de
inteligência do PAS (`src/pas_intelligence/`, `scripts/baseline_avaliacao.py`, `docs/adr/`).
Não substitui os relatórios de ticket em `relatorios/` — cada entrada aponta para a fonte e
resume o que falta fazer. Objetivo: ponto único de partida para decidir o próximo ticket.

Convenção herdada do mapa `pdf-extraction`: cada entrada tem **Onde foi encontrado**, **O
defeito**, **O que falta fazer** e **Severidade** (impacto nos dados/produto, não esforço).

---

## 1. ⏳ A faixa `[−100, 100]` da P2 não tem procedência — **aguardando o dono do produto**

> **O que falta é uma resposta, não investigação.** Quantos itens tem a Parte 2, e como o desconto
> por erro fecha o piso? Perguntado em 2026-07-28; o dono do produto vai olhar junto das outras
> pendências. Enquanto isso o código roda com `P2_MAXIMO = 100.0` / `P2_MINIMO = -100.0`, que são
> chute herdado. A parte do teste deste defeito **já está corrigida** (ver "O que falta fazer").

**Onde foi encontrado:** `pytest tests/test_pas_intelligence.py` —
`TestTargetCalculator::test_guaranteed_scenario` falha. Confirmado por `git stash` em
2026-07-26 que a falha **antecede** as mudanças do ticket 03 (34 passavam / 1 falhava antes,
37 passam / 1 falha depois). Não foi introduzida por aquele trabalho.

**Fato de domínio que resolve a leitura:** no PAS, **nota de prova pode ser negativa** — o
Escore Bruto desconta erro, tal como o Argumento pode ser negativo. O próprio código já
carrega isso: `target_calculator.py:125` trunca `p1_pred` em `max(-20.0, min(20.0, ...))`,
ou seja, P1 ∈ [−20, 20].

**O defeito:** o teste espera que um cenário garantido produza
`status == 'garantido'` e `p2_necessario == 0.0`. Mas o código
(`target_calculator.py:291-304`) trabalha com a faixa simétrica implícita **P2 ∈ [−100, 100]**:

```python
if   p2_necessario >  100:  status = 'impossivel'
elif p2_necessario < -100:  status = 'garantido';  p2_necessario = -100.0
elif p2_necessario >   80:  status = 'improvavel'
else:                        status = 'possivel'
```

Reproduzido com o cenário do próprio teste (histórico máximo, alvo baixo):

```
status = possivel  |  p2_necessario = -99.44  |  arg_pas3_necessario = -99.233
```

Sob P2 com sinal, `-99.44` significa literalmente *"você precisa de pelo menos −99,44 na
Parte 2"* — quase qualquer desempenho serve, mas **não é garantido**, porque ainda existe um
desempenho pior que isso dentro da faixa. Então `'possivel'` está correto e o truncamento em
`-100.0` (piso da faixa) é coerente. **Quem codifica o contrato errado é o teste**, ao presumir
que zero é o mínimo da prova.

**O que falta fazer:**

1. ~~**Corrigir o teste**, não o código.~~ **FEITO em 2026-07-28.** `test_guaranteed_scenario`
   agora usa `arg_alvo=-500` (baixo o bastante para levar `p2_necessario` abaixo do piso) e
   afirma `p2_necessario == P2_MINIMO`. Entrou junto um
   `test_alvo_baixo_mas_dentro_da_faixa_ainda_e_possivel_nao_garantido`, que fixa a fronteira
   que o teste antigo confundia com zero: `arg_alvo=-100` dá ≈ −99,4 e é `'possivel'`, não
   `'garantido'`. `pytest tests/`: **290 passam, 0 falham**.
2. **Verificar qual é o piso real da P2** — segue aberta, e é a pergunta de verdade. Os literais
   viraram as constantes nomeadas **`P2_MAXIMO` / `P2_MINIMO`** em `target_calculator.py`, com
   docstring registrando que não têm fonte no Edital, para que a correção seja uma linha só
   quando o número real aparecer. Se a faixa real for mais estreita (a P2 tem número finito de
   itens, e o piso é `−N` para `N` itens), o ramo `'garantido'` é **código morto** e nenhum aluno
   jamais o alcança. Se for mais larga, o `'impossivel'` dispara cedo demais.
3. Ao mexer, checar o consumidor: `api/services/gestao_service.py:265` compõe
   `eb_nec = path.p1_estimado + path.p2_necessario`, e um `-100.0` ali empurra o EB necessário
   para muito abaixo de zero antes de ir para `calculate_cohort_evolution_probability`.

**Severidade: era baixa para o teste (resolvida), segue média para o piso não documentado.** A
faixa `[−100, 100]` sem procedência decide, sozinha, quando o produto diz "impossível" e quando
diz "garantido".

**Nota de comunicação (não é defeito de lógica):** mesmo correto, o texto
*"Meta alcançável! Você precisa de -99.4 pts na Parte 2"* é ruim de ler. Um aluno nessa
situação entende melhor "praticamente qualquer desempenho na Parte 2 mantém sua aprovação".
Melhoria de mensagem, sem urgência.

---

## 2. `ADR-0007` (baseline dos modelos v1) é inválido: features na ordem errada

**Onde foi encontrado:** ticket 03, ao inspecionar o booster para medir o formato nativo do
LightGBM. Ver `relatorios/03-formato-e-versionamento-do-artefato.md`, seção 2.2.

**O defeito:** os modelos foram treinados com

```
booster.feature_name() → ['EB_PAS1', 'Red_PAS1', 'EB_PAS2', 'Red_PAS2', 'Cresc_EB', 'Cresc_Red']
```

e `scripts/baseline_avaliacao.py:55` os alimenta com

```
FEATURE_COLS_BASE = ["EB_PAS1", "EB_PAS2", "Cresc_EB", "Media_EB", "Std_EB", "CV_EB"]
```

Só a posição 1 coincide; as outras cinco estão trocadas. O `CLAUDE.md` documenta o vetor
correto, o que confirma que o errado é o script de avaliação. O LightGBM aceitou em silêncio
porque recebeu um array NumPy — array não carrega nome de coluna, só posição.

Isso explica os números impossíveis do ADR: `R² = -83.4`, `MAPE = 1.25e+19`, `MaxErr = 31293`
num alvo que vai de 0 a 92. Não é desempenho ruim; é modelo alimentado com lixo.

**Situação em 2026-07-28 (ticket 07): resolvido, por substituição.**

- `FEATURE_COLS_BASE` **foi corrigido** para a ordem real, e o cabeçalho do script agora declara
  que ele está **superado** — não para ressuscitá-lo, mas para o arquivo não seguir sendo uma
  armadilha para quem o rodar sem ler.
- O script **não foi rerodado**, porque o método dele caiu junto com o vetor: KFold aleatório
  sobre `banco_alunos_pas_final.csv` mede interpolação dentro de anos conhecidos, sobre a base em
  que os modelos foram treinados. Ambos os defeitos foram substituídos de uma vez por
  `scripts/baseline_honesto.py`, que mede sobre a régua do ticket 06.
- A linha de base válida agora existe: **RMSE 5,167 em `A3`** (§1 de
  [`07-baseline-honesto.md`](07-baseline-honesto.md)). Ela é a régua que o ADR-0007 prometia ser.
- A suspeita registrada aqui **se confirmou**: a correção não inverteu o ranking a favor dos
  modelos. O `modelo_arg_final` perde para uma regressão linear de duas variáveis mesmo tendo
  visto 95,2% do teste.

**O que falta fazer:** superscrever o ADR-0007 apontando para o ticket 07 como a linha de base
válida. Fica para o ticket 13, junto da promoção.

**Severidade: era alta, agora baixa** — a régua de borracha foi substituída por uma medida.

---

## 3. `p1_pas3_model` e `red_pas3_model` não carregam (causa raiz pendente)

**Onde foi encontrado:** ticket 03, seção 2.1. Já registrado no ADR-0007 em 2026-07-20 como
nota de rodapé (`⚠ Modelo não carregado — incompatibilidade de versão do sklearn`) sem
tratamento.

**O defeito:** ambos falham com `ModuleNotFoundError: No module named '_loss'` no ambiente
atual (`python 3.14.3`, `scikit-learn 1.9.0`). Foram serializados quando o `sklearn` ainda
tinha o módulo interno `_loss` em outro lugar. Os números seguem íntegros no arquivo; a receita
de remontagem é que aponta para o vazio.

**Estado atual — mitigado, não corrigido (2026-07-26):** a degradação deixou de ser silenciosa.
`target_calculator.py` agora registra o motivo em `self.model_load_error`, devolve
`fallback_reason` junto de `method`, grita em log `ERROR`, e levanta `ModelLoadError` com
`PAS_STRICT_MODELS=1`. O `except Exception: pass` de `api/services/gestao_service.py:268` virou
`logger.exception`. **Mas a calculadora reversa continua respondendo por média ponderada em vez
de ML** — a feature segue degradada, agora com aviso.

**O que falta fazer:** ticket 12 regera os artefatos sob o esquema decidido no ticket 03
(manifesto com versões, portão de carregamento no build). A partir daí, ligar
`PAS_STRICT_MODELS=1` em produção. Também revisar o contrato de features de
`predict_stable_components` (`target_calculator.py:96-119`): ele monta 11 colunas — 6 base mais
`delta_p1`, `delta_red`, `delta_p2`, `mean_p1`, `mean_red` — cuja ordem só está registrada num
**comentário**. É exatamente a fragilidade do defeito 2 esperando para se repetir; essa ordem
tem que passar a viver no `manifest.json`.

**Severidade: média.** A feature degrada em vez de mentir, e agora avisa. Mas a calculadora
reversa é um dos produtos anunciados e está sem ML há meses.

---

## 4. ~~Nota não informada é substituída por notas inventadas~~ — **CORRIGIDO em 2026-07-27**

**Onde foi encontrado:** ticket 14 (Aluno sem Etapa 1), lendo o caminho de entrada da Gestão.
`api/schemas/gestao.py:12,16` declara `red_pas1: float = 6.0` e `red_pas2: float = 6.0`;
`landing-page/app/(dashboard)/gestao/page.tsx:31-36` coage o nulo do Supabase com
`Number(s.p1_pas1 ?? 0)` e `Number(s.red_pas1 ?? 6)`. No Preditor público,
`landing-page/components/public/predict/PreditorPage.tsx:413` inicializa o formulário com
`emptyScores() = { p1: "0", p2: "0", red: "0" }`.

**O defeito:** três estados do mundo colapsam no mesmo vetor de entrada — **não fez a Etapa**
(fato definitivo), **não informado** (lacuna de cadastro) e **tirou zero** (raro, mas real numa
nota isolada). E o colapso do segundo não é em zero: é numa Redação **mediana inventada**. Com as
estatísticas de PAS 1 de `gestao_service.py:38` (`mean_red=6,9051`, `std_red=1,8409`):

```
Red = 6,0 → A_red = (6,0 − 6,9051)/1,8409 × 1,00 = −0,49
Red = 0,0 → A_red = (0,0 − 6,9051)/1,8409 × 1,00 = −3,75
```

**+3,26 pontos de Argumento Final presenteados**, na direção do otimismo: o Coordenador Pedagógico
vê o Aluno mais perto da Nota de Corte do que ele está.

**Condição exata de disparo — e ela é mais estreita do que parece.** `??` é *nullish coalescing*:
dispara em `null`/`undefined`, **não em `0`**. Logo:

```js
Number(0 ?? 6)     // → 0   nota zero registrada passa intacta
Number(null ?? 6)  // → 6   só aqui a Redação é inventada
```

- **Hoje, com a Etapa 1 Ausente gravada como `0, 0, 0`** (a convenção do Edital), o defeito **não
  encosta no Aluno sem Etapa 1**. O A1 dele sai correto.
- **Ele dispara** em qualquer linha da `tabela_mestra` com `red_pas1`/`red_pas2` genuinamente NULL
  — o estado "não informado". Quantas linhas são: **não medido**, depende de como a
  `tabela_mestra` é populada.
- **Ele passa a dispararar em 100% do Aluno sem Etapa 1** no instante em que o ADR-0008 for
  implementado, porque ausência declarada significa `None` no lugar de `0`. **Consertar a
  representação sem consertar o coalescing acorda o bug exatamente na classe que a mudança queria
  proteger.**

O `?? 0` das objetivas não é inocente por ser zero: também inventa — afirma "tirou nota
catastrófica" onde a verdade é "não sei". Só é menos visível porque coincide com a convenção do
Edital.

O default do Pydantic é um terceiro disparo, independente: vale quando a **chave falta no JSON**,
não quando ela vem nula. O frontend sempre manda a chave, então está dormente nesse caminho e vivo
para qualquer outro chamador.

**Corrigido em 2026-07-27** (ticket 14), em três arquivos:

- `api/schemas/gestao.py` — `red_pas1`/`red_pas2` perderam o default `6.0`. As seis notas são
  obrigatórias; nota nula ou chave ausente vira **422 nomeando o campo**, verificado
  (`Field required` / `Input should be a valid number`).
- `landing-page/app/(dashboard)/gestao/page.tsx` — o `?? 0` / `?? 6` saiu. As linhas são
  **particionadas** em completas e incompletas; só as completas vão para a API, e as incompletas
  aparecem num aviso nomeando Aluno e campos faltantes. Ninguém recebe número inventado e ninguém
  some da tela em silêncio.
- `landing-page/lib/api.ts` — o erro de `fetchGestao` passou a carregar o corpo da resposta. Sem
  isso, o 422 preciso lia na tela como "API indisponível", e o conserto barulhento voltava a ser
  silencioso.

`pytest`: 248 passam, 2 falham — as duas pré-existentes (defeito 1 e `test_pdf_gen_manual`, que tem
caminho Windows cravado). `eslint` e `tsc --noEmit` limpos nos arquivos tocados.

**O que ainda falta, e é outro defeito:** o Preditor público
(`landing-page/components/public/predict/PreditorPage.tsx:413`) continua nascendo com
`emptyScores() = { p1: "0", p2: "0", red: "0" }`. Ali o zero é digitado por um humano, então a
correção não é propagar nulo — é o **controle de ausência declarada** do
[ADR-0008](../../../docs/adr/0008-aluno-sem-etapa-1-atendido-com-funcao-propria.md), que é decisão
de interface e não entrou neste conserto.

**Severidade: era média (dependia de quantas linhas da `tabela_mestra` têm Redação NULL — nunca
medido, e provavelmente nenhuma, porque a carga vem direto do Cebraspe). O motivo de consertar
agora foi outro:** a mudança de representação do ADR-0008 — ausência como `None` no lugar de `0` —
transformaria este defeito num erro de 100% de cobertura exatamente na classe que a mudança
pretendia proteger. Consertado antes, a ordem deixou de importar.

---

## 5. O roteador do ensemble é cego à direção do Momentum

**Onde foi encontrado:** ticket 14, ao registrar o Momentum como hipótese central do produto.
`src/pas_intelligence/ensemble.py:24-55` roteia por Volatilidade (CV), que é `std/mean` — grandeza
sem sinal.

**O defeito:**

```
CV([30, 35]) = 7,69%   ← subiu 5 pontos
CV([35, 30]) = 7,69%   ← caiu 5 pontos
```

Idênticos. O Aluno que subiu e o Aluno que caiu recebem a **mesma decisão de roteamento**. A
direção existe apenas em `c_eb`/`c_red`, que são features dos modelos, não do roteador. Ou seja: o
mecanismo que decide *qual modelo usar* é cego exatamente à hipótese que motivou o produto.

Caso limite relacionado, do mesmo módulo: sobre `[0, eb_pas2]` o CV devolve **exatamente 100%**
para qualquer `eb_pas2`. Não é volatilidade alta — é a assinatura de grandeza indefinida (o
Momentum do Aluno sem Etapa 1). O roteador não tem como distinguir os dois casos.

**O que falta fazer:** entra como evidência no ticket 10, que julga se o meta-modelo por
volatilidade paga o nível de indireção que custa. Se ficar, o roteador precisa de grandeza com
sinal; se sair, o defeito morre junto.

**Severidade: média.** Não produz número errado por si só — os modelos roteados recebem `c_eb` e
`c_red` e enxergam a direção. Mas a escolha de *qual* modelo confiar é feita com metade da
informação, e o ticket 10 mede num arranjo que já está sob julgamento.

---

## 6. `TRIENNIUM_STATS` não bate com os Editais — a API calcula o Argumento com números que não são do Cebraspe

**Onde foi encontrado:** ticket 04, ao verificar se `A1` e `A2` podem ser calculados exatos para o
Aluno vivo. Ver `relatorios/04-alvo-canonico-argumento-ou-tres-notas.md`, §4 da conversa e §9.

**O defeito:** `api/services/gestao_service.py:36-51` carrega médias e desvios num dicionário
próprio, e eles divergem do `pas_constants.OFFICIAL_STATS`, que vem dos Editais:

| triênio / etapa | `TRIENNIUM_STATS` (P2) | Edital | |
|---|---|---|---|
| 2023-2025 PAS1 | 25,3330 / 14,6860 | 25,333 / 14,686 | bate |
| 2023-2025 PAS2 | 29,2750 / 14,2913 | 29,275 / 14,604 | média bate, **desvio não** |
| 2022-2024 PAS1 | 20,7094 / 13,5819 | 20,406 / 13,533 | **não bate** |
| 2022-2024 PAS2 | 30,3477 / 13,2532 | 29,980 / 13,213 | **não bate** |
| 2022-2024 PAS3 | 32,0862 / 14,1289 | 31,740 / 14,063 | **não bate** |

As de 2022-2024 têm quatro casas decimais e desvio sistematicamente maior que o oficial — cara de
**calculadas de uma amostra de alunos**, não copiadas do Edital. Consequência: o Argumento que a
API produz não bate com o do Edital, e a divergência é maior justamente no desvio, que é o
denominador de todo z-score.

**Estado da fonte (verificado em 2026-07-27):** os números certos **já existem** para todos os
triênios que a API serve hoje. `pas_constants.OFFICIAL_STATS` tem as 24 chaves `(ano, etapa)` dos
8 triênios, e a extração `saida-nova/medias_desvios.csv` (75 linhas) cobre os cinco triênios cuja
tabela sai em Edital avulso — 2016/2018 a 2020/2022, ou seja 15 chaves já absorvidas pelo
`OFFICIAL_STATS`. **Não é mais uma pergunta em aberto: é uma troca de dicionário.**

O que o `medias_desvios.csv` **não** resolve são as chaves do triênio vivo: `(2024,1)` e
`(2025,2)` não estão nele (ele para em 2020/2022) nem no `OFFICIAL_STATS`. Essas continuam
dependendo da extração dos Editais por Etapa — item bloqueante no `map.md`.

**O que falta fazer:** apagar `TRIENNIUM_STATS` e consumir `OFFICIAL_STATS` direto, chaveado por
`(ano, etapa)` em vez de por string de triênio.

**Severidade: alta.** Enquanto `A1` e `A2` eram só entrada de um modelo que previa o Argumento
inteiro, um erro ali se diluía. Na rota canônica do ADR-0009 eles são a parte **exata** da conta —
`Argumento Final = A1 + 2·A2 + 3·Â3` — e um desvio errado no denominador contamina ⅗ do peso, sem
nada para compensar. Alta severidade com conserto barato: a fonte certa já está em disco.

---

## 7. ~~O override de P1/Redação é ignorado em silêncio se só um dos dois for preenchido~~ — **CORRIGIDO em 2026-07-27**

**Onde foi encontrado:** ticket 04, ao definir o contrato do Estimador Auxiliar.

**O defeito:** `src/pas_intelligence/target_calculator.py:262`

```python
if p1_override is not None and red_override is not None:
```

É **`and`**. O Aluno que mexe só na caixa da Redação e deixa a de P1 em branco tem **os dois**
overrides descartados, e a conta volta a usar o Estimador Auxiliar sem nenhum aviso. Ele vê na
tela o número que digitou e um P2 necessário que não corresponde a ele.

O caso não é hipotético nem raro: a Redação é o estimador **mais sensível** — 1 ponto de erro nela
move o P2 necessário em ~0,95 ponto, contra 0,60 do P1 — então é a caixa que o Aluno tem mais
motivo para mexer sozinha.

**Corrigido em 2026-07-27**, em `src/pas_intelligence/target_calculator.py`. Cada override passou
a valer por si; o Estimador Auxiliar só é consultado para o componente que ficou em branco, e só é
consultado quando algum ficou:

```python
if p1_override is None or red_override is None:
    previsao = self.predict_stable_components(notas_existentes)

p1_pred  = p1_override  if p1_override  is not None else previsao['p1_pred']
red_pred = red_override if red_override is not None else previsao['red_pred']
```

Coberto por `tests/test_pas_intelligence.py::TestTargetCalculator::test_override_parcial_e_respeitado`,
que verifica os quatro casos (nenhum, só P1, só Redação, ambos) e — o que importa de verdade — que
o override sozinho **move o P2 necessário**, não só o texto exibido.

`pytest tests/test_pas_intelligence.py`: 38 passam, 1 falha — a falha é o **defeito 1** desta mesma
lista, pré-existente e não relacionada.

**Severidade: era média, e bloqueante para o ADR-0009** — a decisão de restringir o override ao
caminho reverso só faz sentido se o override **funcionar** no caminho reverso.

---

## 8. ⚠ PII de Aluno real publicada no repositório remoto

**Onde foi encontrado:** ticket 07, ao rastrear a proveniência do `13,49`.

**O defeito:** o commit o commit-raiz da PII (SHA não citado aqui de propósito — ver ticket 15) cria `docs/notas/calibracao-modelo-arg-final.md`, que contém uma
tabela com o **nome completo de 6 Alunos reais** e a chance de aprovação calculada para cada um.

O commit **não é ancestral do `HEAD`** e o arquivo não existe na árvore de trabalho atual — mas
ele é alcançável pela branch `feat/proof-section` **e por `origin/feat/proof-section`**, ou seja,
está publicado. Sobreviveu às 4 rodadas de expurgo de 2026-07-25.

Isso viola a restrição dura de privacidade do mapa: *"nenhum dado de aluno vai para arquivo
commitado, relatório, teste ou exemplo"*.

**O que falta fazer:** decisão do dono do produto. Reescrever uma branch já publicada é
destrutivo e não foi feito nesta sessão. O conteúdo técnico da nota (a descoberta MAE-vs-RMSE) já
está preservado sem PII no §6 de [`07-baseline-honesto.md`](07-baseline-honesto.md), então nada
de valor se perde ao removê-la.

**Severidade: alta.** É a única restrição do mapa marcada como dura.

---

## 9. A incerteza mostrada ao Aluno está 17% mais estreita do que deveria

**Onde foi encontrado:** ticket 07, §6.

**O defeito:** `ARG_FINAL_MAE = 13.49` é um **MAE**, calculado em `calculate.py:81` sobre o
triênio 2023/2025 da base em que o próprio modelo foi treinado. Ele é passado como o parâmetro
`rmse` de `calculate_approval_probability` em 6 lugares, e o docstring de `statistics.py` afirma
que é RMSE.

Medido na régua, o `modelo_arg_final` tem **MAE 12,93** (que confirma a origem do número) e
**RMSE 16,26** em Argumento Final. Usar o MAE como desvio-padrão da normal estreita a
distribuição e deixa **toda probabilidade confiante demais** — o Aluno que merece ouvir 70% ouve
mais.

Já havia sido descoberto em 2026-07-24 (commit o commit-raiz da PII (SHA não citado aqui de propósito — ver ticket 15), RMSE 18,01 medido em 2023/2025), com
decisão explícita registrada de **não aplicar a correção**.

**O que falta fazer:** ticket 11. Não é só trocar 13,49 por 16,26 — a largura precisa virar
**por Aluno** e **por classe**, e a constante está duplicada em 6 arquivos.

**Severidade: alta para o produto, e endereçada.** É o Portão 3 do critério de aceite.

---

## 10. `ensemble.py` é código morto, e a documentação o descreve como se rodasse

**Onde foi encontrado:** ticket 07, §5.

**O defeito:** `ensemble.predict_with_dynamic_ensemble` — o ensemble por Coeficiente de Variação
da volatilidade — **não é chamado em lugar nenhum**: nem em `api/services/`, nem no
`app/streamlit_app.py`. O arranjo que a tela usa de verdade é o **meta-modelo roteador**
(`meta_model.joblib`), que escolhe um dos quatro modelos por Aluno.

O `CLAUDE.md` e o mapa descrevem o ensemble dinâmico como o mecanismo de produção. A pergunta
central do ticket 07 ("o ensemble por volatilidade se justifica?") foi feita sobre código que
nunca rodou.

Medido nas linhas limpas, nenhum dos dois arranjos bate o melhor componente sozinho, e o roteador
manda **75% dos Alunos para o `modelo_rf`** — o único que memorizou.

**O que falta fazer:** o ticket 10 decide a família e o arranjo. Se o ensemble sair, `ensemble.py`
sai junto. O `CLAUDE.md` precisa ser corrigido de qualquer forma.

**Severidade: média.** Não quebra nada em produção — mas fez o mapa inteiro raciocinar sobre o
mecanismo errado.
