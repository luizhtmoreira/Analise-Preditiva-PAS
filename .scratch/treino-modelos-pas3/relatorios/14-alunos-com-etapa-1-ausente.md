# Relatório — Ticket 14: alunos com Etapa 1 ausente são fora do treino, ou caso que o produto precisa servir?

**Ticket:** `.scratch/treino-modelos-pas3/issues/14-alunos-com-etapa-1-ausente.md`
**Status:** concluído
**Tipo:** decisão de produto (HITL) — nenhum código de produção foi alterado
**Sessão:** `/grill-with-docs`, 2026-07-27
**Evidência:** relatórios 01 e 02 deste mapa, leitura de `api/`, `landing-page/` e
`src/pas_intelligence/`, e o conhecimento de domínio do dono do produto
**Privacidade:** só agregados e contagens. Nenhum nome, inscrição ou linha individual.

---

## 1. Veredito

> **É ausência, não zero real — e o produto atende essa classe.** A previsão do Argumento Final
> exige função própria porque o **Momentum** é indefinido para ela; o **Quanto Falta** já a
> atende corretamente hoje, por aritmética, sem tocar em modelo nenhum.

Registrado em [`docs/adr/0008-aluno-sem-etapa-1-atendido-com-funcao-propria.md`](../../../docs/adr/0008-aluno-sem-etapa-1-atendido-com-funcao-propria.md).
Termos novos em `CONTEXT.md`: **Etapa Ausente**, **Momentum**, **Aluno sem Etapa 1**, e a
Volatilidade (CV) foi afiada para dizer que é cega à direção.

---

## 2. Ausência ou zero real — fechado sem arqueologia de Edital

O checkbox pedia "evidência do Edital". O relatório 02 já havia declarado que a mecânica da regra
antiga não é reconstruível com o que existe em `data/pdfs/`. O fechamento veio de outro lugar e é
mais forte: **a regra do PAS, dita pelo dono do produto, prevê exatamente o padrão observado no
dado.**

Regra: pode-se faltar à Etapa 1 e seguir no programa; quem falta à Etapa 2 fica impedido de fazer
a Etapa 3; quem falta à Etapa 3 não entra no Resultado Final.

| Etapa zerada (`0,000/0,000/0,000`) | Previsto pela regra | Observado nos 66.313 |
|---|---|---:|
| Etapa 1 | ocorre | **5.768** |
| Etapa 2 | impossível | **0** |
| Etapa 3 | impossível | **0** |

O relatório 02 tinha observado os dois zeros e só sabia explicar o da Etapa 3. O da Etapa 2 ficava
sem causa; agora tem uma, e é regulamentar. **Uma hipótese que acerta as três células de uma tabela
que ela não foi construída para explicar não é inferência frágil — é confirmação.**

Evidência de apoio, já disponível antes desta sessão:

- **Impossibilidade estatística do zero real.** Tirar exatamente `0,000` nas três notas
  simultaneamente — P1 de língua, P2 com desconto por erro, e Redação — é evento de probabilidade
  praticamente nula para quem sentou na prova. Aqui são 5.768 casos.
- **Taxa estável em 8 coortes** (6,7% em 2016/2018 a 11,6% em 2022/2024, tabela 5.1 do relatório
  02), inclusive nos triênios recentes onde não há regra generosa. Estabilidade em 8 coortes é
  assinatura de classe estrutural.
- **O bônus antigo** (mediana +2,704 e +3,549 acima do z de zero em 2016/2018 e 2017/2019):
  instituição não compensa quem tirou zero; compensa quem não fez.

**Corolário que apaga um ramo inteiro:** não existe, e nunca vai existir, Aluno sem Etapa 2 no
Resultado Final do PAS 3. O produto nunca precisa de um modelo genérico de "faltou alguma etapa".
A única classe real é uma.

**Não reconstruído, e assumido como não necessário:** a mecânica exata da regra generosa de
2016/2018 e 2017/2019. Ela só importaria para *recuperar* aquelas 1.483 linhas, e nenhuma decisão
depende disso.

---

## 3. Contagem final para o ticket 05

Derivada das tabelas dos relatórios 01 e 02, sem rodar nada. **A conferir contra o CSV quando o
ticket 05 materializar o dataset** — se qualquer uma das três somas não bater, é sinal de que um
dos dois relatórios contou diferente e a divergência precisa ser explicada antes de treinar.

| Recorte | Linhas |
|---|---:|
| Resultado Final extraído, 8 triênios | 66.313 |
| `checksum_fecha == True` (filtro do ticket 01) | 64.298 |
| … com Etapa 1 **presente** | **60.013** |
| … com Etapa 1 **Ausente** | **4.285** |
| Etapa 1 Ausente com checksum falhando (alvo contaminado pela regra antiga) | 1.483 |

Os 1.483 saem por terem o **alvo** contaminado, não a feature: o Argumento Final impresso neles
segue uma regra que ninguém reconstruiu. Dos 4.285 limpos, **2.747 estão nos três triênios
recentes** — os mais parecidos com o Aluno de hoje. Existe material limpo e recente para servir
essa classe.

**O ticket 05 materializa uma tabela de 64.298 linhas com a coluna `etapa_1_ausente`, e nunca
deleta os 4.285.** A escolha entre um modelo e dois é medição do ticket 10, não filtro do 05.

---

## 4. A classe é atendida — e a decisão foi separada em duas

O dono do produto confirmou que existem Alunos sem Etapa 1 nas Escolas Parceiras, então a classe
está no funil comercial, não só na estatística do Cebraspe.

Duas decisões que pareciam uma:

- **Atender todos** — decisão de produto, tomada: nenhum Aluno recebe recusa.
- **Treinar em todos** — *não* decidida aqui, por desenho. Treinar num dataset misto para ser
  "justo" com 9% pode piorar o modelo para os 91%, e isso não se sabe por argumento. Vai para o
  ticket 10, sobre a régua do ticket 06.

Fundir as duas seria escolher a conclusão antes da medida — o mesmo erro que a nota do mapa sobre
"a régua vem antes de tudo que mede" existe para impedir.

### 4.1 Por que a classe precisa de função própria, e não é caso do mesmo modelo

O produto foi construído sobre uma hipótese: **quem sobe muito da Etapa 1 para a Etapa 2 tende a ir
bem na Etapa 3** (*Momentum*). Para o Aluno sem Etapa 1 essa grandeza não é zero, não é baixa —
**é indefinida**. Não existe "de onde" ele subiu.

Isso resgata, com o sentido certo, a afirmação do ticket sobre o colapso do CV. A Volatilidade
sobre `[0, eb_pas2]` devolve `std/mean = 100%` para **qualquer** `eb_pas2`:

```
CV([0, 20]) = 100%      CV([0, 45]) = 100%      CV([0, 80]) = 100%
```

Isso não é "volatilidade altíssima". É a assinatura matemática de grandeza indefinida — o
mecanismo está informando corretamente que não tem nada a dizer. (Ressalva de precisão: o caminho
que serve o Aluno hoje, `api/services/predict_service.py`, **não** usa o ensemble nem o CV — chama
`_eb_model` e `_arg_model` direto. O colapso é do treino e do caminho legado do Streamlit, não da
API. Mas o argumento conceitual vale para os dois.)

**Alternativa considerada e rejeitada pelo dono do produto:** um modelo único com features apenas
da Etapa 2, que atenderia 100% dos Alunos e faria a classe desaparecer como problema. Rejeitada
porque apagaria o Momentum de todo mundo para acomodar 9% — *"se tem dado do 1, por que
descartar?"*. Fica registrada como caminho fechado, para não ser reproposta como simplificação no
ticket 09.

---

## 5. O que o app faz com essa classe — as duas telas se comportam de forma oposta

O produto tem duas telas, e a mesma classe de Aluno cai em regimes diferentes em cada uma.

| | **Quanto Falta** (EB necessário) | **Preditor** (Arg. previsto + chance) |
|---|---|---|
| Natureza | aritmética exata | previsão |
| Aluno sem Etapa 1 hoje | **já correto por construção** | fora de distribuição |
| O que falta | 2 linhas de conserto + tabela `(2024,1)` | alvo, features e incerteza |

### 5.1 O Quanto Falta já está correto, e ninguém sabia

`target_calculator.py:242-259`:

```python
arg_pas1 = calculate_argument_etapa(notas['P1_PAS1'], notas['P2_PAS1'], notas['Red_PAS1'], stats_pas1)
arg_pas3_necessario = (arg_alvo - arg_pas1 - 2 * arg_pas2) / 3
```

Com `0/0/0`, `calculate_argument_etapa` produz **exatamente o z de zero** — o mesmo número que o
Cebraspe aplica de 2018/2020 em diante (medido no relatório 02: `A1 = −18,228` em 2018/2020, língua
inglesa). O Aluno sem Etapa 1 não é caso especial ali: é um Aluno com A1 muito negativo, e a
fórmula lida com isso sozinha. Ele fica **melhor servido na tela do Quanto Falta do que na do
Preditor**, porque ali não há previsão sobre a parte ausente.

Decomposição que explica por quê:

```
AF = 1·A1  +  2·A2  +  3·A3
      ↑         ↑        ↑
  z de zero   notas    única
  — fixo,     reais    incógnita
  independe   dele
  dele
```

### 5.2 O único defeito na tela que já funciona

`api/schemas/gestao.py:12` e `landing-page/app/(dashboard)/gestao/page.tsx:33` inventam
**`Red_PAS1 = 6.0`** onde o Edital diz `0,000`. Com as estatísticas de PAS 1 carregadas em
`gestao_service.py:38` (`mean_red=6,9051`, `std_red=1,8409`):

```
Red = 6,0 → A_red = (6,0 − 6,9051)/1,8409 × 1,00 = −0,49
Red = 0,0 → A_red = (0,0 − 6,9051)/1,8409 × 1,00 = −3,75
```

**+3,26 pontos de Argumento Final presenteados**, na direção do otimismo: o Coordenador Pedagógico
vê o Aluno mais perto do corte do que ele está.

**Ressalva de disparo, verificada depois da sessão:** `??` é *nullish coalescing* — dispara em
`null`/`undefined`, **não em `0`**. Com a Etapa 1 Ausente gravada como `0, 0, 0` (convenção do
Edital), o defeito **não atinge o Aluno sem Etapa 1 hoje**; ele atinge o estado "não informado"
(Redação NULL na `tabela_mestra`), cuja frequência não foi medida. Mas ele passa a atingir **100%
da classe** no instante em que a ausência declarada do ADR-0008 substituir `0` por `None`.
Consertar a representação sem consertar o coalescing acorda o bug exatamente na classe que a
mudança pretendia proteger — **a ordem importa**. Ver defeito 4 em `defeitos-pendentes.md`.

---

## 6. Ausência é declarada, nunca inferida

O sistema não pode fazer o que o Edital faz. O Cebraspe imprime `0,000` sem ambiguidade porque
**sabe quem compareceu**. E a forma da evidência inverte conforme o documento:

| Documento | Como a ausência aparece |
|---|---|
| Resultado Final do PAS 3 (o que o pipeline extrai hoje) | **afirmação positiva** — imprime `0,000` |
| Resultado por etapa (PAS 1, PAS 2 isolados) | **ausência de registro** — o Aluno não está na lista |

O Aluno vivo — o que usa o app — está no meio do triênio, e o Resultado Final do PAS 3 dele ainda
não existe. As notas dele vêm dos Editais por etapa, onde a ausência é um **silêncio**. E
"não está na lista" não é afirmação sobre o Aluno: é observação sobre a busca, com três causas, das
quais duas são defeitos abertos no mapa `pdf-extraction` — nome quebrado por espaço (ticket 13) e
nome divergente entre Editais (ticket 18).

**Inferir ausência de um silêncio cuja causa conhecida é um bug seu é construir sobre o defeito.**

Consequência de schema: as notas viram `Optional[float] = None` (sem default `0`, sem default
`6.0`), e a ausência da Etapa 1 é um campo próprio, declarado. Isso separa três estados hoje
colapsados no mesmo vetor: **não fez** (fato do mundo), **não informado** (lacuna de cadastro),
**tirou zero** (raro, mas real numa nota isolada).

Nenhuma planilha de terceiro trava essa mudança: os dados são todos públicos, extraídos do
Cebraspe pelo próprio dono do produto. Não há contrato externo a renegociar.

---

## 7. O que este ticket entrega para os próximos

### 7.1 Ticket 04 (alvo canônico) — uma tensão formulada, não resolvida

O custo de atender essa classe **é decidido pelo alvo**:

- **Alvo = Argumento Final direto** → o modelo aprende `PAS1+PAS2 → AF` com a contribuição do A1
  embutida no aprendizado. O Aluno sem Etapa 1 fica fora dessa função e exige modelo separado.
- **Alvo = as 3 notas da Etapa 3 + fórmula oficial** → A1 e A2 saem por aritmética exata, a
  penalidade de ~18 pontos aparece sozinha e correta, e resta prever só o A3.

Isso **conflita** com a §8 do relatório 02, que argumenta a favor do Argumento Final porque ele é
estável entre triênios (média ~3–5, desvio ~50 nos oito) enquanto o EB da Etapa 3 varia 35%. Os
dois argumentos são bons e apontam para lados opostos. O ticket 04 decide — mas decide **sabendo**
que está escolhendo o custo desta classe junto, em vez de descobrir depois.

### 7.2 Ticket 06 (esquema de validação)

O holdout carrega a subpopulação, estratificado por `etapa_1_ausente`. **Todo candidato reporta
dois números**, um por classe. Um modelo que melhora a média piorando os 9% não pode passar
despercebido.

### 7.3 Ticket 09 (features)

O conjunto de features **não pode** dropar as da Etapa 1 como simplificação — decisão de produto,
não resultado empírico (§4.1). O Momentum precisa estar representado **com sinal**, não só via
Volatilidade.

### 7.4 Ticket 10 (família de modelo)

- "Aceita valor faltante nativamente" é **critério de seleção com peso**, não desempate. Escolher
  linear ou MLP fecha a porta desta classe e obriga pipeline separado; GBM a mantém quase de graça.
- Medir **um modelo com roteamento de faltante** contra **dois modelos separados**, sobre o mesmo
  holdout do ticket 06.
- O roteador atual (`meta_model` por CV) é **cego à direção do Momentum** — ver defeito 5.

### 7.5 Ticket 11 (incerteza calibrada)

A incerteza é **por classe**, no mínimo duas. Mostrar probabilidade para o Aluno sem Etapa 1 com um
RMSE emprestado da classe majoritária produz probabilidade errada mesmo com previsão pontual certa.

### 7.6 Extração por etapa (ainda sem ticket — névoa do mapa)

Quando existir, ela precisa trazer **duas** coisas, não uma:

1. as notas de PAS 1 e PAS 2 dos Alunos vivos;
2. as **médias e desvios das etapas vivas**. `OFFICIAL_STATS` tem 24 chaves e falta `(2024,1)`,
   `(2025,1)`, `(2025,2)` — não porque o Cebraspe não publicou, mas porque a tabela foi montada a
   partir dos Editais de PAS 3, e o do triênio 24-26 só sai em 2027. Sem `(2024,1)` o Quanto Falta
   não tem como calcular o A1 do Aluno sem Etapa 1 do triênio vivo.

E ela nasce com `etapa_1_ausente` como campo de primeira classe, derivado de evidência
cruzada (presente no Edital da Etapa 2, ausente no da Etapa 1), **nunca de notas zeradas**.

**Teste de aceite pronto, com dado que já existe:** dos 865 registros com Etapa 1 Ausente no
triênio 2023/2025, quantos estão ausentes do Edital de resultado da Etapa 1 de 2023? Se
essencialmente todos, a leitura de ausência ganha prova documental e o casamento por nome ganha
uma taxa de acerto medida numa população de resposta conhecida. Duas entregas por um trabalho só.

---

## 8. Limitações

- **As contagens da §3 são derivadas dos relatórios 01 e 02, não recontadas nesta sessão.** O
  ticket 05 confere contra o CSV.
- **Nenhum código foi alterado.** Os defeitos 4 e 5 foram registrados, não corrigidos.
- **A regra generosa de 2016/2018 e 2017/2019 continua não reconstruída** (limitação herdada do
  relatório 02), e as 1.483 linhas afetadas ficam fora por alvo contaminado.
- **O ~18 pontos de penalidade é o valor medido em 2018/2020** (`−18,228`, língua inglesa). Para
  cada ano ele é outro número, e para o triênio vivo depende de `(2024,1)`, que ainda não existe na
  tabela.
- **A hipótese do Momentum não foi medida** — é a premissa de produto que motivou a decisão, não um
  resultado. Quanto as features da Etapa 1 de fato acrescentam, dado que a Etapa 2 é conhecida, é
  medição do ticket 09. A decisão de manter a Etapa 1 vale independentemente do resultado, mas o
  número é interessante.

---

## 9. Glossário

- **Etapa Ausente** — Etapa que o Aluno não realizou. O Edital publica as três notas como `0,000`,
  o que significa ausência, não desempenho zero. Só a Etapa 1 pode sê-lo em quem chega ao Resultado
  Final do PAS 3.
- **Aluno sem Etapa 1** — a classe de produto correspondente. Substitui "aluno que só fez o PAS 2",
  que é impreciso: ele vai fazer o PAS 3, e é por isso que é usuário.
- **Momentum** — direção e tamanho da evolução de uma Etapa para a seguinte. Grandeza **com sinal**.
  Hipótese central do produto. Indefinida para o Aluno sem Etapa 1.
- **Volatilidade (CV)** — `std/mean × 100` sobre os EBs anteriores. Mede **magnitude** e é **cega à
  direção**: `[30,35]` e `[35,30]` dão o mesmo CV. Não é sinônimo de Momentum, e confundir os dois
  é o defeito 5.
- **Fora de distribuição** (*out of distribution*) — entrada cujo padrão não aparece no treino. Não
  gera erro: o modelo responde com confiança normal um número sem lastro. Árvore não extrapola —
  ela cola a resposta na folha mais próxima que viu —, então a previsão fica presa ao pior PAS 1 do
  treino, com magnitude que ninguém mediu.
- **Valor faltante** (`NaN`, *missing value*) — marcação explícita de "este número não existe",
  diferente de zero. GBM modernos (LightGBM, HistGradientBoosting) aprendem sozinhos para que lado
  da árvore mandar o faltante; regressão linear e MLP não aceitam — daí o critério do ticket 10.
- **Estratificar** — garantir que uma divisão (treino/holdout) preserve a proporção de um grupo em
  cada parte, em vez de deixar ao sorteio. Sem estratificar por `etapa_1_ausente`, o holdout pode
  sair com poucos Alunos da classe e o número dela vira ruído.
- **Regime** — conjunto de linhas geradas pela mesma regra. As 1.483 antigas são outro regime de
  *alvo*; o Aluno sem Etapa 1 é outro regime de *feature*. Misturar regimes num modelo só é
  legítimo, mas tem que ser medido, não presumido.
