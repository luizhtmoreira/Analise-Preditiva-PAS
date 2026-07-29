# Publicar o lado público do Vetor PAS

**Origem:** `.scratch/publicar-site/map.md` (mapa de wayfinder, Passo 1 resolvido em 2026-07-29)
**Criado:** 2026-07-29
**Rodada:** lado público apenas — Preditor PAS 3, Calculadora de Estratégia, Análise Temporal e
landing, servidos por uma API hospedada. O B2B fica fora (ver *Out of Scope*).

---

## Problem Statement

O Vetor PAS tem um modelo treinado, avaliado e promovido — `models/pas3/`, RMSE 5,009 em `A3`,
com a incerteza medida e guardada no manifesto. Ele **mora só no disco do dono do produto**.

Enquanto isso, quem procura o produto hoje é o Aluno do triênio 2024-2026, que faz o PAS 3 em
2026 — a **Turma viva**. Para ele, o Preditor não responde nada: ele devolve
`modelo_disponivel: False`. O motivo é correto e proposital (`model_package.py:63`), mas o efeito
é que o produto não atende exatamente quem o procura.

A recusa acontece porque o Argumento Final é `A1 + 2·A2 + 3·Â3`, e `A1` e `A2` são **conta
exata**, não previsão: dependem da média e do desvio-padrão que o Cebraspe publica por Etapa.
O Cebraspe só publica esses números **depois do PAS 3** — um Edital por triênio, com as três
Etapas juntas. Para 2024-2026 esse Edital sai em 2026. Até lá, `(2024, Etapa 1)` e
`(2025, Etapa 2)` não existem no `OFFICIAL_STATS`, e o `stats_da_prova` levanta
`EstatisticaOficialAusenteError`.

Somam-se a isso quatro coisas que impedem a publicação mesmo se o Preditor respondesse:

1. **A API não está hospedada.** Roda em `localhost:8000`, sem Dockerfile e sem Space. O modelo
   e os CSVs não estão no git, então uma máquina nova sobe sem nada. E o CORS — a regra que diz
   de quais endereços o navegador pode chamar a API — libera `"https://*.vercel.app"` como texto
   literal, que o Starlette não trata como curinga; `vetorpas.com.br` nem está na lista. O
   Preditor chama a API do navegador, então isso falha na primeira requisição em produção.
2. **A API serve dados velhos.** Lê `data/notas_corte_pas.csv` (2.307 linhas, base ad-hoc) enquanto
   a frente de extração já produziu 5.225 Notas de Corte, incluindo o triênio 2023-2025 que não
   existe na base velha. O modelo novo está sendo servido contra corte antigo.
3. **A Calculadora de Estratégia está degradada e invisível.** Ela vive só na branch
   `feat/nextjs-frontend`, e os dois `.joblib` que estimam P1 e Redação **não carregam** no
   ambiente atual (`ModuleNotFoundError: No module named '_loss'`). O erro é engolido, e a
   Calculadora responde por média ponderada sem que nada na tela diga isso.
4. **A parte "exata" da conta não é exata.** `api/services/gestao_service.py` carrega um
   `TRIENNIUM_STATS` próprio cujos números **divergem dos Editais** (2022-2024 Etapa 1: 20,7094
   contra 20,406 do Cebraspe). E o Aluno declara **uma** língua estrangeira, enquanto o PAS
   registra uma **por Etapa** — 13,9% dos Alunos trocam de língua entre a Etapa 1 e a 2.

O Passo 1 do mapa mediu a saída para o item central: dá para inferir média e desvio dos
**Editais isolados de Etapa** — o "Resultado final nos itens do tipo D e na prova de redação",
publicado no ano da prova, que lista nota por candidato mas não diz a língua de ninguém.
A inferência **serve, com correção**: sai +7,87 pontos de Argumento Final acima do verdadeiro,
mas o erro é um **degrau** (a média do erro e a média do valor absoluto são o mesmo número,
7,867), não ruído. Subtraindo o degrau, o erro médio cai para 1,14 e o máximo para 4,19 — contra
uma Largura de Incerteza de 15,03.

## Solution

Cinco frentes, na ordem em que destravam umas às outras.

**1. O `OFFICIAL_STATS` passa a cobrir a Turma viva.** Os Editais isolados de Etapa de
`(2024, Etapa 1)` e `(2025, Etapa 2)` viram duas entradas novas, marcadas explicitamente como
**derivadas** — porque quando o Edital de verdade sair em 2026 esses números serão substituídos e
as previsões vão mexer. O Deslocamento medido no Passo 1 é calibrado sobre pelo menos quatro
triênios e aplicado **na média estimada, dentro da entrada derivada**, para que `A1`, `A2`, o
Argumento Final e a Calculadora herdem a correção sem que nenhum deles saiba que ela existe.

O extrator que hoje é script descartável em `.scratch/` vira módulo de verdade em
`src/pas_extraction/`, com fixture sintética e teste, no mesmo padrão de `medias_desvios.py`.
Não é operação de uma vez: a calibração roda de novo a cada triênio novo, e em 2026 o Edital
real substitui as entradas derivadas.

**2. A parte exata da conta passa a ser exata mesmo.** O `TRIENNIUM_STATS` e o
`STATS_PAS3_TREND` são apagados; tudo lê `OFFICIAL_STATS` pela porta única que já existe
(`stats_da_prova`). E a língua estrangeira passa a ser **por Etapa** em todo o caminho — entrada,
schema, formulário e cálculo.

**3. A API é hospedada de verdade.** Dockerfile, Space privado no Hugging Face, o pacote de
modelo assado na imagem no build e promovido por commit de ponteiro — o domicílio decidido no
ticket 03 e nunca construído. Mais o CORS consertado, com a lista de origens vindo do ambiente.

**4. Os dados velhos saem.** As duas fontes novas da frente de extração entram em produção, com
uma varredura de plausibilidade antes, porque um corte implausível no Preditor público vira uma
probabilidade absurda na tela de um Aluno.

**5. As branches se unificam e o público vai para a `main`.** Incluindo a Calculadora de
Estratégia, cujos dois bloqueadores caíram: o estimador de P1 e Redação vira **aritmética** (o
Estimador Auxiliar), o que **remove** o carregamento de `.joblib` em vez de consertá-lo; e a
faixa da Parte 2, que era o chute `[−100, 100]`, vira número medido sobre 8 triênios e ~64 mil
Alunos.

A Calculadora entra deixando o código **menor** do que está.

---

## User Stories

### O Aluno da Turma viva

1. Como Aluno do triênio 2024-2026, quero que o Preditor me responda, para que o produto sirva
   justamente a quem está prestes a fazer a prova.
2. Como Aluno, quero que o número que vejo esteja corrigido do viés conhecido, para não receber
   uma estimativa sistematicamente otimista sobre a minha própria aprovação.
3. Como Aluno, quero saber quando a minha previsão está apoiada em número derivado em vez de
   número oficial, para entender por que ela pode mudar quando o Edital sair.
4. Como Aluno que trocou de língua estrangeira entre a Etapa 1 e a Etapa 2, quero informar as
   duas, para que o meu Argumento não seja calculado com a estatística da língua errada.
5. Como Aluno que fez a mesma língua nas duas Etapas, quero que o segundo campo já venha
   preenchido com o primeiro, para não digitar duas vezes o que é a mesma resposta em 86% dos
   casos.
6. Como Aluno sem Etapa 1, quero declarar essa ausência em vez de digitar zeros, para que o
   modelo saiba a diferença entre "não fiz a prova" e "fiz e tirei zero".
7. Como Aluno, quero ver a minha chance por curso calculada com a incerteza real do modelo, e não
   com uma constante herdada de um modelo aposentado.

### O Aluno usando a Calculadora de Estratégia

8. Como Aluno com um curso-alvo em mente, quero saber quanto preciso tirar na Parte 2 do PAS 3
   para alcançá-lo, para transformar uma ambição em meta concreta.
9. Como Aluno, quero que a Calculadora estime a minha P1 e a minha Redação a partir do que eu já
   fiz nas Etapas 1 e 2, para não ter que adivinhar dois números antes de ver o terceiro.
10. Como Aluno, quero poder sobrescrever a estimativa da Redação sozinha, sem que isso descarte
    também a estimativa de P1, porque a Redação é o componente que mais move o resultado.
11. Como Aluno, quero ver o resultado contra **cinco anos reais** de Etapa 3 em vez de um único
    ano extrapolado, para que a faixa entre eles seja a minha incerteza em vez de um número falso
    de precisão.
12. Como Aluno, quero que o ano mais recente venha em destaque entre os cinco, para ter uma
    leitura principal sem perder a faixa.
13. Como Aluno cuja meta é inalcançável, quero ouvir "impossível" só quando for aritmeticamente
    impossível, e não porque uma constante chutada disse isso.
14. Como Aluno cuja meta já está garantida pelo histórico, quero ouvir isso em português, e não
    "você precisa de -99,4 pts na Parte 2".
15. Como Aluno, quero que a Calculadora e o Preditor concordem sobre o meu `A1` e o meu `A2`,
    porque eles são a mesma conta exata sobre as mesmas notas.

### Quem visita o site

16. Como visitante, quero abrir `vetorpas.com.br` e ver a landing atual funcionando, sem
    regressão visual nem de conteúdo.
17. Como visitante, quero usar o Preditor e a Calculadora sem criar conta, porque são o lado
    público do produto.
18. Como visitante, quero ver a Análise Temporal com a série oficial das provas e a evolução da
    Nota de Corte por curso, para entender o histórico antes de olhar a minha própria previsão.
19. Como visitante em qualquer navegador, quero que a chamada à API funcione, sem erro de CORS
    que aparece como "API indisponível".

### O dono do produto

20. Como dono do produto, quero que uma máquina limpa suba a API completa, para que o produto não
    dependa de um diretório que existe só no meu disco.
21. Como dono do produto, quero promover um modelo novo por commit de ponteiro, para que reverter
    seja um comando e não uma cópia manual de arquivo.
22. Como dono do produto, quero que o `/health` responda numa URL pública, para saber que o
    serviço está de pé sem abrir o site.
23. Como dono do produto, quero que as entradas derivadas do `OFFICIAL_STATS` estejam marcadas no
    próprio dado, para que ninguém descubra por acidente, em 2026, por que as previsões mudaram.
24. Como dono do produto, quero saber se o Deslocamento é estável entre triênios **antes** de
    publicar, porque se ele não for, o Preditor para a Turma viva volta a ser questão aberta.
25. Como dono do produto, quero que nenhum nome de Aluno real seja publicado em nenhuma branch da
    `main`, porque é a única restrição do mapa marcada como dura.
26. Como dono do produto, quero que as Notas de Corte novas passem por uma varredura de
    plausibilidade antes de irem ao ar, para que um corte de 199.162,872 não vire uma
    probabilidade absurda na tela de alguém.
27. Como dono do produto, quero que o visual novo da landing possa ir a produção sozinho, sem
    esperar o resto desta rodada.

### Quem mantém o código

28. Como quem mantém o código, quero um caminho só para ler média e desvio oficiais, para que o
    `A1` que a API mostra seja o `A1` com que o modelo foi treinado.
29. Como quem mantém o código, quero que o extrator de Editais de Etapa tenha teste sobre fixture
    sintética, porque ele vai rodar de novo todo ano.
30. Como quem mantém o código, quero que o teste de paridade treino/runtime cubra o caso de língua
    trocada, porque hoje ele passa por ser cego a essa dimensão.
31. Como quem mantém o código, quero que a Calculadora não carregue `.joblib` nenhum, para que a
    classe inteira de defeito "artefato serializado com outra versão de biblioteca" desapareça.
32. Como quem mantém o código, quero as origens de CORS vindo do ambiente, para que DEV e PROD não
    sejam a mesma lista editada à mão.
33. Como quem mantém o código, quero que o `ensemble.py` não volte no merge, porque ele foi
    aposentado pelo ADR-0011 e nunca chegou a rodar em produção.

---

## Implementation Decisions

### A. `ExamStats` aceita uma Parte 1 misturada e carrega procedência

**O problema de forma.** O ticket 12 tornou `parte_1` um campo obrigatório com as **três** línguas
(inglesa, francesa, espanhola), porque o Cebraspe normaliza a Parte 1 separadamente por língua e
agrupar as três embute viés contra quem fez espanhol ou francês. Mas o Edital isolado de Etapa
**não diz a língua de nenhum candidato** — só dá a Parte 1 misturada. Preencher as três exigiria
inventar valores.

**Decisão.** `ExamStats` passa a admitir duas formas de Parte 1, e a diferença é explícita no
dado, não implícita numa convenção:

- **por língua** — as três línguas, como hoje, quando vem do Edital de média e desvio;
- **misturada** — um único par média/desvio, marcado como tal, quando vem do Edital isolado de
  Etapa.

`ExamStats` ganha também um campo de **origem** (`"edital"` ou `"derivada"`). As duas entradas
novas são `derivada`; as 24 existentes são `edital`.

**O custo dessa escolha está medido, e é o que a torna aceitável:** usar a Parte 1 misturada
custa **0,46 ponto de Argumento Final em média**, máximo 3,21, com **viés zero** — é ruído, não
erro sistemático. A Parte 1 pesa 0,72 numa conta que soma 10, e a média misturada cai praticamente
em cima da média da inglesa, que é 66% a 73% da população.

**As propriedades derivadas `m_p1`/`dp_p1` continuam existindo** com o mesmo contrato (média
simples das três línguas; para a forma misturada, o próprio valor). O único consumidor delas é
`api/services/analytics_service.py`, e a interface dele não muda.

### B. `stats_da_prova` continua sendo a porta única

A costura já existe e não se mexe nela: `stats_da_prova(ano, etapa, lingua)` em
`training_dataset.py` é o **ponto único de leitura** do `OFFICIAL_STATS` para quem vai calcular um
Argumento de Etapa, e treino e runtime passam pela mesma função de propósito.

Quando a entrada é de Parte 1 misturada, `stats_da_prova` devolve a estatística misturada
**qualquer que seja a língua pedida**, em vez de levantar erro. A alternativa — recusar — devolveria
o produto ao estado que esta rodada existe para sair.

O cache de módulo `_STATS_POR_ANO_ETAPA_LINGUA`, montado no import, passa a acomodar as duas
formas.

### C. O Deslocamento vive na média estimada da entrada derivada

**Decisão.** A correção é aplicada **na média estimada, dentro da entrada derivada do
`OFFICIAL_STATS`** — não no Argumento Final no fim da conta.

Isto segue o que o mapa registrou: a causa do erro está localizada e medida em `m_p2` da Etapa 2
(−4,61), não espalhada pela conta. Corrigir na origem significa que `stats_da_prova`,
`model_package`, `training_dataset`, `target_calculator` e a API **não mudam uma linha** por causa
do Deslocamento — herdam a correção de graça, e não existe um segundo lugar onde a conta é
ajustada. A alternativa (subtrair 7,87 do Argumento Final) obrigaria o caminho reverso da
Calculadora a somar de volta, criando duas correções para manter em sincronia.

**A causa, para constar.** O Edital isolado de Etapa 2 de 2024 tem 16.339 candidatos; os
concluintes daquele triênio são 8.703. O Cebraspe calcula a média da Etapa 2 sobre os
**concluintes** — o que faz sentido, já que ele só publica o Edital de média e desvio depois do
PAS 3, quando já sabe quem chegou ao fim. Estimando sobre os 16.339 pegamos metade a mais de
gente, e essa metade é mais fraca: 0,31 desvio-padrão de diferença.

**Não dá para resolver por filtro, e isso já foi testado.** Sete recortes da lista do Edital
(tirando faltoso, nota zero, redação zero, tipo D zero) e nenhum reproduz o oficial. O desvio erra
por −1,5 em todos eles: para o desvio subir seria preciso gente com mais dispersão, para a média
subir seria preciso gente com nota mais alta, e as duas direções se contradizem. Existe uma
população que o Cebraspe usa e que não temos. **Isso não bloqueia**: o Deslocamento corrige o
efeito sem que a causa esteja explicada.

**O Deslocamento é por Etapa, não global.** Os pontos medidos até agora:

| Validação | Erro em `m_p2` |
|---|---:|
| (2022, Etapa 1) | −1,35 |
| (2023, Etapa 1) | −2,18 |
| (2024, Etapa 2) | −4,61 |

### D. Portão de calibração — o critério que decide se esta rodada continua

**O único risco vivo do Passo 2** é o Deslocamento não ser estável entre triênios. Hoje ele está
medido em **um** triênio para a Etapa 2 e dois para a Etapa 1, e os dois da Etapa 1 já divergem
entre si (+1,23 em 2022, +1,81 em 2023). Sem mais pontos, a correção é um número solto.

**Decisão — o portão, verificável em código.** A calibração roda sobre **pelo menos quatro
triênios** com Edital isolado de Etapa **e** Edital oficial de média e desvio. Aplicando o
Deslocamento médio por Etapa, o **maior** erro residual em Argumento Final, sobre todos os
triênios de validação, tem que ficar **abaixo de 5,009** — um RMSE do modelo, não três.

Hoje esse máximo está em **4,19** com os três pontos existentes. O portão é apertado de propósito:
o limiar frouxo (`3 × RMSE = 15,03`) já foi atendido pelo erro **bruto** de 7,87, e passar por ele
não é evidência de nada.

**Se o portão não fechar,** o Preditor volta a recusar para a Turma viva e as entradas derivadas
não entram. Isso reordena o mapa e é decisão do dono do produto, não do código — o portão
apenas o torna visível antes de publicar, em vez de depois.

A calibração produz um relatório com os triênios usados, os offsets por Etapa, a dispersão entre
eles e o erro residual — o mesmo padrão de `relatorio_official_stats.py`.

### E. O extrator de Editais de Etapa vira módulo

`src/pas_extraction/editais_de_etapa.py`, com fixture sintética e teste, seguindo o padrão de
`medias_desvios.py` e `notas_corte.py`. Saída: `medias_desvios_etapa.csv`, que alimenta a geração
das entradas derivadas.

Nota: `src/pas_extraction/` é gitignored (lógica de extração não é pública) enquanto
`tests/test_pas_extraction_*.py` é rastreado. O módulo novo segue esse arranjo, que já é o do
resto da frente.

**O que o Passo 1 provou de bônus, e que sustenta essa promoção:** seis Editais, ~19,5 mil
registros cada, **zero falhas** no checksum embutido (`EB parte 1 + EB parte 2 = somatório`), e as
notas batem em **99,63%** com o CSV para os Alunos que aparecem nos dois lados.

**Armadilha de documento que precisa virar código:** "Retificação" no nome do arquivo **não diz**
se o Edital é parcial ou completo. O Edital 8 de 2023 (retificação) tem 827 registros e não
serve; o Edital 7 do mesmo ano tem 19.505. Em 2022 foi o contrário: o Edital original não trazia
os escores brutos das Partes 1 e 2, que só apareceram na retificação. O módulo tem que **contar os
registros e recusar** um documento parcial, em vez de deixar a conferência para o humano.

### F. `TRIENNIUM_STATS` e `STATS_PAS3_TREND` são apagados

`api/services/gestao_service.py` carrega médias e desvios num dicionário próprio que **diverge dos
Editais** — 2022-2024 Etapa 1 com `m_p2=20,7094` contra 20,406 do Cebraspe, com quatro casas
decimais e desvio sistematicamente maior, cara de calculado de uma amostra de Alunos em vez de
copiado do Edital.

Isso deixa de ser opcional nesta rodada porque `get_strategy_prediction` — a Calculadora — consome
`triennium_stats`. Publicar a Calculadora sobre ele significa pôr no ar duas telas que calculam o
mesmo `A1` e o mesmo `A2` com números diferentes.

Tudo passa a ler `OFFICIAL_STATS`, chaveado por `(ano, etapa)` em vez de por string de triênio.
O `STATS_PAS3_TREND` (regressão linear que extrapola uma prova que ainda não aconteceu) sai junto,
substituído pelos Anos-Âncora (decisão H).

### G. A língua estrangeira passa a ser por Etapa

**O defeito.** O `resultado_final.csv` grava `lingua_e1`, `lingua_e2` e `lingua_e3` — uma por
Etapa, porque é assim que o Cebraspe registra, e o treino respeita isso. O runtime não:
`EntradaDePrevisao` tem **um** campo `lingua`, aplicado às duas Etapas. Os schemas fecham o
contrato no mesmo formato. O produto inteiro presume que a língua é atributo do Aluno; ela é
atributo do par **(Aluno, Etapa)**.

**Quantos trocam, medido sobre as 64.298 linhas limpas:** 8.950 (**13,9%**), e a troca não é
uniforme — **72% são inglesa → espanhola**. Não é ruído de extração, é movimento real e de mão
única da coorte.

**Quanto custa:** o Aluno declara uma língua e a Etapa que não casa sai com a estatística errada —
até 1,04 ponto de Argumento Final via `A1`, até **3,79** via `A2` (que entra com peso 2).

**Decisão.**

- `EntradaDePrevisao` carrega `lingua_e1` e `lingua_e2`. `_argumentos_exatos` consome cada uma na
  sua Etapa.
- `PredictInput` troca `lingua` por `lingua_e1` e `lingua_e2`, **ambas obrigatórias, sem default**.
  Não há alias de compatibilidade: o único cliente é o nosso próprio frontend, que está sendo
  reescrito nesta mesma rodada, e um default silencioso é exatamente o viés que o ticket 04 §5.3
  se propôs a eliminar.
- **O formulário** pré-preenche o segundo campo com o primeiro, visivelmente e editável. O
  pré-preenchimento é de interface, não de contrato — a API continua exigindo os dois.
- `api/schemas/gestao.py`: o default vira por Etapa, sem mudar a natureza da dívida já declarada e
  aceita no relatório 13 §6.2.
- **Etapa 3:** o caminho reverso da Calculadora precisa da estatística da Parte 1 da Etapa 3, que
  o Aluno ainda não fez. Usa-se **`lingua_e2`** como a língua provável da Etapa 3, e não
  `lingua_e1`, porque a troca é de mão única e a última língua declarada é a melhor evidência
  disponível. *Assunção explícita, registrada aqui para ser revisitada se algum dia medirmos
  `lingua_e2 → lingua_e3`.*

**Por que nada pegou até agora, e o que impede a regressão.**
`tests/test_model_package.py::test_o_runtime_monta_as_mesmas_features_que_o_treino` existe
exatamente para prender desencontro entre treino e runtime, e passa porque o fixture crava
`"inglesa"` nas três Etapas — com a língua constante, as duas portas concordam por construção. O
teste **é cego a esta dimensão**, e estendê-lo com um caso de língua trocada é o item que impede a
correção de voltar atrás na próxima refatoração.

### H. A Calculadora de Estratégia: aritmética no lugar de `.joblib`

**O Estimador Auxiliar substitui os dois modelos.** Em vez de prever P1 e Redação do PAS 3 com
`p1_pas3_model.joblib` e `red_pas3_model.joblib`, prevê-los por **média ponderada de z-scores** —
aritmética sobre notas que já temos. Um z-score é a nota do Aluno expressa em desvios-padrão em
relação à média daquela prova; ele é o que permite comparar uma nota de 2024 com uma de 2025 sem
herdar a dificuldade de cada prova.

```
Â3                ← única previsão do modelo
P1̂, R̂ed           ← Estimador Auxiliar (média ponderada dos z das Etapas 1 e 2) + override do Aluno
P2                = resolvido:  z_p2 = (A3 − 0,72·z_p1 − 1,00·z_red) / 8,28
```

O peso é 1 para a Etapa 1 e 2 para a Etapa 2, o mesmo da média ponderada que o código já usa como
fallback — o que muda é que a média passa a ser feita na escala padronizada e depois reconvertida
para a escala da Etapa 3, em vez de somar notas de provas com dificuldades diferentes.

**Erro medido nos três triênios recentes:** 1,47 ponto em P1 e 1,36 na Redação. Com `A3` fixo, o
erro de P1 é amortecido em 60% (move o P2 necessário em ~0,59); o da Redação passa quase inteiro
(~1,29) — e é exatamente por isso que a caixa de override da Redação é a que mais importa.

**O que sai de `target_calculator.py`,** por remoção e não por conserto: `_carregar_modelo`,
`model_load_error`, `_registrar_degradacao`, `ModelLoadError`, o `PAS_STRICT_MODELS` e o import de
`joblib`. Isso mata o defeito 3 de `defeitos-pendentes.md` — o `ModuleNotFoundError: _loss` que
hoje faz a Calculadora responder por média ponderada sem avisar ninguém — e elimina a classe
inteira de defeito "artefato serializado com outra versão de biblioteca" deste módulo.

**A faixa da Parte 2 deixa de ser chute.** As constantes `P2_MAXIMO = 100.0` / `P2_MINIMO = -100.0`
decidem sozinhas quando o produto diz "impossível" e quando diz "garantido", e não têm procedência
em Edital nenhum. Medido na Etapa 3, 8 triênios, ~64 mil Alunos:

| | Chute atual | Medido |
|---|---:|---:|
| Piso de P2 | −100 | **0,24** (0% negativo em 8 triênios) |
| Teto de P2 | +100 | **85,6** (o maior de 64 mil Alunos) |
| P2 no percentil 99,9 | — | ~78 |
| Teto de `EB = P1 + P2` | — | 92,3 |

O teto teórico continua 100 porque o fator de normalização existe para que acertar tudo dê 100 —
mas ele é de `P1 + P2` **juntos**, e a P1 sozinha já come até 8,5.

**A faixa é por Etapa, e essa distinção fica no código:** nos Editais de Etapa 2, 2,3% dos
candidatos ficaram abaixo de zero (o pior em −19,6); na Etapa 3, zero em 64 mil. A Calculadora
resolve para a Etapa 3, então usa a faixa da Etapa 3.

**Os quatro status passam a ter significado medido:**

- **impossível** — a nota necessária passa de `100 − P1̂`. Aritmética, não opinião. (Com a faixa
  antiga, uma nota necessária de 95 era classificada como "possível".)
- **improvável** — passa de 85,6, o recorde histórico. Existe no papel, nunca aconteceu.
- **garantido** — a nota necessária fica abaixo do piso de 0,24. Com a faixa antiga esse ramo era
  praticamente código morto; com a medida ele volta a significar algo verdadeiro: *"praticamente
  qualquer desempenho na Parte 2 mantém a sua aprovação"*.
- **possível** — o resto.

A mensagem do ramo `garantido` deixa de exibir o valor truncado (`"você precisa de -99,4 pts"`) e
passa a dizer em português o que aconteceu. Isso resolve a "nota de comunicação" registrada no
defeito 1.

**O que a Calculadora *não* precisa, e essa confusão custou meses:** o modelo de correção item a
item (110 itens na Parte 2, tipos A/B/C/D com pesos 1/2/2/3, desconto por erro, fator de
normalização). Isso alimenta o **Simulador de Itens**, que é outra tela e depende de saber quantos
itens de cada tipo tinha cada prova — dado que **não sai em Edital**, só no caderno de questões.
Ver *Out of Scope*.

### I. Ano-Âncora: cinco anos reais na tela, nenhum ano extrapolado

Um **Ano-Âncora** é um ano real e já publicado usado como cenário: *"e se a minha Etapa 3 for como
a de 2023?"*. É a decisão 3 do relatório 04 — nada de projetar a prova futura.

**Decisão.** A Calculadora devolve **cinco resultados**, um por Ano-Âncora, com o mais recente em
destaque. Os Anos-Âncora são as cinco chaves `(ano, 3)` mais recentes do `OFFICIAL_STATS` — hoje
2025, 2024, 2023, 2022 e 2021.

Cada Ano-Âncora varia **duas** coisas juntas, porque separá-las produziria um cenário que nunca
existiu:

- a média e o desvio da Etapa 3 daquele ano (`stats_pas3`);
- a Nota de Corte do curso no triênio correspondente (Ano-Âncora 2025 → triênio 2023-2025).

A faixa entre os cinco resultados **é** a incerteza sobre a prova futura, mostrada em vez de
escondida atrás de um número de falsa precisão.

Isto substitui o `STATS_PAS3_TREND` (decisão F) e amplia o `StrategyResponse`, que passa a
carregar uma lista em vez de um resultado único.

### J. Hospedagem: Dockerfile, Space e CORS

**O domicílio do ticket 03, que nunca foi construído.** Repositório privado no Hugging Face,
artefato assado na imagem no build, promoção por commit de ponteiro. Hoje o pacote existe só na
máquina do dono do produto; máquina nova sobe sem modelo, e reverter é cópia manual.

A imagem precisa de três coisas que o git não carrega, porque `models/`, `data/` e `*.csv` são
todos gitignored:

- `models/pas3/` — o pacote promovido (modelo + manifesto);
- os CSVs de `data/` — Notas de Corte e banco populacional;
- `assets/` — só mais tarde, no B2B, para os templates whitelabel.

**CORS.** A lista atual tem `"https://*.vercel.app"` como texto literal, que o Starlette **não**
trata como curinga, e `vetorpas.com.br` não está nela. As origens passam a vir do ambiente:
lista explícita para os domínios de produção (`vetorpas.com.br` e `www.vetorpas.com.br`) e
expressão regular para os deploys de preview da Vercel — que é o mecanismo que o Starlette
realmente oferece para isso.

### K. Troca dos CSVs

**Nota de Corte.** `data/notas_corte_pas.csv` (2.307 linhas) sai; o `notas_corte.csv` da extração
(5.225 linhas, incluindo o triênio 2023-2025 que não existe na base velha) entra. Os dois têm
esquemas diferentes — o novo usa nomes minúsculos (`trienio`, `sistema_nome`, `curso`,
`nota_corte`) e o `gestao_service` espera `Trienio`, `Sistema_Nome`, `Curso_Limpo`, `Min`. A
tradução acontece em `load_resources`, que já é o ponto único de carga.

**O CSV novo carrega `inscricao` e `nome` de Alunos reais.** Ele é gitignored, mas vai para dentro
da imagem Docker. A imagem serve só o corte agregado — as duas colunas de PII são **descartadas na
carga**, não embarcadas.

**Discrepância a resolver no ticket, não aqui.** O mapa registra "4.786 cortes"; o arquivo em disco
medido em 2026-07-29 tem **5.225 linhas**, das quais **4.986** com `checksum_fecha == True` e 5.154
não parciais. Nenhum dos três recortes dá 4.786. Quem implementar **mede** em vez de herdar o
número — e o recorte escolhido (linhas limpas ou todas) é decisão do ticket, com o critério escrito.

**Ordem interna que não dá para inverter:** antes de promover o `notas_corte.csv`, é preciso fechar
o **ticket 14 da frente de extração** (validação de formato do campo de classificação). Sem ele,
cortes implausíveis passam — o caso conhecido é MEDICINA/Darcy/Universal em 2020-2022 saindo com
`199.162,872`. Um corte desses no Preditor público vira uma probabilidade absurda na tela de um
Aluno.

**Base populacional.** `resultado_final.csv` (66.313 registros, 8 triênios) tem 510 linhas (0,77%)
com nota de escala corrompida — `eb_p2` chegando a 39.617. **Todas as 510 falham o
`checksum_fecha`**, então o filtro é único e resolve tudo: `checksum_fecha == True` deixa 64.298 de
66.313. A contaminação está só nos cinco triênios mais antigos; os três recentes estão limpos.

### L. Merge das branches

**Direção.** `feat/nextjs-frontend` (o portal) vem para cima de `feat/pdf-extraction` (o modelo), e
não o contrário: a branch do portal está 52 commits atrás da `main` e **deleta** `model_package.py`,
`training_dataset.py`, `training_pipeline.py`, `validation.py` e `dataset_pas3.py` enquanto
**ressuscita** `ensemble.py`, que o ADR-0011 aposentou e que nunca chegou a rodar em produção.

**Cinco conflitos, dois de verdade:** `api/services/predict_service.py` e `PreditorPage.tsx`,
porque os dois lados reescreveram o Preditor por motivos diferentes — um trocou o miolo para `A3` +
incerteza do manifesto, o outro acrescentou semestre, curso-alvo e persistência do Aluno logado.
Os outros três são mecânicos.

**O que a branch do portal traz e é para manter:** `CalculadoraPage.tsx`, o endpoint
`/api/predict/strategy` com schema e serviço, `PublicHeader.tsx`, os fluxos de recuperação de
senha, a página de perfil e a reescrita da landing.

**Bloqueador de privacidade na `feat/proof-section`.** O commit o commit-raiz da PII (SHA não citado aqui de propósito — ver ticket 15) cria
`docs/notas/calibracao-modelo-arg-final.md`, com uma tabela contendo o **nome completo de 6 Alunos
reais** e a chance de aprovação de cada um. Verificado em 2026-07-29: o arquivo **está na árvore de
trabalho** da `feat/proof-section` e da `origin/feat/proof-section` — não é só histórico. Mergear
essa branch para a `main` como está **publica PII**, violando a única restrição do mapa marcada
como dura. O conteúdo técnico da nota (a descoberta MAE-vs-RMSE) já está preservado sem PII no §6
de `07-baseline-honesto.md`, então nada de valor se perde ao removê-la.

Fora isso, `feat/proof-section` tem 14 commits de visual da landing, já contém a `main` inteira e
não conflita com nada — **pode ir a produção sozinha**, a qualquer momento, assim que a PII sair.

---

## Testing Decisions

### O que faz um bom teste aqui

Testa-se **comportamento externo**: o número que sai da porta, a recusa que a API devolve, o
registro que o extrator aceita ou rejeita. Não se testa a forma interna de montar o vetor de
features nem a assinatura de função privada — a única exceção deliberada é o teste de paridade
treino/runtime, que existe justamente porque o desencontro entre os dois **não levanta erro**,
devolve número errado com cara de certo.

Dado sintético do começo ao fim. Nenhuma linha de Aluno real entra em teste, fixture ou exemplo —
`tests/fixtures/` é gitignored porque contém fixtures geradas de Editais reais, e o padrão vigente
é o `conftest.diretorio_do_pacote`, que treina um LightGBM de brinquedo sobre ruído e o escreve em
disco no formato do pipeline.

### Módulos testados e a arte prévia de cada um

| O que | Prior art | O que se acrescenta |
|---|---|---|
| Forma nova do `ExamStats` e as entradas derivadas | `tests/test_pas_extraction_official_stats.py` | uma entrada de Parte 1 misturada é legível por `stats_da_prova` em qualquer língua; `origem` sai no dado |
| Extrator de Editais de Etapa | `tests/test_pas_extraction_medias_desvios.py`, `fixtures.py` | fixture sintética de Edital isolado; o checksum embutido rejeita registro corrompido; um Edital parcial (poucos registros) é **recusado** |
| Calibração do Deslocamento | `scripts/baseline_honesto.py` como precedente de medição com portão | o portão da decisão D é uma asserção, não uma leitura de relatório |
| Língua por Etapa | `tests/test_model_package.py::test_o_runtime_monta_as_mesmas_features_que_o_treino` | **estender com um caso de língua trocada** — é o item que impede a regressão |
| Preditor para a Turma viva | `tests/test_api_predict.py` (já tem `TRIENIO_DA_TURMA_VIVA`) | a constante inverte de sentido: 2024-2026 passa a **responder**, e um triênio sem Edital continua recusando |
| Calculadora sem `.joblib` | `tests/test_pas_intelligence.py::TestTargetCalculator` | nenhum `joblib.load` no caminho; os quatro status contra a faixa medida; os cinco Anos-Âncora saem na resposta |
| Override parcial | `test_override_parcial_e_respeitado` (já existe) | continua passando após a troca do estimador — é o teste que prova que a remoção não mexeu no contrato |
| CORS | — (novo) | `vetorpas.com.br` é aceito; uma origem qualquer é recusada; um preview `*.vercel.app` é aceito |
| Varredura de plausibilidade das Notas de Corte | `tests/test_pas_extraction_notas_corte.py` | nenhum corte promovido fora da faixa observada de Argumento Final |

### Testes que existem hoje e não podem regredir

`pytest tests/` está em **290 passam, 0 falham**. Dois testes documentam contratos que esta rodada
toca de perto e que são a rede de segurança das decisões F, G e H:

- `test_guaranteed_scenario` e `test_alvo_baixo_mas_dentro_da_faixa_ainda_e_possivel_nao_garantido`
  fixam a fronteira entre `garantido` e `possível`. Ao trocar a faixa da P2 pelos valores medidos,
  os dois **mudam de valor esperado** — e essa mudança tem que ser deliberada e escrita, não um
  ajuste até o teste ficar verde.
- `test_o_runtime_monta_as_mesmas_features_que_o_treino` é o único teste que prende desvio entre
  treino e runtime; ele fica mais forte na decisão G, não mais fraco.

### O que se verifica fora do `pytest`

- `/health` respondendo numa URL pública;
- o Preditor funcionando **num navegador** contra essa URL (é onde o CORS falha, e ele não falha em
  teste de servidor);
- uma máquina limpa reproduzindo o deploy sem cópia manual de arquivo;
- o deploy da Vercel verde.

---

## Out of Scope

Nada disto some — só não bloqueia publicar o lado público.

**Todo o B2B.** Upload da base da escola e gravação no Supabase (era a tela "Análise Temporal" do
Streamlit e é o onboarding inteiro do cliente), geração de PDF de verdade (arquivo, lote em ZIP,
templates whitelabel, PDF de cursos, PDF de comparação). Inventário completo em
`app/INVENTARIO-STREAMLIT.md`.

**O Simulador de Itens** — a tela que traduz "acertei tantos do tipo C" em nota. Depende do modelo
de correção item a item e da contagem de itens por tipo de cada prova, que **não sai em Edital**:
só no caderno de questões, com um parser que ainda não existe. Vive só em `feat/nextjs-frontend`
(`simulador_itens.py`, commit `d5d97ed`), com máximos hardcoded e sem o desconto por erro nem o
fator de normalização. E, ao contrário do que se supunha, **não bloqueia a Calculadora**.

**Explicar a população que o Cebraspe usa na Etapa 1.** Ela fica entre a lista do Edital e a dos
concluintes, e nenhum dos sete recortes testados a reproduz. O Deslocamento corrige o efeito sem
que a causa esteja explicada, e isso basta para esta rodada.

**O defeito do nome quebrado** — 2,71% dos nomes saem com espaço no meio da palavra. O mesmo
defeito aparece nos números dos Editais de Etapa (`2. 046`, `1 6.005`), e ali o checksum embutido o
neutraliza. Não toca nenhum número; aparece quando o nome do Aluno for impresso em relatório —
ou seja, no B2B. Ticket 13 da extração.

**O Streamlit quebrado** pelo ticket 13. Com o inventário em disco, consertá-lo virou opcional.

**Superscrever o ADR-0007.** Dívida do ticket 13, não bloqueia publicação.

**Medir `lingua_e2 → lingua_e3`.** A decisão G assume que a língua da Etapa 3 é a da Etapa 2; a
medição que confirmaria isso é um ticket próprio.

---

## Further Notes

### A ordem entre as frentes

O Passo 3 (hospedagem) e o Passo 4 (CSVs) **não dependem** do Passo 2 e podem andar em paralelo
desde já. Estão depois na ordem porque são trabalho que não muda de forma conforme a resposta do
Passo 2 — adiantá-los nunca é errado, só não é o que decide.

O que **não** pode inverter: o ticket 14 da extração antes da promoção do `notas_corte.csv`; e o
portão de calibração antes de as entradas derivadas irem ao ar.

### ADRs que esta rodada toca

- **ADR-0009** (Alvo Canônico = Argumento da Etapa 3) — a decisão C existe para preservá-lo: `A1`
  e `A2` continuam sendo a parte exata da conta, e a correção entra antes deles, não depois.
- **ADR-0012** (Largura fixa por classe) — inalterado; a Calculadora passa a ser o segundo consumidor
  da mesma largura do manifesto.
- **ADR-0008** (Aluno sem Etapa 1) — o controle de ausência declarada no formulário do Preditor
  continua pendente e entra junto da reescrita do formulário no Passo 5.
- **Candidato a ADR novo:** a forma do `ExamStats` com Parte 1 misturada e a marca de procedência.
  É decisão difícil de reverter (muda a forma de um dado que 24 entradas já usam) e merece registro
  formal, não só um comentário no `pas_constants.py`.

### Números desta rodada, para conferência

| | |
|---|---:|
| RMSE do modelo em `A3` | 5,009 |
| Largura de Incerteza em Argumento Final (`3 × σ`) | 15,03 |
| Deslocamento bruto medido | +7,87 |
| Erro residual após corrigir — médio / p95 / máximo | 1,14 / 2,62 / 4,19 |
| Custo da Parte 1 misturada — médio / máximo | 0,46 / 3,21 |
| Custo da língua única em vez de por Etapa — máximo | 3,79 |
| Erro do Estimador Auxiliar — P1 / Redação | 1,47 / 1,36 |
| Faixa medida da P2 na Etapa 3 — piso / teto | 0,24 / 85,6 |
| Entradas no `OFFICIAL_STATS` — antes / depois | 24 / 26 |
| Notas de Corte — antes / depois (linhas do arquivo) | 2.307 / 5.225 |
| idem, só as com `checksum_fecha` | — / 4.986 |
| Linhas limpas do `resultado_final.csv` | 64.298 de 66.313 |

### Glossário

- **`A1`, `A2`, `A3` (Argumento de Etapa):** a nota de uma Etapa já padronizada pela média e pelo
  desvio daquele ano. `Argumento Final = A1 + 2·A2 + 3·A3`. `A1` e `A2` são conta exata; só `A3` é
  previsto. Como `A2` entra com peso 2, um erro na Etapa 2 vale o dobro de um erro na Etapa 1.
- **Turma viva:** o triênio 2024-2026, que faz o PAS 3 em 2026. É quem procura o Preditor hoje.
- **Edital isolado de Etapa:** o "Resultado final nos itens do tipo D e na prova de redação" de uma
  Etapa 1 ou 2, publicado no ano da prova. Lista nota por candidato — inscrição, nome, EB parte 1,
  EB parte 2, somatório, nota tipo D, nota de redação — mas **não a língua estrangeira**.
- **Média e desvio oficiais:** os números que o Cebraspe publica por Edital e que entram na conta do
  Argumento. Um Edital por triênio, com as três Etapas, e só depois do PAS 3.
- **População do Cebraspe:** o conjunto de candidatos sobre o qual ele calcula essa média. Na Etapa
  2 são os **concluintes do triênio**, não quem fez a prova — daí o erro de −4,61.
- **Deslocamento:** o degrau de +7,87 pontos de Argumento Final entre o estimado e o oficial. É
  sistemático, não ruído, e é por isso que subtraí-lo funciona.
- **Largura de Incerteza:** quanto o modelo tipicamente erra, usada como o desvio-padrão da normal
  que produz a probabilidade de aprovação. Um número por classe de Aluno, vindo do manifesto do
  pacote.
- **Estimador Auxiliar:** prever P1 e Redação do PAS 3 pela média ponderada dos z-scores das Etapas
  anteriores, em vez de por modelo. Aritmética, não ML.
- **z-score:** a nota expressa em desvios-padrão em relação à média daquela prova. Serve para
  comparar notas de provas com dificuldades diferentes.
- **Ano-Âncora:** um ano real e já publicado usado como cenário — *"e se a minha Etapa 3 for como a
  de 2023?"*. Substitui a projeção de uma prova que ainda não aconteceu.
- **Escore Bruto (EB):** `P1 + P2`, já normalizado para que acertar 100% das duas partes juntas dê
  100. Por isso o teto da P2 sozinha não é 100: é `100 − P1`.
- **Faixa de P2:** os limites que decidem quando a Calculadora diz "impossível" e quando diz
  "garantido". Eram o chute `[−100, 100]`; agora são medidos, e são **por Etapa** — a Etapa 2 admite
  nota negativa, a Etapa 3 não.
- **Nota de Corte:** o menor Argumento Final entre os aprovados de um curso, num Sistema de
  Concorrência, na última chamada.
- **Space (Hugging Face):** onde a API Python vai rodar. A Vercel só hospeda o Next.js; modelo
  Python não roda lá.
- **CORS:** a regra que diz de quais endereços o navegador pode chamar a API. Se `vetorpas.com.br`
  não estiver na lista da API, o navegador recusa a chamada.
- **Simulador de Itens:** tela que traduz contagem de acertos por tipo de item (A/B/C/D) em nota.
  Outra coisa que a Calculadora, e fora desta rodada.
