# Glossário de estudo — termos que apareceram nos tickets

Documento vivo. Cada termo aqui apareceu numa conversa de decisão e **não estava claro na hora**.
A regra: explicar em português, com o exemplo deste projeto, e dizer por que importa.

Acrescentar termos novos conforme os tickets forem abrindo, em vez de reexplicar na conversa.

---

## Parte 1 — Como um modelo vira arquivo

### Serializar

Um modelo treinado é um objeto vivo na memória do computador: árvores de decisão com milhares
de nós, pontos de corte, pesos. **Serializar** é gravar isso em disco para usar amanhã sem
retreinar. O caminho inverso é *carregar* (ou *desserializar*).

### `pickle`

O serializador embutido do Python. Ele não grava "uma tabela de números" — grava **instruções de
como remontar o objeto**: "importe a classe `RandomForestRegressor` da biblioteca `sklearn`, crie
uma instância, coloque estes valores nestes atributos".

Duas consequências vêm daí:

1. **Depende da biblioteca continuar igual.** Se o `sklearn` mudar um nome interno entre versões,
   a instrução de remontagem aponta para um lugar que não existe mais.
2. **Carregar é executar código.** Um arquivo malicioso pode conter instruções arbitrárias. Por
   isso a regra geral: nunca carregar `pickle` de fonte desconhecida. *(No Vetor PAS isso quase
   não pesa — você produz e consome o próprio arquivo. O risco real aqui é o item 1.)*

### `joblib`

Um empacotamento do `pickle`, otimizado para arrays numéricos grandes. É o formato dos arquivos
em `models/`. Herda as duas propriedades do `pickle` acima.

**Caso real neste projeto:** `p1_pas3_model.joblib` e `red_pas3_model.joblib` não carregam mais.
Erro: `ModuleNotFoundError: No module named '_loss'`. Foram salvos com um `sklearn` em que
existia um módulo `_loss`; no `sklearn` 1.9 ele mudou de lugar. Os números estão intactos no
arquivo — a receita de remontagem é que aponta para o vazio.

### Formato nativo (do LightGBM)

Alguns frameworks têm um jeito próprio de gravar o modelo, que não usa `pickle`. O LightGBM grava
em **texto puro**: dá para abrir num editor e ler os nomes das features, o número de árvores, os
pontos de corte. Não executa código ao carregar, e o LightGBM se compromete a ler arquivos
antigos.

Limite: só cobre o modelo do LightGBM. Um `StandardScaler` do sklearn não tem formato nativo.

### ONNX

Um formato de intercâmbio: converte o modelo num "grafo" que roda num programa próprio
(`onnxruntime`), sem precisar do `sklearn` nem do `LightGBM` instalados. Serve para desacoplar o
ambiente de treino do ambiente que serve as previsões.

Custo: um passo de conversão a mais (que pode falhar), e a conversão trabalha com números de
menor precisão — diferença na terceira casa decimal, irrelevante numa nota de 0 a 100.

### Opset

O número de versão do conjunto de operações do ONNX, gravado dentro do arquivo. Serve para o
programa que lê dizer "eu suporto até a versão N". É compatibilidade **explícita** — o contrário
do `pickle`, onde a incompatibilidade só aparece como erro obscuro na hora de carregar.

---

## Parte 2 — Como um modelo vira produto

### Unidade versionável

O conjunto de arquivos que **tem que viajar junto e ser trocado junto**.

Exemplo do projeto: `modelo_linear.joblib` recebe dados já normalizados por `scaler.joblib`. Se
você retreinar o modelo e esquecer de regravar o scaler, a API carrega os dois, não dá erro
nenhum, e responde errado. Os dois são uma unidade só.

A decisão do ticket 03 foi: a unidade é o **pacote inteiro** de uma rodada de treino. Não existe
"meio pacote" em produção.

### Manifesto

Um arquivo de metadados que viaja junto do modelo, respondendo "de onde isto veio":
qual dataset (com hash), qual commit de código, quais hiperparâmetros, qual métrica de holdout,
quais versões de biblioteca. É o que transforma um arquivo binário anônimo em algo rastreável.

### Imutabilidade

Uma versão publicada nunca muda de conteúdo. Se precisa mudar, vira versão nova. É a propriedade
que o Dropbox não dá: arrastar um arquivo por cima sobrescreve, e o de ontem some sem deixar
registro.

### Endereço por conteúdo / checksum

Um resumo matemático do arquivo (um "hash", tipo SHA-256). Se um único byte mudar, o resumo muda.
Serve para provar que o arquivo que chegou é o mesmo que saiu.

### Linhagem (*lineage*)

A cadeia que liga o modelo de volta ao dado e ao código que o geraram. Sem linhagem você tem um
arquivo que prevê números e nenhuma forma de saber por quê.

### Commit / SHA / revisão / tag

Vocabulário de git, usado aqui para versionar modelo em vez de código:

- **commit** — uma gravação no histórico, com autor e data.
- **SHA** — o identificador único e imutável daquele commit (aquele código longo tipo `3f8a1c...`).
- **revisão** — o jeito de pedir "me dá o repositório neste ponto exato do histórico".
- **tag** — um apelido legível para um commit (`v2`, `producao`), útil porque ninguém decora SHA.

### *Pin* (fixar / cravar)

Registrar a versão **exata** de alguma coisa, em vez de "a mais recente".

`scikit-learn` no `requirements.txt` significa "instale a mais nova que existir hoje" — o que
muda com o tempo sem você mandar. `scikit-learn==1.9.0` significa "esta, sempre". Hoje o
`requirements.txt` do projeto tem 22 linhas e **zero** versões cravadas.

### Git LFS

Extensão do git para arquivos grandes. O git normal guarda o conteúdo inteiro de cada versão no
histórico e engasga com binários de centenas de MB. O LFS guarda, no lugar do arquivo, um bilhete
com o checksum, e o conteúdo fica num servidor à parte.

### Promoção e *rollback*

- **Promover** — colocar uma versão nova de modelo em produção.
- ***Rollback*** — voltar para a anterior porque a nova deu problema.

O padrão maduro: o "o que está em produção agora" é um ponteiro versionado no git. Promover é um
commit; reverter é desfazer esse commit. O histórico do git vira o histórico de deploys, de
graça, com autor e data.

### *Registry* de modelos

Um serviço que guarda modelos com versão, metadados e estado ("este é o de produção"). MLflow,
Weights & Biases, o registry da AWS. **Nota honesta:** quase toda essa cerimônia existe para
resolver problemas de escala e de time — muitos modelos, muitas pessoas treinando, auditoria
regulatória. Uma pessoa com um retreino por ano não precisa. As propriedades importam; a
ferramenta é embalagem.

---

## Parte 3 — Como o modelo chega ao ar

### Imagem

A "caixa lacrada" que o servidor executa: um pacote com o Python, as bibliotecas e o teu código,
tudo dentro. Você monta uma vez ao publicar; o provedor liga e desliga essa caixa conforme a
demanda.

### *Build* (montagem) × *boot* (ligar)

- **Build** — montar a caixa. Acontece uma vez, ao publicar. Demora minutos. Você está olhando.
- **Boot** — ligar a caixa. Acontece toda vez que o servidor acorda. Tem que ser rápido. Ninguém
  está olhando.

A decisão do ticket 03 foi assar o modelo dentro da caixa no **build** — para que o boot não
dependa de rede, e para que o teste de carregamento aconteça enquanto você ainda está olhando.

### Hibernação e *cold start*

O Hugging Face Spaces gratuito **desliga** o servidor após 48h sem acesso. Quando alguém entra
depois disso, o servidor precisa acordar do zero antes de responder — isso é o **cold start**, e
o aluno vê como tela girando. É por isso que o ADR-0004 previu o UptimeRobot cutucando o site a
cada 5 minutos.

### *Secret* / token

Uma senha guardada na configuração do servidor, fora do código. O `HF_TOKEN` é o que autoriza
baixar de um repositório privado. Nunca vai para dentro do repositório de código.

### *Smoke test*

Um teste mínimo que só pergunta "isto sequer liga?". No caso: carregar o modelo e fazer uma
previsão de mentira. Não mede qualidade — mede se o artefato está vivo.

### Falha silenciosa × falha barulhenta

**Silenciosa:** o sistema falha e continua respondendo como se nada fosse. É o
`target_calculator.py:66` de hoje — o modelo não carrega, o código imprime um aviso que ninguém
lê, e passa a responder com média ponderada em vez de ML. O aluno recebe um número e acredita.

**Barulhenta:** o sistema recusa-se a funcionar. É pior de manhã e melhor no resto da vida.
Servidor caído você conserta; número errado entregue como certo você nunca descobre.

---

## Parte 4 — Como se mede um modelo

*(Decidido pelo ticket 06 — é a régua dos tickets 07 a 13.)*

### Holdout

Um pedaço dos dados **separado antes do treino** e nunca mostrado ao modelo. Serve para medir
desempenho em dados que ele não decorou. Medir no dado de treino é como corrigir a prova com o
gabarito ao lado.

Cuidado: a palavra tem dois usos neste mapa. Em cada **dobra**, o triênio previsto é um holdout
*temporário* — ele volta a ser treino na dobra seguinte. O **holdout lacrado** é outra coisa.

### Dobra (*fold*)

Uma rodada da validação: um par *"treinei com estes triênios, prevejo aquele"*. Cinco dobras
devolvem cinco números em vez de um — e é a diferença entre eles que mostra se o erro é estável
entre anos ou se um ano específico foi um desastre. Com um número só você não tem como saber.

### Validação deslizante (*rolling-origin*)

Esquema em que a fronteira entre treino e teste **anda um triênio para a frente** a cada dobra,
sempre treinando no passado e prevendo o futuro. Imita o que o produto faz de verdade: prever
uma Etapa 3 que ainda não aconteceu, num ano que não está na base.

```
dobra 1 · treina 2016/2018, 2017/2019       → prevê 2018/2020
dobra 2 · treina os anteriores + 2018/2020  → prevê 2019/2021
dobra 3 · treina os anteriores + 2019/2021  → prevê 2020/2022
dobra 4 · treina os anteriores + 2020/2022  → prevê 2021/2023
dobra 5 · treina os anteriores + 2021/2023  → prevê 2022/2024
──────────────────── LACRE ────────────────────
        · treina tudo até 2022/2024         → prevê 2023/2025
```

O contrário é o **split aleatório**, que sorteia linhas sem olhar o ano e mistura triênios entre
treino e teste — mede interpolação dentro de anos conhecidos, que é uma coisa que o produto
nunca faz.

### Holdout lacrado

O triênio **2023/2025**, que não entra em treino nenhum e cujo resultado é olhado **uma única
vez**, no ticket 13. "Lacrar" não é termo técnico, é a palavra da casa: fica guardado e ninguém
abre.

O motivo é aritmético, não disciplinar. Olhar o número, mexer no modelo, olhar de novo e mexer de
novo transforma o triênio em mais uma dobra — você deixa de **medir** o modelo e passa a
**ajustá-lo até o número ficar bonito**. Depois disso não sobra nenhum ano limpo, e o próximo só
existe quando sair o Edital de 2027.

Por isso a regra de uso é escrita **antes** de o número ser conhecido: abrir o lacre produz um
número, não uma decisão. Ou promove com ele, ou desiste da rodada — mexer no modelo depois de
ver o resultado queima o lacre.

### Receita

O conjunto de escolhas que define um modelo antes de ele encontrar dado: quais features, qual
família, quais hiperparâmetros, como trata a Etapa 1 ausente. **O que as dobras medem é a
receita, não o arquivo.**

É o que permite embarcar em produção um modelo que nunca foi medido: mede-se a receita seis vezes
em anos que ela não viu, e depois roda-se a mesma receita com todo o dado. Analogia: você não
consegue medir o aluno que ainda não fez a prova — você mede o **método** nas turmas que já
terminaram e diz o número do método para o aluno de hoje.

### Modelo medido × modelo embarcado

O **medido** é o treinado até 2022/2024, o único contra o qual existe um ano limpo para comparar.
O **embarcado** é o treinado nos 8 triênios, que vai para produção porque usa também o triênio
mais parecido com o Aluno vivo — e que, por construção, não tem contra o quê ser medido.

Decisão do ticket 06: **embarca-se o dos 8**, e o manifesto escreve a frase inteira, sem esconder
nada: *"erro X — medido sobre a mesma receita treinada até 2022/2024; o arquivo embarcado foi
treinado até 2023/2025 e não foi medido"*.

### MAE (erro absoluto médio)

Média de quanto o modelo erra, em pontos, ignorando o sinal. `MAE = 13` significa "erra 13 pontos
na média". É a métrica mais fácil de explicar a um humano.

### RMSE (raiz do erro quadrático médio)

Parecido com o MAE, mas eleva os erros ao quadrado antes de somar — o que faz erros grandes
pesarem desproporcionalmente. Usa-se quando errar muito de vez em quando é pior do que errar
pouco sempre.

**Onde isso já aparece no projeto:** `statistics.py` usa `RMSE = 13.49` como a incerteza da
previsão para calcular probabilidade de aprovação. Esse número é resíduo do modelo antigo e fica
errado por construção quando o modelo trocar (ticket 11).

### R² (coeficiente de determinação)

Compara o modelo contra o chute mais burro possível (responder sempre a média).

- `R² = 1` → acerta perfeitamente.
- `R² = 0` → empata com o chute da média.
- `R² < 0` → **pior que o chute da média**.

**Caso real:** o ADR-0007 registrou `R² = -83.4` para o `modelo_mlp`. Isso não é "modelo ruim" —
é sinal de que algo está quebrado no experimento. E estava: o script de avaliação alimentava os
modelos com as features na ordem errada.

### Baseline

A régua contra a qual o modelo novo é comparado. Sem baseline válido, "melhorou" não quer dizer
nada. O ADR-0007 era o baseline deste mapa e está inválido (ticket 07).

---

## Parte 5 — Vocabulário de modelagem

### Família de modelo

O tipo de algoritmo. Regressão linear é uma família; floresta aleatória (`RandomForest`) é outra;
*gradient boosting* (LightGBM, XGBoost, HistGradientBoosting) é outra; rede neural (`MLP`) é
outra.

### GBM / *gradient boosting*

Família que constrói muitas árvores pequenas em sequência, cada uma corrigindo o erro que as
anteriores deixaram. "**Um GBM único**" significa usar um só modelo dessa família, em vez do
arranjo de vários. LightGBM é uma implementação de GBM.

### *Ensemble*

Usar vários modelos juntos e combinar as respostas. É o arranjo atual: `lgbm`, `rf`, `linear` e
`mlp`, com o peso de cada um definido pela volatilidade do aluno.

### Meta-modelo

Um modelo cuja função é escolher **qual dos outros modelos** usar para cada aluno. É o
`meta_model.joblib`. É um nível de indireção a mais — e o ticket 10 vai julgar se ele paga.

### Nota negativa (Escore Bruto e Argumento)

No PAS, **nota de prova pode ser negativa** — o Escore Bruto desconta erro, então um
desempenho ruim leva abaixo de zero, e o Argumento (que normaliza pela média do ano) fica
negativo para quem está abaixo da média. Zero **não** é o piso.

Isso importa em qualquer lugar que trate nota como grandeza positiva: `target_calculator.py`
trunca P1 em `[-20, 20]` corretamente, e trabalha com P2 em `[-100, 100]` — faixa cuja origem
não está documentada (ver defeito 1 em `relatorios/defeitos-pendentes.md`).

### Features

As colunas de entrada que o modelo lê para prever. O `modelo_lgbm` foi treinado com seis:
`EB_PAS1, Red_PAS1, EB_PAS2, Red_PAS2, Cresc_EB, Cresc_Red`.

**Armadilha real deste projeto:** quando você passa as features como uma lista de números pura,
o modelo não confere nome nenhum — lê por **posição**. Passar a lista na ordem errada não gera
erro, gera número errado. Foi exatamente o que aconteceu no `scripts/baseline_avaliacao.py`, e
foi o que invalidou o ADR-0007.

### Hiperparâmetro

Configuração escolhida **antes** de treinar, que não é aprendida do dado: quantas árvores, qual
profundidade máxima, qual taxa de aprendizado. Precisa entrar no manifesto, porque é parte da
receita para reproduzir o modelo.

### XAI (IA explicável) × transparência do artefato

Duas coisas diferentes, e a confusão é comum:

- **Transparência do artefato** — conseguir abrir o arquivo e ver o que ele é, sem executar
  código. É auditoria de procedência: "este arquivo é mesmo o que eu penso que é?".
- **XAI** — responder *por que este aluno recebeu esta previsão*. "A previsão de 68 vem
  principalmente da nota alta no PAS 1, puxada para baixo pela queda no PAS 2."

O texto nativo do LightGBM dá a primeira, não a segunda. São 100 árvores encadeadas: legível por
máquina, não compreensível por humano. **Legível ≠ explicável.**

### Fora de distribuição (*out of distribution*)

Uma entrada cujo padrão **não aparece no treino**. O caso do projeto: o Aluno sem Etapa 1 chega ao
modelo com `EB_PAS1 = 0` e `Cresc_EB = +35`, combinação que nenhuma linha de treino tem.

O que torna isso perigoso é o que **não** acontece: não dá erro, não dá aviso, não vem com
confiança menor. O modelo responde um número com a mesma cara de todos os outros.

E árvore tem uma propriedade agravante: **ela não extrapola**. Uma reta, pedindo `x = 0` quando só
viu `x` perto de 30, pelo menos estende a reta. Uma árvore só sabe dizer "cai na folha dos menores
que eu vi" — então a resposta gruda no pior PAS 1 que existiu no treino, seja lá qual for. Pode até
sair na direção certa por acidente (o Aluno sem Etapa 1 *é* prejudicado), com magnitude que
ninguém mediu.

### Valor faltante (`NaN`, *missing value*)

Marcar explicitamente "este número não existe", que é **diferente de zero**. Zero é um valor;
faltante é a ausência de valor.

Importa aqui porque as famílias de modelo se dividem nisso:

- **LightGBM, HistGradientBoosting** — aceitam nativamente. Durante o treino a árvore aprende, em
  cada nó, para que lado mandar quem está faltando. É informação aprendida, não chute.
- **Regressão linear, MLP** — não aceitam. Exigem preencher o buraco com alguma coisa antes, e essa
  coisa é sempre uma invenção.

Por isso o ticket 10 não pode tratar "aceita faltante" como desempate: escolher linear ou MLP
**fecha a porta** do Aluno sem Etapa 1 e obriga um pipeline separado.

### Estratificar

Garantir que uma divisão (treino/holdout) preserve a proporção de um grupo em cada parte, em vez de
deixar por conta do sorteio.

Sem estratificar por `etapa_1_ausente`, o holdout pode sair com poucos Alunos dessa classe, e a
métrica deles vira ruído — dá para "melhorar o modelo" piorando os 9% sem que apareça em número
nenhum.

### Regime

Um conjunto de linhas geradas pela **mesma regra**. O projeto tem dois exemplos e eles são de tipos
diferentes:

- **Regime de alvo** — as 1.483 linhas antigas cujo Argumento Final impresso segue a regra generosa
  de Etapa ausente que ninguém reconstruiu. O *rótulo* delas é de outro mundo.
- **Regime de feature** — o Aluno sem Etapa 1. O rótulo está certo; a *entrada* é de outro mundo.

Misturar regimes num modelo só é legítimo — às vezes é até melhor, porque dá mais dado. Mas tem que
ser **medido**, nunca presumido.

### Momentum × Volatilidade

Duas grandezas que parecem a mesma e não são:

- **Volatilidade (CV)** — `std/mean`. Quanto as notas **variam**. Não tem sinal.
- **Momentum** — para onde e quanto o Aluno **andou**. Tem sinal.

```
CV([30, 35]) = 7,69%   ← subiu 5
CV([35, 30]) = 7,69%   ← caiu 5
```

O CV não distingue os dois. O `meta_model` roteia por CV, então ele decide qual modelo usar sem
saber se o Aluno subiu ou caiu — cego exatamente à hipótese que motivou o produto (defeito 5).

### SHAP

A técnica de XAI que reparte a previsão entre as features, dizendo quanto cada uma empurrou para
cima ou para baixo. Para árvores existe cálculo exato e barato — no LightGBM, é
`predict(X, pred_contrib=True)`. É matéria-prima de produto para o Vetor PAS, e está anotada como
névoa no `map.md`, não como escopo deste mapa.
