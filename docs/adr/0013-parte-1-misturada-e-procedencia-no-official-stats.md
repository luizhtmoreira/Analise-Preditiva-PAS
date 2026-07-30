# A Parte 1 pode vir misturada, e toda entrada declara sua procedência

O `OFFICIAL_STATS` passa a guardar a Parte 1 de uma Etapa em **duas formas**, e cada entrada passa a declarar de **onde veio**. Nenhum valor existente muda; nenhuma entrada nova entra por esta decisão.

O Cebraspe normaliza a Parte 1 separadamente por língua estrangeira, e agrupar as três embute viés sistemático contra quem fez espanhol ou francês. Por isso a forma canônica é e continua sendo **por língua** — as três, sem parcial. Mas a única fonte disponível para a Turma viva é o **Edital isolado de Etapa** (o "Resultado final nos itens do tipo D e na prova de redação"), que lista nota por candidato e **não diz a língua de ninguém**. Dele só sai a Parte 1 **misturada**. Preencher as três a partir dele exigiria inventar valores.

Isso **complementa e não substitui** o critério do ADR-0009 ("usa-se estatística por língua onde o spread entre elas é estável e agrupada onde não é — critério medido e reavaliado a cada Edital"). Aquele critério decide, com o dado por língua em mãos, quando vale agrupar. Este decide o que fazer quando o dado por língua **não existe**. Uma entrada misturada não é o resultado de um agrupamento escolhido; é o que a fonte publicou.

O custo de usar a misturada está medido e é o que a torna aceitável: **0,46 ponto de Argumento Final em média, máximo 3,21, com viés zero** — é ruído, não erro sistemático. A Parte 1 pesa `0,72` numa conta que soma `10`, e a média misturada cai praticamente em cima da média da inglesa, que é 66% a 73% da população.

A diferença entre as formas fica **no tipo** (`Parte1PorLingua` / `Parte1Misturada`), nunca numa convenção implícita como contar chaves de dicionário — as duas formas têm três chaves, então contar deixou de discriminar e o teste que contava passou a checar o tipo. `ExamStats.parte_1` é declarado `Parte1`, sem união com `dict`, para que o tipo declarado seja o tipo real; as 24 entradas passaram a envolver o dicionário em `Parte1PorLingua(...)`, uma mudança mecânica que não toca um valor sequer. As duas formas são `Mapping` das três línguas oficiais, o que mantém todos os leitores atuais funcionando: `stats.parte_1[lingua]`, `.items()`, `set(stats.parte_1)`, `m_p1`/`dp_p1`. Na forma misturada as três chaves respondem o mesmo par média/desvio — e só as três: uma língua fora da lista continua sendo `KeyError`. O `m_p1` — que nunca foi um número oficial, só o agregado que `api/services/analytics_service.py` lê — vira o próprio valor.

`stats_da_prova(ano, etapa, lingua)` continua sendo o **ponto único de leitura**, e sobre uma entrada misturada devolve a estatística misturada **qualquer que seja a língua pedida**, em vez de levantar erro. Recusar devolveria o Preditor ao estado de não atender a Turma viva, que é o estado de que esta rodada existe para sair. Treino e runtime continuam entrando pela mesma porta, senão o `A1` que a API mostra deixa de ser o `A1` com que o modelo foi treinado.

A **procedência** (`Origem.EDITAL` / `Origem.DERIVADA`) existe porque, quando o Edital de médias e desvios de 2026 sair, os números derivados serão substituídos e as previsões vão mexer. Isso precisa estar registrado no dado, não descoberto depois por quem estranhar uma previsão que mudou sozinha. As 24 entradas atuais são todas `EDITAL`, que é também o default — a forma derivada nasce de um único ponto de construção programática, que declara.

Note que **misturada e derivada são eixos independentes**: o Edital isolado de Etapa é um Edital, então uma entrada pode ser perfeitamente `Parte1Misturada` com `Origem.EDITAL`.

## Considered Options

- **Manter a Parte 1 sempre por língua e replicar o valor misturado nas três**: descartado — funciona igual na conta, mas apaga do dado a informação de que aquilo não é o valor da língua; o próximo leitor não teria como saber que está lendo uma estimativa.
- **Distinguir as formas por convenção (uma chave só no dicionário, ou uma chave `"misturada"`)**: descartado — convenção implícita que todo leitor precisa conhecer e que nenhum verificador de tipo cobra.
- **Aceitar `dict` cru em `ExamStats.parte_1` e converter no `__post_init__`, deixando as 24 entradas literalmente intocadas**: descartado — mantinha o tipo declarado como `Union[Parte1, dict]` para sempre, exatamente a ambiguidade que esta decisão existe para remover. Envolver as 24 em `Parte1PorLingua(...)` é mecânico e não muda valor nenhum.
- **`stats_da_prova` recusar entrada misturada e forçar o chamador a tratar**: descartado — é exatamente a recusa que impede o Preditor de atender a Turma viva hoje, e empurra a decisão para cada chamador em vez de resolvê-la uma vez.
- **`origem` sem default, obrigando as 24 entradas a declarar**: descartado — o risco de uma entrada derivada nascer marcada como `EDITAL` por esquecimento fica contido porque elas vêm de um construtor só, não de literais escritos à mão.
- **Não registrar procedência e resolver quando o Edital sair**: descartado — a substituição vai mexer nas previsões de Alunos reais, e descobrir isso depois é caro.
