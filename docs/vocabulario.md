# Vocabulário do PAS

Os termos que aparecem nesta documentação, em ordem de quem está chegando agora. Nenhuma
definição aqui pressupõe conhecimento técnico.

## O programa

**PAS** — Programa de Avaliação Seriada da UnB. Processo seletivo em três provas anuais, feitas ao
longo dos três anos do ensino médio.

**Etapa** — cada uma das três provas anuais (PAS 1, PAS 2, PAS 3). Elas têm pesos diferentes: a
terceira vale três vezes mais que a primeira.

**Triênio** — o conjunto das três etapas de uma mesma turma. O triênio 2023/2025 fez PAS 1 em
2023, PAS 2 em 2024 e PAS 3 em 2025.

**Etapa ausente** — etapa que o aluno não fez. O edital publica as notas dela como zero, mas isso
significa **ausência, não desempenho zero** — uma distinção que muda completamente a conta. Faltar
à primeira etapa é permitido e o aluno segue no programa; cerca de 9% dos alunos do resultado
final estão nessa situação.

## As notas

**Escore Bruto (EB)** — a nota crua de uma prova, somando as duas partes objetivas. É a nota que
o aluno reconhece, mas é a menos útil para comparar anos diferentes, porque uma prova fácil infla
o escore bruto de todo mundo.

**Argumento de Etapa** — a nota de uma etapa já **descontada da dificuldade da prova daquele
ano**. É calculada comparando o aluno com todos os outros candidatos do mesmo ano. Por isso um
argumento de etapa de 2019 e um de 2024 são diretamente comparáveis, e dois escores brutos não são.

**Argumento Final** — a pontuação cumulativa que a UnB usa para classificar. Combina as três
etapas com pesos 1, 2 e 3. É por esse número, e só por ele, que se passa ou não se passa.

**Nota de corte** — o menor Argumento Final que garantiu vaga em um curso na última chamada. Não
existe "a nota de corte do curso": existe uma por **curso**, por **sistema de concorrência** e por
**semestre de entrada**.

**Sistema de concorrência** — a modalidade em que o candidato disputa: universal, cota para
negros, ou uma das oito combinações de escola pública com renda, cor e deficiência. São dez ao
todo, e cada uma tem sua própria nota de corte.

**Semestre de entrada** — primeiro ou segundo semestre letivo. Não é uma escolha do aluno: todos
fazem a mesma prova e concorrem de uma vez. Quem não entra no primeiro continua disputando o
segundo, e por isso o corte do segundo semestre é **sempre** mais baixo. Na prática, o aluno
disputa tudo ao mesmo tempo, e entra se limpar qualquer um dos cortes.

## A previsão

**Largura de Incerteza** — o quanto o modelo costuma errar. Hoje, cerca de 15 pontos de Argumento
Final. É o que transforma uma previsão isolada em uma faixa honesta, e é o que sustenta a conta de
probabilidade. Muda a cada retreinamento e viaja dentro do próprio arquivo do modelo.

**Chance de aprovação** — a probabilidade de o Argumento Final do aluno ficar acima da nota de
corte, calculada a partir da previsão e da Largura de Incerteza. Quando dizemos 80%, acontece 80%
das vezes — ver [Como sabemos que acerta](confianca/como-sabemos-que-acerta.md).

**Quanto falta** — o caminho inverso: dada a nota de corte, qual a nota mínima necessária na
última etapa. É aritmética exata, não previsão, e por isso continua confiável mesmo para o aluno
cuja previsão é incerta.

**Semáforo de risco** — a classificação visual do risco de um aluno: verde, amarelo, vermelho. Há
um quarto estado, cinza, para quando falta dado e a resposta honesta é "não sei".

**Ano-âncora** — um ano real e já publicado, usado como cenário para a pergunta "e se a minha
terceira etapa for como aquela?". Cada ano-âncora amarra junto a nota de corte daquele ano e a
dificuldade da prova daquele mesmo ano — nunca uma combinação que não aconteceu. Substitui a
projeção do ano futuro: a diferença entre os anos-âncora **é** a incerteza, mostrada em vez de
estimada.

**Procedência** — de onde veio a média e o desvio de uma prova: **edital**, quando o Cebraspe já
publicou, ou **derivada**, quando ainda não publicou e o número foi calculado por nós. A
procedência fica registrada dentro do próprio dado e aparece como aviso na tela, porque quando o
edital verdadeiro sair os números derivados serão substituídos e as previsões vão se mexer.

**Volatilidade** — o tamanho da oscilação entre as etapas já feitas pelo aluno, sem direção. Subir
três pontos e cair três pontos dão a mesma volatilidade.

**Momentum** — a evolução do aluno de uma etapa para a seguinte, **com** direção. É a hipótese
central do produto: quem sobe muito da primeira para a segunda etapa tende a ir bem na terceira.

!!! warning "Por que o momentum é medido em argumento, e não em escore bruto"
    Porque medir em escore bruto confunde "o aluno evoluiu" com "a prova ficou mais fácil". Nos
    60 mil alunos da base, as duas medidas **discordam sobre o sinal em 17% dos casos** — e em
    2022/2024, cuja segunda etapa foi muito mais fácil, a discordância chega a 39%.

## Como o modelo foi medido

**Validação deslizante** — medir o modelo treinando no passado e prevendo o futuro, repetidamente.
É o oposto de sortear alunos aleatoriamente para teste, que mediria uma tarefa que o produto nunca
executa.

**Holdout lacrado** — um triênio inteiro separado no começo do projeto e mantido fora de todo
treino e de toda decisão, aberto uma única vez no final para medir. É a medida mais próxima de
"o que teria acontecido de verdade" que se pode obter sem esperar mais um ano.

**Calibração** — a propriedade de a probabilidade dita corresponder à frequência observada. Um
sistema pode acertar muito e ser mal calibrado (confiante demais); o inverso também existe. Nós
medimos e publicamos as duas coisas.

## Do lado da escola

**Escola parceira** — a instituição contratante.

**Whitelabel** — a personalização do sistema e dos relatórios com o logotipo, as cores e a
identidade da escola. O aluno recebe um documento da escola dele.
