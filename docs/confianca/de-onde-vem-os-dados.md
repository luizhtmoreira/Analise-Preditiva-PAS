# De onde vêm os dados

Esta é a primeira pergunta que um professor cético deve fazer, e ela tem uma resposta curta:
**os dados vêm dos editais oficiais do Cebraspe**, os mesmos PDFs que a banca publica no site
depois de cada etapa do PAS. Nada é comprado, nada é estimado por amostra, nada vem de
formulário preenchido por aluno.

## O que já foi lido

| | |
|---|---|
| **Editais processados** | 77 PDFs oficiais (o corpus segue crescendo a cada publicação) |
| **Alunos** | 66.313 registros de resultado final |
| **Triênios cobertos** | 8, de 2016/2018 a 2023/2025 |
| **Convocações** | 33.386 registros de quem foi chamado, em qual chamada |
| **Notas de corte** | 4.786, uma por curso × sistema de concorrência × semestre |
| **Tamanho dos editais** | de 242 a 419 páginas cada |

!!! info "Triênio"
    Um **triênio** é o conjunto das três provas anuais que uma mesma turma faz ao longo do ensino
    médio. O triênio 2023/2025 é a turma que fez PAS 1 em 2023, PAS 2 em 2024 e PAS 3 em 2025.

## Por que isso importa mais do que parece

Um edital de resultado final do PAS tem centenas de páginas com uma linha por candidato. Ler
esses PDFs à mão não é viável, e ler errado é pior do que não ler: um número trocado vira uma
previsão errada na tela de um aluno real.

Por isso a leitura dos editais não é um script que "extrai e confia". Cada registro passa por
**seis conferências automáticas** antes de ser aceito, e cada linha carrega no próprio arquivo o
resultado das conferências que fez — quem consome os dados pode filtrar por confiança em vez de
acreditar.

## A conferência mais importante: o edital confere a si mesmo

Esta é a parte que costuma convencer quem entende do assunto.

O edital publica, para cada aluno, as nove notas das três provas **e também** o Argumento Final
já calculado. O Argumento Final é obtido a partir dessas nove notas por uma fórmula pública, que
o próprio edital descreve. Então dá para refazer a conta e comparar com o número impresso.

Se o valor recalculado bate com o publicado, **doze campos daquele aluno foram verificados de uma
vez só** — as nove notas, o argumento e os dados da tabela oficial usada na conta. Se não bate,
alguma coisa foi lida errada, e a linha é marcada.

!!! info "Argumento Final"
    A pontuação cumulativa que a UnB usa para classificar os candidatos do PAS. Vale
    `1 × primeira etapa + 2 × segunda etapa + 3 × terceira etapa`, sobre notas já padronizadas
    pela dificuldade da prova de cada ano. É por esse número que se passa ou não se passa.

Resultados dessa conferência:

- **8.499 de 8.499 registros fecham** num edital completo de 242 páginas conferido inteiro.
- **96,96% fecham** no acervo inteiro, com as falhas espalhadas por todo o corpus — o padrão de
  quem tem alguns registros danificados, e não o de quem está usando a fórmula errada.
- Nos **três triênios mais recentes, nenhuma falha**: 100% das linhas fecham.

A fórmula usada na conferência não foi adivinhada a partir do texto do edital. Ela foi
**reconstruída estatisticamente** a partir dos dados e conferida: os pesos recuperados batem com
os oficiais com erro máximo de 0,005 em todos os triênios testáveis. Isso também provou uma
coisa útil — **a fórmula não mudou entre 2016 e 2025**, o que autoriza usar a série histórica
inteira.

## As outras cinco conferências

| Conferência | O que ela pega |
|---|---|
| **Sequência de classificação** | Um aluno que o leitor simplesmente nunca extraiu. É o ponto cego de todas as outras: se a lista de um curso vai de 1 a 900 e falta o 447, alguém sumiu. |
| **Ordem alfabética** | Dois registros colados um no outro dentro da mesma página. |
| **Formato do número** | Um número partido por um espaço no meio (`"1 7.539"` no lugar de `17.539`), defeito comum em texto extraído de PDF. |
| **Coerência das cotas** | Um perfil de cota impossível. Foi essa conferência, e só ela, que pegou um defeito real: o número da página do PDF sendo lido no lugar de um campo do aluno. As outras não viam nada de errado — o valor era um número bem formado. |
| **Cruzamento entre editais** | O mesmo número de inscrição aparecendo em editais diferentes com nome diferente. Só 10 casos em cerca de 100 mil cruzamentos — e nenhum deles é defeito nosso: reextraindo o PDF, o texto bate letra por letra com o impresso. |

## Rastreabilidade e repetibilidade

Duas propriedades que a documentação de um fornecedor de dados deveria sempre declarar:

**Toda linha sabe de onde veio.** Cada registro carrega o arquivo de origem, o número do edital,
o triênio e a página. Qualquer número mostrado na tela pode ser rastreado até a página exata do
PDF oficial que o originou.

**Rodar duas vezes dá o mesmo resultado.** A extração é determinística — duas execuções sobre os
mesmos PDFs produzem arquivos idênticos byte a byte, verificado por comparação direta. Onde
haveria empate, existe uma regra fixa de desempate, nunca uma escolha arbitrária.

## Duas informações que recuperamos e não existem impressas

Além de ler o que está escrito, o pipeline **deduz** dois dados que o Cebraspe não publica:

- **Qual língua estrangeira cada aluno fez em cada etapa.** O edital não diz. Descobrimos testando
  qual das combinações possíveis faz a conta de conferência fechar. Cerca de 20% dos alunos trocam
  de língua entre uma etapa e outra — o que significa que tratar a língua como fixa por aluno,
  o caminho intuitivo, erraria em um quinto da base.
- **O perfil de cota declarado por cada aluno.** Também não é publicado diretamente, mas pode ser
  deduzido do padrão das dez classificações que o edital publica. Dos 512 padrões teoricamente
  possíveis, apenas 10 aparecem nos dados — exatamente os 10 que a legislação de cotas permite.
  O modelo reproduz a realidade sem sobra e sem falta.

## Uma correção que vale contar

Antes de os editais serem lidos, o sistema usava médias e desvios **estimados** a partir de uma
base de alunos, e não os oficiais. Ao comparar com os editais, 82 de 84 campos divergiam — e as
42 médias estavam **todas** acima do valor oficial, porque a base de origem só continha alunos
não eliminados.

O efeito disso era um erro de **0,56 a 1,28 ponto de Argumento Final** em toda previsão do
sistema, sempre na mesma direção. Depois da substituição pelos números oficiais, a mesma
ferramenta de comparação acusa **diferença zero nas 96 comparações**.

Contamos isso porque é o tipo de erro que passa despercebido para sempre em um produto que não
confere seus próprios dados contra a fonte.

---

**A seguir:** [Como sabemos que acerta](como-sabemos-que-acerta.md) — o que acontece depois de os
dados estarem limpos.
