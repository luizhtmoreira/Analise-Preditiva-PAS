# O que a escola recebe

O Vetor PAS tem duas faces. Uma é **aberta ao aluno**, sem login, e existe porque é assim que a
ferramenta chega até ele. A outra é **da coordenação**, e é o que a escola contrata.

## Para a coordenação pedagógica

### Semáforo de risco da turma

A lista dos alunos da escola classificada por risco de não passar no curso pretendido: verde,
amarelo, vermelho. A coordenação carrega as notas da turma e recebe de volta o quadro completo,
em vez de olhar aluno por aluno.

Existe um quarto estado, cinza — **"sem previsão"**. Ele aparece quando falta dado para responder
com honestidade sobre aquele aluno. A alternativa seria empurrá-lo para "alto risco", o que seria
uma afirmação que ninguém mediu sobre uma pessoa real.

### A escola contra a população inteira

A turma da escola comparada com **todos os candidatos do PAS** daquele triênio, não com uma
amostra. Como a base vem dos editais oficiais, a população de referência é literalmente todo mundo
que fez a prova.

É a resposta para "a minha turma está indo bem?" com um referencial externo, em vez do referencial
interno da própria escola de anos anteriores.

### Comparação entre grupos

Duas turmas, dois turnos, duas metodologias — comparadas estatisticamente. Serve para saber se uma
diferença observada entre grupos é real ou é variação normal.

### Relatórios com a marca da escola

Relatórios individuais em PDF, com o logotipo e a identidade visual da instituição, para entregar
ao aluno e à família. O aluno recebe um documento da escola dele, não de um fornecedor.

!!! warning "Estado atual"
    A geração de relatórios funciona hoje na ferramenta interna e é operada por nós. A migração
    para o portal, com emissão em lote pela própria coordenação, ainda não está concluída.

## Para o aluno

Estas funcionalidades são abertas e gratuitas — inclusive para alunos de escolas que não são
clientes. Isso é intencional: é o canal pelo qual a ferramenta chega às escolas.

**Calculadora de previsão.** O aluno informa as notas que já tirou e o curso que quer, e recebe a
estimativa do Argumento Final, a chance de aprovação e a distância até a nota de corte.

**Quanto falta.** O cálculo inverso: dada a nota de corte do curso, qual a nota mínima que ele
precisa tirar na última etapa. É a informação mais acionável do produto, porque continua exata
mesmo quando a previsão é incerta.

**Painel multi-curso.** Para o aluno cadastrado: salvar as notas uma vez e acompanhar vários
cursos ao mesmo tempo, em vez de refazer a conta a cada curso.

**Análise histórica de notas de corte.** A evolução real dos cortes por curso ao longo dos anos.
Não é uma projeção de tendência: são os cinco anos mais recentes, cada um com a nota de corte
daquele ano amarrada às estatísticas de dificuldade da prova daquele mesmo ano. A variação entre
eles **é** a incerteza sobre o ano que vem, mostrada em vez de estimada.

!!! info "Nota de corte"
    O menor Argumento Final que garantiu vaga em um curso na última chamada. Não é um número por
    curso: é um número **por curso, por sistema de concorrência e por semestre de entrada**. Quem
    concorre por cota disputa outro corte, e o segundo semestre sempre corta mais baixo que o
    primeiro.

### Os dez sistemas de concorrência, não só o universal

O aluno se declara em qualquer um dos **dez sistemas de concorrência do edital do PAS** —
universal, cota para negros, e as oito combinações de escola pública com renda, cor e deficiência
— e a conta é feita contra o corte **daquele** sistema.

Isso é menos comum do que parece. Ferramentas que tratam "a nota de corte do curso" como um número
único estão, na prática, respondendo sempre pelo sistema universal, que é o corte mais alto. Para
um aluno cotista, isso significa uma resposta pessimista demais sobre a própria chance — e para a
coordenação, um aluno classificado como risco alto sem motivo.

A rigor, o aluno concorre em **tudo ao mesmo tempo**: no universal, no sistema da cota dele, no
primeiro e no segundo semestre. Ele entra se limpar qualquer um deles, e por isso o limiar que
importa é o **menor** entre todos — nunca um corte escolhido.

## Por que a parte gratuita é generosa

Vale explicar, porque a pergunta aparece. O aluno tem acesso à previsão sem pagar e sem que a
escola dele seja cliente. A ferramenta chega até ele por conta própria, e chega junto com o nome
da escola quando a escola é parceira.

Para a escola contratante, o que muda não é o aluno ter acesso — é a **coordenação** passar a
enxergar a turma inteira de uma vez, comparar com a população real e entregar o documento com a
marca da instituição.

---

**A seguir:** [Privacidade e dados dos alunos](privacidade.md) — a página que deveria ser lida
antes de qualquer decisão de contratação.
