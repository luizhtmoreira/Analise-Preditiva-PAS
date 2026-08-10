# Privacidade e dados dos alunos

O aluno do PAS é, na maioria, menor de idade, e o dado que tratamos é desempenho escolar
identificável. Isso coloca este assunto acima de qualquer funcionalidade: uma escola não deveria
contratar um fornecedor de análise preditiva sem ler esta página.

Ela descreve o que o sistema faz hoje, não o que pretendemos fazer.

## O princípio

> O dado não entra — em vez de entrar e a gente lembrar de não olhar.

Essa frase é a regra de arquitetura do projeto, e ela tem uma consequência concreta: as proteções
não são de acesso, são de **fronteira**. Não adianta ter permissão bem configurada em cima de um
arquivo que não deveria estar ali. Onde possível, o dado identificável simplesmente não chega ao
lugar.

## O que sai do nosso disco

Os arquivos completos, com nome de aluno, ficam em armazenamento privado e **nunca são embarcados
no sistema nem publicados**.

O que vai para o servidor que atende as requisições é uma versão reduzida — internamente chamada
de **Derivado de Deploy** — que contém apenas as colunas que a aplicação de fato lê, e **nenhuma
coluna que identifique um aluno pelo nome**. A redução é grande o bastante para ser visível:
de 24,2 MB para 4,5 MB.

!!! success "Isso é verificado por teste automatizado, não por procedimento"
    Existe um teste que **abre o arquivo gerado** e confirma que a coluna de nome não está lá.
    Ele verifica o arquivo escrito, não a intenção do script que escreveu — a distinção importa,
    porque é exatamente assim que esse tipo de proteção costuma falhar silenciosamente.

    A lista de colunas permitidas tem dono único no código, lida tanto por quem publica quanto
    por quem consome. Não existem duas listas para manter em sincronia.

## O que o sistema lê em funcionamento

Nome e número de inscrição **não são lidos**. As colunas são excluídas antes de qualquer linha
entrar na memória do servidor — não é um filtro aplicado depois, é uma leitura que nunca traz o
campo.

O critério de aceite verificado na publicação foi literal: **nenhum nome e nenhum número de
inscrição de aluno é servido por nenhum endereço público do sistema.** As poucas telas públicas
que exibem casos reais usam identificações genéricas (*Aluno A*, *Aluno B*), e a calculadora
pública só mostra de volta o que o próprio visitante digitou.

## O treinamento do modelo

Nenhum dado de aluno sai da máquina onde o modelo é treinado. O arquivo de identificação que
acompanha cada modelo treinado registra a **impressão digital criptográfica** do conjunto de
dados usado — um código de 64 caracteres que permite provar que um determinado modelo foi
treinado sobre um determinado arquivo, sem carregar nenhum conteúdo desse arquivo.

No conjunto de treino, nome e inscrição não existem. Sobrevive apenas um identificador
pseudonimizado, usado para reconhecer que duas linhas são da mesma pessoa em triênios diferentes.

## Testes e exemplos

Nenhuma linha de aluno real entra em teste, exemplo ou material de demonstração.

Os testes que precisam de um edital em PDF **não recortam um edital verdadeiro** — eles geram um
PDF novo, com candidatos inventados e notas inventadas, na hora de rodar. A razão é direta: um
edital real lista nota por candidato, e recortar um pedaço dele para usar como teste embutiria
dado de aluno de verdade dentro do código-fonte, para sempre.

## O código que lê os editais fica fora do controle de versão

O módulo que processa os PDFs oficiais é mantido fora do repositório de código, porque a pasta em
que ele trabalha contém editais e planilhas com dados de alunos reais. É uma decisão que custa
comodidade de desenvolvimento e compra uma garantia: esse material não tem como ser publicado por
descuido junto com o código.

Pelo mesmo motivo, os modelos treinados, as bases de dados e os modelos de relatório das escolas
são todos mantidos fora do repositório e explicitamente excluídos das imagens do servidor.

## Separação entre escolas

Os dados de uma escola nunca são acessíveis a contas de outra escola. O isolamento é feito no
próprio banco de dados, por regras de acesso por linha — a política é aplicada pelo banco, não
pela aplicação, de modo que uma falha de programação na aplicação não expõe dados de outro
cliente.

## Sobre a LGPD

Duas posições registradas, porque uma escola tem o direito de saber que elas foram pensadas antes
e não depois:

**Finalidade declarada antes da coleta.** Registrar a trajetória de notas de uma população
majoritariamente menor de idade não é impedimento — é o próprio serviço — mas a finalidade
precisa constar da política de privacidade **antes** de a base encher, e não depois. Corrigir
retroativamente uma base já cheia é o caso caro.

**Pedido de exclusão é uma operação, não várias.** O e-mail do aluno não é copiado para tabelas
secundárias; ele vive em um único lugar, ligado por referência. Isso existe para que apagar os
dados de um aluno seja uma única ação, sem cópias espalhadas que alguém precise lembrar de
apagar junto.

## O que ainda não está fechado

- O registro de consultas feitas por visitantes **não cadastrados** é uma decisão em aberto, e
  está tratada como o que é: coleta de comportamento sem cadastro, que exige base legal própria.
- A pseudonimização usada no conjunto de treino é uma função criptográfica sem sal, o que a torna
  teoricamente reversível por força bruta para quem já tivesse a lista de números de inscrição.
  Ela protege contra leitura casual, não contra um atacante determinado que já tenha a base.
