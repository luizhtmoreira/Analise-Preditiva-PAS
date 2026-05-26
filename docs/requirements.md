# Requisitos do Sistema

Esta seção detalha as especificações do **Vetor PAS** através de Requisitos Funcionais (RF) e Requisitos Não-Funcionais (RNF).

## Requisitos Funcionais (RF)

Os Requisitos Funcionais descrevem **o que** o sistema deve fazer. Eles representam as funcionalidades que agregam valor direto ao usuário final (coordenadores pedagógicos e escolas parceiras).

| ID | Descrição | Prioridade |
|---|---|---|
| **RF01** | O sistema deve possuir uma tela de login seguro para autenticação de coordenadores e diretores. | Alta |
| **RF02** | O sistema deve adaptar dinamicamente o logotipo e a paleta de cores de acordo com a escola autenticada (arquitetura *Multi-tenant / Whitelabel*). | Alta |
| **RF03** | O sistema deve permitir o *upload* (importação) de dados de turmas de alunos via arquivos tabulares (.csv ou .xlsx). | Alta |
| **RF04** | O sistema deve classificar os alunos graficamente em um **Semáforo de Risco** (Verde, Amarelo ou Vermelho) baseado em sua proximidade do curso-alvo. | Alta |
| **RF05** | O sistema deve prever a pontuação bruta final do aluno no PAS 3 utilizando os históricos das etapas passadas (PAS 1 e PAS 2). | Alta |
| **RF06** | O sistema deve possuir uma Calculadora de Metas que realize engenharia reversa para indicar qual a nota mínima exata que o aluno precisa tirar no PAS 3. | Alta |
| **RF07** | O sistema deve apresentar a probabilidade matemática (chance em porcentagem) do aluno ser aprovado no curso e cota desejados. | Média |
| **RF08** | O sistema deve permitir testes A/B ou comparação estatística de desempenho médio entre turmas diferentes. | Média |
| **RF09** | O sistema deve gerar relatórios consolidados em formato PDF com a marca da escola. | Alta |
| **RF10** | O sistema deve permitir a geração e download em lote (ZIP) de múltiplos PDFs de forma automatizada para uma turma inteira. | Alta |
| **RF11** | O sistema deve disponibilizar um histórico interativo das notas de corte dos últimos triênios para consulta. | Baixa |

## Requisitos Não Funcionais (RNF)

Os Requisitos Não-Funcionais descrevem **como** o sistema deve fazer. Eles representam restrições, atributos de qualidade, arquitetura e tecnologias adotadas.

| ID | Categoria | Descrição |
|---|---|---|
| **RNF01** | **Tecnologia** | O *backend* do sistema deve ser escrito na linguagem Python (versão 3.10 ou superior). |
| **RNF02** | **Interface UI** | A camada de visualização e interface (*Dashboard*) deve ser totalmente renderizada utilizando o framework `Streamlit`. |
| **RNF03** | **Persistência** | O armazenamento de entidades, históricos de aprovação e autenticação de usuários deve ser feito no BaaS `Supabase` (PostgreSQL). |
| **RNF04** | **Inteligência** | A estimativa de notas deve utilizar um modelo preditivo baseado em um *Ensemble Dinâmico* (englobando Regressão Linear e LightGBM). |
| **RNF05** | **Performance** | O tempo de carregamento inicial do dashboard após o login não deve ultrapassar 3 segundos em conexões padrão. |
| **RNF06** | **Performance** | O cálculo do *Ensemble Dinâmico* para uma turma importada de 50 alunos deve ser processado em menos de 5 segundos. |
| **RNF07** | **Segurança** | Os dados sensíveis dos alunos (histórico escolar) de uma escola jamais devem estar acessíveis a contas de escolas diferentes (isolamento rígido de inquilinos / Row Level Security). |
| **RNF08** | **Documentação** | Os algoritmos de conversão do Edital Cebraspe (`argument_calculator.py`) devem estar cobertos por testes unitários (`pytest`). |
