# Análise Preditiva PAS

Sistema preditivo para estudantes e escolas parceiras do PAS/UnB. Calcula a probabilidade de aprovação de alunos em cursos da Universidade de Brasília com base no histórico de desempenho nas etapas do PAS.

## Language

### Participantes

**Aluno**:
Estudante do ensino médio inscrito no PAS/UnB. Pode usar as features públicas anonimamente ou criar conta (Aluno Cadastrado) para acessar o Painel Multi-Curso.
_Avoid_: usuário, estudante

**Aluno Cadastrado**:
Aluno que criou conta no sistema. Obrigatoriamente vinculado a uma escola no cadastro (campo com autocomplete). Tem acesso ao Painel Multi-Curso e ao "Quanto Falta".
_Avoid_: usuário logado, aluno autenticado

**Coordenador Pedagógico**:
Profissional de uma escola parceira que acessa as features B2B do sistema para acompanhar a turma.
_Avoid_: professor, usuário, admin

**Escola Parceira**:
Instituição de ensino contratante do serviço. Sinônimo de tenant no contexto técnico.
_Avoid_: cliente, escola contratante, organização

**Tenant**:
Identificador de uma Escola Parceira no sistema. Armazenado como campo `tenant` no perfil Supabase de cada Coordenador Pedagógico. Determina logo, template PDF e configurações visuais (whitelabel).
_Avoid_: escola, cliente, organização, domain_key

### Programa

**PAS**:
Programa de Avaliação Seriada da Universidade de Brasília. Processo seletivo em três etapas anuais (PAS 1, PAS 2, PAS 3) cursadas ao longo do ensino médio.

**Etapa**:
Uma das três provas anuais do PAS (1, 2 ou 3). Cada etapa tem peso diferente no cálculo do Argumento Final.
_Avoid_: fase, série, ano

**Triênio**:
Conjunto das três etapas do PAS cursadas por um mesmo grupo de alunos (ex: triênio 2023–2025).
_Avoid_: ciclo, turma

**Etapa Ausente**:
Etapa do PAS que o Aluno não realizou. O Edital de Resultado Final publica as três notas dessa Etapa como `0,000` — o que significa **ausência, não desempenho zero**. Só a Etapa 1 pode ser ausente em quem chega ao Resultado Final do PAS 3: faltar à Etapa 1 é permitido e o Aluno segue no programa, mas quem falta à Etapa 2 fica impedido de fazer a Etapa 3, e quem falta à Etapa 3 não aparece no Resultado Final.
_Avoid_: nota zero, etapa zerada, aluno faltante, etapa em branco

### Pontuação

**EB (Escore Bruto)**:
Nota bruta de uma etapa do PAS, calculada como soma das partes P1 e P2 da prova objetiva.
_Avoid_: nota, pontuação, score

**Argumento Final**:
Pontuação cumulativa ponderada das três etapas do PAS usada pelo UnB para classificação. Calculado com os pesos oficiais Cebraspe: P1 × 0,72 + P2 × 8,28 + Redação × 1,00.
_Avoid_: nota final, pontuação final, score final

**Nota de Corte**:
Argumento Final mínimo exigido para aprovação em um curso específico na última chamada do PAS. É um número **por curso e por Sistema de Concorrência**, não um por curso: quem concorre por cota disputa outro corte. "Última chamada" é a última em que aquele *sistema* teve convocado — um sistema pode parar na 1ª chamada enquanto o curso vai até a 3ª. Derivada em `src/pas_extraction/notas_corte.py` (ticket 10), cruzando o Resultado Final (nota) com a Convocação (quem foi chamado).
_Avoid_: mínimo, cutoff, nota de corte do curso (sem o sistema)

### Risco e predição

**Semáforo de Risco**:
Classificação visual do risco de reprovação de um Aluno em relação ao curso-alvo. Três estados: Baixo Risco (verde), Médio Risco (amarelo), Alto Risco (vermelho).
_Avoid_: status, classificação, cor

**Volatilidade (CV)**:
Coeficiente de Variação calculado sobre os EBs das etapas anteriores do Aluno (`std/mean × 100`). Drive principal na escolha do modelo de predição pelo ensemble. Mede **magnitude** de dispersão e é cega à **direção**: subir de 30 para 35 e cair de 35 para 30 produzem o mesmo CV. Não é sinônimo de Momentum.
_Avoid_: variação, instabilidade, desvio, momentum

**Momentum**:
Direção e tamanho da evolução do Aluno de uma Etapa para a seguinte. Hipótese central do produto: quem sobe muito da Etapa 1 para a Etapa 2 tende a ir bem na Etapa 3. É grandeza **com sinal**, ao contrário da Volatilidade (CV). Indefinido — não zero — para o Aluno sem Etapa 1.
_Avoid_: crescimento, variação, tendência, volatilidade

**Aluno sem Etapa 1**:
Aluno cuja Etapa 1 é uma Etapa Ausente: fez a Etapa 2, fará a Etapa 3, não fez a Etapa 1. É uma trajetória permitida pelo PAS e uma classe que o produto atende — não um cadastro incompleto. Para ele o Momentum é indefinido, e por isso a previsão do Argumento Final exige função própria; já o Quanto Falta é aritmética exata e vale sem alteração. Representa 8,7% do Resultado Final histórico.
_Avoid_: aluno que só fez o PAS 2, aluno incompleto, aluno com nota zero, outlier

### Produto

**Feature Pública**:
Funcionalidade acessível a qualquer visitante sem autenticação: Preditor PAS 3 (primeiro curso) e Análise Temporal.
_Avoid_: feature gratuita, feature aberta

**Feature B2B**:
Funcionalidade exclusiva para Coordenadores Pedagógicos autenticados: Gestão de Ativos, Análise da Escola vs. População, Comparação Entre Grupos, Gerador de PDFs.
_Avoid_: feature premium, feature paga, feature restrita

**Painel Multi-Curso**:
Feature exclusiva para Alunos Cadastrados. Permite salvar os inputs uma vez e ver probabilidade de aprovação + Quanto Falta para N cursos simultaneamente. Gatilho de login: tentativa de adicionar o segundo curso no Preditor.
_Avoid_: comparação de cursos, dashboard do aluno

**Quanto Falta**:
Cálculo reverso por curso: dado os EBs fixos do PAS 1 e PAS 2 do Aluno Cadastrado, qual EB mínimo ele precisa no PAS 3 para atingir a Nota de Corte. Calculado via `target_calculator`. Disponível para todos os cursos no Painel Multi-Curso.
_Avoid_: meta, objetivo, score necessário

**Soft Gate**:
Mecanismo de conversão do Preditor: o primeiro curso é calculado livremente; ao tentar adicionar um segundo curso, o sistema solicita login/cadastro. O Aluno já viu o valor antes de ser solicitado a criar conta.
_Avoid_: paywall, bloqueio, gate

**Dashboard de Prospecção**:
Área interna em `/admin`, protegida por variável de ambiente, acessível apenas pelo dono do produto. Mostra ranking de escolas por número de Alunos Cadastrados com filtro por período. Serve para identificar Escolas Prospecto.
_Avoid_: admin, painel interno, analytics

**Escola Prospecto**:
Escola não-contratante que aparece no ranking da Dashboard de Prospecção com volume relevante de Alunos Cadastrados. Sinal de demanda orgânica para abordagem comercial.
_Avoid_: lead, escola potencial, prospect

**Whitelabel**:
Personalização visual do sistema por Tenant: logo, template PDF e cores. Configurado via campo `tenant` no perfil Supabase.
_Avoid_: customização, branding, tema

### Pré-Lançamento & Estratégia de Landing Pages

**Estratégia de Landing Pages & Branches**:
O projeto gerencia o ciclo de pré-lançamento e o MVP através de duas branches principais de frontend:
- **`feature/landing-page-temporaria` (Temporária)**: Contém a landing page de pré-lançamento e captura de leads. É a versão ativa na URL de produção (`vetorpas.com.br`) temporariamente.
- **`feat/nextjs-frontend` (Principal)**: Contém a landing page definitiva e os dashboards do portal do MVP. Em desenvolvimento constante; substituirá a versão temporária quando finalizada.
- **`main`**: A landing page atualmente presente nesta branch é obsoleta e não será utilizada.

**Lista de Espera (Waitlist)**:
Formulário de captura de interesse antecedendo o lançamento do MVP do Vetor PAS. Coleta Alunos interessados para acesso antecipado.
_Avoid_: pré-cadastro, newsletter

**Lead da Lista de Espera**:
Registro de um Aluno interessado contendo Nome, E-mail, Escola e Curso Pretendido na UnB. Armazenado na tabela `waitlist` do Supabase.
_Avoid_: contato, inscrito

**História do Fundador**:
Seção de conexão institucional no pré-lançamento relatando a experiência pessoal do fundador como ex-estudante do PAS 3 e a motivação para criar a solução.
_Avoid_: sobre nós, quem somos

**Mobile-First Design**:
Diretriz de design para a Landing Page Temporária: foco prioritário na experiência em smartphones (layouts empilhados, touch targets otimizados, formulário responsivo rápido), considerando que a maioria dos estudantes do PAS 3 acessará via celular (Instagram/WhatsApp).
_Avoid_: versão mobile simples, adaptação mobile
