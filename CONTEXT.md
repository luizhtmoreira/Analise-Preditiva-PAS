# Análise Preditiva PAS

Sistema preditivo para estudantes e escolas parceiras do PAS/UnB. Calcula a probabilidade de aprovação de alunos em cursos da Universidade de Brasília com base no histórico de desempenho nas etapas do PAS.

## Language

### Participantes

**Aluno**:
Estudante do ensino médio inscrito no PAS/UnB. Pode usar as features públicas anonimamente ou criar conta para manter histórico.
_Avoid_: usuário, estudante

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

### Pontuação

**EB (Escore Bruto)**:
Nota bruta de uma etapa do PAS, calculada como soma das partes P1 e P2 da prova objetiva.
_Avoid_: nota, pontuação, score

**Argumento Final**:
Pontuação cumulativa ponderada das três etapas do PAS usada pelo UnB para classificação. Calculado com os pesos oficiais Cebraspe: P1 × 0,72 + P2 × 8,28 + Redação × 1,00.
_Avoid_: nota final, pontuação final, score final

**Nota de Corte**:
Argumento Final mínimo exigido para aprovação em um curso específico na última chamada do PAS.
_Avoid_: mínimo, cutoff

### Risco e predição

**Semáforo de Risco**:
Classificação visual do risco de reprovação de um Aluno em relação ao curso-alvo. Três estados: Baixo Risco (verde), Médio Risco (amarelo), Alto Risco (vermelho).
_Avoid_: status, classificação, cor

**Volatilidade (CV)**:
Coeficiente de Variação calculado sobre os EBs das etapas anteriores do Aluno (`std/mean × 100`). Drive principal na escolha do modelo de predição pelo ensemble.
_Avoid_: variação, instabilidade, desvio

### Produto

**Feature Pública**:
Funcionalidade acessível a qualquer visitante sem autenticação: Preditor PAS 3 e Análise Temporal. Alunos podem fazer login opcionalmente para salvar histórico.
_Avoid_: feature gratuita, feature aberta

**Feature B2B**:
Funcionalidade exclusiva para Coordenadores Pedagógicos autenticados: Gestão de Ativos, Análise da Escola vs. População, Comparação Entre Grupos, Gerador de PDFs.
_Avoid_: feature premium, feature paga, feature restrita

**Whitelabel**:
Personalização visual do sistema por Tenant: logo, template PDF e cores. Configurado via campo `tenant` no perfil Supabase.
_Avoid_: customização, branding, tema
