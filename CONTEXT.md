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

**Semestre de Ingresso**:
1º ou 2º semestre letivo em que a vaga é oferecida. **Não é uma escolha do Aluno e não é uma prova separada**: todos fazem a mesma prova e concorrem de uma vez, e existe um limite de vagas para o 1º semestre. Quem não se classifica dentro dele **continua vivo** disputando as vagas do 2º. Por isso a Nota de Corte do 2º semestre é sempre a mais baixa das duas — conferido em **1.317 de 1.317** chaves curso+sistema que oferecem os dois, com o corte do 1º acima do 2º por 31,5 pontos de Argumento Final na mediana.
Consequência para qualquer cálculo de aprovação: o Aluno concorre em **tudo** ao mesmo tempo — Universal, o Sistema de Concorrência da cota dele, 1º e 2º semestre — e entra se limpar qualquer um. O limiar dele é o **menor corte entre todos eles**, nunca um corte escolhido.
_Avoid_: semestre escolhido, opção de semestre, segunda chamada, turno

**Etapa Ausente**:
Etapa do PAS que o Aluno não realizou. O Edital de Resultado Final publica as três notas dessa Etapa como `0,000` — o que significa **ausência, não desempenho zero**. Só a Etapa 1 pode ser ausente em quem chega ao Resultado Final do PAS 3: faltar à Etapa 1 é permitido e o Aluno segue no programa, mas quem falta à Etapa 2 fica impedido de fazer a Etapa 3, e quem falta à Etapa 3 não aparece no Resultado Final.
_Avoid_: nota zero, etapa zerada, aluno faltante, etapa em branco

### Pontuação

**EB (Escore Bruto)**:
Nota bruta de uma etapa do PAS, calculada como soma das partes P1 e P2 da prova objetiva.
_Avoid_: nota, pontuação, score

**Argumento de Etapa (A1, A2, A3)**:
Pontuação padronizada de **uma** Etapa: a soma dos z-scores das três partes multiplicados pelos pesos oficiais — `[(P1−média)/desvio]×0,72 + [(P2−média)/desvio]×8,28 + [(Redação−média)/desvio]×1,00`. Por ser feito de z-scores, **já nasce descontado da dificuldade da prova daquele ano** — ao contrário do EB. Para o Aluno que já fez PAS 1 e PAS 2, `A1` e `A2` são **aritmética exata**, nunca previsão; só `A3` é desconhecido.
_Avoid_: argumento parcial, nota padronizada, score da etapa

**Argumento Final**:
Pontuação cumulativa ponderada das três etapas do PAS usada pelo UnB para classificação. Vale `1×A1 + 2×A2 + 3×A3` sobre os Argumentos de Etapa. É **estável entre triênios** por construção (média ~3–5, desvio ~50 nos 8 triênios medidos), porque a padronização absorve a diferença de dificuldade das provas.
_Avoid_: nota final, pontuação final, score final

**Nota de Corte**:
Argumento Final mínimo exigido para aprovação em um curso específico na última chamada do PAS. É um número **por curso, por Sistema de Concorrência e por Semestre de Ingresso** — nunca um por curso: quem concorre por cota disputa outro corte, e o 1º e o 2º semestre têm cortes diferentes. "Última chamada" é a última em que aquele *sistema* teve convocado — um sistema pode parar na 1ª chamada enquanto o curso vai até a 3ª. Derivada em `src/pas_extraction/notas_corte.py` (ticket 10), cruzando o Resultado Final (nota) com a Convocação (quem foi chamado).
_Avoid_: mínimo, cutoff, nota de corte do curso (sem o sistema e sem o semestre)


### Risco e predição

**Ano-Âncora**:
Ano real e já publicado usado como cenário para responder "e se a minha Etapa 3 for como aquela?". Um Ano-Âncora amarra **junto** a Nota de Corte daquele ano (concorrência) e as estatísticas da prova daquele ano (dificuldade) — nunca uma combinação que não aconteceu. O produto mostra os cinco mais recentes, com o último em destaque. Substitui a projeção do ano futuro por regressão: a diferença entre os Anos-Âncora **é** a incerteza de dificuldade, mostrada em vez de estimada.
_Avoid_: cenário, projeção, tendência, ano de referência

**Semáforo de Risco**:
Classificação visual do risco de reprovação de um Aluno em relação ao curso-alvo. Três estados: Baixo Risco (verde), Médio Risco (amarelo), Alto Risco (vermelho).
_Avoid_: status, classificação, cor

**Alvo Canônico**:
A única grandeza que o modelo prevê e da qual **todo** número mostrado ao Aluno é derivado. É o `A3` — o Argumento da Etapa 3. Argumento Final, EB e escore necessário saem dele por aritmética, e por isso não podem se contradizer na tela.
_Avoid_: target, variável-resposta, output do modelo

**Estimador Auxiliar**:
Regra ou modelo que estima P1 e Redação da Etapa 3 com o único fim de **repartir** o Alvo Canônico entre as três partes da prova, para que o resultado possa ser falado em escore em vez de em desvio-padrão. Não é fonte de verdade: trocá-lo não muda o Argumento previsto nem a probabilidade, só a apresentação. Pode ser sobrescrito pelo próprio Aluno.
_Avoid_: modelo de P1, modelo de redação, submodelo, previsão de nota

**Largura de Incerteza**:
O quanto o modelo costuma errar, usado como a dispersão da conta de probabilidade de aprovação (`X ~ N(previsão, σ²)`). É **um número por classe de Aluno** — `14,965` para quem fez a Etapa 1 e `15,475` para o **Aluno sem Etapa 1**, em pontos de Argumento Final — e **não varia por Aluno**: a largura por Aluno foi medida e desloca a probabilidade em no máximo 3 pontos percentuais (ADR-0012). Vive no manifesto do pacote de modelo, nunca no código, e é medida em `A3`, valendo `3×` em Argumento Final. Não é o `RMSE = 13,49` de `statistics.py`, que era um MAE de um modelo aposentado. Os valores acima são os do pacote promovido em 2026-07-28 e **mudam a cada retreino** — leia sempre o manifesto, nunca este parágrafo, se o número importa para uma conta.
_Avoid_: RMSE, margem de erro, intervalo de confiança, desvio do modelo

**Edital isolado de Etapa**:
O "Resultado final nos itens do tipo D e na prova de redação" de uma Etapa: lista a nota de cada candidato e **não diz a língua estrangeira de ninguém**. É a única fonte disponível para as Etapas da Turma viva, e é diferente do **Edital de médias e desvios**, que publica média e desvio já separados por língua. Dele só sai a Parte 1 Misturada.
_Avoid_: edital de notas, edital parcial, resultado por etapa, edital

**Parte 1 Misturada**:
Média e desvio da Parte 1 de uma Etapa calculados sobre as três línguas estrangeiras juntas, porque a fonte não diz quem fez qual. É uma **forma do dado**, marcada como tal e distinguível da forma por língua pelo tipo, nunca por contagem de chaves. Preencher as três a partir dessa fonte exigiria inventar valores. O custo está medido: **0,46 ponto de Argumento Final em média, máximo 3,21, com viés zero** — ruído, não erro sistemático (ADR-0013).
_Avoid_: parte 1 agrupada, média geral da parte 1, m_p1, parte 1 sem língua

**Procedência**:
De onde veio a média e o desvio de uma `(Ano, Etapa)`: **Edital** (publicado pelo Cebraspe) ou **Derivada** (inferida enquanto o Edital não sai). Fica registrada no próprio dado porque, quando o Edital de verdade sair, os números derivados serão substituídos e as previsões de Alunos reais vão mexer. É eixo independente da Parte 1 Misturada: o Edital isolado de Etapa é um Edital.
_Avoid_: origem do dado, fonte, estimado vs oficial, provisório

**Volatilidade**:
Dispersão **absoluta** entre os Argumentos de Etapa já realizados do Aluno (`|A2 − A1|`), em pontos de Argumento. Mede **magnitude** e é cega à **direção**: subir 3 pontos e cair 3 pontos produzem a mesma Volatilidade. Não é sinônimo de Momentum. **Não é mais um Coeficiente de Variação**: dividir pela média — o que o CV fazia para comparar Alunos de níveis diferentes — é ao mesmo tempo impossível e desnecessário na escala de Argumento. Impossível porque a média do par é ~0 (mediana 0,12) e negativa em 49,3% da base, o que faz a divisão disparar e trocar de sinal; desnecessário porque o Argumento **já** é medido em desvios-padrão da turma, então a comparabilidade entre níveis que a divisão buscava já vem pronta.
_Avoid_: CV, coeficiente de variação, variação, instabilidade, momentum

**Momentum**:
Direção e tamanho da evolução do Aluno de uma Etapa para a seguinte, medido em **Argumento de Etapa** (`A2 − A1`), nunca em EB. Hipótese central do produto: quem sobe muito da Etapa 1 para a Etapa 2 tende a ir bem na Etapa 3. É grandeza **com sinal**, ao contrário da Volatilidade. Indefinido — não zero — para o Aluno sem Etapa 1. Medi-lo em EB confunde "o Aluno evoluiu" com "a prova ficou mais fácil": nos 60.013 Alunos da população limpa, EB e Argumento **discordam sobre o sinal em 17,2% dos casos**, chegando a 39,4% no triênio 2022/2024, cuja Etapa 2 foi muito mais fácil.
_Avoid_: crescimento, variação, tendência, volatilidade, delta de EB

**Aluno sem Etapa 1**:
Aluno cuja Etapa 1 é uma Etapa Ausente: fez a Etapa 2, fará a Etapa 3, não fez a Etapa 1. É uma trajetória permitida pelo PAS e uma classe que o produto atende — não um cadastro incompleto. Para ele o Momentum é indefinido, e por isso a previsão do Argumento Final exige função própria; já o Quanto Falta é aritmética exata e vale sem alteração. Representa 8,7% do Resultado Final histórico.
_Avoid_: aluno que só fez o PAS 2, aluno incompleto, aluno com nota zero, outlier

**Aluno Repetente**:
Aluno que cursa o PAS mais de uma vez, refazendo as três Etapas num Triênio posterior e
concorrendo de novo — frequentemente ao mesmo curso. É uma trajetória permitida pelo programa e
uma classe que o produto atende, **não uma duplicata de registro**: as duas passagens são pessoas
iguais em provas diferentes, com notas próprias. Representa 0,46% do Resultado Final histórico
(144 Alunos), concentrado nos Triênios recentes.
_Avoid_: duplicata, aluno duplicado, linha repetida, inscrição repetida, outlier

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

**Simulador de Itens**:
Sub-ferramenta da Calculadora de Estratégia/Quanto Falta. Converte o EB necessário ($X$) no PAS 3 em combinações exatas ou simuladas de acertos por tipo de item (Tipo A, Tipo B, Tipo C). Protegida por Soft Gate (exibe preview em teaser para não-cadastrados e exige login para interação completa).
_Avoid_: simulador de questões, simulador de acertos

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

**Tela de Perfil (`/perfil`)**:
Área exclusiva do Aluno Cadastrado para visualização de dados da conta (e-mail, escola vinculada), atualização da escola e acionamento de logout.
_Avoid_: página do usuário, minha conta, configurações de usuário

### Pré-Lançamento & Estratégia de Landing Pages

**Estratégia de Landing Pages & Branches**:
O projeto gerencia o ciclo de pré-lançamento e o MVP através de duas branches principais de frontend:
- **`feature/landing-page-temporaria` (Temporária)**: Contém a landing page de pré-lançamento e captura de leads. É a versão ativa na URL de produção (`vetorpas.com.br`) temporariamente.
- **`feat/nextjs-frontend` (Portal, incorporada)**: trazia a landing definitiva, o Preditor com semestre/curso alvo, a Calculadora de Estratégia, o header público, a recuperação de senha e a tela de perfil. Foi integrada em `feat/pdf-extraction` (ticket 10 da rodada *Publicar o Site*); o tronco unificado é `feat/pdf-extraction`.
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
