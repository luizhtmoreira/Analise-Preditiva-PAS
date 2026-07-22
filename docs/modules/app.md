# Interface Web & Portal (Next.js)

O diretório `landing-page/` abriga a interface de usuário principal do **Vetor PAS**, desenvolvida usando o framework **Next.js (App Router)**. Ela unifica o site institucional (landing page pública) e os portais autenticados para Alunos e Escolas.

## 🏗️ Estrutura de Rotas (App Router)

As rotas são organizadas de acordo com as seguintes regras de visibilidade:

### 1. Rotas Públicas
*   `/` — Landing page de marketing principal com foco mobile-first e formulário de lista de espera (Waitlist).
*   `/predict` — Calculadora interativa de predição do PAS 3 para o primeiro curso.
*   `/temporal` — Análise estatística de notas de corte e oscilações ao longo de triênios passados.
*   `/auth/login` e `/auth/cadastro` — Telas de autenticação integradas com o Supabase.

### 2. Rotas Protegidas (Alunos & Escolas)
*   `/app/layout.tsx` — Layout compartilhado para as áreas autenticadas contendo o menu lateral (`Sidebar.tsx`) e controle de tenant.
*   `/app/relatorios` — Interface para emissão, busca e download de relatórios individualizados (PDFs).
*   `/app/escola` — Análise de desempenho agregado de turmas vs. população total de candidatos.
*   `/app/gestao` — Painel pedagógico contendo o Semáforo de Risco, lista de alunos e percentuais de aprovação.
*   `/app/comparacao` — Módulo para comparar médias estatísticas de turmas diferentes.

---

## 🔐 Autenticação e Whitelabel (Multi-Tenant)

O fluxo de personalização de marca (*tenant mapping*) no frontend funciona da seguinte forma:

1. O usuário (aluno ou coordenador) faz o login e uma sessão segura é estabelecida através do **Supabase Auth**.
2. A aplicação recupera as informações do perfil do usuário contendo o identificador `tenant` (escola vinculada).
3. Com base no identificador de tenant, a aplicação altera dinamicamente os elementos visuais da interface (cores de botões, barras laterais e cabeçalho, logotipos e referências da instituição).
4. Essa configuração do tenant é compartilhada com o backend FastAPI no momento em que as requisições de PDFs whitelabel são solicitadas.

---

## 🛠️ Conectividade com o Backend (FastAPI)

Toda a lógica de inferência pesada, regressão linear, LightGBM e geração física de PDFs ReportLab não roda diretamente no Next.js:
- O Next.js consome endpoints REST expostos pela API FastAPI (hospedada no Hugging Face).
- Os endpoints chamados incluem `/predict`, `/analytics` e `/gestao`.
- As credenciais de conexão e URLs da API são injetadas em tempo de build/execução via variáveis `.env.local`.

---

## 📊 Streamlit Dashboard (Legado/Admin)
O dashboard Streamlit legado em `app/streamlit_app.py` é mantido para fins de prototipagem rápida e visualizações internas rápidas pelo administrador da plataforma, mas a interface de produção voltada para as escolas contratantes é o Next.js.
