# 📘 Developer & Agent Handbook: Vetor PAS

Este documento serve como o manual definitivo do sistema **Vetor PAS**. Ele foi projetado para dar a qualquer desenvolvedor ou agente de Inteligência Artificial um entendimento completo e profundo de todo o contexto técnico, de negócios e arquitetural do projeto.

---

## 1. Visão Geral do Produto e Modelo de Negócio

O **Vetor PAS** é uma plataforma de inteligência pedagógica B2B/B2C focada em prever o desempenho e calcular probabilidades de aprovação de estudantes no **PAS/UnB** (Programa de Avaliação Seriada da Universidade de Brasília). 

### O Problema
O PAS é um vestibular seriado em três etapas (PAS 1, PAS 2 e PAS 3) com um edital de cálculo de notas extremamente complexo e confuso (envolvendo notas padronizadas baseadas nas médias e desvios padrão de cada ano). Isso gera grande ansiedade nos alunos sobre quanto eles precisam tirar na última etapa para passar.

### A Solução
O Vetor PAS resolve essa dor usando modelos de Machine Learning (treinados em dados históricos reais de mais de 48 mil candidatos) para:
1.  **Predição de Desempenho:** Prever qual será a nota final do aluno no PAS 3 baseado nas notas que ele tirou no PAS 1 e 2.
2.  **Quanto Falta (Engenharia Reversa):** Calcular exatamente qual a nota mínima (Escore Bruto) que o aluno precisa tirar no PAS 3 para alcançar a nota de corte do curso desejado.
3.  **Gestão Pedagógica B2B:** Permitir que escolas parceiras façam o upload de turmas inteiras e visualizem o risco dos alunos através de um **Semáforo de Risco** (Verde = Baixo Risco, Amarelo = Médio Risco, Vermelho = Alto Risco).

### 📢 Construindo em Público (Build in Public)
O Vetor PAS está sendo desenvolvido sob a filosofia **Build in Public**. Isso significa que:
- O progresso do desenvolvimento, ideias de design, validações e desafios técnicos são compartilhados abertamente com a comunidade.
- A **Landing Page Temporária** com o formulário de lista de espera (waitlist) foi colocada em produção na `main` desde cedo para validar o interesse e coletar leads qualificados organicamente enquanto o MVP principal (na branch `feat/nextjs-frontend`) é construído.

---

## 2. Estratégia de Branches e Landing Pages

O ecossistema de branches do Git é dividido estrategicamente para gerenciar o pré-lançamento e o MVP:

1.  **`main` (Página de Espera - PRODUÇÃO)**:
    *   **Propósito**: Contém o site de pré-lançamento do projeto com o formulário de lista de espera (waitlist) e a história do fundador.
    *   **Status**: **100% Implementado e Implantado**. É a branch de produção oficial conectada à Vercel. Qualquer alteração aqui é implantada automaticamente na URL pública ([vetorpas.com.br](https://vetorpas.com.br)).
2.  **`feat/nextjs-frontend` (Painel e Landing Page Principal - DESENVOLVIMENTO)**:
    *   **Propósito**: Contém a landing page principal definitiva e a interface completa de dashboards do portal do aluno e da escola.
    *   **Status**: **Em desenvolvimento constante (Local)**. Esta branch ainda não foi mergeada para a `main`, portanto suas páginas não estão acessíveis em produção.
3.  **`feat/grill`**:
    *   **Propósito**: Branch secundária de testes e experimentações.

---

## 3. Arquitetura do Sistema e Status de Deploy

O sistema opera de forma desacoplada em três camadas. Abaixo está a especificação técnica e o status de deploy de cada uma:

```mermaid
graph TD
  User[Usuário / Coordenador / Aluno] -->|Acessa via Navegador| Next[Next.js Frontend - Vercel]
  Next -->|Autenticação e Relações do Perfil| Supa[(Supabase - PostgreSQL)]
  Next -->|Consome inferência e relatórios| Fast[FastAPI Backend - Hugging Face Spaces]
  Fast -->|Carrega modelos .joblib| ML[Motor de IA - pas_intelligence]
  Fast -->|Consulta Notas de Corte / RLS| Supa
  Fast -->|Gera documentos PDF| RL[ReportLab Generator]
  RL -->|Retorna PDF| User
```

### Camadas Tecnológicas e Status

1.  **Frontend (Diretório `landing-page/`)**:
    *   **Tecnologia**: **Next.js (App Router)** com React, TypeScript e TailwindCSS (v4).
    *   **Status de Deploy**: **Ativo em Produção na Vercel**. A branch `main` serve a Landing Page Temporária de espera. A branch `feat/nextjs-frontend` (portal completo) é executada apenas localmente (`http://localhost:3000`) por enquanto.
2.  **Backend API (Diretório `api/`)**:
    *   **Tecnologia**: **FastAPI** (Python).
    *   **Status de Deploy**: **Apenas Local (localhost:8000)**. A hospedagem no **Hugging Face Spaces** (via Docker) está planejada e decidida na arquitetura (ADR 0004), mas ainda não foi ativada em produção. A URL de produção da Vercel precisará apontar para o link do Space através da variável `API_URL` assim que o deploy for feito.
3.  **BaaS / Banco de Dados / Auth**:
    *   **Tecnologia**: **Supabase** (PostgreSQL).
    *   **Status de Deploy**: **Ativo e Conectado em Produção**. O banco de dados e o provedor de autenticação estão funcionando tanto localmente quanto na URL de produção.
4.  **Processador Analítico (Core - `src/pas_intelligence/` e `src/pdf_generator.py`)**:
    *   **Tecnologia**: Modelos serializados `.joblib` em Python executados com Scikit-Learn/LightGBM e geração de PDFs com ReportLab.
    *   **Status de Deploy**: **100% Implementado**. Integrado à API local do FastAPI. O deploy final em produção ocorrerá junto com a API do FastAPI no Hugging Face Spaces.
5.  **Dashboard Legado/Admin (`app/streamlit_app.py`)**:
    *   **Tecnologia**: **Streamlit** (Python).
    *   **Status de Deploy**: **Apenas Local**. Utilizado internamente para administração e testes rápidos de novas lógicas.


---

## 4. Estrutura de Diretórios e Código

A raiz do monorepo é estruturada da seguinte forma:

```
├── .streamlit/             # Configurações de tema e segredos do Streamlit
├── api/                    # Código do backend FastAPI
│   ├── main.py             # Entrada principal, CORS e inicialização do FastAPI
│   ├── routers/            # Endpoints REST (predict.py, analytics.py, gestao.py)
│   ├── schemas/            # Schemas de validação de dados Pydantic
│   └── services/           # Lógica que encapsula a chamada ao motor preditivo
├── app/
│   └── streamlit_app.py    # Dashboard Streamlit (legado/admin interno)
├── assets/
│   └── templates/          # Logotipos de escolas e templates de PDFs Whitelabel
├── docs/                   # Documentação detalhada em Markdown
│   ├── adr/                # Architectural Decision Records (ADRs 0001 a 0007)
│   ├── modules/            # Detalhamento de cada módulo do sistema
│   ├── architecture.md     # Visão geral da stack
│   ├── identidade-visual.md# Design tokens da UnB (HEX, Tailwind, config)
│   └── requirements.md     # Requisitos Funcionais e Não-Funcionais
├── landing-page/           # Frontend Next.js
│   ├── app/                # Rotas da aplicação (App Router)
│   │   ├── (public)/       # Rotas públicas (/predict, /temporal, landing page)
│   │   ├── auth/           # Login e cadastro do Supabase
│   │   └── (dashboard)/    # Rotas privadas /app/* (gestao, relatorios, escola)
│   ├── components/         # Componentes React (UI, brand, dashboard)
│   ├── lib/                # Utilitários, chamadas Supabase e tipos
│   └── README.md           # Guia de setup do frontend
├── models/                 # Modelos de Machine Learning (.joblib) - Gitignored
├── scripts/                # Scripts utilitários de importação de dados e mocks
├── src/                    # Código core do backend em Python
│   ├── pas_intelligence/   # O "cérebro" de Inteligência Artificial do sistema
│   ├── data_processing/    # Limpeza e preparação de dados
│   ├── extract_pas1_pdf.py # Parsers para ler PDFs de resultados do Cebraspe
│   ├── extract_pas2_html.py# Parsers para extrair notas de páginas HTML
│   └── pdf_generator.py    # Gerador de relatórios ReportLab PDF
├── tests/                  # Testes unitários do backend (Pytest)
├── CONTEXT.md              # Linguagem ubíqua e conceitos de produto
├── README.md               # Instruções gerais de inicialização do projeto
└── requirements.txt        # Dependências de bibliotecas Python
```

---

## 5. Regras de Negócio e Cálculos (O Edital Cebraspe)

As notas no PAS/UnB não são somas simples. O Cebraspe calcula a nota em relação à média e ao desvio padrão de cada prova no ano respectivo.

*   **Escore Bruto (EB):**
    O Escore Bruto de uma etapa é a soma ponderada das questões objetivas (dividida entre Parte 1 e Parte 2).
*   **Nota Padronizada (NP):**
    Calculada como:
    $$NP = 10 \times \frac{EB - \mu}{\sigma} + 50$$
    Onde $\mu$ é a média oficial e $\sigma$ é o desvio padrão da etapa naquele ano específico.
*   **Argumento Final (AF):**
    O Argumento Final é a pontuação acumulada ponderada das três etapas:
    $$AF = NP_1 \times 0.72 + NP_2 \times 8.28 + NP_3 \times 1.00$$
    *Nota: A redação das etapas 1 e 2 não tem peso direto, mas a redação do PAS 3 possui um cálculo integrado ao edital.*
*   **Engenharia Reversa ("Quanto Falta"):**
    O módulo `target_calculator.py` recebe a nota de corte alvo (Argumento Final) e os EBs conhecidos do aluno no PAS 1 e PAS 2. Ele resolve a equação reversamente para encontrar o $EB_3$ necessário. Como as médias e desvios oficiais do PAS 3 ainda não foram divulgados, o sistema projeta esses dados usando regressão linear histórica antes de calcular o alvo.

---

## 6. O Motor de IA (`src/pas_intelligence/`)

O coração preditivo do Vetor PAS é implementado através de um **Ensemble Dinâmico** que orquestra 4 algoritmos de Machine Learning:

### O Ensemble Dinâmico (`ensemble.py`)
Em vez de usar o mesmo modelo para todos, o sistema decide o peso de cada algoritmo baseado na **Volatilidade (Coeficiente de Variação - CV)** do histórico de Escores Brutos do aluno nas Etapas 1 e 2:
*   **Baixa Volatilidade (Aluno Estável)**: Se as notas do aluno são lineares e previsíveis, o modelo de **Regressão Linear** assume o maior peso.
*   **Alta Volatilidade (Aluno Errático)**: Se o histórico do aluno oscila bruscamente, os modelos baseados em árvores (**LightGBM** e **Random Forest**) e redes neurais (**MLP**) ganham pesos maiores devido à capacidade de capturar padrões não-lineares.
*   Essa ponderação dinâmica é ajustada via curva sigmoide.

### Modelos Serializados (`models/`)
Os modelos foram treinados com base em uma amostra real de **48.758 alunos** de 7 triênios históricos (2016 - 2024):
*   `modelo_lgbm.joblib` / `modelo_arg_final.joblib` — Modelos LightGBM campeões em acurácia para dados tabulares erráticos.
*   `modelo_rf.joblib` — Regressor de Floresta Aleatória.
*   `modelo_linear.joblib` — Regressor Linear simples e estável.
*   `modelo_mlp.joblib` — Rede neural Multi-Layer Perceptron (100, 50).
*   `meta_model.joblib` — Classificador RandomForest que ajuda a ponderar/escolher a melhor combinação de modelos base por aluno.
*   `scaler.joblib` / `meta_scaler.joblib` — Scalers StandardScaler para normalização de features.

---

## 7. Emissão de Relatórios Whitelabel (`src/pdf_generator.py`)

A entrega de valor tangível para as Escolas Parceiras é o Relatório Pedagógico Whitelabel individualizado em PDF.
*   O motor utiliza a biblioteca **ReportLab** em Python.
*   Ele renderiza vetores e textos sobre templates em formato PDF localizados na pasta `assets/templates/`.
*   O sistema substitui dinamicamente as paletas de cores do documento e injeta o logo correspondente à escola baseando-se nas configurações do `tenant` associadas ao coordenador requisitante.
*   Suporta **geração em lote**: processa uma turma inteira de alunos na memória, comprime os PDFs em um único arquivo `.zip` e entrega o link para download.

---

## 8. Banco de Dados (Supabase)

O Supabase gerencia a consistência de persistência do ecossistema:
*   `waitlist` — Tabela temporária para registrar leads interessados na lista de espera. Contém: `nome`, `email`, `escola`, `curso_pretendido`.
*   `profiles` — Tabela estendida do Supabase Auth contendo informações adicionais do usuário (`nome`, `escola_id`, `role` [aluno/coordenador]).
*   `tenants` — Tabela que gerencia as Escolas Parceiras ativas. Mapeia a ID da escola para os caminhos de marca (caminho_logo, paleta_cores).

---

## 9. Como Executar e Testar Localmente

### 1. Backend FastAPI (`api/`)
```bash
# Com o ambiente virtual Python ativo (.venv)
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```
Swagger UI disponível em `http://localhost:8000/docs`.

### 2. Frontend Next.js (`landing-page/`)
```bash
cd landing-page
npm install
# Crie e preencha o arquivo .env.local
npm run dev
```
Interface disponível em `http://localhost:3000`.

### 3. Dashboard Streamlit (`app/`)
```bash
# Na raiz do projeto com o .venv ativo
streamlit run app/streamlit_app.py
```

### 4. Executar Testes
```bash
pytest tests/
```
