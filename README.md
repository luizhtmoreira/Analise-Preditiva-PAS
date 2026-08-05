# 📈 Análise Preditiva PAS — Vetor PAS

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge&logo=lightgbm&logoColor=white)](#)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Produção-success?style=for-the-badge)](#)

*Plataforma de inteligência pedagógica e predição de resultados para o Programa de Avaliação Seriada (PAS) da Universidade de Brasília.*
</div>

---

## 🎯 Visão do Produto

O **Análise Preditiva PAS** é uma solução multi-tenant (whitelabel) desenvolvida para capacitar coordenadores pedagógicos de escolas preparatórias com **inteligência de dados**. 

Ao invés de intuição, a plataforma utiliza modelos de *Machine Learning* treinados com dados históricos reais para responder a duas perguntas fundamentais na jornada do aluno:
1. **Predição de Desempenho:** *"Baseado no histórico, qual será a nota final deste aluno no PAS 3?"*
2. **Engenharia Reversa (Meta):** *"Dado o curso dos sonhos, quanto este aluno precisa tirar na próxima etapa para ser aprovado?"*

O sistema personaliza a experiência para cada escola parceira, adaptando logotipos, identidades visuais e templates de relatórios PDF.

---

## ✨ Principais Funcionalidades

- 🌐 **Páginas por Curso:** landing pages públicas, sem login, com o histórico de notas de corte e a previsão do curso — o principal canal de aquisição orgânica.
- 🚦 **Semáforo de Risco:** Classificação instantânea de alunos (Verde / Amarelo / Vermelho) em relação aos seus cursos-alvo.
- 🔮 **Motor de Predição:** Estimativa da nota de Etapa 3 (`A3`) por um único modelo LightGBM, combinada por aritmética exata com as notas já sabidas do aluno em Etapa 1 e 2.
- 🎯 **Calculadora de Metas:** Cálculo matemático reverso definindo a nota exata necessária no PAS 3 para um curso-alvo.
- 📄 **Geração de Relatórios (PDF):** Emissão de relatórios pedagógicos customizados com a marca da escola (hoje só no painel Streamlit legado).
- 📈 **Análise de Cortes e Probabilidades:** Histórico profundo de notas de corte e cálculo de chance percentual de aprovação por cota (10 Sistemas de Concorrência do Edital).

---

## 🧠 Inteligência Artificial e Modelagem

O núcleo de predição é treinado sobre uma base de **66.313 registros** (8 triênios, de 2016/2018 a 2023/2025), com **60.013 linhas limpas** após validação de checksum.

### Um modelo, mais aritmética
Para um aluno que já sentou PAS 1 e PAS 2, os Argumentos de Etapa `A1` e `A2` são **aritmética exata** — não há nada a prever ali. Só `A3` (Etapa 3) é previsto, por um único **LightGBM** (400 árvores, com faltante nativo para o aluno sem Etapa 1, em vez de zero literal):

```
Argumento Final = A1 + 2·A2 + 3·A3
```

Um *ensemble* de 4 modelos (LightGBM + MLP + Regressão Linear + Random Forest, roteado pela volatilidade do histórico do aluno) foi testado e aposentado: ganhava apenas 0,10% do seu melhor componente sozinho — dentro do ruído entre dobras de validação. A probabilidade de aprovação (`P(X > corte)`) usa a incerteza medida do próprio modelo (a *Largura de Incerteza*, por turma, versionada no manifesto do pacote) como desvio-padrão dessa normal.

O regime de validação é janela expansiva (5 dobras) com um holdout de 2023/2025 lacrado — usado uma única vez, na promoção do modelo em produção.

---

## 🏗️ Arquitetura e Stack Tecnológica

O sistema foi desenhado para ser rápido, escalável e de fácil manutenção:

- **Frontend Web / App:** `Next.js` (React, TypeScript e TailwindCSS) hospedado na Vercel para a landing page institucional e portal do aluno/escola.
- **Backend API:** `FastAPI` (Python) hospedado no Render, buildado a partir de um repositório de deploy dedicado que empacota `api/` + `src/pas_intelligence/`; serve as predições, relatórios e a curva histórica de cortes via API REST.
- **Dashboard de Administração / Admin:** `Streamlit` (legado, roda só localmente — não versionado nem publicado) para análises rápidas e controle pedagógico interno.
- **Banco de Dados / Auth:** `Supabase` usado pelo frontend Next.js para PostgreSQL, Row Level Security (RLS) e gerenciamento de sessões; a API não depende do Supabase, lê os dados de CSVs derivados e do pacote de modelo.
- **Machine Learning:** `LightGBM` (pacote versionado em texto nativo, com manifesto de proveniência) para prever `A3`; o restante do Argumento Final é aritmética direta.
- **Distribuição de artefatos:** os artefatos treinados (modelo + CSVs derivados, sem PII) moram num repositório privado no Hugging Face Hub e são buscados só no build da imagem da API — nunca no boot.
- **Geração de Documentos:** `ReportLab` injetando dados em templates PDF dinâmicos (consumido hoje só pelo painel Streamlit legado).

---

## 🚀 Como Executar Localmente

### Pré-requisitos
- Python 3.10 ou superior.
- Node.js 18 ou superior.
- Git instalado.
- `models/pas3/` (pacote de modelo) e os CSVs de `data/` no disco — ambos gitignored; veja a seção de artefatos abaixo se estiverem ausentes.
- Credenciais de acesso ao projeto no Supabase (só necessárias para o frontend, passo 3).

### 1. Clonar e Configurar o Repositório
```bash
git clone https://github.com/luizhtmoreira/Analise-Preditiva-PAS.git
cd Analise-Preditiva-PAS
```

### 2. Configurar o Backend (FastAPI + ML Core)
```bash
# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate # No Windows use: .venv\Scripts\activate

# Instale as dependências Python (requirements-api.txt é o subconjunto que a API
# de fato usa; requirements.txt inclui também o que o Streamlit legado precisa)
pip install -r requirements-api.txt

# Inicie o servidor FastAPI local — não depende de Supabase nem de .env;
# lê o pacote de modelo em models/pas3/ e os CSVs derivados em data/
uvicorn api.main:app --reload --port 8000
```
O backend estará disponível em `http://localhost:8000`. Sem `models/pas3/` e os CSVs de `data/`
no disco, o `lifespan` de `api/main.py` falha ao subir — esses artefatos são buscados no build
da imagem de produção (ver seção de Arquitetura) e não estão no repositório público.

### 3. Configurar o Frontend (Next.js)
```bash
cd landing-page

# Instale as dependências do Node
npm install

# Crie um arquivo .env.local baseado em .env.local.example com as chaves:
# NEXT_PUBLIC_SUPABASE_URL=sua_url
# NEXT_PUBLIC_SUPABASE_ANON_KEY=sua_chave
# NEXT_PUBLIC_API_URL=http://localhost:8000

# Inicie o servidor de desenvolvimento
npm run dev
```
O frontend estará disponível em `http://localhost:3000`.

### 4. Executar o Dashboard Streamlit (Opcional/Legado)
Se precisar executar o dashboard Streamlit legado:
```bash
# Retorne à raiz do projeto com o ambiente virtual ativo
cd ..
streamlit run app/streamlit_app.py
```
Disponível em `http://localhost:8501`.

---

## 🧪 Testes

O projeto utiliza o `pytest` para assegurar a integridade dos cálculos do edital, previsões e regras multi-tenant do backend (todos sobre dados sintéticos, sem PII):
```bash
pytest tests/

# Um único arquivo
pytest tests/test_pas_intelligence.py
```
