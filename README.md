# 📈 Análise Preditiva PAS — Vetor PAS

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-F37021?style=for-the-badge&logo=lightgbm&logoColor=white)](#)
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

- 📊 **Dashboard Interativo:** Painel completo para análise de turmas, com importação de dados e visualizações ricas.
- 🚦 **Semáforo de Risco:** Classificação instantânea de alunos (Verde / Amarelo / Vermelho) em relação aos seus cursos-alvo.
- 🔮 **Motor de Predição:** Estimativa de notas utilizando um *ensemble* dinâmico de 4 modelos de IA.
- 🎯 **Calculadora de Metas:** Cálculo matemático reverso definindo a nota exata necessária no PAS 3.
- 📄 **Geração de Relatórios (PDF):** Emissão em lote ou individual de relatórios pedagógicos customizados com a marca da escola.
- 📈 **Análise de Cortes e Probabilidades:** Histórico profundo de notas de corte e cálculo de chance percentual de aprovação por cota.

---

## 🧠 Inteligência Artificial e Modelagem

O coração da plataforma é um robusto sistema de predição treinado em uma base de **48.758 alunos** (triênios de 2016 a 2024).

### Ensemble Dinâmico
A plataforma não confia em um único modelo, mas orquestra quatro algoritmos diferentes de acordo com a **volatilidade** (estabilidade) do histórico do aluno:
- **LightGBM (Modelo Campeão)**
- **Redes Neurais (MLP)**
- **Regressão Linear**
- **Random Forest**

Se o aluno tem um desempenho estável, a regressão linear ganha peso. Se o histórico é errático, modelos baseados em árvores (LightGBM) assumem o protagonismo, garantindo uma precisão superior à média humana.

---

## 🏗️ Arquitetura e Stack Tecnológica

O sistema foi desenhado para ser rápido, escalável e de fácil manutenção:

- **Frontend / Dashboard:** `Streamlit` para prototipação e entrega rápida de visualizações de dados.
- **Backend / Processamento:** `Python` nativo com regras de negócio focadas em cálculos do edital do Cebraspe.
- **Banco de Dados / Auth:** `Supabase` fornecendo PostgreSQL e armazenamento remoto seguro.
- **Machine Learning:** `Scikit-Learn` e `LightGBM` (modelos serializados em `.joblib`).
- **Geração de Documentos:** `ReportLab` injetando dados em templates PDF dinâmicos.

---

## 🚀 Como Executar Localmente

### Pré-requisitos
- Python 3.10 ou superior.
- Git instalado.
- Credenciais de acesso ao projeto no Supabase.

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/luizhtmoreira/Analise-Preditiva-PAS.git
   cd Analise-Preditiva-PAS
   ```

2. **Crie e ative um ambiente virtual (recomendado):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate # No Windows use: .venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as Variáveis de Ambiente:**
   Crie um arquivo `.env` na raiz do projeto com o seguinte formato:
   ```env
   SUPABASE_URL=sua_url_do_supabase
   SUPABASE_KEY=sua_chave_do_supabase
   ENV=DEV
   ```

5. **Inicie a Aplicação:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

A plataforma estará disponível no seu navegador em `http://localhost:8501`.

---

## 🧪 Testes

O projeto utiliza o `pytest` para assegurar a integridade dos cálculos do edital, previsões e regras multi-tenant:
```bash
pytest tests/
```
