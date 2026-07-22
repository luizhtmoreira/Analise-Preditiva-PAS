# ⚡ Vetor PAS — Backend FastAPI

Este diretório contém a API do backend do **Vetor PAS**, desenvolvida em **FastAPI** (Python). Ela serve de ponte entre a aplicação frontend Next.js e o motor de predição baseado em Machine Learning.

---

## 🚀 Como Iniciar Localmente

### 1. Pré-requisitos
- Python 3.10 ou superior.
- Ambiente virtual configurado e ativo na raiz do projeto.
- Dependências instaladas (`pip install -r requirements.txt`).

### 2. Configurar Variáveis de Ambiente
Certifique-se de configurar o arquivo `.env` na raiz do repositório principal com as chaves do Supabase:
```env
SUPABASE_URL=sua-url-do-supabase
SUPABASE_KEY=sua-chave-do-supabase
ENV=DEV
```

### 3. Rodar a API
Execute a API a partir da raiz do projeto:
```bash
uvicorn api.main:app --reload --port 8000
```

A documentação interativa da API (Swagger UI) estará disponível em [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📂 Estrutura de Pastas

*   `main.py` — Ponto de entrada do FastAPI. Inicializa a aplicação, configura CORS para o Next.js e importa as rotas.
*   `routers/` — Definições de rotas de API HTTP.
    *   `predict.py` — Rotas de cálculo preditivo e probabilidades de aprovação do aluno.
    *   `analytics.py` — Estatísticas de notas de corte e comparação histórica.
    *   `gestao.py` — Endpoints B2B de gestão escolar (dados de turmas, semáforo de risco).
*   `schemas/` — Schemas de validação de dados Pydantic (Request/Response bodies).
*   `services/` — Camada de serviço/lógica de negócio. Invoca as funções core de `src/pas_intelligence` e manipula os resultados.

---

## 🧠 Integração com o Motor de IA

Os endpoints importam as bibliotecas de inteligência diretamente de `src/pas_intelligence/`:
- **Previsões de notas:** Chamam os modelos serializados `.joblib` em `models/` através das funções do `ensemble.py`.
- **Calculadora reversa:** Utiliza o `target_calculator.py` para responder "quanto falta" para o aluno atingir a aprovação.
- **Relatórios Whitelabel:** Gera PDFs via ReportLab, aplicando logos dinamicamente e retornando o arquivo final como um stream de bytes para o frontend.
