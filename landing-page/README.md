# 🌐 Vetor PAS — Frontend Next.js

Este diretório contém o frontend principal da plataforma **Vetor PAS**, desenvolvido com **Next.js (App Router)**, **TypeScript** e **TailwindCSS (v4)**.

O app engloba a Landing Page institucional pública, telas de login e cadastro, e a área logada de alunos e escolas parceiras (coordenadores).

---

## 🚀 Como Iniciar Localmente

### 1. Pré-requisitos
- Node.js 18+ instalado.
- Backend FastAPI rodando localmente (ou apontando para staging/produção).
- Banco de Dados Supabase configurado.

### 2. Configurar Variáveis de Ambiente
Crie um arquivo `.env.local` na raiz desta pasta com as seguintes chaves (veja o `.env.local.example` como referência):

```env
NEXT_PUBLIC_SUPABASE_URL=https://seu-projeto.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=seu-anon-token-do-supabase
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Instalar e Rodar
```bash
# Instalar dependências
npm install

# Iniciar o servidor local
npm run dev
```

Abra [http://localhost:3000](http://localhost:3000) no seu navegador.

---

## 📂 Estrutura de Pastas

*   `app/` — Rotas e páginas (Next.js App Router).
    *   `(public)/` — Rotas abertas a qualquer visitante (Preditor simplificado em `/predict` e análise histórica em `/temporal`).
    *   `auth/` — Telas de autenticação (`/login`, `/cadastro`, `/entrar`).
    *   `(dashboard)/` — Rotas privadas (`/app/*`) protegidas pelo `middleware.ts`.
        *   `gestao/` — Semáforo de Risco e listagem de alunos da escola.
        *   `relatorios/` — Geração e download de relatórios em PDF.
        *   `escola/` — Painel analítico de desempenho da escola vs. população geral.
        *   `comparacao/` — Comparador estatístico de turmas.
*   `components/` — Componentes reutilizáveis do React.
    *   `ui/` — Componentes de UI genéricos (botões, inputs, cards, selects).
    *   `brand/` — Marca institucional (logotipos, curvas gaussianas).
    *   `public/` — Componentes das páginas públicas.
    *   `dashboard/` — Componentes da área administrativa.
*   `lib/` — Utilitários, conexões com Supabase e definições de tipos TypeScript.
*   `public/` — Ativos estáticos (SVG, logos, imagens).

---

## 🎨 Whitelabeling e Customização de Temas

A identidade visual dinâmica (multi-tenant) é gerenciada no arquivo `/app/layout.tsx` e herdada pelas páginas internas:
- O layout lê o identificador do `tenant` associado ao perfil do usuário logado via Supabase.
- Elementos visuais (como logotipo, cores de fundo e bordas) se ajustam dinamicamente baseados na correspondência da escola parceira.

---

## 🧪 Build e Verificação
Para validar se o projeto compila sem erros de tipografia ou linters antes de realizar o deploy:
```bash
npm run build
```
