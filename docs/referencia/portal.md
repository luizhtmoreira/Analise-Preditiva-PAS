# Portal web

`landing-page/` — aplicação Next.js (App Router) que reúne o site institucional, as
funcionalidades públicas e o portal autenticado.

!!! note
    Antes de modificar este diretório, leia `landing-page/AGENTS.md` — a versão do Next.js em uso
    tem mudanças de API que quebram padrões antigos.

## Rotas

### Públicas, sem login

| Rota | O que é |
|---|---|
| `/` | Landing institucional, mobile-first |
| `/predict` | Calculadora de previsão do PAS 3 |
| `/calculadora` | Cálculo reverso — quanto falta na última etapa |
| `/temporal` | Histórico de notas de corte por curso e ano |
| `/auth/entrar`, `/auth/cadastro` | Autenticação via Supabase |
| `/auth/esqueci-senha`, `/auth/redefinir-senha` | Recuperação de senha |

### Autenticadas

| Rota | Para quem |
|---|---|
| `/perfil` | Aluno cadastrado — conta, escola vinculada, logout |
| `/gestao` | Coordenação — semáforo de risco da turma |
| `/escola` | Coordenação — escola contra a população do triênio |
| `/comparacao` | Coordenação — comparação estatística entre grupos |
| `/relatorios` | Coordenação — emissão de relatórios |

## Whitelabel

O usuário faz login e a aplicação recupera do perfil o identificador de **tenant** (a escola
vinculada). Com base nele, altera logotipo, cores e referências institucionais, e repassa o tenant
ao backend no momento em que um relatório é solicitado.

## Conexão com o backend

Toda a inferência roda na API FastAPI; o Next.js consome as rotas REST descritas em
[Arquitetura](arquitetura.md). O endereço da API é injetado por variável de ambiente no build.

!!! warning "Armadilha conhecida"
    Se a variável de ambiente da API não estiver definida no build da Vercel, o fallback aponta
    para `localhost` e a aplicação chama o endereço errado em produção — sem que o CORS acuse
    nada, porque a requisição nem chega a sair do navegador para o host certo.

## Painel Streamlit legado

`app/streamlit_app.py` é a ferramenta interna original. Não é versionada nem publicada, roda
apenas localmente e está sendo substituída por este portal. Hoje ela ainda é o único lugar onde a
geração de relatórios em PDF está operacional.
