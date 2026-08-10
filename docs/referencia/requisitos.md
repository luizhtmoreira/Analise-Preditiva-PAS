# Requisitos do sistema

## Requisitos funcionais

| ID | Descrição | Estado |
|---|---|---|
| **RF01** | Login seguro para coordenadores, diretores e alunos | Entregue |
| **RF02** | Adaptação dinâmica de logotipo e paleta por escola parceira (multi-tenant / whitelabel) | Entregue |
| **RF03** | Importação de turmas por arquivo tabular (`.csv` / `.xlsx`) | Entregue |
| **RF04** | Semáforo de risco com quatro estados (verde, amarelo, vermelho e cinza para "sem previsão") | Entregue |
| **RF05** | Previsão do Argumento da Etapa 3 a partir das etapas anteriores | Entregue |
| **RF06** | Cálculo reverso: nota mínima necessária na última etapa para um curso-alvo | Entregue |
| **RF07** | Probabilidade de aprovação por curso, sistema de concorrência e semestre | Entregue |
| **RF08** | Comparação estatística de desempenho entre grupos | Entregue |
| **RF09** | Relatórios em PDF com a marca da escola | Parcial — só na ferramenta interna |
| **RF10** | Geração e download em lote (ZIP) para uma turma inteira | Parcial — só na ferramenta interna |
| **RF11** | Histórico interativo de notas de corte por curso e ano | Entregue |
| **RF12** | Declaração de sistema de concorrência pelo aluno, nos dez sistemas do edital, com o corte correspondente | Entregue |
| **RF13** | Sinalização, na tela, de quando uma previsão usa estatística derivada em vez de edital publicado | Entregue |

## Requisitos não funcionais

| ID | Categoria | Descrição |
|---|---|---|
| **RNF01** | Tecnologia | Backend em Python 3.10+ com FastAPI, em contêiner Docker no Render |
| **RNF02** | Interface | Frontend em Next.js (React, TypeScript, Tailwind) na Vercel |
| **RNF03** | Persistência | Contas, perfis e cadastros em Supabase (PostgreSQL) |
| **RNF04** | Inteligência | Um único modelo LightGBM prevê o Argumento da Etapa 3; o restante do Argumento Final é aritmética exata ([ADR-0011](../adr/0011-lightgbm-unico-com-faltante-nativo-substitui-o-ensemble.md)) |
| **RNF05** | Incerteza | A probabilidade de aprovação usa a Largura de Incerteza medida do próprio modelo, versionada no manifesto do pacote — nunca uma constante em código ([ADR-0012](../adr/0012-largura-fixa-por-classe-em-vez-de-incerteza-por-aluno.md)) |
| **RNF06** | Validação | Todo modelo é medido por validação deslizante de janela expansiva, com holdout lacrado ([ADR-0010](../adr/0010-validacao-deslizante-com-holdout-lacrado.md)) |
| **RNF07** | Reprodutibilidade | Todo pacote de modelo carrega manifesto com hash do dado, commit, versões e métricas; o treino é determinístico byte a byte |
| **RNF08** | Privacidade | Nenhum nome ou número de inscrição é embarcado no servidor ou servido por endereço público; nenhum dado de aluno real entra em teste, fixture ou exemplo |
| **RNF09** | Segurança | Isolamento rígido entre escolas por Row Level Security no banco |
| **RNF10** | Desempenho | Carregamento do painel após login em até 3 segundos, descontado o boot frio da hospedagem gratuita |
| **RNF11** | Testes | Os cálculos do edital, as previsões e as regras multi-tenant são cobertos por `pytest`, sobre dados sintéticos |
