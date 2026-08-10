# Decisões de arquitetura

Um **ADR** (*Architecture Decision Record*) é o registro de uma decisão técnica: o que foi
decidido, por quê, e o que se perdeu ao decidir assim. Eles são escritos no momento da decisão e
não são reescritos depois — quando uma decisão é revista, um ADR novo substitui o antigo, e o
antigo continua no lugar com a marca de substituído.

Publicamos estes registros por uma razão simples: eles são a prova de que as escolhas do produto
foram medidas antes de serem tomadas. Quase todos carregam o número que sustentou a decisão e o
custo que ela impôs.

## Se você só for ler três

**[ADR-0009 — O alvo canônico é o Argumento da Etapa 3](0009-alvo-canonico-argumento-da-etapa-3.md)**
: Por que o sistema prevê um único número e deriva todo o resto por aritmética. A decisão que
  garante que os números na tela não se contradigam. Antes dela, duas rotas do mesmo produto
  discordavam sobre a aprovação de 1 em cada 9 alunos.

**[ADR-0010 — Validação deslizante com holdout lacrado](0010-validacao-deslizante-com-holdout-lacrado.md)**
: A régua do projeto. Define como qualquer afirmação de acurácia é medida, e institui o triênio
  lacrado. É o documento que torna o número de acerto cobrável.

**[ADR-0012 — Largura de incerteza fixa por classe](0012-largura-fixa-por-classe-em-vez-de-incerteza-por-aluno.md)**
: Por que a incerteza usada na conta de probabilidade viaja dentro do arquivo do modelo, e não como
  constante no código. Documenta também a rejeição de uma abordagem mais sofisticada porque ela
  foi medida e movia o resultado em 0,2 ponto percentual.

## Por assunto

### Produto e interface

| | |
|---|---|
| [0001](0001-monorepo-nextjs-unificado.md) | Monorepo Next.js unificado para landing e dashboard |
| [0002](0002-fastapi-backend-python.md) | FastAPI como backend Python separado |
| [0003](0003-split-features-publico-b2b.md) | Split de features: público para alunos, B2B para coordenação |
| [0005](0005-shadcn-recharts.md) | shadcn/ui + Recharts como stack de interface |
| [0006](0006-soft-gate-segundo-curso.md) | Pedir login no segundo curso, não após o primeiro resultado |

### Modelo e medição

| | |
|---|---|
| [0008](0008-aluno-sem-etapa-1-atendido-com-funcao-propria.md) | O aluno sem Etapa 1 é atendido, e a ausência é declarada |
| [0009](0009-alvo-canonico-argumento-da-etapa-3.md) | O alvo canônico é o Argumento da Etapa 3 |
| [0010](0010-validacao-deslizante-com-holdout-lacrado.md) | Validação deslizante, com o triênio mais recente lacrado |
| [0011](0011-lightgbm-unico-com-faltante-nativo-substitui-o-ensemble.md) | Um LightGBM único substitui o ensemble de quatro modelos |
| [0012](0012-largura-fixa-por-classe-em-vez-de-incerteza-por-aluno.md) | Largura de incerteza fixa por classe de aluno |
| [0013](0013-parte-1-misturada-e-procedencia-no-official-stats.md) | A Parte 1 pode vir misturada, e toda entrada declara sua procedência |
| [0007](0007-baseline-modelos-v1.md) | Baseline dos modelos v1 — **superscrito**, mantido como histórico |

### Publicação e hospedagem

| | |
|---|---|
| [0014](0014-api-no-render-com-derivado-sem-pii-e-repo-de-deploy.md) | A API roda no Render, servindo um derivado sem dados pessoais |
| [0004](0004-deploy-vercel-hugging-face-spaces.md) | Vercel + Hugging Face Spaces — **substituído** pelo 0014 na parte do backend |
