# Split de features: público (Alunos) vs. B2B (Coordenadores)

O sistema original era totalmente autenticado (apenas Coordenadores Pedagógicos de Escolas Parceiras tinham acesso). Decidimos abrir duas features publicamente — Preditor PAS 3 e Análise Temporal — para que Alunos possam usá-las sem conta. As demais features (Gestão de Ativos, Análise da Escola vs. População, Comparação Entre Grupos, Gerador de PDFs) permanecem exclusivas para Coordenadores autenticados.

A motivação é dupla: ampliar o topo do funil de aquisição de escolas parceiras (coordenadores que descobrem o produto via alunos) e criar uma base de usuários Alunos para uma eventual linha de produto B2C. Alunos podem criar conta opcionalmente para manter histórico de predições — sem conta o Preditor funciona de forma anônima.

## Consequências

- A landing page precisa converter duas audiências distintas com CTAs diferentes.
- A auth do Next.js tem dois níveis: sem auth (features públicas), auth de Aluno (histórico), auth de Coordenador com tenant (features B2B).
