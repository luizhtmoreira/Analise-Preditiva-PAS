# FastAPI como backend Python separado

Todo o processamento de ML e lógica de negócio do PAS permanece em Python (FastAPI), exposto como API consumida pelo Next.js. A alternativa era reescrever a lógica em TypeScript e usar apenas API Routes do Next.js. Escolhemos manter Python porque os modelos treinados (`.joblib`) são LightGBM, RandomForest e HistGradientBoosting — não há runtime Node.js capaz de executá-los. Reescrever também descartaria a lógica validada de Argumento Final, ensemble por volatilidade e geração de PDF via ReportLab. O Next.js atua puramente como camada de UI e BFF leve.

## Consequências

- CORS deve ser configurado entre o domínio Vercel (Next.js) e o domínio Hugging Face Spaces (FastAPI).
- O FastAPI é stateless: nenhum estado de sessão é mantido entre requests. O Next.js gerencia estado de UI com Zustand e cache de servidor com React Query.
