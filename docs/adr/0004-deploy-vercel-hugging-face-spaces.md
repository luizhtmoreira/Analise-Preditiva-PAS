# Deploy: Vercel (Next.js) + Hugging Face Spaces (FastAPI)

> **Status: parcialmente substituído pelo ADR-0014 (2026-07-31).** A parte do FastAPI está morta por
> fato, não por preferência: hospedar Docker Spaces deixou de ser gratuito (`402 Payment Required` —
> exige assinatura PRO), e a premissa "CPU Basic, gratuito" abaixo não vale mais. O backend passou
> para o Render. **A Vercel para o Next.js continua valendo**, e os repositórios do Hugging Face
> Hub continuam sendo o Domicílio Versionado dos artefatos — o que caiu foi o Spaces como lugar de
> *rodar* a API, não o Hub como lugar de *guardar* o modelo.

O Next.js é hospedado na Vercel (Hobby tier). O FastAPI com os modelos ML é hospedado no Hugging Face Spaces (CPU Basic, gratuito). A alternativa era um VPS único (Hetzner/DigitalOcean, ~$6–12/mês) ou Vercel + Railway (~$25–35/mês).

Escolhemos esta combinação por restrição de custo: o projeto está em fase pré-receita. O Hugging Face Spaces é infraestrutura desenhada para servir modelos ML via API, suporta FastAPI com Docker, e oferece Git LFS para os arquivos `.joblib` (atualmente gitignored e hospedados no Dropbox).

## Consequências

- O Hugging Face Space hiberna após 48h de inatividade. Um job de keep-alive via UptimeRobot (monitoramento gratuito a cada 5 minutos) previne cold starts.
- Os modelos `.joblib` precisarão ser movidos do Dropbox para um repositório Hugging Face (Git LFS).
- Quando o projeto gerar receita, migrar o FastAPI para Railway ou um VPS elimina a dependência de keep-alive e melhora latência.
