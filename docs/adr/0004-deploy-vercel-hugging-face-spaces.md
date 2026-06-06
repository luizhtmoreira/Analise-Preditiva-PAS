# Deploy: Vercel (Next.js) + Hugging Face Spaces (FastAPI)

O Next.js é hospedado na Vercel (Hobby tier). O FastAPI com os modelos ML é hospedado no Hugging Face Spaces (CPU Basic, gratuito). A alternativa era um VPS único (Hetzner/DigitalOcean, ~$6–12/mês) ou Vercel + Railway (~$25–35/mês).

Escolhemos esta combinação por restrição de custo: o projeto está em fase pré-receita. O Hugging Face Spaces é infraestrutura desenhada para servir modelos ML via API, suporta FastAPI com Docker, e oferece Git LFS para os arquivos `.joblib` (atualmente gitignored e hospedados no Dropbox).

## Consequências

- O Hugging Face Space hiberna após 48h de inatividade. Um job de keep-alive via UptimeRobot (monitoramento gratuito a cada 5 minutos) previne cold starts.
- Os modelos `.joblib` precisarão ser movidos do Dropbox para um repositório Hugging Face (Git LFS).
- Quando o projeto gerar receita, migrar o FastAPI para Railway ou um VPS elimina a dependência de keep-alive e melhora latência.
