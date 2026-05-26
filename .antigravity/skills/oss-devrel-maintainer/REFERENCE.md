# REFERENCE — OSS Maintainer Vetor PAS

## Mapa da Arquitetura

```
vetor-pas/
├── app/
│   └── streamlit_app.py        # Dashboard B2B — pode ser público (sem lógica de IA)
├── src/
│   ├── pas_intelligence/       # ⛔ PROPRIETÁRIO — repositório privado separado
│   │   ├── ensemble.py         # Ensemble dinâmico (LightGBM + RF + MLP)
│   │   ├── argument_calculator.py  # Fórmula exata do Edital Cebraspe
│   │   ├── target_calculator.py    # Engenharia reversa de metas
│   │   ├── statistics.py
│   │   └── recommender.py
│   ├── parsers/
│   │   ├── extract_pas1_pdf.py # Pode ser público (depende de PDF público do Cebraspe)
│   │   └── extract_pas2_html.py
│   └── pdf_generator.py        # Público (lógica de layout, sem dados de clientes)
├── assets/
│   └── templates/              # ⛔ PRIVADO — logos e cores dos clientes (dados B2B)
├── models/
│   └── *.pkl                   # ⛔ PRIVADO — modelos treinados (ativo proprietário)
├── .env.example                # ✅ Público — placeholders sem valores reais
└── docker-compose.yml          # ✅ Público — setup da comunidade
```

## Isolamento dos Módulos Proprietários

### Estratégia recomendada: Pacote privado instalável

```python
# Em vez de código aberto:
from src.pas_intelligence.ensemble import predict  # ❌ expõe código

# Use pacote privado instalado via pip:
from vetorpas_core import predict  # ✅ código fechado, interface pública
```

Publicar `vetorpas-core` no PyPI privado (ex: GitHub Packages) ou como wheel fechado. O repositório público importa o pacote mas não contém o código-fonte.

### Alternativa: Git Submodule privado

```bash
git submodule add git@github.com:vetorpas/core-private.git src/pas_intelligence
```

O submódulo privado não é clonado por contribuidores externos. A interface pública (`predict(pas1, pas2) -> float`) é documentada no README sem expor a implementação.

## Licenciamento — Detalhamento

### AGPL-3.0 (recomendação primária)

- Qualquer serviço que rode o código modificado em servidor deve publicar as modificações
- Protege contra "SaaS forks" — concorrente que hospedar o Vetor PAS modificado é obrigado a abrir o diff
- Módulos proprietários ficam **fora** do escopo da licença (repositório separado)
- Referência: como Grafana, Nextcloud e Plausible usam AGPL como escudo comercial

### BSL 1.1 — Business Source License (alternativa)

```
Change Date: 2028-01-01
Change License: Apache 2.0
Use Limitation: Production use requires a commercial license from Vetor PAS Ltda.
```

Código fica público para leitura e contribuição, mas uso em produção por terceiros é comercialmente bloqueado até a Change Date. Após a data, vira Apache 2.0 automaticamente.

### CLA — Contributor License Agreement

Se usar AGPL: considerar CLA simples (ex: DCO — Developer Certificate of Origin via `git commit -s`). Necessário para manter o direito de relicenciar módulos futuramente.

## Templates de Governança

### `.github/PULL_REQUEST_TEMPLATE.md`

```markdown
## Descrição
<!-- O que este PR faz? -->

## Tipo de mudança
- [ ] Bug fix
- [ ] Nova feature
- [ ] Documentação
- [ ] Refatoração

## Checklist
- [ ] Testes adicionados/atualizados
- [ ] `make test` passa localmente
- [ ] Sem credenciais ou dados reais incluídos
- [ ] Documentação atualizada se necessário

## Issue relacionada
Closes #
```

### `SECURITY.md`

```markdown
# Política de Segurança

## Reportar uma Vulnerabilidade

**NÃO abra uma issue pública.** Envie um e-mail para: security@vetorpas.com.br

Resposta esperada em até 72 horas. Divulgação coordenada após patch.
```

## Docker — Requisitos de Onboarding

O `docker-compose.yml` público deve:

1. Subir o Streamlit apontando para uma instância Supabase local (ex: `supabase/cli` self-hosted) — sem exigir conta na nuvem para testar
2. Usar `vetorpas-core` como stub/mock quando o pacote privado não estiver disponível
3. Ter um volume de dados de exemplo (`data/sample_students.csv`) com dados fictícios gerados por Faker

```yaml
# docker-compose.yml (exemplo mínimo)
services:
  app:
    build: .
    env_file: .env
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
  db:
    image: supabase/postgres:15
    environment:
      POSTGRES_PASSWORD: localdev
```

## Métricas de Saúde do Repositório (DevRel KPIs)

| Métrica | Meta nos primeiros 90 dias |
|---|---|
| Time to First PR externo | < 30 dias |
| Issues fechadas / abertas | > 0.7 |
| Tempo médio de resposta a issues | < 48h |
| Stars | indicador de awareness, não de saúde |
| Setup time (novo contribuidor) | < 5 minutos |

## Referências Externas

- [Open-Core Model — Joseph Jacks](https://opencoreventures.com/blog/2023-open-core-definition/)
- [AGPL como estratégia SaaS — Plausible](https://plausible.io/blog/open-source-saas)
- [git-filter-repo (reescrever histórico)](https://github.com/newren/git-filter-repo)
- [trufflehog — varredura de secrets](https://github.com/trufflesecurity/trufflehog)
- [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)