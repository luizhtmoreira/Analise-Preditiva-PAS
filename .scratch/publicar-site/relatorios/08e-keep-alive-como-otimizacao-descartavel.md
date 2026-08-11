# Relatório — Ticket 08e: Keep-alive, como otimização descartável

**Ticket:** `.scratch/publicar-site/issues/08e-keep-alive-como-otimizacao-descartavel.md`
**Status:** concluído
**Data:** 2026-08-11

---

## 1. `/health` já era barato

`api/main.py` — a rota é `{"status": "ok"}` puro, não toca no pacote de modelo nem nos CSVs. O
carregamento de recursos (`load_resources()`) roda uma vez no `lifespan`, no boot do processo, não a
cada requisição. Nenhuma mudança precisou entrar aqui; só verificação de código, não presunção.

## 2. Primeira tentativa falhou, e foi medida falhando

A primeira implementação foi um workflow do GitHub Actions
(`.github/workflows/keep-alive.yml`) com `cron: "*/10 * * * *"` batendo em `/health`. Depois de
publicado, **7 horas de observação real** mostraram que o `schedule` do GitHub não é pontual: os
disparos vieram a **50–90 minutos** de intervalo, não a 10 — acima do limiar de hibernação do Render
(15 min ociosos). Confirmação direta: `/health` voltou a bater **32,9 s** de Boot Frio com o workflow
ativo e já tendo disparado outras vezes antes. GitHub documenta que `schedule` pode atrasar sob
carga; na prática, para um intervalo de minutos, o atraso não é exceção — é a regra.

Isso não foi descoberto por leitura de documentação, foi por medição ao vivo, exatamente o padrão que
o `08d` já tinha estabelecido para esse tipo de afirmação.

## 3. Correção: cron-job.org, configurado pelo dono do produto

Como o keep-alive precisa de um timer de verdade, e criar conta em serviço de terceiro não é algo que
um agente deve fazer sozinho (é login e senha de uma conta que fica com o dono do produto, não
comigo), pedi para o dono configurar manualmente: conta gratuita em cron-job.org, cronjob batendo em
`https://api.vetorpas.com.br/health` a cada 5 minutos.

**Verificado depois, não presumido:** deixei passar duas janelas de ~25 minutos sem eu mesmo chamar
`/health`, para não mascarar o teste, e então chequei:

| Momento | `time_total` | Interpretação |
|---|---|---|
| Antes do cron-job.org (GitHub Actions sozinho) | 32,9 s | Boot Frio — hibernou |
| ~25 min depois do cron-job.org configurado | 1,25 s | Acordado |
| ~25 min depois, segunda janela | 0,35 s | Acordado |

O workflow do GitHub Actions **não foi removido** — fica como camada redundante gratuita, mas o
comentário no topo do arquivo e o ADR-0014 agora deixam explícito que ele sozinho não é confiável
para isso, para que ninguém no futuro presuma que ele já resolve o problema.

## 4. Consumo de horas — a conta anterior era otimista

O ADR-0014 citava "~730 h de 750 h" para o serviço sempre-acordado. Recalculado com precisão: um mês
de 31 dias sempre-acordado consome **744 h** — a folga real é de **6 horas (0,8%)**, não os ~20 h que
a cifra anterior sugeria. Meses de 30 dias dão 30 h de folga; fevereiro, 78 h. Isso está registrado no
ADR-0014, sob a mesma seção do keep-alive, porque é o mesmo tipo de aritmética que decide se o plano
gratuito aguenta.

## 5. Decisão

O produto não passa a depender do keep-alive em nenhum ponto do código — ele é inteiramente externo
(GitHub Actions + cron-job.org, nenhum dos dois referenciado por `api/` ou `landing-page/`). Se
qualquer um dos dois parar, o comportamento é exatamente o do `08d`: Boot Frio raro vira Boot Frio
normal, com o frontend já preparado para esperar.

## 6. Arquivos alterados

- `.github/workflows/keep-alive.yml` — novo, camada redundante (não a principal)
- `docs/adr/0014-api-no-render-com-derivado-sem-pii-e-repo-de-deploy.md` — mecanismo real
  documentado (GitHub Actions medido falhando + cron-job.org como correção), conta de horas
  recalculada com precisão
- `.scratch/publicar-site/issues/08e-keep-alive-como-otimizacao-descartavel.md` — critérios marcados

## 7. O que fica fora do escopo

- Não criei conta em nenhum serviço de terceiro — é decisão e credencial do dono do produto, não de
  um agente.
- Não medi o pior caso (deploy em andamento, restart de plataforma) — fora do alcance de observação
  deste ticket; o ticket só pede que a falha desses cenários **não quebre** o produto, o que já é
  verdade por construção (nada no código depende do ping).
