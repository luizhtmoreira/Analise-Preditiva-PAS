# A API roda no Render, a partir de um Repo de Deploy, servindo um Derivado sem PII

> **Status:** aceito. **Substitui o ADR-0004** na parte do backend (a hospedagem do Next.js na
> Vercel continua valendo).

O ADR-0004 escolheu Hugging Face Docker Spaces por "CPU Basic, gratuito". Ao publicar de verdade em
2026-07-31, o Hugging Face devolveu `402 Payment Required`: hospedar Docker Spaces passou a exigir
assinatura PRO. A decisão não estava errada quando foi tomada — ela **envelheceu sem ninguém
notar**, e é esse o defeito que este ADR tenta não repetir.

**Decidido:** a API FastAPI roda no **Render**, plano gratuito, buildando um Dockerfile a partir de
um **Repo de Deploy** dedicado, baixando em tempo de build um **Derivado de Deploy** dos artefatos
(sem a coluna `nome`) a partir de repositórios privados no Hugging Face Hub, e é alcançada pelo
frontend em `api.vetorpas.com.br` — nunca pelo endereço `onrender.com`.

## O critério que guiou a escolha

Não foi "qual tier gratuito é mais generoso" — foi **qual erro é mais barato de corrigir**. O
ADR-0004 machucou porque a saída custava caro. Toda escolha abaixo foi feita para que trocar de
plataforma custe um registro DNS e um `git push`, não uma reescrita.

## Decisões, e por quê

**Render, não Cloud Run.** O Render gratuito **não consegue cobrar** — não tem cartão. O Cloud Run
gratuito é uma cota dentro de uma conta que tem cartão: estourou, vira fatura. Para um produto
pré-receita operado por uma pessoa, "impossível me cobrar por construção" vale mais que
"provavelmente não vai me cobrar". Se o Preditor viralizar num grupo de cursinho, no Render ele fica
lento; no Cloud Run ele fica caro.

**Fly.io e Railway saíram por fato, não por gosto.** O Fly encerrou o tier gratuito em 2024 (hoje é
trial de 2 h de VM ou 7 dias); o Railway encerrou o dele em 2023 (mínimo ~$1/mês). Nenhum dos dois
atende à restrição "de graça neste primeiro momento". O Render é o único da lista com tier
permanente e **sem cartão**.

**A PII sai na origem, não na leitura.** `resultado_final.csv` (24,2 MB, 45 colunas, 66.313 Alunos
reais) e `notas_corte.csv` carregam `nome`. O código já era cuidadoso — o `usecols` de
`gestao_service` e `analytics_service` nunca lê `nome` — mas `usecols` protege a *leitura*, não o
*arquivo*: o CSV inteiro ia para dentro da imagem. Passa a ser o `publicar_pacote.py` que sobe o
Derivado, e o Domicílio Versionado nunca vê `nome`. É a mesma fronteira que este repo já adota em
`src/pas_extraction/`: **o dado não entra**, em vez de entrar e a gente lembrar de não olhar.
Efeito medido: 24,2 MB → **4,5 MB** (10 colunas), 4,1 MB já filtrado por `checksum_fecha`.

**Dois repositórios no Hugging Face, não um.** `Luiz1912/vetor-pas-dados` acumulava dois papéis —
backup dos CSVs de extração e insumo de deploy. Cortar na origem destruiria o backup. Passa a haver
o cru privado (backup explícito, ninguém baixa em build) e o Derivado (o que o Ponteiro aponta).
Mesma lição da invariante dos parsers no `CLAUDE.md`: backup por decisão, não por acidente.

**O download continua em build-time.** O comentário do Dockerfile — *"nunca no boot, porque o boot
não pode depender de rede"* — vale **mais** no Render que valia no HF: lá hibernar era raro (48 h),
aqui é rotina (15 min ociosos). Um boot que depende de rede passaria a significar "o Hugging Face
fora do ar derruba a API a cada Boot Frio". O relatório do ticket 08 assumia que o Render obrigaria
a mudar isso, porque "secrets de build são pagos no Render". **Isso é falso** — o Render suporta
`--mount=type=secret` (Secret Files, montados em `/etc/secrets/`) em todos os planos, inclusive o
gratuito. A linha que já está no Dockerfile sobrevive. Esse erro era da mesma família do que matou o
ADR-0004: fato externo herdado sem verificação.

**Repo de Deploy, não o monorepo conectado.** O fluxo padrão do Render é clonar um repositório Git —
exatamente o que `publicar_space.py` foi escrito para evitar, entregando árvore e história a um
terceiro (a história deste repo já teve PII em commits órfãos, e o force-push não despublica). A
lista `PERMITIDOS` de 9 padrões já existe, testada e revisada; só troca o destino de `upload_folder`
para um `git push` num repositório que nasce vazio. De brinde vêm deploy automático a cada push,
histórico e rollback por revert.

**`api.vetorpas.com.br`, nunca `onrender.com`.** Gravar o nome da plataforma no build do Next.js
tornaria a saída cara: trocar de hospedagem exigiria **redeployar a landing**, e o projeto na Vercel
é CLI-only, sem integração Git — um deploy manual que ninguém roda há meses, sob pressão, com a API
fora. Com domínio próprio, sair do Render é um CNAME. O DNS está na Hostinger sob controle do dono
(`cosmos/nova.dns-parking.com`), a Vercel só ocupa o apex e o `www`, e `api.` está livre. O Render
dá TLS gratuito em domínio próprio em todos os planos.

## Consequências

- **O Boot Frio é comportamento normal, não anomalia.** O frontend precisa saber esperar. Medido: a
  API entra com **297 MB de 512 MB** do teto gratuito em repouso (213 MB só de importar
  `pandas + scipy + lightgbm`; os 4,5 MB do Derivado economizam imagem e download, **não RAM**), com
  **0,1 vCPU**. O spin-up de 30–50 s que outros medem é o piso, não a estimativa.
- **O keep-alive é otimização descartável, nunca dependência.** O Render **não suporta oficialmente**
  manter serviço gratuito acordado; a posição deles é "migre para plano pago". Esta seção citava
  UptimeRobot "desde o dia 1" — nunca foi de fato configurado. O ticket 08e (2026-08-10) tentou
  primeiro um workflow do GitHub Actions (`.github/workflows/keep-alive.yml`, `cron: "*/10 * * * *"`)
  e **mediu que ele falha na prática**: em 7 h de observação os disparos vieram a cada 50–90 min, não
  a cada 10 — GitHub não garante pontualidade de `schedule`, e o gap supera o limiar de hibernação de
  15 min do Render (`/health` voltou a bater 32 s de Boot Frio com o workflow ativo). O mecanismo real
  é o **cron-job.org**, configurado manualmente pelo dono do produto (não por automação — é conta e
  senha de terceiro), batendo em `/health` a cada 5 min. Verificado contra o serviço em produção após
  duas janelas ociosas de ~25 min sem nenhuma chamada minha: `/health` respondeu em 1,25 s e depois em
  0,35 s — nunca subiu para a casa dos 30 s. O workflow do GitHub Actions continua no repositório como
  camada redundante (não atrapalha, só não é confiável sozinho). Nenhum dos dois elimina Boot Frio —
  só o torna raro. Deploy, restart da plataforma, OOM kill, ou a conta do cron-job.org sendo
  desativada continuam produzindo boot do zero — o produto volta ao comportamento do ticket 08d, não
  quebra. Apoiar a experiência do Aluno nele seria repetir o ADR-0004 com outro fornecedor.
- **Um único serviço gratuito por workspace, com folga apertada.** Sempre-acordado num mês de 31
  dias consome **744 das 750 horas/mês** — 6 h de folga (0,8%), não os ~730 h citados antes (essa
  conta era otimista; a folga real varia com o mês: 30 h num mês de 30 dias, 78 h em fevereiro). Um
  segundo serviço (staging, cron) estoura a cota e suspende tudo até virar o mês.
- **`/health` precisa ser barato de propósito** — ele é batido a cada 10 minutos, nos seus 0,1 vCPU,
  e não pode tocar no modelo nem nos CSVs.
- **Risco concreto no frontend:** metade das chamadas de `landing-page/lib/api.ts` é server-side
  (`process.env.API_URL` — Gestão, Analytics, `/api/temporal`) e passa por uma Function da Vercel,
  que tem timeout próprio no plano Hobby. Se o Boot Frio exceder esse timeout, essas páginas
  **quebram com erro**, não ficam lentas — e só em produção, só depois de ociosidade. As chamadas
  client-side (Preditor, Calculadora) não têm esse problema. O timeout precisa ser conferido e
  configurado.
  **Resolvido no ticket 08d (2026-08-08):** Boot Frio medido contra `api.vetorpas.com.br/health`
  hibernada de verdade (não estimado de terceiro): **32,4 s**. O timeout de Function no plano Hobby
  (confirmado na documentação oficial da Vercel, sem `vercel.json` neste repo que o reduza) é **300
  s, fixo** — não é um valor a configurar aqui, é o teto do plano. Margem de quase 10×; a folga
  contra o piso documentado de 30–50 s também se sustenta mesmo num boot mais lento do que o medido
  numa única amostra. Não há gatilho para antecipar o Starter de $7/mês por este motivo. O
  `/temporal` ganhou proteção extra por já usar `revalidate: 3600` — uma regeneração que falhar por
  Boot Frio serve a página estática anterior em vez de erro. Relatório completo em
  `.scratch/publicar-site/relatorios/08d-boot-frio-medido-e-o-frontend-que-sabe-esperar.md`.
- **O Starter de $7/mês é a saída documentada, com gatilho escrito:** sem spin-down, 0,5 vCPU. O
  gatilho é o lançamento — não "quando doer".
- **O `CMD` passa a ler `$PORT`** (o Render injeta, default 10000) em vez do `7860` cravado. O
  `useradd -u 1000` fica como boa prática, deixa de ser "convenção do HF".

## Alternativas rejeitadas

- **Assinar HF PRO (~$9/mês)** — o menor atrito de todos: rodar `publicar_space.py` de novo e
  encerrar. Rejeitada pela restrição de custo agora, não por mérito. Continua sendo uma opção
  legítima no lançamento, e a decisão acima não a bloqueia.
- **Cloud Run** — melhor Boot Frio (segundos) e teto de escala maior. Rejeitada pelo cartão na
  conta, pela cota gratuita valer só em regiões dos EUA, e por bem mais maquinário (gcloud, Artifact
  Registry, IAM) para uma pessoa manter sozinha.
- **Publicar o Derivado num repositório HF público** — eliminaria o secret e simplificaria o
  Dockerfile, e o dono do produto autorizou (os Editais do Cebraspe já publicam nota e inscrição
  abertamente). Mantida como **plano B**, não como plano A: publicar é irreversível, e um CSV
  público baixado uma vez está fora para sempre. Enquanto o caminho privado funciona de graça, gastar
  essa opção não compra nada.
- **Imagem pré-buildada empurrada para um registry** — preservaria o mesmo princípio do Repo de
  Deploy. Rejeitada porque o ganho é ilusório: o build do Render roda em máquina separada da
  instância gratuita, então não compra velocidade — compra cross-compilation `linux/amd64` no Apple
  Silicon e um registry para manter.
- **Baixar os artefatos em runtime, no boot** — o que o relatório do ticket 08 assumia como
  inevitável. Rejeitada porque a premissa que a forçava (secrets de build pagos) é falsa, e porque
  no Render ela é *pior* que era no HF: transformaria o Hugging Face em dependência de cada Boot
  Frio.
