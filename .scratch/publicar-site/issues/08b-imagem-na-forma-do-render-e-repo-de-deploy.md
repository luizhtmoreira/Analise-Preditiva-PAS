# 08b — A imagem na forma do Render, entregue por um Repo de Deploy

**What to build:** o artefato de deploy sai deste repositório sem levar a árvore nem a história do
monorepo junto, e chega numa forma que o Render consegue buildar e rodar.

Duas metades da mesma fatia: o Dockerfile deixa de assumir Hugging Face Spaces, e o publicador
deixa de falar com o HF.

**A forma do Render.** O `CMD` passa a ler `$PORT` (o Render injeta, default 10000) em vez do
`7860` cravado. O secret de build passa a ser lido de onde o Render monta Secret Files. O
`useradd -u 1000` **fica** — deixa de ser "convenção do HF Spaces" e passa a ser boa prática, e o
comentário deve dizer isso, senão o próximo leitor apaga.

**O download continua em build-time, e a justificativa ficou mais forte.** O comentário do
Dockerfile — *"nunca no boot, porque o boot não pode depender de rede"* — vale **mais** no Render
que valia no HF: lá hibernar era raro (48 h), aqui é rotina (15 min ociosos), então um boot
dependente de rede transformaria o Hugging Face em dependência de cada Boot Frio. O relatório do
ticket 08 assumia que o Render obrigaria a mudar isso porque "secrets de build são pagos no
Render" — **isso é falso**: o Render suporta `--mount=type=secret` em todos os planos, inclusive o
gratuito (ADR-0014). A linha que já existe sobrevive; muda o caminho de montagem.

**O Repo de Deploy.** O fluxo padrão do Render é clonar um repositório Git — exatamente o que o
`publicar_space.py` foi escrito para evitar, porque a história deste repo já teve PII em commits
órfãos e force-push não despublica (ticket 15). O publicador troca de destino: em vez de
`upload_folder` para o HF, um `git push` de snapshot num repositório que **nasce vazio**. A lista
`PERMITIDOS` de 9 padrões já existe, testada e revisada — ela não muda.

O Repo de Deploy nunca é editado à mão. Ele carrega um `README.md` de uma linha dizendo isso, pela
mesma razão que o `.scratch/parser-backup/` carrega a instrução dele no `CLAUDE.md`: dois lugares
com o mesmo código convidam alguém a editar o errado.

## Nota para quem implementar

**O caminho exato onde o Render monta Secret Files é fato externo. Confira na documentação do
Render; não responda de memória.** Vale para qualquer outro detalhe de plataforma que aparecer neste
ticket — porta padrão, nome de variável, limite de plano.

Isso não é zelo genérico. Este ticket existe porque um fato externo não verificado já custou o
ticket 08 **duas vezes**:

1. o ADR-0004 escolheu Hugging Face Spaces por "CPU Basic, gratuito" — política que mudou, e só se
   descobriu ao receber `402 Payment Required` depois de toda a infraestrutura pronta;
2. o relatório do ticket 08 registrou que "secrets de build são pagos no Render" e concluiu daí que
   os artefatos teriam de ser baixados em runtime. **Era falso**, e essa premissa sozinha quase
   reescreveu o Dockerfile sem necessidade.

Conhecimento de modelo sobre preço, cota e caminho de plataforma envelhece em silêncio. Verifique.

**Blocked by:** None — can start immediately.

**Status:** done — commit `85365fe`

- [x] A imagem sobe e `/health` responde com a porta vindo do ambiente, e continua respondendo com
      o default quando a variável não existe
- [x] O secret de build é lido de onde o Render o monta, e a etapa de busca falha com mensagem
      clara — não com crash genérico — quando ele falta
- [x] Nenhum comentário no Dockerfile afirma que o destino é Hugging Face Spaces; o que é convenção
      herdada e ficou por mérito próprio está marcado como tal
- [x] O Repo de Deploy recebe exatamente os padrões de `PERMITIDOS` e nada além — verificado sobre o
      repositório publicado
- [x] O Repo de Deploy não tem nenhum commit vindo da história do monorepo
- [x] Publicar duas vezes seguidas produz um segundo commit de snapshot, não um conflito
- [x] O Repo de Deploy diz, nele mesmo, que é gerado e não deve ser editado à mão
