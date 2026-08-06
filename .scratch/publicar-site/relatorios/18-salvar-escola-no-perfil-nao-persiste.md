# Relatório — Ticket 18: salvar escola no perfil não persiste

**Ticket:** `.scratch/publicar-site/issues/18-salvar-escola-no-perfil-nao-persiste.md`
**Status:** concluído — hipótese confirmada ao vivo pelo dono do produto em 2026-08-06: trocou a
escola em `/perfil`, e o JSON de `raw_user_meta_data` no painel do Supabase mudou
**Onde vive o trabalho:** nenhuma mudança de código; só investigação

---

## 1. Causa raiz

**Não é o `updateUser()` que falha.** `alunos_perfis` tem uma coluna `escola` (confirmado no
painel do Supabase pelo dono do produto — print anexado à conversa) que **nenhum código do repo
escreve**. Busquei `escola` em:

- `PreditorPage.tsx` e `CalculadoraPage.tsx` (os dois únicos arquivos que fazem `upsert` em
  `alunos_perfis`) — o `upsert` grava `p1_pas1`, `p2_pas1`, `cota`, `trienio`, `curso_alvo` etc.,
  mas **nunca `escola`** (`PreditorPage.tsx:789-801`)
- todo `api/` (backend Python) — nenhuma referência a `alunos_perfis` no Python, e nenhuma a
  `escola` fora de `parte_1`/línguas (não relacionado)
- histórico do git para os dois arquivos acima — `escola` nunca apareceu neles, em nenhum commit

Ou seja: a coluna `alunos_perfis.escola` é **órfã** — existe no schema (provavelmente criada
manualmente no SQL Editor, como o resto do schema deste projeto, ver
`.scratch/historico-de-consultas-supabase.md`), mas não tem nenhum caminho de escrita ativo no
código atual.

`PerfilAlunoClient.handleSaveEscola` (`components/profile/PerfilAlunoClient.tsx:33-50`) grava em
`auth.users.raw_user_meta_data.escola` via `supabase.auth.updateUser({ data: { escola } })` — é
exatamente o lugar que o **ticket 17** já documentou como o correto para este dado ("`escola` mora
em `user_metadata`... uma tela dedicada"). `updateUser` é uma chamada GoTrue, não passa por RLS do
Postgres — não há mecanismo plausível para ela retornar `error: null` e não persistir.

**Hipótese fechada: isto é o mesmo falso positivo que o próprio ticket já alertava** (linhas
26-29) — só que a forma como se manifestou é mais sutil do que "checou a tabela errada porque a
coluna não existe": a coluna `escola` **existe** em `alunos_perfis`, então é natural checar ali
primeiro — só que ela nunca é escrita, então qualquer edição em `/perfil` vai parecer "não
persistir" para sempre, mesmo que `raw_user_meta_data` esteja atualizando corretamente.

## 2. Confirmação ao vivo — 2026-08-06

O dono do produto testou com a conta existente (`lht.unb@gmail.com`, `raw_user_meta_data.escola`
partindo de `"sagrado"`), trocou a escola em `/perfil` e conferiu no painel: `raw_user_meta_data`
mudou. Hipótese confirmada — não é bug de persistência.

## 2a. O que não pude verificar sozinho, e por quê

O checklist do ticket pede confirmar a mudança em `raw_user_meta_data` diretamente no painel. Não
tenho como fazer isso nesta rodada:

- Criar uma conta de Aluno de teste esbarra em confirmação de e-mail obrigatória
  (`mailer_autoconfirm: false` no projeto real, confirmado via
  `GET /auth/v1/settings`) — e o envio do e-mail de confirmação falhou nos meus testes
  (`AuthApiError: Error sending confirmation email`), então nem por aí dava para logar.
- Não tenho a `service_role` key nem acesso ao painel do Supabase para ler `auth.users` direto.
- O dono do produto decidiu, quando perguntado, que esta rodada fica só no diagnóstico por código.

**Para fechar o ticket:** no painel do Supabase, abrir `Authentication → Users → (usuário de
teste) → raw_user_meta_data` (não `Table Editor → alunos_perfis`), salvar uma escola nova em
`/perfil`, e conferir se o campo `escola` daquele JSON muda. Pela leitura do código, deve mudar.

## 3. Decisão em aberto (não é escopo deste ticket, mas nasce dele)

`alunos_perfis.escola` fica órfã: ou alguém tem um plano para lê-la (dashboard de escola/coordenador
ainda não escrito?) e o código do Preditor/Calculadora precisa passar a escrevê-la também, ou é
lixo de schema e vale marcar para dropar. Mesma pergunta em aberto que o **ticket 17** fez para
`cota` — talvez valha resolver as duas juntas, já que é o mesmo padrão (`user_metadata` vs.
coluna dedicada).

## 4. Checklist do ticket

- [x] O defeito está reproduzido e a causa raiz identificada (não só contornada) — causa raiz:
      confusão entre duas colunas `escola` (uma viva em `user_metadata`, uma órfã em
      `alunos_perfis`), não uma falha de persistência do `updateUser`
- [x] Trocar a escola em `/perfil` reflete em `raw_user_meta_data` no Supabase, verificado
      diretamente no painel — confirmado ao vivo pelo dono do produto (ver §2)
- [x] Diagnóstico documentado por leitura de código (não há teste de integração com Supabase Auth
      no repo, e a rodada não incluiu conta de teste real)
- [x] Nenhum código foi alterado — `pytest tests/`, `eslint` e `tsc --noEmit` continuam no estado
      anterior à rodada
