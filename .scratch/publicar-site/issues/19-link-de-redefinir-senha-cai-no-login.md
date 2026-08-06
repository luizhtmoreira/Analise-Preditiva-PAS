# 19 — Link de "esqueci minha senha" cai na tela de login, sem erro

**What to build:** clicar no link de redefinição de senha recebido por e-mail leva o Aluno até
`/auth/redefinir-senha` — hoje ele volta para `/auth/entrar`, sem nenhuma mensagem explicando
por quê.

## O que se sabe

Achado durante a investigação do ticket 18 (dono do produto tentou logar numa conta antiga,
esqueceu a senha, pediu redefinição e caiu de volta no login).

**O caminho de código** (`app/auth/callback/route.ts`):

```ts
if (code) {
  const { error } = await supabase.auth.exchangeCodeForSession(code);
  if (!error) return NextResponse.redirect(`${origin}${next}`); // /auth/redefinir-senha
}
return NextResponse.redirect(`${origin}/auth/entrar?error=auth_callback_failed`);
```

Isto é: ou o link não chegou com `?code=` (não é o caso — `EsqueciSenhaForm.tsx:21` monta
`redirectTo` com `?next=/auth/redefinir-senha` corretamente), ou `exchangeCodeForSession(code)`
retornou erro. **De qualquer forma, o fallback é mudo:** `app/auth/entrar/page.tsx` nunca lê o
parâmetro `?error=auth_callback_failed` — o usuário só vê a tela de login normal, sem pista do
que aconteceu. Essa ausência de mensagem é, por si, um defeito de UX independente da causa raiz
(está fora do escopo deste ticket consertar — ver "O que resolver").

## Causa raiz confirmada — 2026-08-06

Duas perguntas ao dono do produto fecharam a investigação:

1. O teste foi em produção (`vetorpas.com.br`), não em `localhost` — **hipótese 1 (URL de
   redirect fora da allowlist) descartada.**
2. O e-mail de redefinição foi aberto **no app do Gmail**, não no mesmo navegador em que o reset
   foi pedido — **hipótese 2 confirmada.** O app do Gmail abre links no seu próprio navegador
   embutido (WebView), um contexto de armazenamento separado do navegador/aba onde
   `resetPasswordForEmail` rodou. O `code_verifier` do PKCE (gerado e salvo *localmente* no
   navegador que pediu o reset) não existe nesse WebView — `exchangeCodeForSession` falha por
   design, não por bug de configuração.

**Isto não é uma falha de configuração — é uma limitação estrutural do fluxo atual.** PKCE exige
o mesmo navegador nas duas pontas; abrir o link de um cliente de e-mail (app do Gmail, Outlook
etc., que quase sempre usam WebView próprio) quebra essa premissa quase sempre — não é um caso
raro, é o caminho mais comum de um usuário real clicar num link de e-mail.

**A correção recomendada pela própria documentação do Supabase para este cenário:** trocar a
verificação de `exchangeCodeForSession(code)` (exige o `code_verifier` local) por
`verifyOtp({ token_hash, type: "recovery" })` (valida o token em si, sem depender de nada salvo
no navegador que pediu o reset) — isso significa gerar o link de e-mail com `token_hash` em vez
de `code`, e trocar a rota `/auth/callback` (ou uma nova, ex. `/auth/confirm`) para usar
`verifyOtp` em vez de `exchangeCodeForSession` nesse fluxo especificamente. Login por senha e
cadastro por e-mail/senha não são afetados — é só o caminho de recovery (e, pelo mesmo motivo,
provavelmente o de confirmação de cadastro) que precisa migrar.

## Duas hipóteses para a causa raiz (histórico da investigação, já resolvidas acima)

O fluxo usa PKCE por padrão (`@supabase/ssr` define `flowType: "pkce"` em
`createBrowserClient`/`createServerClient`), o que abre duas possibilidades:

1. **URL de redirect não cadastrada no Supabase para o ambiente testado.** Painel do projeto
   (`Authentication → URL Configuration`, verificado pelo dono do produto em 2026-08-06):
   `Redirect URLs` tem **exatamente 1 entrada**, `https://vetorpas.com.br/auth/callback` — nada
   para `localhost`. Se o teste que reproduziu o defeito foi feito rodando `npm run dev`
   (`http://localhost:3000/auth/callback`), o GoTrue rejeita esse `redirect_to` por não bater com
   a allowlist e cai no `Site URL` padrão — o que bate com "voltou pro login" sem erro visível.
   **Se o teste foi em produção (`vetorpas.com.br`), esta hipótese cai** — confirmar em qual
   ambiente o defeito foi reproduzido antes de investigar mais.
2. **`code_verifier` do PKCE ausente no navegador que abriu o link.** O `code_verifier` fica
   salvo no navegador que *pediu* a redefinição; se o link do e-mail foi aberto num
   navegador/dispositivo diferente (comum: pedir no Chrome, abrir o e-mail e clicar num
   navegador in-app do cliente de e-mail), `exchangeCodeForSession` falha por não achar o
   verifier — mesmo sintoma, causa diferente.

Não foi possível testar ao vivo nesta rodada (sem senha da conta usada no teste, sem acesso ao
painel do Supabase para forçar/inspecionar o resultado do `exchangeCodeForSession`).

## O que resolver

1. Migrar o fluxo de recovery (e provavelmente o de confirmação de cadastro, mesmo problema) de
   `exchangeCodeForSession(code)` para `verifyOtp({ token_hash, type: "recovery" | "signup" })` —
   isso muda o `redirectTo` passado a `resetPasswordForEmail`/`signUp` (ou o template de e-mail no
   painel do Supabase, dependendo de qual `{{ .ConfirmationURL }}` está configurado hoje) para
   carregar `token_hash` em vez de depender de `code`, e a rota de callback a processar de acordo.
2. `/auth/entrar` deveria mostrar uma mensagem quando `?error=auth_callback_failed` estiver
   presente ("O link expirou ou é inválido — peça um novo"), em vez de falhar em silêncio — vale
   manter mesmo depois da migração, como rede de segurança para qualquer falha futura desse link.

**Blocked by:** Nenhum — pode começar imediatamente. Achado durante a investigação do ticket 18,
mas é um defeito à parte.

**Status:** ready-for-agent

- [x] Causa raiz confirmada — PKCE `code_verifier` ausente ao abrir o link num WebView diferente
      do navegador que pediu o reset (confirmado: teste em produção, link aberto no app do Gmail)
- [ ] Clicar no link de redefinição de senha, do e-mail até o formulário de nova senha, funciona
      ponta a ponta em produção — inclusive abrindo o link num app de e-mail (Gmail, Outlook),
      não só copiando a URL para o mesmo navegador
- [ ] `/auth/entrar` mostra uma mensagem quando chega com `?error=auth_callback_failed`, em vez de
      falhar em silêncio
- [ ] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
