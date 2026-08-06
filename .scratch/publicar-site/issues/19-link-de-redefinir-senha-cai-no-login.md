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

## Duas hipóteses para a causa raiz (não confirmadas)

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

1. Confirmar qual das duas hipóteses é a causa raiz (ambiente do teste é o primeiro dado que
   falta).
2. Se for a hipótese 1: cadastrar a URL de dev (`http://localhost:3000/auth/callback`) na
   allowlist do Supabase, se o objetivo é também testar esse fluxo localmente.
3. Independente da causa: `/auth/entrar` deveria mostrar uma mensagem quando
   `?error=auth_callback_failed` estiver presente ("O link expirou ou é inválido — peça um novo"),
   em vez de falhar em silêncio — isso não corrige a causa raiz, mas transforma qualquer falha
   futura desse link (inclusive uma terceira causa ainda não cogitada) em algo diagnosticável pelo
   próprio usuário, sem precisar abrir o console.

**Blocked by:** Nenhum — pode começar imediatamente. Achado durante a investigação do ticket 18,
mas é um defeito à parte.

**Status:** ready-for-agent

- [ ] Causa raiz confirmada (ambiente do teste + qual das duas hipóteses, ou uma terceira)
- [ ] Clicar no link de redefinição de senha, do e-mail até o formulário de nova senha, funciona
      ponta a ponta em produção
- [ ] `/auth/entrar` mostra uma mensagem quando chega com `?error=auth_callback_failed`, em vez de
      falhar em silêncio
- [ ] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
