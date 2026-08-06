# 18 — Salvar escola no perfil não persiste

**What to build:** o Aluno troca a escola em `/perfil`, clica em salvar, e a mudança **fica** —
hoje, segundo relato do dono do produto, ela não pega no banco.

## O que se sabe

O caminho de código, em `components/profile/PerfilAlunoClient.tsx:33-50`
(`handleSaveEscola`), está estruturalmente correto na leitura:

```ts
const { error } = await supabase.auth.updateUser({
  data: { escola: escola.trim() },
});
if (!error) {
  setEscolaSavedMsg("Escola atualizada com sucesso!");
  ...
}
```

Isto é: chama `supabase.auth.updateUser` (que grava em `user_metadata`, não em
`alunos_perfis` — ver ticket 17, que documenta esse mesmo padrão para `escola`), e só mostra a
mensagem de sucesso se não vier erro. Não achei, por inspeção, o motivo do defeito — este ticket
é para **reproduzir e diagnosticar**, não corrigir um ponto já identificado.

**Cuidado com falso positivo antes de investigar o código:** `escola` não é uma coluna de
`alunos_perfis` — ela vive em `auth.users` → `raw_user_meta_data` (metadados de autenticação do
Supabase), uma tabela diferente da que provavelmente é olhada primeiro num client SQL. Confirmar
onde a verificação foi feita antes de assumir que o `updateUser` falhou de verdade.

## Como reproduzir e diagnosticar

1. Logar como Aluno, abrir `/perfil`, trocar a escola, salvar.
2. Conferir se a mensagem "Escola atualizada com sucesso!" aparece (indica que `error` veio
   `null` — ou seja, o Supabase Auth respondeu sem erro).
3. Se a mensagem aparece mas o valor não muda: o `updateUser` está retornando sucesso mas não
   persistindo (sessão desatualizada, JWT não sendo refeito, ou o valor sendo lido de um cache
   antigo em algum outro lugar que exibe `escola`) — abrir o painel do Supabase Auth e olhar
   `raw_user_meta_data` do usuário de teste diretamente, não uma tabela derivada.
4. Se a mensagem não aparece (ou aparece um erro silencioso): o problema está antes, no próprio
   `updateUser` — checar RLS, políticas de Auth, ou o client do Supabase sendo instanciado sem a
   sessão certa.

**Blocked by:** Nenhum — pode começar imediatamente. Achado durante a investigação do ticket 17,
mas é um defeito à parte (persistência, não modelo de dado).

**Status:** diagnosticado — 2026-08-06, relatório em
`.scratch/publicar-site/relatorios/18-salvar-escola-no-perfil-nao-persiste.md`. Causa raiz: não é
o `updateUser()` que falha — `alunos_perfis` tem uma coluna `escola` órfã (nenhum código escreve
nela) que é onde o dono do produto checou; `raw_user_meta_data.escola` é o lugar certo e o código
grava lá corretamente. Falta só a confirmação final no painel (`Authentication → Users`, não
`Table Editor`), que ficou com o dono do produto.

- [x] O defeito está reproduzido e a causa raiz identificada (não só contornada)
- [ ] Trocar a escola em `/perfil` reflete em `raw_user_meta_data` no Supabase, verificado
      diretamente no painel — não presumido pela mensagem de sucesso da tela
- [x] Diagnóstico documentado por leitura de código (não há teste de integração com Supabase Auth
      no repo, e não foi possível logar com conta de teste real nesta rodada — e-mail de
      confirmação obrigatório e envio de e-mail falhando no projeto)
- [x] `pytest tests/`, `eslint` e `tsc --noEmit` verdes (nenhum código alterado nesta rodada)
