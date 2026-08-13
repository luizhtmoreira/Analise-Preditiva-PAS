# Relatório — Ticket 17: cota devia ser atributo declarado do Aluno, não subproduto de cálculo

**Ticket:** `.scratch/publicar-site/issues/17-cota-devia-ser-atributo-declarado-do-aluno.md`
**Status:** implementado — `pytest`, `eslint`, `tsc --noEmit` e `next build` verdes; **não**
verificado ao vivo em produção (sem credencial de Aluno de teste disponível nesta sessão)
**Onde vive o trabalho:** `landing-page/components/profile/PerfilAlunoClient.tsx`,
`landing-page/components/public/calculadora/CalculadoraPage.tsx`,
`landing-page/components/public/predict/PreditorPage.tsx`

---

## 1. Decisão de forma

O ticket deixava em aberto se `cota` ficaria em `alunos_perfis` (write-gated ao perfil) ou
migraria para `user_metadata`, junto de `escola`. **O dono do produto decidiu: `user_metadata`,**
espelhando `escola` byte a byte — mesmo lugar, mesma tela de edição, mesmo padrão de
`updateUser`. Decisão tomada antes de codar, não inferida do repositório.

Isso também evita recriar a armadilha do ticket 18: `alunos_perfis.escola` é hoje uma coluna
órfã (nada escreve nela, o valor real vive em `user_metadata`). Se `cota` tivesse ficado em
`alunos_perfis`, o próximo passo natural seria a mesma pergunta de novo daqui a duas rodadas.

## 2. O que foi implementado

1. **`PerfilAlunoClient.tsx`** — novo card "Sistema de Concorrência", espelho do card de
   Escola: `<select>` com os 10 sistemas de `lib/cotas.ts`, `handleSaveCota` grava via
   `supabase.auth.updateUser({ data: { cota } })`, botão desabilitado até uma cota conhecida
   ser escolhida.
2. **`CalculadoraPage.tsx` e `PreditorPage.tsx`** — o efeito de carregamento passa a ler
   `user.user_metadata?.cota` (não mais `profile.cota` de `alunos_perfis`) como valor padrão do
   seletor; o aviso `cotaSalvaDesconhecida` dispara da mesma forma se o valor declarado não bater
   com nenhum dos 10 sistemas atuais.
3. **O mesmo par de arquivos** — `cota` saiu do payload do `upsert` em `alunos_perfis` que roda a
   cada cálculo. O `<select>` da simulação continua livre para explorar qualquer cenário
   hipotético sem tocar no valor declarado — a separação que o ticket pediu.
4. A coluna `alunos_perfis.cota` deixou de ser lida e escrita em qualquer lugar do repo (grep
   confirma). Ela não é apagada do banco — só passa a não ser mais tocada pelo código, igual
   `escola` desde o ticket 18.

## 3. Verificação

- `pytest tests/` — 486 passed (backend nunca tocou `alunos_perfis`; nenhum teste é relevante a
  este ticket, todos passavam antes também).
- `npx eslint` nos 3 arquivos — 0 erros/avisos novos (os 2 erros e alguns avisos que aparecem já
  existiam antes desta mudança, confirmado revertendo o diff e rodando de novo).
- `npx tsc --noEmit` — limpo.
- `npm run build` (Next.js) — build de produção completo sem erros, `/perfil` gera normalmente.
- **Não testado ao vivo**: não há credencial de Aluno de teste nesta sessão para logar, salvar
  uma cota no perfil e confirmar que ela sobrevive a uma simulação com cota diferente. Recomendo
  um teste manual de 2 minutos antes de considerar fechado — mesmo padrão que o ticket 19 só
  revelou defeitos reais ao testar ao vivo.

## 4. Achado que não bloqueou o commit, mas vale checar

Os dois `upsert` em `alunos_perfis` agora **não enviam mais `cota`**, inclusive na primeira
inserção de um Aluno novo (antes, o valor default do estado local, `"Sistema Universal"`, sempre
viajava junto). Se a coluna `alunos_perfis.cota` tiver `NOT NULL` **sem default** no schema do
Supabase, o primeiro cálculo de um Aluno novo pode falhar no insert. Não consegui inspecionar o
schema sem a `service_role` key (a anon key não expõe o OpenAPI da tabela). Dado que hoje toda
linha já está em "Sistema Universal", é provável que exista um default — mas é uma checagem de
30 segundos no painel do Supabase que vale fazer antes de dar como fechado.

## 5. Checklist do ticket

- [x] `cota` declarada do Aluno vive num único lugar (`user_metadata`), editável numa tela de
      declaração explícita (perfil), não como efeito colateral de um cálculo
- [x] Preditor e Calculadora carregam essa cota declarada como valor padrão do seletor
- [x] Trocar a cota no seletor de uma simulação não sobrescreve a cota declarada do Aluno —
      `cota` saiu dos dois `upsert`, e nenhum outro `auth.updateUser` existe fora do perfil
- [x] Existe uma ação explícita (tela de perfil) para o Aluno atualizar a cota declarada
- [x] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
- [ ] Verificação ao vivo (login real → salvar cota → simular com cota diferente → cota
      declarada intacta) — pendente, ver §3
- [ ] Confirmar no painel do Supabase se `alunos_perfis.cota` tem default — ver §4
