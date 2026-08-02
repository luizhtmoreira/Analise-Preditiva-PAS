# 17 — Cota devia ser atributo declarado do Aluno, não subproduto de cálculo

**What to build:** a cota de concorrência do Aluno vira um dado que ele **declara**, como a
escola — não um valor que muda sozinho toda vez que ele roda uma simulação.

## O defeito

Hoje `cota` é gravada em `alunos_perfis` como **efeito colateral de calcular**: toda vez que o
Aluno logado clica em "calcular" no Preditor ou na Calculadora, o formulário inteiro — incluindo
a cota que estava selecionada naquele momento — é gravado via `upsert`
(`CalculadoraPage.tsx:348`, `PreditorPage.tsx:746`). Não existe um botão "salvar minha cota":
ela é sobrescrita a cada cálculo, junto das notas digitadas.

**Compare com `escola`.** `escola` mora em outro lugar (`user_metadata.escola`, nos metadados de
autenticação do Supabase, não em `alunos_perfis`) e só muda numa tela dedicada —
`app/perfil/page.tsx` (`PerfilAlunoClient.tsx`) — onde o Aluno escolhe a escola num combobox e
clica em salvar. É uma declaração intencional, isolada de qualquer simulação.

**A consequência prática.** Se um Aluno de Cota para Negros abre a Calculadora só para comparar
"e se eu fosse Universal?", rodar esse cálculo grava "Sistema Universal" como se fosse a cota
real dele — sobrescrevendo o dado verdadeiro em silêncio. Isso provavelmente explica por que,
hoje, **toda linha em `alunos_perfis` está em "Sistema Universal"**: não é que ninguém declarou
cota, é que nenhuma simulação com cota diferente do padrão "gruda" — a próxima simulação (ou o
padrão da tela, que é sempre Universal) sobrescreve de novo.

**Confirmado no banco em 2026-08-02** (dono do produto, olhando os dados reais): não há hoje
nenhum registro de cota diferente de Universal — coerente com a hipótese acima, não uma
observação de acaso.

## O que resolver

Duas coisas que hoje são uma só precisam virar duas:

1. **Cota declarada do Aluno** — o valor persistente, editado numa tela de declaração (a mesma
   tela de perfil que já tem `escola`, ou uma seção nova nela). É o que o Preditor e a
   Calculadora usam como padrão ao carregar.
2. **Cota selecionada nesta simulação** — o que o Aluno escolhe no `<select>` da tela para
   explorar um cenário hipotético ("e se eu fosse X?"). Isso **não pode** sobrescrever a cota
   declarada — é read do padrão, não write de volta.

A pergunta de forma que este ticket precisa responder: `cota` fica em `alunos_perfis` (editada só
a partir do perfil, nunca do submit do cálculo) ou migra para `user_metadata` junto de `escola`?
As duas resolvem o defeito; a decisão é sobre consistência com o resto do dado do Aluno, não
sobre corrigir o bug.

**Sem risco de migração de dado:** como todo registro hoje está em "Sistema Universal" (o
default, nunca uma declaração real), não há valor legítimo para preservar na troca de esquema.

**Blocked by:** Nenhum — pode começar imediatamente. Achado ao investigar o ticket 16, mas é
independente dele.

**Status:** ready-for-agent

- [ ] `cota` declarada do Aluno vive num único lugar, editável numa tela de declaração explícita
      (perfil), não como efeito colateral de um cálculo
- [ ] Preditor e Calculadora carregam essa cota declarada como valor padrão do seletor
- [ ] Trocar a cota no seletor de uma simulação **não** sobrescreve a cota declarada do Aluno
- [ ] Existe uma ação explícita (na tela de perfil) para o Aluno atualizar a cota declarada
- [ ] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
