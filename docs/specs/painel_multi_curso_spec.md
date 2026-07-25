# Especificação Técnica: Painel Multi-Curso & Soft Gate

> **Status:** Planejado / Roteiro de Implementação  
> **Domínio:** Aluno Cadastrado | Vetor PAS  
> **Referência de Domínio:** [CONTEXT.md](../../CONTEXT.md)

---

## 1. Visão Geral

O **Painel Multi-Curso** é uma funcionalidade exclusiva para **Alunos Cadastrados**. Ele permite que o aluno salve seu histórico de notas (PAS 1, PAS 2 e Redação) uma única vez e acompanhe simultaneamente suas probabilidades de aprovação e o cálculo de **Quanto Falta** para múltiplos cursos da Universidade de Brasília (UnB).

O mecanismo de conversão para transformar visitantes em Alunos Cadastrados é o **Soft Gate**.

---

## 2. Fluxo do Soft Gate & Transição de Cadastro

### 2.1 Experiência do Visitante (Deslogado)
1. O visitante acessa `/predict`, escolhe o 1º curso alvo, digita suas notas e clica em **"Calcular minha previsão"**.
2. O sistema exibe o resultado preditivo completo para o 1º curso.
3. No final da tela, é exibido o card de incentivo com o botão **"+ Adicionar curso"**.
4. Ao clicar em **"+ Adicionar curso"**, abre-se a modal do **Soft Gate**:
   - Explicita o benefício de comparar múltiplos cursos simultaneamente.
   - Oferece os botões *"Criar conta gratuita"* (`/auth/cadastro?next=/predict`) e *"Já tenho conta — entrar"* (`/auth/entrar?next=/predict`).

### 2.2 Preservação do Curso Alvo no Login/Cadastro
- Antes de redirecionar o visitante para a tela de login ou cadastro, o frontend deve salvar temporariamente no `localStorage` (ex: chave `vetor_pas_draft_simulation`):
  - `curso_alvo` (nome do curso, cota, triênio)
  - `notas` (PAS 1 P1/P2/Red, PAS 2 P1/P2/Red)
- Após a confirmação da autenticação (`/auth/callback` ou pós-login em `/auth/entrar`), se existir um rascunho em `vetor_pas_draft_simulation`:
  1. O sistema vincula automaticamente o `curso_alvo` à lista de cursos salvos da conta do aluno.
  2. Limpa o rascunho do `localStorage`.
  3. Redireciona o **Aluno Cadastrado** diretamente para o **Painel Multi-Curso** com seu curso preservado.

---

## 3. Comportamento do Painel Multi-Curso (Aluno Cadastrado Logado)

### 3.1 Interface quando Logado
1. Quando o usuário autenticado como **Aluno Cadastrado** acessa `/predict`:
   - Se ele já possui cursos salvos, o sistema carrega suas simulações automaticamente.
   - O botão **"+ Adicionar curso"** **NÃO abre a modal de login**.
   - Em vez disso, o botão abre um seletor inline (Combobox de Cursos) para adicionar um 2º, 3º ou N-ésimo curso.

### 3.2 Exibição Comparativa das Simulações
Para cada curso salvo na lista do aluno, o painel renderiza um card comparativo contendo:
- **Identificação do Curso:** Nome, Campus, Turno e Cota.
- **Probabilidade de Aprovação:** Semáforo de Risco (Baixo, Médio, Alto).
- **Nota de Corte Prevista / Histórica.**
- **Quanto Falta:** Escore Bruto (EB) e P2 mínima necessária no PAS 3 para alcançar a nota de corte.
- **Ações:** Opção de excluir o curso da comparação ou torná-lo o curso principal.

---

## 4. Modelagem de Dados & Persistência (Supabase)

### Tabela Sugerida: `aluno_cursos_salvos`
```sql
CREATE TABLE public.aluno_cursos_salvos (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    curso VARCHAR(255) NOT NULL,
    cota VARCHAR(100) DEFAULT 'Sistema Universal',
    trienio VARCHAR(20) DEFAULT '2024-2026',
    semestre VARCHAR(10) DEFAULT '1°',
    is_primary BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Policy de segurança RLS (Row Level Security)
ALTER TABLE public.aluno_cursos_salvos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Alunos gerenciam apenas seus próprios cursos salvos"
ON public.aluno_cursos_salvos
FOR ALL USING (auth.uid() = user_id);
```

---

## 5. Roteiro de Implementação Futura (Passo a Passo)

1. **[BD/Supabase]:** Criar a tabela `aluno_cursos_salvos` com suporte a RLS por `auth.uid()`.
2. **[Frontend - Draft Storage]:** Adicionar helper para salvar `curso_alvo` e notas no `localStorage` ao acionar o Soft Gate.
3. **[Frontend - Pós Login Handler]:** No callback de auth ou `PublicHeader`/`PreditorPage`, verificar e sincronizar o rascunho com o banco.
4. **[Frontend - Componente Multi-Curso]:** Atualizar `PreditorPage.tsx` para carregar e alternar múltiplos cursos quando `isLoggedIn` for verdadeiro, permitindo adicão inline sem acionar a modal de login.
