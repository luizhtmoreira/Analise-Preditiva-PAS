# 08a — Derivado de Deploy: o dado que sai do disco não tem nome de Aluno

**What to build:** o que a API hospedada consome é um **Derivado de Deploy** — reduzido às colunas
que ela de fato lê, sem nenhuma coluna que identifique um Aluno pelo nome. O bruto continua
existindo, como backup explícito, e nunca é embarcado.

Hoje o `resultado_final.csv` (24,2 MB, 45 colunas, 66.313 Alunos reais) e o `notas_corte.csv` vão
inteiros para dentro da imagem, com a coluna `nome`. O código já é cuidadoso — o `usecols` de
`gestao_service` e `analytics_service` nunca lê `nome` — mas **`usecols` protege a leitura, não o
arquivo**. A fronteira certa é a mesma que este repo já adota em `src/pas_extraction/`: o dado não
entra, em vez de entrar e a gente lembrar de não olhar (ADR-0014).

Medido: **24,2 MB → 4,5 MB** com as 10 colunas lidas; 4,1 MB já filtrado por `checksum_fecha`.

**O Domicílio Versionado passa a ter dois papéis separados.** Hoje `Luiz1912/vetor-pas-dados`
acumula backup dos CSVs de extração **e** insumo de deploy — cortar na origem destruiria o backup.
Passa a haver o cru privado (backup por decisão, ninguém baixa em build) e o Derivado (o que o
Ponteiro aponta). Mesma lição da invariante dos parsers no `CLAUDE.md`.

**Só a metade `dados` do Ponteiro muda.** `models/pas3/` não tem PII e não é derivado.

**A lista de colunas precisa nascer com dono.** Ela hoje existe duplicada nos dois serviços que
leem os CSVs. Se virar três lugares, um ticket futuro que passe a ler uma coluna nova quebra em
produção com `KeyError`, e nada liga esse erro ao script de publicação. Esse é o mesmo defeito que o
code review do ticket 08 já corrigiu uma vez, quando fez o `ponteiro.json` virar fonte única dos
nomes de arquivo de cada artefato. **Prefactor antes de publicar qualquer coisa.**

**Não é escopo:** publicar o Derivado num repositório público. Foi autorizado pelo dono do produto
como **plano B** e continua sendo — mas publicar é irreversível, e enquanto o caminho privado
funcionar de graça, gastar essa opção não compra nada (ADR-0014).

## Nota para quem implementar

Cortar coluna é trivial. **O que não é trivial é onde a fonte única da lista vai morar** — isso é
decisão de seam, não faxina.

O modo de falha esperado aqui é entregar a lista em dois lugares com um comentário dizendo "manter
em sincronia". **Isso é reprovação**, não uma solução aceitável com ressalva: é literalmente o
defeito que este ticket existe para consertar. Duas cópias com um aviso continuam sendo duas cópias,
e o `KeyError` chega em produção do mesmo jeito — só que agora com um comentário atestando que
alguém sabia.

Se o lugar certo para a lista não for óbvio, isso é motivo para **perguntar**, não para duplicar.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] A lista de colunas do Derivado é **fonte única em código**, lida tanto por quem publica quanto
      pelos serviços que consomem — não duplicada
- [ ] O Derivado não contém `nome`, em nenhum dos dois CSVs — verificado sobre o arquivo publicado,
      não sobre a intenção do script
- [ ] O cru continua existindo num repositório próprio, privado, que **nenhuma etapa de build lê**
- [ ] O Ponteiro aponta para as revisões do Derivado; a metade `modelo` fica intacta
- [ ] A API sobe contra o Derivado e responde igual: `/health`, `/api/predict` com previsão real e
      `modelo_disponivel: true`, e a Análise Temporal com os mesmos números de antes
- [ ] Reverter continua sendo voltar o Ponteiro — documentado em passos executáveis
