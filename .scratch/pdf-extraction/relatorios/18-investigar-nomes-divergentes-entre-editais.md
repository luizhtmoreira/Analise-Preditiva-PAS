# Relatório — Ticket 18: Investigar nomes divergentes entre Editais

**Ticket:** `.scratch/pdf-extraction/issues/18-investigar-nomes-divergentes-entre-editais.md`
**Status:** concluído (investigação — nenhum código alterado)
**Onde vivem as evidências brutas:** este relatório não reproduz nome real de Aluno (convenção
do ticket 13, §3.4) — os 10 casos são referenciados por inscrição, Edital e página; quem
precisar do texto exato pode reabrir `data/pdfs/...` nas páginas citadas.

---

## 1. O que foi pedido

Classificar, um a um, os 10 casos em que a mesma inscrição aparece com nome divergente em
Editais diferentes do mesmo triênio (achado do ticket 08), depois do ticket 13 (reparo de nome
quebrado por espaço) já ter rodado sobre o corpus inteiro — para decidir se o achado está
inteiramente explicado por aquele defeito ou se sobra causa raiz distinta.

## 2. Ponto de partida: o próprio ticket 13 já tinha respondido a pergunta

`relatorios/13-reparo-nome-quebrado-por-espaco.md`, §4, registra: *"A reconciliação cruzada
entre Editais (10 inscrições com nome divergente) não muda com este ticket —
`schema.canonizar` já removia espaço na comparação, então um nome quebrado por espaço sempre
canonizava igual à versão intacta; o defeito era invisível a essa camada antes e depois."*

Ou seja: a hipótese que motivou este ticket (item 6 de `defeitos-pendentes.md`, "candidato a
ser inteiramente explicado pelo ticket 13") já estava refutada antes deste ticket começar —
`reconciliar_nomes` compara `canonizar(nome)`, e `canonizar` já ignora espaço, então uma
palavra quebrada por espaço nunca poderia ter causado uma dessas 10 divergências. O trabalho
deste ticket passou a ser puramente forense: achar a causa raiz real de cada um dos 10.

Confirmado empiricamente: o relatório de validação em `saida-nova/` (gerado em 2026-08-05,
já com o reparo do ticket 13 ativo — cabeçalho reporta 1.865 nomes reparados) lista exatamente
os mesmos 10 casos, com os mesmos textos divergentes.

## 3. Método

Os 10 pares de nome foram lidos em `saida-nova/resultado_final.csv` e `saida-nova/convocacao.csv`
(não da tabela do relatório — ver nota na seção 5.2) e, para os casos não-triviais, o texto
bruto da página do PDF de origem foi reextraído diretamente (`pypdf`, mesmo `extraction_mode`
que o extrator de produção usa por Família — `plain` para Resultado Final, `layout` para
Convocação) para confirmar se a string capturada bate com o que está impresso no Edital.

## 4. Classificação dos 10 casos

| Inscrição | Padrão | Confirmado no PDF fonte? | Classificação |
|---|---|---|---|
| 16108039 | sobrenome com acento (`"...á"`) sai correto no Resultado Final; a mesma pessoa na Convocação aparece com as duas letras finais trocadas e sem acento | Sim — reextraído da Convocação, o texto bruto já traz a forma corrompida | (b) causa raiz distinta |
| 16109359 | idêntico ao anterior | Sim | (b) |
| 16126338 | idêntico ao anterior | Sim | (b) |
| 16131344 | idêntico ao anterior | Sim | (b) |
| 16148432 | idêntico ao anterior | Sim | (b) |
| 16151517 | idêntico ao anterior (aparece em 2 Convocações diferentes, ambas corrompidas do mesmo jeito) | Sim | (b) |
| 17196132 | idêntico ao anterior | Sim | (b) |
| 16105688 | sobrenome completamente diferente entre Resultado Final e Convocação (não é ruído de caractere — são palavras diferentes) | Sim — os dois lados batem, cada um, com o texto bruto do seu próprio PDF | (b) |
| 16116602 | Resultado Final tem um sobrenome a mais no fim do nome do que a Convocação | Sim — a Convocação, reextraída, realmente não tem essa palavra | (b) |
| 21177086 | uma letra duplicada num sobrenome, num dos dois Editais | Sim — reextraído, o Edital mais antigo já traz a letra duplicada | (b) |

**Nenhum dos 10 casos é (a) resolvido pelo reparo do ticket 13.** Todos são (b), e por uma razão
comum: em nenhum dos 10 casos o parser (`pas_extraction`) introduziu a divergência —
reextraindo o texto bruto de cada PDF envolvido, a string capturada pelo pipeline bate,
byte a byte, com o que está impresso no Edital. A divergência já existe **na fonte**, antes de
qualquer processamento nosso.

## 5. Achados por grupo

### 5.1 Sete casos: mesmo padrão de corrupção, sempre do lado da Convocação

Os sete primeiros casos da tabela compartilham uma assinatura idêntica: o sobrenome termina
com uma sílaba acentuada (ex.: padrão `"...á"`); no Resultado Final esse sobrenome sai correto
(sem acento, mas na ordem certa de letras); na Convocação do mesmo triênio, a mesma pessoa
aparece com as duas letras finais **trocadas de posição** e em maiúsculas — ex.: onde o
Resultado Final tem `"...sa"`, a Convocação correspondente tem `"...as"`.

Reextraindo o texto bruto de 4 PDFs de Convocação diferentes (Ed_33, Ed_34, Ed_37, Ed_38 —
triênios 2016/2018 e 2017/2019), a forma corrompida já está lá, linha a linha, na posição
alfabética certa da listagem — não é o parser reordenando nada; é o texto que o `pypdf` extrai
do PDF. Como o mesmo padrão se repete idêntico em 4 arquivos independentes, de anos diferentes,
sempre no mesmo tipo de sílaba final e sempre do lado da Convocação (nunca do Resultado Final),
a hipótese mais provável é um defeito no processo de geração desses PDFs pela Cebraspe (ex.:
fonte ou codificação que não trata bem aquele caractere acentuado nesse template específico de
documento) — não um defeito de extração nosso. Não há texto adicional a recuperar: o dado
correto simplesmente não está presente no PDF de Convocação.

### 5.2 Nota lateral: a tabela do relatório de validação pareia nome e origem incorretamente

Ao conferir o caso 16108039 contra o texto da tabela em `saida-nova/relatorio_validacao.md`
(seção 6), a ordem dos dois nomes na coluna "Nomes encontrados" pareceu inicialmente invertida
em relação à coluna "Origem". Causa: `reconciliacao.reconciliar_nomes` monta `nomes` com
`tuple(sorted({...}))` — ordem alfabética das strings — enquanto `proveniencias` preserva a
ordem de inserção (Resultado Final antes de Convocação). As duas colunas **não são arrays
paralelos**; um leitor humano tende a assumir que são. Não afeta nenhum dado (os CSVs de saída
estão corretos, foi só a leitura da tabela que confundiu por um momento durante esta
investigação) — mas vale registrar como possível ajuste de clareza no relatório, caso o pacote
volte a ser mexido (não aberto como ticket: é só um detalhe de exibição, sem impacto em dado).

### 5.3 Caso 16105688: os dois Editais discordam de verdade

Reextraindo os dois PDFs (Resultado Final e Convocação, mesmo triênio), cada lado bate
exatamente com o que está impresso no seu próprio Edital — mas os dois Editais têm sobrenomes
diferentes para a mesma inscrição. Não há nada para o parser reconciliar aqui: a fonte é
inconsistente consigo mesma.

### 5.4 Caso 16116602: um Edital tem um sobrenome a mais que o outro

O Resultado Final grafa o nome completo com um sobrenome extra no final; a Convocação do mesmo
triênio, reextraída, genuinamente não traz essa palavra — a linha termina antes dela (a lista de
Convocação está em uma única linha por candidato, sem quebra, e o texto simplesmente para ali).
Sem outra fonte para conferir qual dos dois é o nome completo real, fica registrado como
divergência de fonte, não como perda de texto pelo pipeline.

### 5.5 Caso 21177086: os dois registros nem são do mesmo triênio

Este par vem de dois Editais de **triênios diferentes** (2021/2023 e 2022/2024) — a
reconciliação cruzada (`reconciliacao.reconciliar_nomes`) casa por inscrição em todos os grupos
passados, sem restringir por triênio. Não se sabe, a partir dos dados disponíveis, se são a
mesma pessoa prestando o PAS em dois triênios diferentes (inscrição reaproveitada) ou uma
coincidência de número entre duas pessoas — o nome quase idêntico (uma letra duplicada de
diferença) sugere a primeira hipótese, mas não é conclusivo. De todo modo, a letra duplicada já
está no texto bruto do Edital mais antigo — não é o parser duplicando nada.

## 6. Conclusão

Os 10 casos permanecem 10 casos depois do ticket 13 — confirmando o que o próprio relatório do
ticket 13 já previa. Nenhum é um defeito do `pas_extraction`: em todos os 10, o texto extraído
bate exatamente com o texto impresso no PDF de origem, verificado por reextração direta. A
causa raiz — quando dá para identificar uma — está nos Editais da Cebraspe (um padrão de
corrupção de acento na geração dos PDFs de Convocação, em 7 dos 10 casos; inconsistência ou
typo do próprio documento oficial nos outros 3), não em nada que o pipeline faça. Não há
correção de código a fazer: não existe segundo dado, dentro do que o Edital disponibiliza, para
decidir qual grafia está certa.

**Item 6 de `defeitos-pendentes.md` fechado por esta investigação** — substituído pelo item 7,
que registra esta conclusão (ver arquivo).
