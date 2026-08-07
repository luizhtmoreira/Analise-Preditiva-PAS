# Defeitos pendentes conhecidos — pipeline `pas_extraction`

Registro consolidado dos defeitos **documentados e ainda não corrigidos** no pipeline de
extração dos Editais PAS/UnB (`src/pas_extraction/`). Não substitui os relatórios de ticket
em `relatorios/` — cada entrada abaixo aponta para a fonte original e resume o que falta
fazer. Objetivo: dar um ponto único de partida para decidir o próximo ticket, em vez de
precisar reler os 12 relatórios inteiros.

Convenção: cada entrada tem **Onde foi encontrado**, **O que falta fazer** e **Severidade**
(estimativa de impacto nos dados/produto, não de esforço).

## 1. Nome quebrado por espaço inserido no meio da palavra

**Onde foi encontrado:** achado ao comparar `resultado_final.csv` (saída nova) contra
`data/banco_alunos_pas_final.csv` (base antiga, ad-hoc) por `(inscrição, triênio)` —
2026-07-26. Havia um indício isolado anterior em `scripts/NOTES.md`, seção 9 (o exemplo
`"Daniela F erreira Miguel Pereira"`, citado ali como 1 dos 3 casos que sobraram do checksum
de língua por etapa), mas nunca foi tratado como classe de defeito com impacto medido — só
como anedota.

**O defeito:** o extrator novo insere um espaço dentro de uma palavra do campo `nome` (ex.:
`"Isabella"` → `"Isabell a"`, `"Sousa"` → `"Sou sa"`, `"Arthur"` → `"Arth ur"`), ou ocasionalmente
um espaço duplicado sem quebra de palavra. Medido cruzando os 57.118 pares casáveis entre as
duas bases:

- **1.797 nomes divergem** entre as duas bases (3,1% dos pares casados).
- **1.796 dessas 1.797 (99,9%) são diferenças *puramente* de espaço em branco** — mesmas
  letras, mesma ordem, confirmado normalizando os dois lados com `re.sub(r"\s+", "", s)` e
  comparando.
- Dessas 1.796: **1.690 são quebra de palavra real** (uma palavra virou duas), **106 são
  espaço duplicado** (sem quebra de palavra).
- **Em 100% dos 1.796 casos é o lado NOVO que tem a palavra quebrada/espaço extra** — 0 casos
  em que o defeito estivesse do lado antigo. Não é ruído simétrico dos dois parsers; é uma
  característica específica do extrator novo.
- Taxa sobre a base inteira: **1.796 / 66.313 registros do `resultado_final.csv` = 2,71%**.
- Ocorre nos **8 de 8 Editais de Resultado Final** do corpus (proporcional ao volume de cada
  um), não concentrado em um PDF específico — não é artefato de um arquivo isolado.

**Hipótese de causa raiz:** mesma classe de corrupção já catalogada no protótipo
(`scripts/NOTES.md`, "ARMADILHA B(c) — números partidos por whitespace", e a nota da seção 2
sobre `"abaix o"` no lugar de `"abaixo"`) — o `pypdf`/`pdfplumber` ocasionalmente injeta um
espaço espúrio no meio de um token durante a extração de texto. O ticket 02 deu tratamento a
essa corrupção **só para os 9 campos numéricos** (`_formato_numerico_valido` detecta e o
reparo de números partidos entra em cena — ver ticket 04, achado "todos os 758 registros
reparados fecham o checksum"). O campo `nome` nunca recebeu o equivalente: `convocacao.py`
colapsa espaços duplicados (`_ESPACOS_RE.sub(" ", nome.strip())`) mas isso não resolve o caso
de quebra de palavra (`"Isabell a"` continua com um espaço "válido"), e `resultado_final.py`
não parece ter nem esse tratamento parcial.

**Por que passa pelas validações existentes sem ser pego:**
- `campos_formato_invalido` só valida os 9 campos numéricos (ticket 02).
- O checksum do Argumento Final (ticket 04) é cego a isso — o nome não entra na conta.
- `ordem_alfabetica_quebrada` (ticket 02) só dispara se a quebra mudar a ordem relativa dos
  nomes dentro do curso; um espaço extra raramente move o nome o suficiente para isso.
- A reconciliação cruzada entre Editais (ticket 08) casa por **inscrição**, não por nome, então
  não é afetada — mas qualquer relatório ou tela que exiba o nome do aluno, é.

**Impacto:** não contamina nenhum valor numérico (score, argumento, nota de corte) — é
isolado ao campo `nome`. Afeta exibição do nome do aluno em relatórios/PDF para o usuário
final e qualquer futura lógica que dependa do nome como chave (nenhuma hoje, pelo que se sabe).

**Status:** não corrigido, sem ticket aberto. Candidato a ticket de follow-up: normalizar
`nome` com uma heurística de reparo de palavra partida (ex.: comparar contra o padrão "duas
palavras curtas coladas por um espaço onde uma delas tem 1-3 letras" — cuidado: nomes
legitimamente compostos por partícula curta, como "de", "da", "e", não devem ser fundidos).

## 2. Validação de formato do campo de classificação — pendente

**Onde foi encontrado:** `relatorios/08-rodada-completa-deterministica.md`, §3.3 (o achado que
explodiu o `resultado_final.csv` para 6,4 GB); reafirmado em
`relatorios/10-notas-de-corte-por-sistema-de-concorrencia.md` ("Follow-up herdado do ticket
08", linhas ~241 e ~382-383).

**O defeito:** o campo de classificação (ranking do aluno em cada Sistema) não passa pela
mesma validação de formato exato que os 9 campos numéricos de nota (`_formato_numerico_valido`,
ticket 02) — é só `_WS.sub("", v)` seguido de `int()`. Um dígito colado (ex. número de página
vazando para o campo, mesma classe de corrupção do defeito 3 abaixo) produz uma posição
implausível (ex. 6 dígitos num curso com ~900 classificados).

**Evidência de que ainda está ativo:** ao comparar `notas_corte.csv` (novo) contra
`data/notas_corte_pas.csv` (antigo) em 2026-07-26, apareceu exatamente esse padrão: MEDICINA,
Darcy Ribeiro, Universal, 2020/2022 → `nota_corte = 199.162,872` (implausível). O próprio
pipeline já marca esse registro com `checksum_fecha=False`, então não é um erro silencioso,
mas o valor absurdo ainda vai para o CSV de saída.

**O que falta fazer:** validação de formato do campo de classificação em
`resultado_final._montar_registro`, simétrica à que já existe para os 9 campos numéricos —
sinalizaria `campos_formato_invalido` no registro específico em vez de só neutralizar o
efeito colateral na camada de buracos (que é o que o `_buracos_por_sistema` faz hoje, com o
limite de plausibilidade `3× observado + 50`).

## 3. 10ª classificação lida como número da página seguinte — pendente

**Onde foi encontrado:** `relatorios/06-deducao-das-cotas-declaradas.md`, §3.

**O defeito:** quando um registro é o último de uma página e seu 22º campo (10ª classificação)
só começa na página seguinte, `resultado_final._separar_registro` lê o número da página no
lugar do valor real, porque o `pypdf` emite o número da página no início do texto de cada
página e o parser trabalha sobre o blob já concatenado, sem noção de fronteira de página.

**Impacto medido:** 8 de 10 casos conhecidos no corpus (66.313 registros) são pegos pela
checagem de fecho de cota e saem marcados com `cota_padrao_suspeito=True` — não silencioso.
Os outros 2 caem em padrões que continuam sendo fecho válido (ex. `{1,9,10}`) e ficam
invisíveis a essa camada.

**O que falta fazer:** dar consciência de fronteira de página a `_separar_registro`
(território dos tickets 01/05). Recomendado como ticket de follow-up próprio — misturar com o
ticket 06 (dedução de cota) esconderia os dois problemas no mesmo commit.

## 4. Ponto cego de posição máxima em `_buracos_por_sistema` — limitação de técnica

**Onde foi encontrado:** `relatorios/02-validacoes-estruturais-por-registro.md`, §3.4.

**O defeito:** `_buracos_por_sistema` infere N (total esperado de candidatos) como
`max(posicoes)`, porque nenhum Edital declara esse número. Se o registro perdido for
justamente o de classificação N (o último do Sistema), `max` encolhe junto com ele e a
checagem não vê buraco nenhum.

**Por que não tem correção prevista:** não existe fonte independente do total real de
candidatos por Sistema nos dados disponíveis — é limitação da técnica, não bug corrigível
sem uma fonte de dado externa (fora do escopo declarado). Documentado e testado
explicitamente (`test_registro_de_posicao_maxima_perdido_e_um_ponto_cego_conhecido`) em vez
de escondido.

## 5. Falta de teste ponta a ponta para Nota de Corte — FECHADO pelo ticket 17

**Onde foi encontrado:** `relatorios/10-notas-de-corte-por-sistema-de-concorrencia.md`,
seção de limitações conhecidas.

**A lacuna:** não existia teste que produzisse uma Nota de Corte a partir de PDFs
ponta-a-ponta. Fixtures de Edital real carregam dado de aluno identificável — a restrição de
[[project_parser_privacy]] impede commitá-las — e por isso a regra de derivação só tinha 41
testes sintéticos diretos (construindo `RegistroResultadoFinal`/`RegistroConvocacao` à mão,
sem passar por PDF).

**Resolvido pelo ticket 17** (`tests/test_pas_extraction_notas_corte_e2e.py`): mesma saída que
o ticket 02 usou para a mesma restrição de privacidade — `fixtures.gerar_pdf_texto_sintetico`
monta dois PDFs (Resultado Final + Convocação, mesmo triênio, 4 inscrições que se cruzam) a
partir de texto **inteiramente inventado**, dentro do próprio teste, sem gravar nada no
repositório nem exigir `data/pdfs` local. O teste roda os extratores reais
(`pipeline.extrair_edital`, `convocacao.extrair_edital_convocacao`) e confere que
`derivar_notas_corte` produz o corte esperado, incluindo o caso de empate entre dois
convocados do mesmo Sistema na mesma chamada (desempate pela menor inscrição).

## 6. 10 inscrições com nome divergente entre Editais diferentes — FECHADO pelo ticket 18

**Onde foi encontrado:** `relatorios/08-rodada-completa-deterministica.md` — reconciliação
cruzada por inscrição entre ~100 mil registros de Editais diferentes.

**Investigado a fundo pelo ticket 18** (`relatorios/18-investigar-nomes-divergentes-entre-editais.md`):
não é o defeito 1 (nome quebrado por espaço) — `schema.canonizar` já ignorava espaço na
comparação, então esse defeito nunca poderia ter causado divergência aqui, mesmo antes do
ticket 13 corrigi-lo. Reextraindo o texto bruto dos PDFs envolvidos nos 10 casos, o pipeline
bate exatamente com o que está impresso em cada Edital — a divergência já existe na fonte, não
é introduzida pelo `pas_extraction`. Sem defeito de código para corrigir. Ver item 7.

## 7. (informativo, sem ação de código) Nomes divergentes entre Editais são um problema da fonte, não do parser

**Onde foi encontrado:** ticket 18, investigando o item 6 (acima).

**O achado:** dos 10 casos de nome divergente entre Editais do mesmo triênio (achado do ticket
08), nenhum é causado por extração — em todos, reextraindo o PDF de origem diretamente, o
texto capturado pelo pipeline bate byte a byte com o que está impresso no Edital.

- **7 de 10** compartilham um padrão idêntico: um sobrenome termina numa sílaba acentuada; sai
  correto no Resultado Final, mas na Convocação do mesmo triênio a mesma pessoa aparece com as
  duas letras finais trocadas de posição e sem acento. Confirmado em 4 PDFs de Convocação
  independentes (Ed_33, Ed_34, Ed_37, Ed_38 — dois triênios diferentes), sempre no mesmo padrão,
  sempre do lado da Convocação. Hipótese mais provável: defeito na geração desses PDFs pela
  Cebraspe (fonte/codificação daquele caractere acentuado no template de Convocação), não do
  nosso extrator — o texto correto simplesmente não está no PDF de Convocação para ser
  recuperado.
- **3 de 10** são casos isolados sem padrão comum entre si: dois Editais que discordam de
  verdade sobre o sobrenome de uma inscrição (16105688); um Edital com um sobrenome a menos que
  o outro, texto genuinamente ausente na Convocação (16116602); um par que nem é do mesmo
  triênio (2021/2023 vs 2022/2024, inscrição 21177086) com uma letra duplicada num dos dois
  documentos-fonte.

**Por que não é ticket de correção:** não existe, dentro do que o Edital disponibiliza, uma
segunda fonte para decidir qual grafia está certa — corrigir exigiria um dado externo (ex.:
base de matrícula da UnB), fora do escopo do pipeline de extração de Editais.

**Achado colateral, sem ticket próprio:** a tabela "Reconciliação cruzada" de
`relatorio_validacao.md` lista `nomes` (ordenado alfabeticamente) e `proveniências` (ordem de
inserção) como colunas lado a lado que não são arrays paralelos — um leitor pode assumir
erroneamente que a 1ª variante de nome corresponde à 1ª origem listada. Não afeta nenhum CSV de
saída, só a legibilidade da tabela; ver `reconciliacao.reconciliar_nomes` se algum dia isso for
mexido.

---

**Como manter este arquivo:** ao corrigir um defeito, mover a entrada para o relatório do
ticket que a resolveu (com referência de volta pra aqui) em vez de só apagar. Ao achar um
defeito novo por comparação de dados ou revisão de código, adicionar uma entrada nova seguindo
o mesmo formato antes de abrir ticket.
