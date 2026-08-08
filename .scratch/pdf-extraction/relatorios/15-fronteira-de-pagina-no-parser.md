# Relatório — Ticket 15: Fronteira de Página no Parser

**Ticket:** `.scratch/pdf-extraction/issues/15-fronteira-de-pagina-no-parser.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/resultado_final.py` (`_construir_blob`, novo
`_sem_numero_de_pagina`/`_NUMERO_PAGINA_RE`) — pacote gitignored, ver seção "Por que o
código não está no git" do relatório do ticket 01
**Onde vive o teste:** `tests/test_pas_extraction.py`, classe `TestFronteiraDePagina`
(sintético) + `TestCotaDeclarada.test_registro_na_fronteira_de_pagina_nao_sai_mais_suspeito`
(fixture real, atualizado)

---

## 1. O que foi pedido

O achado do ticket 06 (§3 do relatório): quando o 22º campo de um registro (a 10ª
classificação) só é impresso no início da página seguinte, `_separar_registro` lia o
número da página — que o `pypdf` emite como uma linha isolada no topo de cada página — no
lugar do valor real. Dar consciência de fronteira de página ao parser, para que isso pare
de acontecer.

Critérios de aceite (todos atendidos):

- [x] `_separar_registro` (na prática, `_construir_blob`) tem consciência de fronteira de
      página, não só do blob já concatenado
- [x] Os 8 casos conhecidos no corpus real passam a ter o valor correto na 10ª classificação
- [x] Teste sintético reproduz a borda de página e confirma o valor correto
- [x] Os 2 casos antes invisíveis à checagem de fecho foram reavaliados — ver seção 3

---

## 2. A correção

**Causa raiz confirmada no texto bruto:** o Edital imprime o número da página como uma
linha isolada no topo de cada página, e o `pypdf` extrai isso como o primeiro token do
texto — confirmado despejando `page.extract_text(extraction_mode="plain")` de um Edital
real, página por página:

```
'82 \n \n29.293, 3.128, 3.100, 12.463, 6.467, 2.223, 24.453, 7.417, -15.087, 73, -, ...'
'83 \n \n- / 16102355, Pedro Lucas Souza Oliveira, 5.275, 20.825, 6.722, 3.875, 19. ...'
```

`_construir_blob` concatenava as páginas sem remover esse número — ele entrava no blob
como se fosse dado, e quando um campo (qualquer campo, não só o 22º — ver seção 4) começava
exatamente ali, o parser lia o número de página no lugar do valor.

**A correção** é `_sem_numero_de_pagina`, aplicada ao texto de cada página *antes* da
concatenação: um regex (`^\s*\d+\s+`) remove o primeiro token numérico do início do texto
bruto da página, desde que seguido de espaço em branco (não vírgula) — é essa exigência de
`\s+` (não `,`) que evita casar o começo de um registro novo, cuja inscrição de 8 dígitos é
sempre seguida de vírgula, nunca de espaço:

```python
_NUMERO_PAGINA_RE = re.compile(r"^\s*\d+\s+")

def _sem_numero_de_pagina(texto: str) -> str:
    return _NUMERO_PAGINA_RE.sub("", texto or "", count=1)
```

Aplicado por página, antes de `_normalizar_pagina` (que colapsa espaço/quebra de linha) —
a ordem importa porque `_sem_numero_de_pagina` depende da quebra de linha real entre o
número e o corpo da página para não confundir o número com um campo de dado que também
começa com dígitos (ex. uma inscrição, que é sempre seguida de vírgula, não de espaço).

**Por que a correção geral (na fronteira de página) em vez de pontual (só no 22º campo):**
o ticket já apontava a causa raiz corretamente — o número de página vazando para dentro do
blob — e não seria coerente corrigir esse vazamento só para a posição onde o ticket 06
mediu o sintoma. A correção no ponto de origem (`_construir_blob`) resolve a classe inteira
de corrupção, não uma posição específica dela — e a seção 4 mostra que essa generalidade
importava na prática.

---

## 3. Os 8 casos conhecidos e os 2 antes invisíveis — todos resolvidos

Reprocessei o corpus real inteiro (`data/pdfs/resultado-final-pas3/`, 8 Editais) duas vezes
— uma com o código antes da correção, uma depois — e comparei registro a registro (por
`arquivo_origem` + `inscrição`) as 10 classificações de cada um.

**10 registros têm a 10ª classificação corrigida de um número de página para o valor real**
— exatamente os mesmos 10 que o relatório do ticket 06 media (seção "Limite conhecido da
detecção"):

| Padrão antes (com o número de página) | Era pego pelo fecho? | Padrão depois (corrigido) |
|---|---|---|
| `{1, 10}` (5 registros) | sim, suspeito | `{1}` |
| `{1, 7, 9, 10}` | sim, suspeito | `{1, 7, 9}` |
| `{1, 2, 10}` | sim, suspeito | `{1, 2}` |
| `{1, 3, 5, 7, 9, 10}` | sim, suspeito | `{1, 3, 5, 7, 9}` |
| `{1, 9, 10}` (2 registros) | **não** — fecho válido (PcD) | `{1, 9}` |

Os 8 primeiros já saíam marcados `cota_padrao_suspeito=True` antes da correção (o
comportamento que o ticket 06 entregou: sinalizar, não descartar) — agora saem com o
padrão certo e sem marca, porque o padrão certo *é* válido.

**Os 2 casos antes invisíveis à checagem de fecho** (`{1, 9,10}`, que é indistinguível de
um PcD genuíno porque também é um fecho válido) são resolvidos pela mesma correção, embora
a checagem de fecho não pudesse ter avisado sobre eles: `{1, 9, 10}` antes e `{1, 9}` depois
são os dois padrões válidos — a checagem nunca teria capturado a diferença, só a correção na
origem (fronteira de página) resolve. São as inscrições `18147304` (Ed_37, 2018/2020) e
`16125849` (Ed_31, 2016/2018) — identificados comparando as duas rodadas, não por inspeção
manual do PDF.

**Conclusão do item "2 casos invisíveis" do ticket:** resolvidos pela correção, mas por
mecanismo diferente do que a checagem de fecho oferece — documentado aqui, não fica como
limitação conhecida. Movido o item 3 de `defeitos-pendentes.md` para "resolvido", com
referência de volta para este relatório.

---

## 4. Achado além do escopo do ticket: a corrupção não se limitava ao 22º campo

O ticket (e o relatório do ticket 06 que o originou) descreviam o sintoma só na 10ª
classificação, porque foi ali que a checagem de fecho de cota o pegou. Reprocessando o
corpus inteiro para comparar as duas rodadas, apareceram **52 registros** com alguma
classificação diferente entre antes/depois — bem mais que os 10 do ticket. Os 42 além dos
10 já discutidos não têm o campo inteiro substituído pelo número da página; têm o número da
página **colado sem separador** na frente de um campo que também começa exatamente na
fronteira, por exemplo:

```
antes:  classificacao_1 = 157116   (número de página "157" colado em "116")
depois: classificacao_1 = 116
```

Confirmado como a mesma causa raiz (não um bug novo): o número de página é seguido de
`\n\n` no texto bruto, e é exatamente esse padrão que `_NUMERO_PAGINA_RE` casa e remove —
a correção resolveu esses 42 casos como efeito colateral direto de corrigir na origem, não
por uma segunda mudança.

**Efeito colateral maior: 380 registros que eram descartados silenciosamente voltam a
aparecer.** Comparando as chaves (arquivo + inscrição) das duas rodadas — não só os valores
—, a rodada "antes" tem **66.313** registros e a "depois" tem **66.693**: **380 a mais**,
zero a menos. Nenhuma chave que existia antes sumiu depois. Conferi uma amostra manualmente
(nome, notas, curso, campus — ver `checar_novos.py` do processo de verificação) e são
Alunos genuínos, sem qualquer sinal de corrupção: eram descartados porque a colagem do
número de página num campo numérico ocasionalmente produzia um valor grande o bastante, ou
uma sequência que não batia o `_NUMERO_RE`, para que `_montar_registro` devolvesse `None` —
descarte silencioso de registro inteiro, não um campo sinalizado. A correção não introduziu
lógica nova para recuperar esses 380; eles voltam a aparecer só porque o campo volta a ter o
valor certo, que sempre foi parseável.

Isso não muda o escopo do ticket (a correção pedida é a mesma, na mesma função) nem exige
um ticket novo — é o mesmo defeito, medido de forma mais completa agora que a correção
existe para comparar contra.

---

## 5. Como foi verificado

- **3 testes sintéticos novos** (`TestFronteiraDePagina`, `tests/test_pas_extraction.py`):
  constroem um PDF de 2 páginas com `fixtures.gerar_pdf_texto_sintetico` onde o 22º campo
  de um registro cai exatamente na borda — um com o valor real "7" (garante que não é só o
  caso "-" que passa), um com "-" (o caso do corpus real), e um confirmando que o registro
  seguinte à borda sai intacto. Confirmados **red antes / green depois** da correção
  (revertendo a mudança manualmente e rodando os três — os dois primeiros falham com `83`
  no lugar do valor esperado, exatamente o bug).
- **Teste da fixture real atualizado**: `test_padrao_suspeito_real_sobrevive_na_saida_com_a_marca`
  virou `test_registro_na_fronteira_de_pagina_nao_sai_mais_suspeito` — o próprio comentário
  do teste antigo already avisava "quando ele for corrigido, este teste falha de propósito:
  é o lembrete de revisitá-lo" (ticket 06). Reescrito para fixar o comportamento novo: o
  Aluno da fixture (`resultado_final_cota_suspeita.pdf`, Ed_31 pág. 82-83) agora tem só o
  padrão `{1}`, `perfil_cota="Universal"`, `padrao_suspeito=False`.
- **Corpus real inteiro, duas rodadas** (`data/pdfs/resultado-final-pas3/`, 8 Editais): a
  comparação "antes x depois" registro a registro que identificou os 10 casos do ticket 06
  (8 antes suspeitos + 2 antes invisíveis, seção 3) e os 42+380 da seção 4 rodou num script
  ad-hoc de scratchpad, não commitado — ele exigia reverter a correção em disco para gerar
  a rodada "antes", o que não é reproduzível a partir do código como ele fica depois deste
  ticket. **O que fica commitado e reproduzível** é `scripts/verificar_fronteira_de_pagina_ticket15.py`:
  roda o pipeline sobre o corpus real e confere as duas afirmações que importam para detectar
  regressão futura — zero registros com `cota_padrao_suspeito=True` no corpus inteiro, e as
  duas inscrições antes invisíveis (`18147304`, `16125849`) com o padrão corrigido `{1,9}`.
  Rodado e conferido: `total de registros no corpus: 66693`, `suspeitos: 0`, os dois casos
  `[OK]`.
- **Suíte completa** (`pytest tests/`): **486 passam**, nenhuma regressão.

---

## 6. Escopo deliberadamente fora deste ticket

| Não implementado aqui | Por quê |
|---|---|
| Ticket de follow-up para os 42 casos de colagem sem separador ou os 380 registros recuperados | Não são bugs novos — são o mesmo defeito (número de página vazando pro blob), já corrigido por esta mudança. Documentados aqui para fechar a medição completa, não para abrir trabalho novo. |
| Validação de formato do campo de classificação (defeito 2 de `defeitos-pendentes.md`) | Território do ticket 14, já fechado antes deste. Não tocado aqui. |
| medias_desvios.py, que duplica uma função `_construir_blob` com o mesmo nome | Família diferente, fora do escopo declarado do ticket (que aponta só `resultado_final._separar_registro`/`_montar_registro`). Não investiguei se sofre do mesmo defeito. |

---

## 7. Onde continuar

- Nenhum follow-up pendente para este defeito — ele está fechado, e a seção 4 documenta a
  medição completa do impacto real (52 registros corrigidos, 380 recuperados) para quem for
  comparar `resultado_final.csv` contra uma rodada anterior e notar a diferença de contagem.
- Se `medias_desvios.py` algum dia mostrar sintoma parecido, o defeito e a correção aqui são
  o ponto de partida óbvio — mesma extração `plain`, mesma estrutura de página com número
  isolado no topo.
