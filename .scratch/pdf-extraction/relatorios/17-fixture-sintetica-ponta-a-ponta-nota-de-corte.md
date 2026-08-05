# Relatório — Ticket 17: Fixture sintética ponta a ponta para Nota de Corte

**Ticket:** `.scratch/pdf-extraction/issues/17-fixture-sintetica-ponta-a-ponta-nota-de-corte.md`
**Status:** concluído
**Onde vive o código:** `tests/test_pas_extraction_notas_corte_e2e.py` (arquivo novo, nenhum
outro tocado)

---

## 1. O que foi pedido

Um teste automatizado, executável em CI sem `data/pdfs` local, que roda o pipeline inteiro —
extração de PDF → Resultado Final → Convocação → Nota de Corte — sobre dados 100% inventados
e confere que o corte final sai correto.

Critérios de aceite (todos atendidos — ver seção 4):

- [x] Duas fixtures sintéticas do mesmo triênio (Resultado Final + Convocação), inscrições
      que se cruzam, dados inteiramente inventados
- [x] Comitáveis sem violar [[project_parser_privacy]]
- [x] Roda o pipeline ponta a ponta e confere o corte esperado, calculado manualmente
- [x] Cobre o caso de empate/múltiplos alunos do mesmo Sistema na maior chamada
- [x] Roda em CI sem exigir `data/pdfs` local
- [x] `defeitos-pendentes.md` (item 5) e a seção de limitações do ticket 10 atualizados

---

## 2. Visão geral do que foi entregue

```
tests/test_pas_extraction_notas_corte_e2e.py
    _CABECALHO_RESULTADO_FINAL / _CABECALHO_CONVOCACAO  — texto de página 1 inventado
    _registro_resultado_final / _registro_convocacao    — geram um registro de cada Família
    notas_corte_derivadas (fixture pytest)              — monta os 2 PDFs em tmp_path via
                                                            fixtures.gerar_pdf_texto_sintetico,
                                                            roda extrair_edital +
                                                            extrair_edital_convocacao +
                                                            derivar_notas_corte
    TestPipelineCompletoNotaDeCorte                     — 6 testes (ver seção 4)

.scratch/pdf-extraction/relatorios/defeitos-pendentes.md   — item 5 marcado FECHADO
.scratch/pdf-extraction/relatorios/10-...-concorrencia.md  — seção de limitações atualizada
```

Nenhum arquivo de `src/pas_extraction/` foi tocado — o ticket é só de teste, os dois
extratores (`pipeline.extrair_edital`, `convocacao.extrair_edital_convocacao`) e a derivação
(`notas_corte.derivar_notas_corte`) já existiam e funcionaram sem alteração.

---

## 3. Decisões tomadas e o porquê

### 3.1 PDF gerado em `tmp_path`, não fixture binária commitada

**Decisão:** as duas fixtures não são arquivos `.pdf` em `tests/fixtures/` (como as fatiadas
de Edital real, ex. `resultado_final_22_campos.pdf`) — são geradas dentro do próprio teste, a
cada execução, via `fixtures.gerar_pdf_texto_sintetico`, exatamente como o ticket 02 já fez
para o Edital isolado de Etapa (mesmo motivo: aquele PDF também listaria nome de Aluno em
toda linha se fosse fatiado de um real).

**Porquê:** o requisito de "comitável sem violar a política de privacidade" já está satisfeito
pela própria natureza do texto — inventado, vivendo como string no `.py` versionado — sem
precisar decidir se um binário PDF gerado é ou não seguro de commitar. Gerar em `tmp_path`
também elimina qualquer necessidade de manter um arquivo binário sincronizado com o código do
teste (mudar um campo do formato não deixa uma fixture `.pdf` desatualizada por trás).

### 3.2 Classificações do Resultado Final saem todas "-" (não concorreu)

**Decisão:** os 10 campos de classificação por Sistema, no registro sintético de Resultado
Final, são todos `"-"`.

**Porquê:** `derivar_notas_corte` (`notas_corte.py`) nunca lê `classificacoes` — ela cruza
Resultado Final e Convocação só por `(triênio, inscrição)` para pegar o `argumento_final`, e é
a Convocação (não o Resultado Final) que diz em que Sistema e chamada o Aluno foi chamado.
Preencher classificações plausíveis não mudaria nada sob teste e adicionaria número inventado
sem função — contrariando a diretriz do projeto de não escrever código/dado que a asserção não
usa.

### 3.3 O cenário de empate: dois convocados, mesmo Sistema, mesmo Argumento Final

**Decisão:** das 4 inscrições convocadas no Sistema 1 (Universal) na mesma chamada, duas
(`20100003` e `20100004`) têm exatamente o mesmo Argumento Final (70.000) — o menor dos
quatro.

**Porquê:** é o cenário que o próprio ticket pede ("cobre o caso de empate/múltiplos alunos do
mesmo sistema na maior chamada, mesmo cenário testado sinteticamente no ticket 10") e que
prova o critério de desempate de `derivar_notas_corte`
(`min(candidatos, key=lambda par: (par[1].argumento_final, par[0].inscricao))`): o corte em si
(70.000) seria o mesmo mesmo sem o empate, mas só com duas inscrições empatadas o teste
verifica que o **representante** da linha (`inscricao`/`nome` do corte) é escolhido pela menor
inscrição, e não por ordem de inserção ou acaso de iteração de `dict`.

### 3.4 Validado empiricamente antes de escrever o teste final

**Decisão:** antes de escrever `test_pas_extraction_notas_corte_e2e.py`, rodei dois scripts
descartáveis (fora do repositório, em `/tmp`) que geravam um PDF sintético de cada Família
isoladamente e imprimiam o texto extraído nos dois modos (`layout`/`plain`) usados pelos
parsers reais, para confirmar que `classificar_familia`, `extrair_metadados` e os parsers de
corpo reconheciam o texto inventado antes de montar o cenário de 4 inscrições completo.

**Porquê:** `resultado_final.py` e `convocacao.py` leem a página 1 em modos de extração
diferentes (`layout` vs `plain`) e ancoram em regexes específicos (frase "na seguinte ordem",
`CAMPUS_RE`, `_ANCORA_RE` de 8 dígitos) — escrever o texto às cegas e só descobrir uma
incompatibilidade no teste final teria misturado dois tipos de erro (formato de texto errado
vs. lógica de teste errada) na mesma depuração. Rodar isolado achou um problema real de
primeira tentativa: `"EDITAL N. 99"` não casa `_EDITAL_RE` (`N[ºo°]?` não aceita `.`) — corrigido
para `"EDITAL No 17"` no texto final. Como o número do Edital não entra na chave de
`ChaveCorte` nem em nenhuma asserção deste teste, isso não teria quebrado o teste, mas deixaria
`arquivo_origem`/proveniência com um metadado "desconhecido" sem necessidade.

---

## 4. Como foi verificado

**Os 6 testes novos**, todos passando (`pytest tests/test_pas_extraction_notas_corte_e2e.py -v`):

1. `test_extratores_leem_as_quatro_inscricoes_dos_dois_pdfs` — os dois extratores reais
   (não a derivação direta) leem as 4 inscrições dos 2 PDFs sintéticos.
2. `test_deriva_exatamente_um_corte_para_medicina_sistema_1` — exatamente 1 `NotaCorte`, com
   as 6 dimensões da chave corretas (trienio, campus, curso, turno, sistema, chamada).
3. `test_corte_e_o_menor_argumento_final_entre_os_convocados` — `argumento_final == 70.000`,
   `convocados_na_chamada == convocados_com_argumento == 4`, `parcial is False`.
4. `test_empate_e_desempatado_pela_menor_inscricao` — o representante do corte é `20100003`
   (não `20100004`, o outro empatado).
5. `test_checksum_fecha_e_none_sem_tabela_de_medias_e_desvios` — `checksum_fecha is None`
   (nenhum dos 2 PDFs sintéticos publica a tabela oficial; `None` = "não conferido", nunca
   "conferido e reprovado").
6. `test_nao_sobra_nenhum_grupo_sem_argumento_ou_sem_chamada` — os quatro contadores de
   "o que não deu para derivar" (`DerivacaoNotasCorte`) saem todos vazios/zero.

**Sem `data/pdfs`:** movi `data/pdfs/` para fora do repositório e rodei só este arquivo de
teste — os 6 continuam passando, confirmando que nada aqui depende do corpus local.

**Suíte completa** (`pytest tests/`): 461 passando (455 antes do ticket + 6 novos), 0 falhas,
0 regressões — mesmo resultado antes e depois desta mudança.

---

## 5. Escopo deliberadamente fora deste ticket

- **Cascata de chamada** (sistema sem convocado na última chamada, cai para a anterior) — já
  coberta por teste sintético direto em `test_pas_extraction_notas_corte.py`; o ticket pede só
  o cenário de empate/múltiplos alunos, não uma reprodução de toda a suíte de regra de negócio
  via PDF.
- **Tabela de Médias e Desvios / checksum do Argumento Final** — os PDFs sintéticos não
  publicam essa tabela de propósito (ver 3.2/seção 4, item 5); testar o cruzamento das três
  Famílias ao mesmo tempo (Resultado Final + Convocação + Médias e Desvios) seria escopo do
  ticket 08 (rodada completa), não deste.

---

## 6. Glossário — termos necessários para entender este relatório

- **PDF sintético**: PDF gerado em tempo de teste (`fixtures.gerar_pdf_texto_sintetico`) a
  partir de texto inteiramente inventado — usado quando fatiar um PDF real exporia dado de
  Aluno identificável.
- **Chave de corte / `ChaveCorte`**: as 6 dimensões que identificam uma Nota de Corte — triênio,
  semestre, campus, curso, turno e Sistema de Concorrência (ver docstring de `notas_corte.py`).
- **Representante do corte**: o Aluno cuja linha (`inscricao`/`nome`) é gravada como a Nota de
  Corte quando há empate no Argumento Final — o *valor* do corte seria o mesmo sem o desempate;
  só o registro exibido muda.
- **Checksum `None` vs. `False`**: `None` = a tabela de Médias e Desvios não estava disponível
  para conferir aquele registro (não é erro); `False` = a tabela estava disponível e o
  Argumento Final não bateu com ela (achado real, ver ticket 14).

---

## 7. Onde continuar

Nenhum follow-up aberto por este ticket. O item 5 de `defeitos-pendentes.md` está fechado; os
itens 1, 2, 3 e 6 (nome quebrado por espaço, formato de classificação, fronteira de página,
nomes divergentes entre Editais) continuam pendentes e são candidatos a próximo ticket.
