# Relatório — Ticket 07: Relatório de validação agrupado por padrão

**Ticket:** `.scratch/pdf-extraction/issues/07-relatorio-de-validacao-agrupado-por-padrao.md`
**Status:** concluído
**Onde vive o código:** `src/pas_extraction/relatorio_validacao.py` (novo), com um novo
subcomando `validar` em `cli.py` — pacote gitignored, ver seção "Por que o código não está
no git" do relatório do ticket 01
**Onde vive o teste:** `tests/test_pas_extraction_relatorio_validacao.py` (novo, 24 testes)
**Bloqueado por:** ticket 02 (sequência/ordem alfabética), ticket 04 (checksum), ticket 06
(fecho do reticulado de cotas) — todos já gravam o resultado da validação em cada
`RegistroResultadoFinal`; este ticket só consolida.

---

## 1. O que foi pedido

Um relatório que diz **onde mexer no parser**, agrupando as falhas por padrão em vez de
listar registro por registro — com 122 mil registros, uma lista linear é inútil, mas saber
que 200 registros falharam do mesmo jeito é acionável. A camada do checksum precisa reportar
a **distribuição** dos deltas, não só a taxa de acerto: deltas concentrados em torno de um
mesmo valor indicam fórmula incompleta (o parser está certo, falta um termo no cálculo);
deltas espalhados indicam dado corrompido (o cálculo está certo, o parser quebrou o número).
Regra central: nenhum registro é descartado automaticamente sem que o padrão da falha esteja
explicado.

Critérios de aceite (todos atendidos — ver seção 5):

- [x] As falhas são agrupadas por padrão, não listadas registro a registro
- [x] O relatório reporta a distribuição dos deltas do checksum, não apenas a taxa de acerto
- [x] Falhas com deltas concentrados são distinguidas de falhas com deltas espalhados
- [x] Nenhum registro é descartado automaticamente sem que o padrão da falha esteja explicado
- [x] O relatório cobre as cinco camadas de validação, indicando qual delas pegou cada grupo
- [x] Saída em terminal e em arquivo

---

## 2. Visão geral do que foi entregue

```
gerar_relatorio(registros: Sequence[RegistroResultadoFinal]) -> RelatorioValidacao
  │
  ├─ _camada_checksum ........... agrupa por intervalo de delta (largura 0,05) + calcula
  │                                DistribuicaoDeltas.concentrada
  ├─ _camada_sequencia .......... agrupa por (curso, Sistema, posições faltantes)
  ├─ _camada_ordem_alfabetica ... agrupa por curso
  ├─ _camada_formato_numerico ... agrupa por nome do campo
  └─ _camada_cotas ............... agrupa por perfil de cota deduzido

RelatorioValidacao
  ├─ total_registros, total_sem_checksum
  ├─ grupos: Tuple[GrupoFalha, ...]         (camada, padrão, quantidade, exemplos)
  └─ distribuicao_checksum: Optional[DistribuicaoDeltas]  (total_falhas, faixas, concentrada)

formatar_terminal(relatorio) -> str   # resumo curto
formatar_markdown(relatorio) -> str   # relatório completo, todos os grupos
escrever_relatorio(relatorio, destino)  # grava o markdown em arquivo
```

Novo subcomando `python -m pas_extraction validar [pdfs...] --out <arquivo>` em `cli.py`,
espelhando `extract`/`stats-diff`: descobre PDFs, chama `extrair_edital` em cada um,
concatena todos os registros num relatório só (um padrão que se repete entre Editais também
deve aparecer como um grupo só, não um por Edital), escreve o markdown e imprime o resumo.

---

## 3. Decisões e o porquê

**A "camada" do checksum produz `GrupoFalha` como as outras quatro, mas carrega dado extra
(`DistribuicaoDeltas`) que as outras não têm.** O critério de aceite pede duas coisas
diferentes ao mesmo tempo: que as cinco camadas apareçam de forma uniforme (mesma pergunta —
"qual grupo, detectado por qual camada, com quantos registros") e que o checksum
especificamente reporte uma distribuição, não uma lista de grupos. A saída é ter os dois:
`relatorio.grupos` trata o checksum como mais uma camada (um `GrupoFalha` por intervalo de
delta), e `relatorio.distribuicao_checksum` é o dado estruturado extra — só ele carrega
`concentrada`, que as outras camadas não têm equivalente.

**"Concentrada" é decidida por um limiar de share (30% das falhas num intervalo de 0,05),
não por inspeção visual.** O ticket descreve o sintoma ("empilhadas em torno de 0,7") mas não
dá um número. Um limiar automático é necessário para o relatório dizer "concentrada" ou
"espalhada" sem depender de alguém olhar a tabela — e 30% num intervalo estreito (0,05, contra
uma faixa de deltas que no protótipo ia a várias unidades) é um pico grande demais para ser
ruído aleatório. **Risco assumido:** é um número escolhido, não medido — o ticket não trouxe
um corpus de referência com "concentrada" e "espalhada" rotulados para calibrar contra. Se o
limiar se mostrar errado em uso real (falsos positivos/negativos na classificação), é o
primeiro lugar a ajustar; a constante está isolada em `LIMIAR_CONCENTRACAO` para isso.

**A camada de sequência agrupa por padrão, não por buraco único.** `buracos_classificacao`
(ticket 02) já é uniforme entre todos os registros do mesmo curso que têm classificação num
Sistema — então agrupar por `(curso, Sistema, posições)` naturalmente produz um grupo cuja
`quantidade` é "quantos registros apontam para este buraco", não "quantos buracos existem".
Isso é a semântica certa para "onde mexer no parser": um buraco só, mas 160 registros que o
denunciam, é mais forte como sinal de prioridade do que contar 1.

**Bug de ponto flutuante pego pelos próprios testes: `delta // 0,05` erra o intervalo para
deltas redondos.** `0.7 // 0.05` dá `13.0` em Python (não `14.0`) porque nem 0,7 nem 0,05 têm
representação binária exata — e um delta exatamente redondo é justamente o caso mais comum
de "pico" que este módulo existe para detectar. Corrigido convertendo para milésimos
(inteiros exatos) antes da divisão inteira. Sem os testes sintéticos com deltas como `0.700`,
`0.701`... isso teria passado despercebido até aparecer com dado real.

**`camada.capitalize()` nos títulos perderia o "N" de "sequência 1..N".** `str.capitalize()`
minúsculiza todo o resto da string, não só maiúsculiza a primeira letra — "sequência 1..N"
viraria "sequência 1..n", perdendo a letra que marca "até a posição N". Trocado por um
helper (`_titulo`) que só maiúsculiza o primeiro caractere.

**A reconciliação cruzada entre Editais (6ª camada do spec.md) não entra.** O ticket lista
"cinco camadas" explicitamente e bloqueia só em 02/04/06 — nenhum deles produz o dado de
reconciliação entre Editais diferentes (mesma inscrição, mesmo nome). Documentado em
`ORDEM_CAMADAS` para não ser lido como omissão.

---

## 4. Trade-offs e limitações conhecidas

- **O limiar de concentração (30%) é uma heurística, não uma medida calibrada** — ver seção 3.
- **A largura do intervalo (0,05) é fixa**, não adaptativa ao volume de falhas. Com poucas
  falhas (dezenas), um intervalo largo pode juntar deltas que não têm relação nenhuma entre
  si além de caírem no mesmo bucket; com muitas (milhares), pode ser fino demais para revelar
  um padrão mais amplo. Não foi medido contra o corpus real de 122 mil registros — só contra
  fixtures sintéticas.
- **O CLI reextrai os PDFs em vez de ler o CSV de `extract`.** Decisão deliberada (é
  `RegistroResultadoFinal` com os três campos de validação já preenchidos que o relatório
  consome, e o CSV já achatou isso em texto) mas tem custo: rodar `extract` e `validar` sobre
  o mesmo corpus processa cada Edital duas vezes.

---

## 5. Critérios de aceite — conferência

| Critério | Status | Onde |
|---|---|---|
| Falhas agrupadas por padrão | ✅ | `_grupos_de` + as cinco funções `_camada_*` |
| Distribuição dos deltas, não só taxa | ✅ | `DistribuicaoDeltas.faixas` (histograma completo) |
| Concentrado vs espalhado distinguidos | ✅ | `DistribuicaoDeltas.concentrada` |
| Nenhum descarte automático | ✅ | `gerar_relatorio` só lê e agrupa, nunca filtra `registros` |
| Cobre as cinco camadas, com camada de cada grupo | ✅ | `GrupoFalha.camada` + `ORDEM_CAMADAS` |
| Saída em terminal e arquivo | ✅ | `formatar_terminal` / `escrever_relatorio` + subcomando `validar` |

Smoke test manual contra `tests/fixtures/resultado_final_com_checksum.pdf` (189 registros,
fatia de 6 páginas + cauda do Ed_38): 0 falhas de checksum (distribuição espalhada, vazia),
196 ocorrências de buraco de sequência em 4 padrões (fixture truncada — os buracos são o
efeito esperado de fatiar só uma parte do curso), 21 registros de formato numérico em 7
padrões — números batendo com o que os relatórios dos tickets 01/02/04 já mediram nesta
mesma fixture.

---

## 6. Glossário desta sessão

- **Padrão de falha:** a chave de agrupamento de uma camada — não o registro em si, mas o
  que caracteriza registros que falharam "do mesmo jeito" (o mesmo campo com formato
  inválido, o mesmo intervalo de delta, o mesmo buraco de sequência).
- **Distribuição concentrada:** deltas de checksum empilhados no mesmo intervalo de 0,05 —
  sintoma de fórmula incompleta (o mesmo termo faltando em todo registro daquele padrão).
- **Distribuição espalhada:** deltas de checksum sem intervalo dominante — sintoma de dado
  corrompido (extração quebrada caso a caso, sem relação entre os registros).
- **Grupo de falha (`GrupoFalha`):** camada + padrão + quantidade de registros + exemplos de
  inscrição, para localizar o registro no Edital de origem.
