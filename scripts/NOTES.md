# Protótipo: censo de formatos dos PDFs de editais PAS/UnB

**Status:** respondido. Código do protótipo (`prototype_pdf_census.py`, `prototype_pdf_probe.py`,
`prototype_pdf_census.json`) é descartável — apagar depois de construir o pipeline real.

## Pergunta

Quantas famílias de formato existem entre os 77 PDFs em `data/pdfs`, e quantos parsers
distintos são necessários para extrair os registros de forma confiável?

## Resposta curta

**3 parsers, não 30.** As "diferenças sutis" entre PDFs são, na maioria, ruído de extração
de texto — não variação real de formato. O risco do projeto não está na diversidade de
formatos; está em **duas armadilhas de corrupção silenciosa** (abaixo).

## Achados

### 1. Só existem 3 famílias de documento (77 PDFs)

| Família | PDFs | Formato | Volume |
|---|---|---|---|
| `convocacao_registro` | 65 | tabela em colunas: `inscrição \| nome \| sistema/subsistema`, agrupada por cabeçalho de curso em caixa alta | baixo (~48 pág) |
| `resultado_final_*` | 7 | fluxo de texto, registros separados por ` / `, campos por `,` | **~122k registros** (242–419 pág) |
| `medias_desvio` | 5 | tabela de 1 página com média/desvio por prova | trivial |

O volume real é **~122k registros**, não 70k — as famílias `resultado final` sozinhas já
passam disso.

### 2. Os PDFs declaram o próprio schema — use isso como classificador

Cada edital contém na primeira página a frase `"na seguinte ordem: ..."` que lista os campos
na ordem exata em que aparecem. **Classifique por essa frase, não por heurística de layout.**

Agrupando por essa frase: 12 strings distintas → **6 após normalizar** (remover acento, caixa
e todo não-alfanumérico) → **3 schemas reais**, porque as 3 restantes são só mudança de
redação institucional:

- `"nome do candidato"` → `"nome da pessoa candidata"` (a partir de ~2023/2025)
- `"nota final nos itens do tipo D"` → `"nota provisória nos itens do tipo D"`

As outras diferenças eram puro artefato de extração: espaço no fim da string, `Campus` vs
`campus`, e **`"abaix o"` no lugar de `"abaixo"`** (quebra de palavra virando espaço).

> Implicação: normalização agressiva (`re.sub(r'[^a-z0-9]+','',s)`) antes de comparar
> qualquer string de schema. Sem isso você "descobre" 12 formatos que não existem.

### 3. Schema do fluxo é estável: 22 campos

```
inscrição, nome, eb_p1_e1, eb_p2_e1, red_e1, eb_p1_e2, eb_p2_e2, red_e2,
eb_p1_e3, eb_p2_e3, red_e3, argumento_final, classificação, + 9 campos de cota
```

22 campos em **todos** os triênios (2016/2018 → 2023/2025). Sem drift real de schema.
O agrupamento por `campus/curso/turno` vem de cabeçalhos em caixa alta intercalados no fluxo —
precisa ser carregado como estado durante o parse.

### 4. ARMADILHA A — um PDF, duas seções, dois schemas

Os arquivos `Res_final_tipo_D_redacao` contêm **duas listas diferentes no mesmo PDF**.
Em `Ed_27_PAS_3_2021_2023` (317 páginas), medido página a página:

- páginas 0–98: seção 1, registros de **4 campos** (inscrição, nome, nota tipo D, nota redação)
- página 99 em diante: seção 2, registros de **22 campos** (não eliminados)

A transição é marcada pelo cabeçalho `"2 DO RESULTADO FINAL DOS CANDIDATOS NÃO ELIMINADOS"`.
Um parser que assume um schema por arquivo produz lixo em metade do documento **sem levantar
erro**. O parser precisa ser dirigido por seção, não por arquivo.

### 5. ARMADILHA B — corrupção silenciosa no texto extraído

Três modos de falha confirmados, todos produzindo registros que "parseiam" sem exceção:

**(a) Cabeçalho de curso engolido dentro do registro** — o separador ` / ` some na fronteira
de curso e o cabeçalho entra no meio do fluxo:

```
..., -38.328, 164, -, -, -, -, -, -, -, -, - . ENGENHARIA DE REDES DE COMUNICAÇÃO
(BACHARELADO) 23124716, Alexandre Almeida Santos, 0.000, 20.240, ...
```

**(b) Registro sem número de inscrição** — dois registros grudados e uma inscrição perdida:

```
22101407, Luisa Silva Tomasello, ..., 31.413, 16, -, -, -, Jose Pedro Leao Tavares,
0.000, 34.714, ...
```

(A Luisa deveria ter 9 traços de cota; tem 3, e o nome seguinte aparece sem inscrição.)

**(c) Números partidos por whitespace** — `56.29 1` em vez de `56.291`, `- 58.570` com o
sinal negativo separado, `26.    309`. Um `float()` ingênuo lê `56.29` e perde o dígito,
ou o valor negativo vira positivo. **Este é o pior**: gera um número plausível e errado.

### 6. `extraction_mode`: plain para o fluxo, layout para as colunas

Modo `layout` produz **mais** números partidos que `plain` (74 vs 68 hits na amostra) porque
ele injeta espaços para preservar alinhamento visual. Mas a família `convocacao_registro`
depende de layout para separar as colunas.

→ `plain` para `resultado_final_*`, `layout` para `convocacao_registro`.

### 7. Médias/desvios vêm em DOIS lugares — e são melhores que as constantes atuais

Os PDFs de resultado final trazem a tabela de média/desvio numa seção final
(`"2 Média e desvio padrão"` / `"3 Média e desvio padrão"`), **mas nem todos**:

| Origem | Exemplo |
|---|---|
| cauda do próprio resultado final | `Ed_38_2024` (pág. 241/242), `Ed_27_2021_2023` (pág. 316/317) |
| PDF separado | `Ed_31_2016-2018` não tem cauda → vem em `Ed_32_2016-2018_media_e_desvio_padrao.pdf` |

O pipeline precisa dos dois caminhos, chaveados por `(subprograma, triênio, etapa)`.

**São mais ricos que o `OFFICIAL_STATS` atual.** O edital dá Parte I separada por língua
(Inglesa / Francesa / Espanhola), enquanto `pas_constants.py` tem um `m_p1` único. E os
valores atuais foram *estimados* (`"Gerado automaticamente via análise do
banco_alunos_pas_final.csv"`), não são os oficiais:

| (2022, etapa 1) | `OFFICIAL_STATS` atual | edital Ed_38 |
|---|---|---|
| `m_p2` | 20.709 | **20.406** |
| `dp_p2` | 13.581 | **13.533** |
| `m_red` | 5.888 | **5.849** |

→ A extração **corrige** as constantes do projeto, não só alimenta o banco.

### 8. O argumento final é um CHECKSUM embutido — resolve a validação

Testado em 1.261 registros de `Ed_38_2024`: recalculando o argumento final a partir das
9 notas brutas + médias/desvios da cauda, usando a fórmula do
`argument_calculator.py` (`AF = 1*AP1 + 2*AP2 + 3*AP3`):

```
delta <= 0.005 :  1058/1261  (83.9%)
delta <= 0.05  :  1142/1261  (90.6%)
mediana |delta| = 0.0012
```

A língua estrangeira de cada aluno não está no PDF, mas é **inferível**: testar as 3 e ficar
com a que fecha o checksum (distribuição obtida: 821 inglês, 414 espanhol, 26 francês —
plausível).

> Nota: a fórmula com pesos (1,2,3) em `argument_calculator.py:196` está **correta** — foi
> validada aqui contra dado oficial. A primeira tentativa do protótipo somava as etapas sem
> peso e errava por mediana 16.1; a regressão recuperou w=(0.987, 1.972, 2.994), R²=0.9984.

**Por que isso resolve a validação:** para um registro passar no checksum, as 9 notas *e* o
argumento *e* as médias precisam estar simultaneamente corretos. Um dígito perdido em
`56.29 1`, um cabeçalho engolido, dois registros grudados — nada disso fecha a conta.
É a verificação de 12 campos por meio de um único número, sem olho humano.

### 9. RESOLVIDO — a língua estrangeira é escolhida POR ETAPA, não por aluno

A primeira versão do checksum deixava 16% de falhas, com um cluster sistemático em
`delta ≈ 0.7–0.8`. Delta repetido não é corrupção (corrupção é aleatória) — era a fórmula
incompleta: eu assumia a mesma língua nas 3 etapas.

Testando as 27 combinações (3 línguas × 3 etapas) independentemente por registro:

```
                        língua fixa      língua por etapa
delta <= 0.005          83.9%            99.8%   (1258/1261)
falhas (delta > 0.01)   203              3
```

**17,4% dos alunos trocam de língua entre etapas.** Os 3 que ainda falham são corrupção
confirmada (um deles: `"Daniela F erreira Miguel Pereira"` — nome quebrado).

> Lição que justifica a cautela: se o gate tivesse sido ligado na versão "língua fixa",
> **200 registros bons de 1261 (16%) teriam sido descartados silenciosamente.** É por isso
> que um cluster sistemático de deltas precisa ser explicado *antes* de virar critério de
> descarte — a falha do checksum acusa "fórmula ou dado", e só investigando se sabe qual.

**Tolerância operacional:** `delta <= 0.005`. (51% fecham em ≤0.001; o resto é o
arredondamento oficial de 3 casas por etapa.)

**Bônus:** o checksum *recupera um dado que não está impresso no PDF* — qual língua o aluno
fez em cada etapa. Distribuição obtida: 821 inglês, 414 espanhol, 26 francês.

### 10. Os 9 campos finais SÃO as cotas — a informação já está no resultado final

A ordem declarada completa (`Ed_38_2024`) resolve os 22 campos:

```
campus/curso/turno, inscrição, nome,
eb_p1_e1, eb_p2_e1, red_e1,          (etapa 1)
eb_p1_e2, eb_p2_e2, red_e2,          (etapa 2)
eb_p1_e3, eb_p2_e3, red_e3,          (etapa 3)
argumento_final,
classificação no Sistema Universal,
classificação no Sistema de Cotas para Negros (se houver),
+ 8 classificações no Sistema de Cotas para Escolas Públicas
  (renda ≤1,5 SM e >1,5 SM × PPI/não-PPI × com/sem deficiência)
```

= 2 + 9 + 1 + **10 classificações** = 22 campos. ✔ bate com o observado.

`-` significa "não concorreu naquele sistema". **A cota já está no resultado final** — a
família convocação não é a única fonte. Ela continua no escopo, mas responde outra pergunta:
*quem foi efetivamente chamado* (a convocação tem chamadas sucessivas), não *em que cota
concorreu*.

> Consequência para notas de corte: dá para derivá-las **por curso e por sistema de cota**
> direto do resultado final — o último classificado de cada sistema em cada curso.

### 11. DECISÃO — extrair só a seção 2 dos PDFs "tipo D + redação"

Medido em `Ed_27_2021_2023` (317 páginas, varredura completa):

| | inscrições |
|---|---|
| seção 1 (tipo D + redação, 4 campos) | 9.036 |
| seção 2 (não eliminados, 22 campos) | 7.726 |
| **só na seção 1 = eliminados** | **1.449** |

Decisão do Luiz: **extrair apenas a seção 2.** Custo consciente: perdem-se ~1.4k eliminados
por PDF, e a nota nos *itens do tipo D* (que só existe na seção 1). Aceitável porque os
eliminados só têm 2 notas na seção 1 — não formam o vetor de 9 notas que os modelos usam.

*(139 inscrições apareceram na seção 2 e não na seção 1 — ~1,8%, compatível com a taxa de
corrupção de extração. Deve sumir quando o parser estiver correto; serve de métrica no loop.)*

### 12. A `classificação` é o detector de registro faltante

Resposta à dúvida sobre "reconciliar contagem": ela **não** faz o mesmo que o checksum.

- **checksum** pega registro *errado* — os números não fecham
- **reconciliação** pega registro *ausente* — e o checksum é cego para isso, porque um
  registro que nunca foi extraído não deixa nada para conferir

E o documento se auto-declara: dentro de cada curso, a `classificação no Sistema Universal`
é exatamente `1..N` sem buracos, ordenada por argumento final decrescente. Então:

- buraco na sequência → registro perdido, **e você sabe qual posição**
- número repetido → registro duplicado
- ordem quebrada vs. argumento → registro embaralhado ou grudado

Não precisa de contagem externa. Vale igual para cada sistema de cota separadamente.

### 13. As cotas SÃO inferíveis para todos — inclusive não aprovados

**Os campos de classificação são RANKING, não aprovação.** Todo candidato não eliminado é
ranqueado em todos os sistemas em que concorre, independente de ter passado. Evidência:

- 289 alunos têm cota preenchida **e** classificação universal > 100 (longe de aprovar)
- o pior deles tem argumento final **-104.356** e ainda aparece em `COTA_NEGROS`

Logo a inferência vale para a lista inteira, não só para os aprovados.

**Os sistemas são ANINHADOS, não exclusivos.** Primeira tentativa acusou 8,46% de
"contradição" (aluno em `renda≤1,5` *e* `renda>1,5` ao mesmo tempo) — o teste é que estava
errado, não o dado. Ser ≤1,5 SM habilita a concorrer também às vagas de >1,5 SM; ser PPI
habilita as vagas não-PPI; PcD idem (cascata de remanejamento da Lei 12.711).

O aluno é ranqueado em **todos** os subsistemas que subsome, e seus atributos são os do
subsistema **mais específico** em que aparece.

**Confirmação decisiva:** apenas **8 padrões distintos** observados (de 2⁹ = 512 possíveis),
e todos os 8 são fecho-para-baixo válido do reticulado. **0 violações em 1.843 registros.**

Os 4 atributos binários (`EP`, `renda≤1,5`, `PPI`, `PcD`) geram os 9 sistemas de cota
(1 Negros + 8 Escola Pública); o Universal não é cota. Distribuição na amostra:

| perfil deduzido | n | % |
|---|---:|---:|
| (só universal) | 1308 | 71.0% |
| EP + renda>1.5 + naoPPI | 310 | 16.8% |
| EP + renda>1.5 + PPI | 77 | 4.2% |
| COTA_NEGROS | 64 | 3.5% |
| EP + renda≤1.5 + naoPPI | 57 | 3.1% |
| EP + renda≤1.5 + PPI | 20 | 1.1% |
| EP + renda>1.5 + naoPPI + PcD | 5 | 0.3% |
| EP + renda≤1.5 + naoPPI + PcD | 2 | 0.1% |

`COTA_NEGROS` nunca aparece junto com subsistemas EP — o candidato **opta** por um sistema
ou outro na inscrição.

**LIMITE DA INFERÊNCIA:** para os 71% que aparecem só no Universal, é impossível distinguir
*"não tem direito a cota"* de *"tem direito e optou por não usar"*. O dado registra a opção
declarada, não a elegibilidade. Isso precisa estar explícito no CSV — o campo deve ser
`cotas_declaradas`, nunca `cotas_elegíveis`.

**Colunas derivadas a gravar por aluno** (além das 10 classificações cruas):
`sistema_negros`, `escola_publica`, `renda_baixa`, `ppi`, `pcd` (booleanos) + `perfil_cota`.

## Decisões do Luiz (2026-07-24)

1. **Saída em CSV** (não Supabase direto).
2. **Família convocação entra no escopo** — para saber quem foi aprovado e em que cota;
   alvo inicial são as notas de corte, possivelmente mais depois.
3. **Corrigir `pas_constants.py`** com os valores oficiais dos editais.
4. **Registrar a língua escolhida por etapa para todos os alunos** — derivada via checksum
   (ver seção 9), não está impressa em lugar nenhum do PDF.
5. **Seção 2 apenas**, nos PDFs de duas seções (ver seção 11).

## Consequência para a arquitetura da extração

1. **Classificador por schema declarado** (frase `"na seguinte ordem:"`, canonizada) — barato
   e autoritativo. 3 parsers.
2. **Parse dirigido por seção**, não por arquivo — detectar a troca de regime pelo cabeçalho
   numerado.
3. **Não confiar em "parseou sem erro".** A validação é a parte cara e não é opcional:
   - todo registro tem exatamente 22 (ou 4) campos e inscrição de 8 dígitos
   - nome não contém dígitos; campo numérico casa `^-?\d+\.\d{3}$` **exato** (pega `56.29 1`)
   - reconciliar contagem: nº de registros extraídos vs. nº de candidatos por curso
   - ranges: argumento final e escores dentro dos limites conhecidos do PAS
4. **Loop agêntico**: usar na fase de *desenvolver e corrigir* os 3 parsers contra os casos
   que a validação sinalizar — não como parser de produção registro a registro.
