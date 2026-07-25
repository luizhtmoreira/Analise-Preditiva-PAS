# Pipeline de Extração de Editais PAS/UnB

## Problem Statement

Os dados históricos que alimentam os modelos preditivos do Vetor PAS estão presos em 77
editais em PDF publicados pelo Cebraspe — cerca de 122 mil registros de Alunos com notas de
todas as três Etapas, Argumento Final, classificação e sistema de concorrência.

Hoje esse dado chega ao projeto de três formas ruins:

1. **Extratores ad-hoc.** `scripts/extrator_master.py` e companhia cobrem só a família de
   convocação, têm a lista de arquivos hardcoded (com caminhos Windows absolutos que não
   existem nesta máquina), referenciam PDFs que não estão mais em `data/pdfs`, e não têm
   nenhum teste. Foram declarados obsoletos pelo dono do produto.
2. **Constantes estimadas, não oficiais.** `pas_constants.py` documenta que o `OFFICIAL_STATS`
   foi *"gerado automaticamente via análise do banco_alunos_pas_final.csv"* — ou seja, média
   e desvio-padrão foram inferidos dos dados em vez de lidos dos editais. Para o triênio
   2022/2024, Etapa 1, o projeto usa `m_p2=20.709` enquanto o edital oficial diz `20.406`.
   Todo Argumento Final calculado pelo sistema carrega esse erro.
3. **Nota de Corte sem cobertura de cota.** As Notas de Corte atuais não distinguem sistema de
   concorrência, embora a informação exista no edital.

O obstáculo real não é o volume: é que a extração de PDF corrompe dados **silenciosamente**.
Números partem no meio (`56.29 1` vira 56.29), cabeçalhos de curso são engolidos dentro de
registros, e registros se colam perdendo o número de inscrição. Nada disso levanta exceção —
produz um número plausível e errado. Sem uma forma automática de detectar isso, ninguém pode
conferir 122 mil registros no olho, e o dado entra no modelo com erro invisível.

## Solution

Um pipeline de extração que lê os 77 editais e produz CSVs confiáveis, com **verificação
aritmética automática de cada registro**.

A chave é que os editais carregam o próprio gabarito:

- **O Argumento Final é um checksum.** Recalculando-o a partir das 9 notas brutas mais a
  tabela de média e desvio-padrão do próprio edital, e comparando com o valor impresso,
  verifica-se 12 campos com um único número. **99,8% dos registros fecham** com tolerância de
  0,005; os que não fecham são corrupção real de extração.
- **A classificação é o gabarito de completude.** Dentro de cada curso ela é exatamente
  `1..N` sem buracos, ordenada por Argumento Final decrescente. Buraco = registro perdido, e
  se sabe qual posição.
- **Os editais declaram o próprio schema** na primeira página (`"na seguinte ordem: ..."`),
  o que dá um classificador de formato autoritativo em vez de heurística.

Isso substitui inspeção humana por verificação matemática, e transforma a correção dos parsers
num loop com critério de parada objetivo: itera-se enquanto a taxa de falha de checksum cair.

Como subproduto, o pipeline recupera dois dados que **não estão impressos** em nenhum lugar:
a língua estrangeira que cada Aluno fez em cada Etapa (inferida por qual das três faz o
checksum fechar) e o perfil de cotas declaradas de cada Aluno.

## User Stories

1. Como dono do produto, quero extrair os 77 editais com um comando só, para não depender de
   listas de arquivos mantidas à mão.
2. Como dono do produto, quero que o pipeline descubra sozinho a que família cada edital
   pertence, para não ter que anotar metadados manualmente arquivo por arquivo.
3. Como dono do produto, quero que a família seja determinada pelo schema que o próprio edital
   declara, para que um edital novo com redação diferente seja classificado corretamente sem
   mudança de código.
4. Como dono do produto, quero extrair da família *resultado final* as 9 notas, o Argumento
   Final e as 10 classificações de cada Aluno, para alimentar o treino dos modelos.
5. Como dono do produto, quero que apenas a seção de não eliminados seja extraída dos editais
   que contêm duas seções, porque a seção de itens do tipo D não tem o vetor de 9 notas.
6. Como dono do produto, quero que a transição entre seções seja detectada pelo cabeçalho
   numerado do documento, e não por número de página fixo, para que funcione em qualquer
   edital da família.
7. Como dono do produto, quero extrair a tabela de média e desvio-padrão de cada edital, para
   poder normalizar as notas com os valores oficiais.
8. Como dono do produto, quero que o pipeline busque a tabela de média e desvio-padrão tanto
   na cauda do edital de resultado final quanto num edital avulso, porque triênios diferentes
   publicaram de formas diferentes.
9. Como dono do produto, quero que a média e o desvio-padrão da Parte 1 sejam gravados
   separadamente por língua estrangeira, porque é assim que o edital publica e o dado atual
   agrega indevidamente.
10. Como dono do produto, quero corrigir o `OFFICIAL_STATS` com os valores oficiais extraídos,
    para que o Argumento Final calculado pelo sistema pare de carregar erro de estimativa.
11. Como dono do produto, quero saber exatamente quais valores do `OFFICIAL_STATS` mudaram e em
    quanto, antes de aplicar a correção, porque isso altera a saída de modelos em produção.
12. Como dono do produto, quero extrair da família *convocação* quem foi chamado, em que
    chamada e em que sistema de concorrência, para poder derivar Notas de Corte por cota.
13. Como dono do produto, quero que o triênio, o semestre e o número da chamada sejam lidos do
    conteúdo do edital, e não de uma tabela hardcoded, porque a tabela atual já está
    dessincronizada dos arquivos em disco.
14. Como dono do produto, quero que cada registro extraído tenha seu Argumento Final recalculado
    e comparado com o impresso, para detectar corrupção silenciosa sem inspeção manual.
15. Como dono do produto, quero que a língua estrangeira de cada Etapa seja inferida por qual
    das três faz o checksum fechar, porque essa informação não está impressa no edital.
16. Como dono do produto, quero que a língua seja inferida por Etapa e não por Aluno, porque
    17,4% dos Alunos trocam de língua entre Etapas.
17. Como dono do produto, quero que a classificação dentro de cada curso seja verificada como
    sequência `1..N` sem buracos, para detectar registros que o parser perdeu.
18. Como dono do produto, quero que a ordem alfabética dentro de cada curso seja verificada,
    porque quebra de ordem indica registros colados.
19. Como dono do produto, quero que todo campo numérico seja validado contra o formato exato de
    três casas decimais, para pegar números partidos por espaço.
20. Como dono do produto, quero que o mesmo número de inscrição encontrado em editais diferentes
    seja conferido quanto ao nome, como verificação cruzada independente.
21. Como dono do produto, quero um relatório de validação que agrupe as falhas por padrão em vez
    de listar registro por registro, para saber onde mexer no parser.
22. Como dono do produto, quero que o relatório distinga falhas com deltas concentrados de
    falhas com deltas espalhados, porque as primeiras indicam fórmula incompleta e as segundas
    indicam dado corrompido.
23. Como dono do produto, quero que nenhum registro seja descartado automaticamente por falhar
    no checksum sem que o padrão da falha esteja explicado, porque uma versão anterior da
    fórmula descartaria 16% de registros perfeitamente bons.
24. Como dono do produto, quero registrar as cotas declaradas de cada Aluno, deduzidas do padrão
    de preenchimento das 10 classificações.
25. Como dono do produto, quero que as cotas sejam registradas para todos os Alunos não
    eliminados e não apenas para os aprovados, porque os campos são ranking e não aprovação.
26. Como dono do produto, quero que os quatro atributos (escola pública, renda, PPI, PcD) sejam
    deduzidos do subsistema mais específico em que o Aluno aparece, porque os sistemas são
    aninhados por cascata de remanejamento.
27. Como dono do produto, quero que o campo se chame explicitamente *cota declarada* e nunca
    *cota elegível*, porque para 71% dos Alunos é impossível distinguir quem não tem direito de
    quem tem e optou por não usar.
28. Como dono do produto, quero que padrões de cota que violem a estrutura aninhada sejam
    sinalizados como suspeitos, porque isso indica corrupção de extração.
29. Como dono do produto, quero a saída em CSV, para inspecionar e versionar antes de decidir
    carregar em qualquer banco.
30. Como dono do produto, quero um CSV por família, porque as três têm granularidades
    diferentes.
31. Como dono do produto, quero que cada linha do CSV carregue a proveniência (arquivo de
    origem, edital, triênio, página), para poder auditar qualquer valor de volta até o PDF.
32. Como dono do produto, quero que cada linha carregue o resultado da sua própria validação,
    para poder filtrar por confiança em vez de confiar cegamente.
33. Como dono do produto, quero derivar Notas de Corte por curso e por sistema de concorrência a
    partir do resultado final, porque a informação de cota já está lá.
34. Como dono do produto, quero rodar o pipeline sobre um subconjunto dos editais, para iterar
    rápido durante o desenvolvimento dos parsers.
35. Como dono do produto, quero que rodar o pipeline duas vezes sobre a mesma entrada produza
    exatamente a mesma saída, para poder comparar execuções e detectar regressão.
36. Como dono do produto, quero que o pipeline não dependa de caminhos absolutos de máquina,
    para que rode em qualquer clone do repositório.
37. Como desenvolvedor, quero que os casos reais de corrupção estejam fixados como testes de
    regressão, para que uma correção de parser não reintroduza um problema já resolvido.
38. Como desenvolvedor, quero fixtures pequenas, para que a suíte de testes rode rápido apesar
    de os editais reais terem centenas de páginas.

## Implementation Decisions

### Novo pacote, substituindo os scripts atuais

O código vive num pacote novo, `src/pas_extraction/`, separado de `pas_intelligence` — são
domínios distintos: extração é offline, lê PDF e escreve CSV; `pas_intelligence` é predição
que roda dentro do app Streamlit. A única alteração em `pas_intelligence` é a correção do
`OFFICIAL_STATS`.

Os extratores em `scripts/` (`extrator_master.py`, `extract.py`, `extract_students.py`,
`extrator_teste.py`, `debug_quota_logic.py` e afins) são referência histórica e não devem ser
estendidos. Ficam intactos até o novo pipeline cobrir seus casos, e então são removidos.

Um detalhe do código antigo é aproveitável e vira constante compartilhada: o `mapa_sistemas`
de `extrator_master.py` numera os 10 sistemas de concorrência de 1 a 10, na mesma ordem em que
as classificações aparecem no resultado final. Isso liga as duas famílias.

### Vocabulário novo

Termos que passam a integrar o domínio do projeto, para uso consistente no código e nos docs:

- **Edital**: um PDF publicado pelo Cebraspe. Identificado por número, triênio e data.
- **Família de Edital**: um dos três formatos — *Resultado Final*, *Convocação*, *Médias e
  Desvios*. Determinada pelo schema declarado, não pelo nome do arquivo.
- **Sistema de Concorrência**: um dos 10 sistemas em que um Aluno pode ser classificado
  (Universal, Cota para Negros, e 8 subsistemas de Escola Pública).
- **Cota Declarada**: sistema de concorrência em que o Aluno optou por concorrer. Distinto de
  elegibilidade, que o edital não revela.
- **Checksum do Argumento Final**: verificação que recalcula o Argumento Final a partir das
  notas brutas e o compara com o impresso.

### Classificação por schema declarado

A família é determinada pela frase `"na seguinte ordem: ..."` da primeira página, canonizada
antes de comparar — remover acentos, caixa e todo caractere não-alfanumérico. Sem essa
canonização aparecem 12 schemas distintos onde existem 3; a diferença é ruído de extração
(espaço no fim, `Campus` vs `campus`, `"abaix o"` em vez de `"abaixo"`).

Depois de canonizar, os 6 grupos restantes colapsam em 3 porque as diferenças são de redação
institucional: `"nome do candidato"` virou `"nome da pessoa candidata"` a partir de 2023/2025,
e `"nota final"` virou `"nota provisória"`.

### Modo de extração de texto por família

Contraintuitivo mas medido: `extraction_mode='layout'` produz **mais** números partidos que
`plain` (74 contra 68 hits na amostra), porque injeta espaços para preservar alinhamento
visual. Logo:

- Resultado Final e Médias/Desvios → `plain` (o dado é fluxo separado por ` / `)
- Convocação → `layout` (o dado é colunar e depende do alinhamento)

### Schema do Resultado Final

22 campos, estável em todos os triênios de 2016/2018 a 2023/2025:

```
campus/curso/turno (cabeçalho intercalado, carregado como estado durante o parse)
inscrição, nome,
eb_p1_e1, eb_p2_e1, red_e1,      (Etapa 1)
eb_p1_e2, eb_p2_e2, red_e2,      (Etapa 2)
eb_p1_e3, eb_p2_e3, red_e3,      (Etapa 3)
argumento_final,
classificação no Sistema Universal,
classificação no Sistema de Cotas para Negros,
+ 8 classificações no Sistema de Cotas para Escolas Públicas
```

`-` significa "não concorreu naquele sistema".

### Parse dirigido por seção

Os editais de *resultado final tipo D + redação* contêm duas listas com schemas diferentes no
mesmo arquivo. Medido no Ed_27 (2021/2023, 317 páginas): páginas 0–98 têm registros de 4
campos, e a partir da página 99 registros de 22 campos, com a transição marcada pelo cabeçalho
`"2 DO RESULTADO FINAL DOS CANDIDATOS NÃO ELIMINADOS"`.

Só a seção de não eliminados é extraída. Custo aceito conscientemente: perdem-se ~1.449
Alunos eliminados por edital, que de todo modo só têm 2 notas e não formam o vetor de 9.

Um parser que assumisse um schema por arquivo produziria lixo em metade do documento sem
levantar erro — por isso a seção, e não o arquivo, é a unidade de parse.

### Fórmula do Argumento Final

`AF = 1×AP1 + 2×AP2 + 3×AP3`, onde cada `APn = argumento(P1) + argumento(P2) + argumento(Redação)`
e `argumento(x) = ((x − média) / desvio) × peso`, com `PESO_P1=0,72`, `PESO_P2=8,28`,
`PESO_REDACAO=1,00`.

Essa é exatamente a fórmula já implementada em `argument_calculator.py`, agora **validada
contra dado oficial** — a regressão sobre 1.261 registros recuperou pesos `(0,987, 1,972, 2,994)`
com R²=0,9984. O pipeline reusa essa função em vez de reimplementá-la; é o mesmo cálculo
servindo agora como verificação.

### Inferência de língua por Etapa

A língua estrangeira não está no edital. Testam-se as 27 combinações (3 línguas × 3 Etapas) e
fica a que minimiza o delta do checksum. Resultado medido:

```
                          língua fixa por Aluno   língua por Etapa
delta <= 0,005            83,9%                   99,8%  (1258/1261)
falhas (delta > 0,01)     203                     3
```

17,4% dos Alunos trocam de língua entre Etapas — daí a diferença. Tolerância operacional:
`delta <= 0,005` (51% fecham em ≤0,001; o resto é o arredondamento oficial de 3 casas).

### Dedução das Cotas Declaradas

Os 4 atributos binários — escola pública, renda ≤1,5 salário mínimo per capita, PPI, PcD —
geram os 9 sistemas de cota. Os sistemas são **aninhados, não exclusivos**: ser ≤1,5 SM
habilita a concorrer também às vagas de >1,5 SM, ser PPI habilita as não-PPI, PcD idem
(cascata de remanejamento da Lei 12.711). O Aluno é ranqueado em todos os subsistemas que
subsome, e seus atributos são os do subsistema mais específico em que aparece.

O modelo, vindo do protótipo, com os índices na ordem em que as classificações aparecem:

```python
# atributos exigidos por cada subsistema de Escola Pública
EP_ATTRS = {
    2: {"R", "PPI"},   3: {"R", "PPI", "PcD"},
    4: {"R"},          5: {"R", "PcD"},
    6: {"PPI"},        7: {"PPI", "PcD"},
    8: set(),          9: {"PcD"},
}
# um padrão válido é sempre o fecho para baixo desse reticulado:
def fecho(attrs): return {i for i, need in EP_ATTRS.items() if need <= attrs}
```

Validação decisiva: apenas 8 padrões distintos ocorrem (de 2⁹ = 512 possíveis), e todos os 8
são fecho para baixo válido — **0 violações em 1.843 registros**. Padrão que não seja fecho é
sinal de corrupção e deve ser sinalizado.

`Cota para Negros` nunca coocorre com subsistemas de Escola Pública: o Aluno opta por um
sistema ou outro na inscrição.

Colunas derivadas por Aluno, além das 10 classificações cruas: `sistema_negros`,
`escola_publica`, `renda_baixa`, `ppi`, `pcd` e `perfil_cota`.

### Camadas de validação

Em ordem de poder de detecção, todas automáticas:

1. **Checksum do Argumento Final** — verifica 12 campos de uma vez.
2. **Classificação como sequência `1..N`** por curso e por sistema — detecta registro ausente
   ou duplicado, que é o ponto cego do checksum: um registro que nunca foi extraído não deixa
   nada para conferir.
3. **Ordem alfabética** dentro do curso — quebra indica registros colados.
4. **Formato numérico exato** `^-?\d+\.\d{3}$` — pega números partidos por espaço.
5. **Fecho do reticulado de cotas** — pega padrão de cota impossível.
6. **Reconciliação cruzada entre editais** — mesma inscrição, mesmo nome.

Nenhum registro é descartado automaticamente sem que o padrão da falha esteja explicado. O
relatório agrupa falhas por padrão e reporta a **distribuição dos deltas**, porque deltas
concentrados indicam fórmula incompleta enquanto deltas espalhados indicam dado corrompido —
a distinção que impediu o descarte de 200 registros bons durante o protótipo.

### Saída

Um CSV por família, com colunas de proveniência (arquivo, edital, triênio, página) e o
resultado da validação por linha, de modo que o consumidor filtre por confiança em vez de
confiar cegamente. Execução determinística: mesma entrada, mesma saída, byte a byte.

### Correção do `OFFICIAL_STATS`

A correção é aplicada em duas etapas: primeiro um relatório de diferenças entre os valores
atuais e os oficiais, depois a substituição. Como isso altera a saída de modelos em produção,
o relatório precisa ser revisado antes de a substituição ser aplicada.

A estrutura também muda: o `ExamStats` atual tem um `m_p1` único, mas o edital publica Parte 1
separada por língua. A mudança de forma precisa acomodar isso sem quebrar quem consome hoje.

## Testing Decisions

**O que faz um bom teste aqui:** testa comportamento externo — dado um edital, quais registros
saem e o que a validação diz. Não testa estrutura interna do parser, que vai mudar muito
durante o loop de correção. Testes que travem detalhes de implementação atrapalhariam
exatamente a iteração que o pipeline precisa.

**Costura única:** `extrair_edital(caminho_pdf) -> ResultadoExtracao`, com o resultado contendo
registros, médias e desvios, e relatório de validação. Toda a lógica — classificação de
família, parse por seção, checksum, inferência de língua, dedução de cota — é exercitada por
essa fronteira.

**Fixtures:** PDFs pequenos de 3 a 5 páginas, fatiados uma única vez dos editais reais e
commitados. Fatiar em vez de sintetizar preserva a corrupção real de extração — número
partido, cabeçalho engolido, registro colado — que é justamente o que precisa ser testado.
Fixtures necessárias: resultado final de 22 campos, resultado final com a transição entre as
duas seções, convocação, e médias/desvios.

**Casos de regressão a fixar** a partir das corrupções já identificadas no protótipo: o número
`56.29 1` que deve virar 56.291; o cabeçalho `ENGENHARIA DE REDES DE COMUNICAÇÃO (BACHARELADO)`
engolido no meio de um registro; o par de registros colados em que o segundo perdeu o número de
inscrição; o negativo `- 58.570` com sinal separado.

**Prior art:** `tests/test_pas_intelligence.py` é o teste mais próximo em estilo — testa
`argument_calculator` e `statistics` por comportamento, com valores conhecidos. Os testes de
extração seguem o mesmo formato.

**Cobertura mínima esperada:** para cada família, um teste que verifica a contagem de registros
extraídos e a ausência de falhas de checksum na fixture; para o edital de duas seções, um teste
que verifica que apenas a seção de não eliminados foi extraída; para as cotas, um teste que
verifica o perfil deduzido de um Aluno com padrão conhecido.

## Out of Scope

- **Carga em Supabase.** A saída é CSV. Decidir se e como esses dados entram no banco é
  trabalho posterior.
- **Retreino dos modelos** com os dados extraídos. O pipeline produz o dado; treinar é outro
  ciclo.
- **A seção de itens do tipo D e redação.** Decidida fora de escopo conscientemente, com o
  custo medido (~1.449 Alunos eliminados por edital).
- **Alunos eliminados.** Só entram os não eliminados, consequência da decisão acima.
- **Download automático de editais novos** do site do Cebraspe. Os PDFs entram em `data/pdfs`
  manualmente.
- **Interface visual** para o relatório de validação. Saída em terminal e arquivo.
- **Remoção dos scripts antigos.** Acontece depois que o novo pipeline cobrir os casos deles.
- **Alteração no app Streamlit.** A única mudança fora do pacote novo é o `OFFICIAL_STATS`.

## Further Notes

Todas as descobertas técnicas que embasam este spec estão em `scripts/NOTES.md`, com os números
medidos e o método de cada uma — 13 seções, produzidas por quatro rodadas de protótipo
descartável (`scripts/prototype_pdf_census.py`, `prototype_pdf_probe.py`,
`prototype_checksum.py`, `prototype_cotas.py`). Esses scripts são descartáveis e devem ser
removidos quando o pipeline real existir; o `NOTES.md` fica.

Um alerta que vale repetir, porque é o erro mais provável de se cometer aqui: durante o
protótipo, uma versão do checksum que parecia excelente (83,9% de acerto) teria descartado 200
de 1.261 registros perfeitamente válidos. O que denunciou o problema não foi a taxa de acerto,
foi a **forma da distribuição dos deltas** — as falhas estavam empilhadas em torno de 0,7 em
vez de espalhadas. Qualquer gate de qualidade neste pipeline precisa reportar distribuição, não
só taxa.

O volume real é maior que a estimativa inicial de 70 mil: são ~122 mil registros nas famílias
de resultado final.
