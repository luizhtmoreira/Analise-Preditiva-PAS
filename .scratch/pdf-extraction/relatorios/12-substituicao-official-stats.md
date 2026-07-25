# Relatório — Ticket 12: Substituição do `OFFICIAL_STATS`

**Ticket:** `.scratch/pdf-extraction/issues/12-substituicao-official-stats.md`
**Status:** concluído
**Onde vive o código:** `src/pas_intelligence/pas_constants.py` (único arquivo alterado —
fora do pacote `src/pas_extraction/`, como o ticket exige)
**Onde vive o teste:** `tests/test_pas_intelligence.py`, classe nova `TestExamStats`

---

## 1. O que foi pedido

O Argumento Final calculado pelo sistema deixa de carregar o erro de estimativa documentado
no ticket 11: os valores de `OFFICIAL_STATS`, antes inferidos do `banco_alunos_pas_final.csv`,
são substituídos pelos valores oficiais publicados nos Editais. A forma também muda: o Edital
publica a Parte 1 separada por língua estrangeira (Inglesa/Francesa/Espanhola), enquanto o
`ExamStats` tinha um `m_p1`/`dp_p1` único — a mudança de forma precisa acontecer **sem quebrar**
o único consumidor, `api/services/analytics_service.py`, que lê `s.m_p1` e `s.m_p1 + s.m_p2`.

Critérios de aceite (todos atendidos — ver seção 5):

- [x] Os valores do `OFFICIAL_STATS` são os oficiais extraídos dos Editais, e o comentário
      sobre geração via `banco_alunos_pas_final.csv` sai
- [x] O `ExamStats` acomoda a Parte 1 por língua estrangeira
- [x] `api/services/analytics_service.py` continua funcionando sem alteração de comportamento
      observável (mesma interface de atributos)
- [x] `tests/test_pas_intelligence.py` continua passando
- [x] Nenhuma outra alteração em `pas_intelligence` e nenhuma no app Streamlit

---

## 2. Visão geral do que foi entregue

```
src/pas_intelligence/pas_constants.py
    ValorLingua       — dataclass novo: (media, desvio_padrao) de uma língua estrangeira
    ExamStats         — m_p2/dp_p2/m_red/dp_red continuam campos; m_p1/dp_p1 viram
                        @property calculadas a partir de `parte_1: Dict[str, ValorLingua]`
    OFFICIAL_STATS    — 24 entradas (as 21 antigas corrigidas + o triênio 2023/2025 completo),
                        todos os 6 números por entrada lidos de Edital, não estimados

tests/test_pas_intelligence.py
    TestExamStats     — 3 testes novos (ver seção 5)
```

Nada em `ensemble.py`, `argument_calculator.py`, `target_calculator.py`, `statistics.py` ou
`app/streamlit_app.py` foi tocado — nenhum deles importa `OFFICIAL_STATS`/`ExamStats` (só
`analytics_service.py` importa, confirmado por busca no repositório inteiro antes de editar).

---

## 3. Decisões tomadas e o porquê

### 3.1 Os valores vieram de `coletar_valores_oficiais()`, nunca de transcrição manual

**Decisão:** os 24×6 números que entraram em `pas_constants.py` foram gerados rodando
`pas_extraction.relatorio_official_stats.coletar_valores_oficiais(data/pdfs/*.pdf)` num
script descartável fora do repositório (não commitado), e colados no arquivo a partir da
saída do programa — nunca digitados a partir da leitura de um PDF ou do Markdown do ticket 11.

**Porquê:** é exatamente a recomendação da seção 8 do relatório do ticket 11 — "o ticket 12
recebe uma API em vez de uma tabela para transcrever; a transcrição manual seria a fonte de
erro mais provável". 144 números (24 entradas × 6 campos) transcritos à mão de uma tabela
Markdown é onde um dígito trocado se esconderia sem nenhum teste pegar, porque o teste
compararia o valor errado contra ele mesmo. Rodar o coletor de novo (em vez de copiar do
`official-stats-diff.md` do ticket 11) também teve uma vantagem concreta: o triênio
2023/2025 aparece no relatório do ticket 11 só como "ausente" (seção 4, sem os valores) —
precisei coletar de novo para ter os números das 3 novas entradas, e reaproveitei a mesma
rodada para as 21 antigas por consistência.

### 3.2 Incluir o triênio 2023/2025 — decisão que pedi ao usuário, não tomei sozinho

**Decisão:** as 3 novas entradas (2023/E1, 2024/E2, 2025/E3) entraram no `OFFICIAL_STATS`,
que passa de 21 para 24 chaves.

**Porquê perguntei antes de fazer:** o próprio relatório do ticket 11 (seção 8) marcou isso
como decisão do dono do produto, não do agente — "incluir ou não é escolha, não correção".
Ao contrário da correção dos 21 valores (que é estritamente consertar um erro conhecido),
adicionar dado novo muda o comportamento observável de `analytics_service.get_temporal()`:
a série que alimenta o gráfico de Análise Temporal ganha 3 pontos que não existiam antes.
Isso é exatamente o tipo de decisão que não deve ser resolvida por inferência silenciosa — perguntei, e a resposta foi incluir.

### 3.3 `m_p1`/`dp_p1` viram `@property`, não campos armazenados

**Decisão:** em vez de guardar `m_p1`/`dp_p1` como números escritos à mão ao lado de
`parte_1`, eles são calculados sob demanda como a média simples das três línguas em
`parte_1`.

**Porquê:** o ticket 11 (seção 3.3) já tinha descartado explicitamente qualquer `m_p1`
"oficial" — nenhuma das três línguas é "a" Parte 1, e inventar uma média oficial produziria
um número que não está em Edital nenhum. Mas o ticket 12 ainda precisa de **algum** valor
para `s.m_p1` continuar funcionando (critério de aceite: `analytics_service.py` sem
alteração de comportamento observável). A escolha foi entre (a) escrever o número da média
como mais um literal no `ExamStats`, correndo o risco de ele divergir de `parte_1` se algum
dos três valores for corrigido depois, ou (b) derivá-lo de `parte_1` sempre. Escolhi (b):
elimina por construção a possibilidade de `m_p1` ficar dessincronizado dos três valores que
ele resume, e deixa explícito no código (docstring da property) que é uma aproximação, não
um dado oficial — a mesma transparência que o relatório do ticket 11 pede.

**Alternativa descartada:** usar a Língua Inglesa como proxy única (por ser, supostamente, a
mais comum). Não se sustenta: o próprio `scripts/NOTES.md` (seção 8) mediu a distribuição
real de um Edital via checksum — 821 inglês, 414 espanhol, 26 francês — inglês é maioria mas
está longe de ser quase-unânime, e o ticket 11 já tinha achado que o `m_p1` estimado antigo
ficava mais perto do francês em 7 das 21 entradas. Fixar uma língua widened a superfície de
erro sem ganhar precisão real.

### 3.4 `parte_1` é um campo com valor default (`field(default_factory=dict)`), não obrigatório

**Decisão:** `parte_1: Dict[str, ValorLingua] = field(default_factory=dict)` — tecnicamente
opcional na assinatura do `ExamStats`, embora as 24 entradas do `OFFICIAL_STATS` sempre
preencham as três línguas.

**Porquê:** ordem de campos do dataclass — `m_p2`/`dp_p2`/`m_red`/`dp_red` continuam sem
default (são sempre exigidos), e um campo com default precisa vir depois dos sem-default.
Não adicionei validação de "as três línguas têm que estar presentes" no `__post_init__`
porque isso não pode acontecer com os dados que o módulo carrega hoje (é constante estática,
não input de usuário) — validar uma invariante que a própria definição dos dados já garante
seria a validação desnecessária que as diretrizes do projeto pedem para evitar. O teste
`test_official_stats_tem_24_entradas_com_tres_linguas_cada` cobre a invariante real (as três
línguas presentes em toda entrada de produção) sem precisar de guarda em runtime.

### 3.5 Revisão de código: duplicação extraída, `parte_1` deixou de ter default

**Decisão pós-revisão:** `/code-review` (eixos Standards e Spec, em paralelo) apontou dois
problemas reais no `ExamStats` original: (a) `m_p1` e `dp_p1` repetiam a mesma forma de
cálculo lado a lado; (b) `parte_1: Dict[str, ValorLingua] = field(default_factory=dict)`
tinha um default que nenhum código gerado usa (as 24 entradas sempre têm as três línguas) e,
se algum dia fosse usado, `m_p1`/`dp_p1` explodiriam em `ZeroDivisionError` sobre uma lista
vazia. Corrigi as duas: extraí `_media_parte_1(atributo)` como o cálculo comum das duas
properties, e tirei o default de `parte_1` — agora é campo obrigatório, porque um `ExamStats`
sem Parte 1 não é um dado válido neste domínio, e um objeto que finge aceitar esse estado
inválido é pior do que um que exige o dado na criação.

O eixo Spec também confirmou que a expansão de 21 para 24 chaves muda o número de linhas que
`analytics_service.get_temporal()` devolve — o mesmo ponto que já tinha discutido com o
usuário na seção 3.2 antes de implementar, não um bug novo. Verifiquei manualmente depois da
correção: `get_temporal()` roda sem exceção e devolve 24 `EtapaStat` (antes eram 21).

### 3.6 Rodada de `stats-diff` depois da troca, como teste de fechamento

**Decisão:** depois de escrever o novo `pas_constants.py`, rodei
`python -m pas_extraction.cli stats-diff` de novo (contra um arquivo temporário fora do
repositório) para conferir o que o próprio ticket 11 previu como critério de fechamento.

**Porquê:** é a verificação mais forte disponível — não "os números parecem certos", mas "a
ferramenta que gerou o diff original agora relata diff zero". Resultado: as 96 comparações
1-para-1 (`m_p2`, `dp_p2`, `m_red`, `dp_red` × 24 entradas) todas com `Δ = +0.000`; "sem
cobertura" e "ausentes no OFFICIAL_STATS" ambos zerados (antes eram 0 e 3, respectivamente —
o 3 já não existe porque as 3 entradas entraram). Ver seção 5.

---

## 4. Escopo deliberadamente fora deste ticket

- **Reestimar o `banco_alunos_pas_final.csv`** para corrigir o viés de sobrevivência que o
  ticket 11 documentou — não é o problema que este ticket resolve; a saída é usar o valor
  oficial, que é o que foi feito.
- **Expor a língua estrangeira por Aluno** para que consumidores futuros calculem o
  Argumento usando a língua certa em vez da média das três — depende do ticket 04 (inferência
  de língua por checksum), que ainda não foi implementado.
- **Qualquer mudança em `analytics_service.py`** — o ticket é explícito: "esta é a única
  alteração fora do pacote `src/pas_extraction/`... nada mais em `pas_intelligence` e nada no
  app Streamlit muda". O consumidor foi verificado, não editado.

---

## 5. Como foi verificado

**Critérios de aceite:**

1. *Valores oficiais, comentário de geração via CSV removido* — `pas_constants.py` não tem
   mais a linha `# Gerado automaticamente via análise do banco_alunos_pas_final.csv`; todos
   os 6 números de cada uma das 24 entradas vieram de `coletar_valores_oficiais()` (seção 3.1).
2. *`ExamStats` acomoda Parte 1 por língua* — campo `parte_1: Dict[str, ValorLingua]`, testado
   em `test_m_p1_e_dp_p1_sao_media_das_tres_linguas` e
   `test_official_stats_tem_24_entradas_com_tres_linguas_cada`.
3. *`analytics_service.py` sem alteração de comportamento observável* —
   `test_consumidor_analytics_service_continua_funcionando` replica o uso exato do consumidor
   (`s.m_p1`, `s.m_p1 + s.m_p2`) contra uma entrada real do `OFFICIAL_STATS` de produção.
   Também testado manualmente com `s = OFFICIAL_STATS[(2016, 1)]` → `s.m_p1 = 4.489...`,
   `s.dp_p1 = 2.674...`, `s.m_p1 + s.m_p2 = 28.227...` — todos `float`, nenhuma exceção.
4. *`tests/test_pas_intelligence.py` passando* — 34 testes na suíte (31 antigos + 3 novos),
   1 falha, **pré-existente e sem relação** com este ticket
   (`TestTargetCalculator::test_guaranteed_scenario`, já documentada como tal no relatório do
   ticket 11 — asserção sobre saída de modelo em `models/`, não tocado aqui).
5. *Nenhuma outra alteração em `pas_intelligence`/Streamlit* — `git diff --stat` limitado a
   `src/pas_intelligence/pas_constants.py` e `tests/test_pas_intelligence.py`.

**Suíte completa do projeto** (`pytest tests/`): 109 passando, 2 falhas, ambas
pré-existentes e não relacionadas — `TestTargetCalculator::test_guaranteed_scenario` (modelo
em `models/`) e `test_pdf_gen_manual.py::test_pdf_gen` (caminho Windows hardcoded). Mesmos
dois já documentados nos relatórios dos tickets 09 e 11 — nenhuma regressão introduzida por
este ticket. (Uma rodada intermediária da suíte completa mostrou 3 falhas a mais em
`TestCotaDeclarada`, ticket 06 — investiguei antes de assumir que era deste ticket: isolei
`pas_constants.py` sozinho via `git stash` e as mesmas 3 falhas continuavam presentes sem
minha mudança, e uma nova rodada da suíte completa logo em seguida já não reproduzia — sinal
de uma condição de corrida com outro processo mexendo nas mesmas fixtures durante a execução,
não algo introduzido aqui.)

**Teste de fechamento (seção 3.6):** `python -m pas_extraction.cli stats-diff` rodado de novo
depois da troca — 24 entradas comparadas, 0 sem cobertura, 0 ausentes, 0 divergências, e as
96 diferenças de Parte II/Redação todas em `+0.000` (antes: 82 de 84 divergiam). Confirma que
a troca é exata, não aproximada.

---

## 6. Glossário — termos necessários para entender este relatório

(Os termos já definidos no relatório do ticket 11 — Ano da prova, Triênio, Entrada sem
cobertura, Etapa ausente do `OFFICIAL_STATS`, Agregação indevida, Viés de sobrevivência —
não são repetidos aqui; valem exatamente como lá.)

- **`ValorLingua`**: par (média, desvio-padrão) de uma língua estrangeira específica na Parte
  1 de uma Etapa. Substitui, em `ExamStats`, o que antes era um único `m_p1`/`dp_p1`.
- **`m_p1`/`dp_p1` como `@property`**: no `ExamStats` novo, deixaram de ser números fixos e
  passaram a ser calculados (média simples das três línguas de `parte_1`) toda vez que são
  lidos — não podem ficar dessincronizados dos valores oficiais que resumem.
- **Teste de fechamento**: rodar `stats-diff` de novo depois da substituição, para confirmar
  que a comparação 1-para-1 (Parte II e Redação) zera — é o critério que o próprio ticket 11
  definiu para saber que o ticket 12 terminou corretamente.

---

## 7. Onde continuar

- **Ticket 04 (checksum + língua por etapa)**, quando implementado, é o que permitiria um
  consumidor futuro ler a língua real de um Aluno em vez da média das três em `m_p1`/`dp_p1`
  — hoje ninguém além de `analytics_service.py` lê esses dois campos, então não há urgência.
- **Nota de Corte por Sistema de Concorrência** (ticket 10, spec story 33) — já destravada
  desde o ticket 09; não depende deste ticket, mas fecha o mesmo arco da spec.
