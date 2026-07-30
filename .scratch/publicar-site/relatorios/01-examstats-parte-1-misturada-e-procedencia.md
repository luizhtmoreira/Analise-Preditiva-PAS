# Relatório — Ticket 01: `ExamStats` aceita Parte 1 misturada e carrega procedência

**Branch:** `feat/ticket-01-examstats` (worktree em `../pas-ticket-01`), a partir de `feat/pdf-extraction`
**Commits:** `af2e4f2` (feature) + `aadc818` (correções da revisão)
**Status:** concluído, `pytest tests/` verde

---

## O que mudou

O `OFFICIAL_STATS` passa a poder guardar a Parte 1 de uma Etapa em **duas formas**, e cada
entrada passa a declarar **de onde veio**. Nenhum valor existente mudou; nenhuma entrada nova
entrou.

| Antes | Depois |
|---|---|
| `parte_1: Dict[str, ValorLingua]`, sempre as três línguas | `parte_1: Parte1` — `Parte1PorLingua` ou `Parte1Misturada` |
| forma implícita no formato do dicionário | forma **no tipo**, mais o predicado `.misturada` |
| sem registro de procedência | `origem: Origem` (`EDITAL` / `DERIVADA`), default `EDITAL` |
| `stats_da_prova` só sabia responder por língua | responde a estatística misturada para as três, quando a entrada é misturada |

`api/services/analytics_service.py` **não foi tocado** — `m_p1`/`dp_p1` mantêm o contrato.
`training_dataset._stats_por_ano_etapa_lingua` também não mudou de lógica (só de docstring):
ele continua fazendo `stats.parte_1.items()` e acomoda as duas formas de graça.

---

## Decisões, e por quê

**1. As duas formas são tipos, não convenções.**
O ticket pediu distinguibilidade "sem depender de convenção (ex.: contar chaves do
dicionário)". `Parte1PorLingua` e `Parte1Misturada` são classes irmãs sob uma base `Parte1`.
Descartada a alternativa de marcar a forma com uma chave especial no dicionário: convenção que
todo leitor precisa conhecer e que nenhum verificador de tipo cobra.

**2. As duas formas são `Mapping` das três línguas.**
Essa é a decisão que fez o ticket ser barato. Como as duas respondem `parte_1[lingua]`,
`.items()` e `set(parte_1)`, **todos os leitores existentes continuaram funcionando sem uma
linha de mudança** — inclusive o cache de módulo que o ticket avisou para observar. Na forma
misturada as três chaves devolvem o mesmo par média/desvio; uma língua fora das três continua
sendo `KeyError`.

**3. `parte_1` é declarado `Parte1`, e as 24 entradas foram envolvidas em `Parte1PorLingua(...)`.**
A primeira versão aceitava `dict` cru e convertia no `__post_init__`, para deixar as 24
entradas literalmente intocadas ("expand puro"). A revisão de código apontou, nos dois eixos
independentemente, que isso deixa o tipo declarado como `Union[Parte1, dict]` para sempre —
exatamente a ambiguidade que o ticket existe para remover. Envolver as 24 é mecânico e não
toca um valor sequer, então o custo é zero e a honestidade do tipo é real.

**4. `origem` tem default `EDITAL`.**
As 24 entradas atuais são todas de Edital, e mantê-las sem `origem=` explícito preserva a
promessa de expand puro. O risco (uma entrada derivada nascer marcada como `EDITAL` por
esquecimento) fica contido porque as derivadas do ticket 07 virão de **um** ponto de construção
programática, não de literais escritos à mão.

**5. Misturada e derivada são eixos independentes.**
Foi a confusão mais fácil de cometer ao ler o ticket. O Edital isolado de Etapa **é** um
Edital: uma entrada pode ser `Parte1Misturada` com `Origem.EDITAL`. Tem teste próprio para
travar isso.

**6. `stats_da_prova` não recusa.**
Sobre entrada misturada devolve a estatística misturada qualquer que seja a língua pedida.
Recusar devolveria o Preditor ao estado de não atender a Turma viva — o estado de que esta
rodada existe para sair.

---

## Correções vindas da revisão de código (`aadc818`)

- `misturada` virou `ClassVar[bool]`. Sem isso, se uma subclasse repetisse a anotação, o
  atributo silenciosamente viraria campo de construtor do dataclass.
- `para(lingua)` foi removido e `__getitem__` passou a ser o método abstrato. Antes havia
  duas portas públicas para a mesma leitura (`Middle Man`).
- `LINGUAS_OFICIAIS` existia em duas cópias (`pas_constants` e `model_package`). Agora mora em
  `pas_constants`, junto do dado que indexa, e `model_package` reexporta para não quebrar quem
  já importava de lá.
- O teste `test_official_stats_tem_24_entradas_com_tres_linguas_cada` checava
  `set(stats.parte_1) == {três línguas}`. Como a forma misturada **também** tem três chaves,
  essa asserção parou de discriminar no instante em que o ticket foi implementado. Passou a
  checar o tipo.
- `CONTEXT.md` ganhou os três termos novos, conforme `DEVELOPER_HANDBOOK.md` §4 (é o arquivo
  da linguagem ubíqua; deixar o vocabulário só no ADR e em docstrings é como o glossário
  apodrece).
- O ADR-0013 passou a reconciliar explicitamente com o ADR-0009.

---

## Testes

Baseline medida **neste worktree** (não os 290 do ticket, que foram medidos onde `data/` e
`models/` existem — aqui esses diretórios são gitignored e ausentes, o que produz 78 skips):

| | passam | pulam | falham |
|---|---:|---:|---:|
| antes (`1dcab1a`) | 253 | 78 | 0 |
| depois (`aadc818`) | 263 | 78 | 0 |

Dez testes novos. Nenhum skip novo, nenhuma falha. Rodar com
`PYTHONPATH=<repo-principal>/src` para que `pas_extraction` (untracked, fora do git por causa
do expurgo de PII) seja encontrado.

---

## Glossário

**Edital isolado de Etapa** — o "Resultado final nos itens do tipo D e na prova de redação" de
uma Etapa. Lista a nota de cada candidato e **não diz a língua estrangeira de ninguém**. É a
única fonte disponível para as Etapas da Turma viva. Diferente do **Edital de médias e
desvios**, que publica média e desvio já separados por língua.

**Parte 1 Misturada** — média e desvio da Parte 1 calculados sobre as três línguas juntas,
porque a fonte não diz quem fez qual. É uma *forma do dado*, marcada como tal. Custo medido:
0,46 ponto de Argumento Final em média, máximo 3,21, viés zero — ruído, não erro sistemático.

**Procedência (`Origem`)** — de onde veio a média e o desvio de uma `(Ano, Etapa)`: `EDITAL`
(publicado pelo Cebraspe) ou `DERIVADA` (inferida enquanto o Edital não sai). Registrada no
próprio dado porque, quando o Edital sair, os derivados serão substituídos e as previsões de
Alunos reais vão mexer.

**Expand puro** — metade "expand" do padrão *expand/contract* de migração: adiciona a forma
nova sem quebrar a antiga, e adia a remoção para depois. Aqui significa que nenhuma entrada
nova entra e nenhum consumidor precisa mudar.

**`ClassVar`** — anotação do `typing` que diz "este atributo pertence à classe, não à
instância". Num `dataclass` é o que impede o atributo de virar parâmetro do construtor.

**`Mapping`** — a interface de "coisa que se lê como dicionário" (`obj[chave]`, `.items()`,
`in`, iteração), sem prometer que se pode escrever nela. Implementá-la foi o que permitiu que
as duas formas de Parte 1 fossem lidas pelo mesmo código.

---

## O que fica para os próximos tickets

- **Ticket 07** cria as primeiras entradas `Origem.DERIVADA` e as primeiras
  `Parte1Misturada` reais no `OFFICIAL_STATS`. Este ticket só abriu a forma.
- O `HistoricalStats` que `stats_da_prova` devolve **não carrega a procedência**. Não foi
  pedido, e ninguém a jusante precisa dela hoje. Se o Preditor um dia tiver que avisar o Aluno
  "esta previsão usa número estimado", é aí que o campo entra.
- `mkdocs.yml` só indexa ADRs até o 0007. O 0013 seguiu a mesma sorte dos 0008–0012 — lacuna
  pré-existente, não deste ticket.
