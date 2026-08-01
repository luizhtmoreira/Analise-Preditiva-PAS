# 01 — `ExamStats` aceita Parte 1 misturada e carrega procedência

**What to build:** o `OFFICIAL_STATS` passa a poder guardar média e desvio de uma Etapa cuja fonte
não diz a língua estrangeira de cada candidato, sem inventar valores e sem mentir sobre a
procedência do número.

Hoje `ExamStats.parte_1` é obrigatório com as **três** línguas (inglesa, francesa, espanhola),
porque o Cebraspe normaliza a Parte 1 separadamente por língua e agrupar as três embute viés
sistemático contra quem fez espanhol ou francês (ticket 12 da extração; ticket 04 §5.3 do treino).
Mas o **Edital isolado de Etapa** — o "Resultado final nos itens do tipo D e na prova de redação",
que é a única fonte disponível para a Turma viva — lista nota por candidato e **não diz a língua de
ninguém**. Só dá a Parte 1 misturada. Preencher as três exigiria inventar valores.

Depois deste ticket, uma entrada do `OFFICIAL_STATS` pode ter a Parte 1 em duas formas, e a
diferença fica **explícita no dado**, não numa convenção implícita:

- **por língua** — as três, como hoje, quando vem do Edital de média e desvio;
- **misturada** — um único par média/desvio marcado como tal, quando vem do Edital isolado de Etapa.

E toda entrada passa a declarar sua **origem**: `edital` (as 24 atuais) ou `derivada` (as que
virão no ticket 07). Isso existe porque, quando o Edital de verdade sair em 2026, esses números
serão substituídos e as previsões vão mexer — e isso precisa estar registrado, não descoberto
depois.

Este ticket é **expand puro**: as 24 entradas existentes e todos os seus consumidores continuam
funcionando sem alteração. Nenhuma entrada nova entra aqui.

**O custo da forma misturada está medido e é o que a torna aceitável:** usar a Parte 1 misturada em
vez da língua declarada custa **0,46 ponto de Argumento Final em média**, máximo 3,21, com **viés
zero** — é ruído, não erro sistemático. A Parte 1 pesa 0,72 numa conta que soma 10, e a média
misturada cai praticamente em cima da média da inglesa, que é 66% a 73% da população.

**A costura não muda.** `stats_da_prova(ano, etapa, lingua)` continua sendo o ponto único de
leitura do `OFFICIAL_STATS` para quem vai calcular um Argumento de Etapa — treino e runtime pela
mesma porta, senão o `A1` que a API mostra deixa de ser o `A1` com que o modelo foi treinado.
Quando a entrada é misturada, ela devolve a estatística misturada **qualquer que seja a língua
pedida**, em vez de levantar erro. Recusar devolveria o produto ao estado que esta rodada existe
para sair.

Atenção ao cache de módulo que achata `OFFICIAL_STATS` para `(ano, etapa, língua)` — ele é montado
no import e precisa acomodar as duas formas.

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** ready-for-agent

- [ ] `ExamStats` admite as duas formas de Parte 1, e a forma misturada é distinguível da forma por
      língua sem depender de convenção (ex.: contar chaves do dicionário)
- [ ] `ExamStats` carrega a origem do dado (`edital` / `derivada`); as 24 entradas existentes são
      `edital`
- [ ] `stats_da_prova(ano, etapa, lingua)` devolve a estatística misturada para as três línguas
      quando a entrada é misturada, e continua devolvendo a estatística da língua pedida quando a
      entrada é por língua
- [ ] As propriedades `m_p1` / `dp_p1` mantêm o contrato atual (média simples das três línguas; o
      próprio valor, na forma misturada) e `api/services/analytics_service.py` não muda de interface
- [ ] Teste sobre uma entrada sintética misturada, cobrindo as três línguas pela mesma porta
- [ ] `pytest tests/` continua verde (linha de base: 290 passam, 0 falham)
- [ ] Um ADR registra a decisão — é mudança de forma num dado que 24 entradas já usam, difícil de
      reverter, e não deve viver só como comentário no `pas_constants.py`
