# Relatório — Ticket 07: As entradas derivadas entram e o Preditor responde para a Turma viva

**Ticket:** `.scratch/publicar-site/issues/07-preditor-responde-para-a-turma-viva.md`
**Status:** concluído, `pytest tests/` verde (432 passam)
**Blocked by:** 01 (`ExamStats` com Parte 1 misturada e procedência) e 06 (calibração — aprovada
na 2ª rodada, resíduo 4,366 < 5,009). Os dois já estavam prontos; este ticket só escreveu o dado.

---

## O que mudou

O tracer bullet do dado (`OFFICIAL_STATS`) à tela: um Aluno do triênio 2024-2026 digita as seis
notas e recebe o Argumento Final previsto, em vez da recusa de hoje.

| Camada | Antes | Depois |
|---|---|---|
| `OFFICIAL_STATS` | 24 entradas, todas `Origem.EDITAL` | 26 — as duas novas são `Origem.DERIVADA` |
| `training_dataset` | sem forma de ler procedência | `origem_da_prova(ano, etapa) -> Origem`, porta única |
| `model_package.Previsao` | — | `usa_estatistica_derivada: bool` |
| `PredictResponse` | — | `usa_estatistica_derivada: bool` |
| Preditor (tela) | — | aviso quando a previsão usa estimativa |
| `stats_da_prova`, `model_package`, `training_dataset`, `target_calculator`, API | recusavam a Turma viva (`EstatisticaOficialAusenteError`) | **nem uma linha mudou** — herdam a correção do dado |

A recusa continua existindo: só parou de disparar para `(2024, 1)` e `(2025, 2)`. Um triênio sem
nenhuma cobertura (testes usam `2025-2027`) continua recebendo `modelo_disponivel: false`.

---

## Decisões, e por quê

**1. As duas entradas novas são o resultado de rodar `Correcao.aplicar`, não números
copiados do corpo do ticket.** `scripts/gerar_stats_turma_viva.py` reproduz a conta: extrai
`StatsEmpiricos` dos dois Editais isolados (`Ed_8_2024...` e o PDF de hash da Etapa 2/2025, já
catalogados em `data/pdfs/editais-de-etapa`), monta a calibração de cada Etapa sobre os 6
triênios com Edital oficial, e aplica. Rodá-lo reproduz exatamente os literais escritos em
`pas_constants.py` — é a evidência "rastreável" que o critério de aceite pede para o valor
bruto.

**2. O script filtra `stats_oficiais` para `origem is Origem.EDITAL` explicitamente, em vez de
usar o default de `montar_calibracao`.** Achado ao rodar pela primeira vez *depois* de escrever
as entradas no arquivo: o default (`_STATS_POR_ANO_ETAPA_LINGUA`) achata `OFFICIAL_STATS`
inteiro, que a essa altura já contém as duas `DERIVADA`. Sem o filtro, `montar_calibracao`
comparia o empírico de 2024/1 contra o oficial *que a própria correção produziu* — resíduo
artificial perto de zero, calibração contaminada a cada re-execução futura. Documentado no
docstring do script, não só aqui, porque é a armadilha mais fácil de cair para quem rodar de
novo depois deste ticket ter mergeado.

**3. A Parte 1 das duas entradas é `Parte1Misturada`, sem correção — herdado do ticket 06, não
uma decisão nova deste ticket.** `Correcao.aplicar` já devolve a Parte 1 intocada
(`CorrecaoComponente.INERTE`); este ticket só embrulha o valor bruto (`m_p1`/`dp_p1` do
`StatsEmpiricos`) em `Parte1Misturada(ValorLingua(...))`. O preço é o viés de +0,35 ponto medido
no ticket 06 — não escondido, registrado no comentário de `pas_constants.py`.

**4. `usa_estatistica_derivada` é computado em `model_package._argumentos_exatos`, não em
`predict_service`.** A alternativa era o serviço, depois de receber `A1`/`A2`, chamar
`origem_da_prova` ele mesmo. Ficou no pacote de modelo porque é ele quem já sabe, no mesmo
lugar, quais `(ano, etapa)` foram usados para `A1` e `A2` — repetir essa resolução do lado de
fora seria uma segunda leitura da mesma decisão, podendo divergir se um dia a lógica de
`anos_do_trienio` mudar só de um lado.

**5. `_argumentos_exatos` passou a devolver 3-tupla `(a1, a2, usa_estatistica_derivada)`.**
Custo mecânico: `montar_features` (usado só nos testes de paridade treino/runtime) descarta o
terceiro elemento. A alternativa — um método `_origem_derivada` separado, repetindo
`anos_do_trienio` — duplicaria uma chamada barata só para evitar mudar uma assinatura interna;
preferi mudar a assinatura.

**6. O booleano não carrega texto.** A resposta tem `usa_estatistica_derivada: bool`, e a tela
escreve a frase. Alternativa considerada: a API já mandar o texto pronto (como
`motivo_indisponivel`). Descartada porque aqui não há "motivo" no sentido de erro — é um estado
normal da previsão, e o texto é copy de produto, não diagnóstico; deixá-lo no frontend segue o
padrão de `etapa_1_ausente` (também um booleano puro, texto ao lado na tela).

---

## O que já não precisou mudar

Exatamente a lista que o ticket previu, e por quê ela realmente não mudou:

- **`stats_da_prova`** — já lia `OFFICIAL_STATS` pela porta única; as duas entradas novas
  entram nele por já estarem no dicionário fonte.
- **`model_package._argumentos_exatos`** (a parte que calcula `A1`/`A2`) — já chamava
  `stats_da_prova`; parou de levantar `EstatisticasIndisponiveisError` para 2024-2026 porque a
  chave passou a existir, não porque o código mudou de comportamento.
- **`training_dataset`, `target_calculator`, `gestao_service`** — todos leem `OFFICIAL_STATS`
  pela mesma porta (ticket 05). A Gestão de Ativos e a Calculadora de Estratégia passaram a
  atender a Turma viva de graça; os testes novos (`test_gestao_preve_para_a_turma_viva...`,
  `test_calculadora_preve_para_a_turma_viva...`) só confirmam isso, não implementam nada.

---

## Efeito colateral descoberto ao testar: os outros consumidores de `TRIENIO_DA_TURMA_VIVA`

`tests/test_api_predict.py` usava `TRIENIO_DA_TURMA_VIVA = "2024-2026"` como sinônimo de "não
tem previsão" em oito testes — não só o que o ticket citou explicitamente. Rodar a suíte depois
da mudança no dado quebrou os outros sete (a Gestão parava de mostrar `grey`, a Calculadora
parava de recusar, etc.) — sintoma correto: a mudança de comportamento é real, não um teste
frágil.

Resolução: `TRIENIO_DA_TURMA_VIVA` continua sendo `"2024-2026"`, mas os testes que precisavam de
um triênio **sem nenhuma cobertura** passaram a usar `TRIENIO_SEM_EDITAL = "2025-2027"` — o
triênio seguinte, cujo `(2025, Etapa 1)` não está em `OFFICIAL_STATS` nem como Edital nem como
derivada. Nenhum teste ficou testando uma coisa que deixou de ser verdade.

---

## Testes

Novos, por arquivo:

- `tests/test_pas_intelligence.py` — `OFFICIAL_STATS` tem 26 entradas; as 24 de Edital continuam
  `Parte1PorLingua`; as duas novas são `DERIVADA` + `Parte1Misturada`.
- `tests/test_model_package.py` — `2025-2027` continua recusando (`EstatisticasIndisponiveisError`,
  "2025" na mensagem); `2024-2026` responde com `usa_estatistica_derivada is True`; `2023-2025`
  responde com `usa_estatistica_derivada is False`.
- `tests/test_api_predict.py` — sentido invertido para `TRIENIO_DA_TURMA_VIVA` (responde, com o
  aviso); `TRIENIO_SEM_EDITAL` cobre a recusa que o ticket exige continuar existindo; Gestão e
  Calculadora ganharam um teste positivo cada para a Turma viva, e os que dependiam do triênio
  vivo recusar foram movidos para `TRIENIO_SEM_EDITAL`.

`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` → **432 passed**.

Frontend: `npx tsc --noEmit` sem erros; `npx eslint` nos dois arquivos tocados sem erros novos
(3 warnings pré-existentes, não relacionados).

---

## Nota sobre o estado do repositório encontrado

Ao começar este ticket, a árvore de trabalho já tinha mudanças **não commitadas** de um outro
ticket (13, "língua por etapa" — `lingua_e1`/`lingua_e2` substituindo `lingua` única em
`PredictInput`/`EntradaDePrevisao`, corrigindo o defeito 11 de `defeitos-pendentes.md`). Este
ticket foi implementado em cima dessas mudanças, sem revertê-las — elas não conflitam com o que
o ticket 07 pediu, e a suíte inteira passa com as duas juntas.

---

## Glossário

**Estatística derivada (`Origem.DERIVADA`):** a média/desvio de uma `(Ano, Etapa)` que ainda não
tem Edital do Cebraspe, estimada a partir do Edital isolado de Etapa corrigido (ticket 06). Some
quando o Edital de verdade sai — a previsão muda quando isso acontecer.

**Turma viva:** o triênio 2024-2026, atendido a partir deste ticket. Ainda não tem Edital de
médias e desvios de nenhuma Etapa — as duas entradas derivadas são o que o atende até 2026.

**`usa_estatistica_derivada`:** o booleano que a resposta da API carrega para a tela saber
quando mostrar o aviso de estimativa. Computado em `model_package`, não recalculado no serviço.

**`scripts/gerar_stats_turma_viva.py`:** o script que reproduz os dois números de
`pas_constants.py` a partir dos PDFs — a evidência rastreável exigida pelo critério de aceite
sobre o valor bruto. Não escreve nada; só confere.

**Circularidade de calibração:** o bug que o filtro `origem is Origem.EDITAL` evita — calibrar
contra um valor que a própria calibração produziu, artificialmente perfeito. Só aparece em
reexecuções *depois* que as entradas `DERIVADA` já estão em `OFFICIAL_STATS`.

---

## Code review (`/code-review`)

Standards e Spec rodaram em paralelo. Spec voltou sem achado: os oito itens do checklist foram
conferidos um a um, incluindo re-rodar `scripts/gerar_stats_turma_viva.py` de novo para
confirmar que reproduz os literais de `pas_constants.py`. Standards trouxe três achados de
julgamento, nenhum bloqueante:

1. **`MAPEAMENTO` duplicado** entre `scripts/gerar_stats_turma_viva.py` e
   `scripts/medir_deslocamento.py`. Aceito como está — é um script de evidência, não código de
   runtime, e o docstring já admite a duplicação; extrair um `MAPEAMENTO` compartilhado é
   trabalho de quando um terceiro script precisar da mesma lista, não antes.
2. **`origem_da_prova` sem teste direto.** Corrigido: `tests/test_training_dataset.py` ganhou
   três testes (procedência sintética, procedência real de uma entrada de Edital, erro quando a
   chave falta), no mesmo padrão que já cobria `stats_da_prova`.
3. **A Gestão de Ativos (dashboard B2B) não expõe `usa_estatistica_derivada`.** Real, mas fora
   do escopo deste ticket — o título e o checklist falam da "tela do Preditor", o produto
   público. Registrado aqui como recomendação para quem tocar a Gestão de Ativos depois: a
   coordenação vê o mesmo Argumento estimado sem o aviso que o Aluno vê.

## O que fica para depois

- Quando o Edital de médias e desvios de 2024-2026 sair (2026), trocar as duas entradas por
  `Parte1PorLingua` + `Origem.EDITAL` com os valores oficiais — e apagar o comentário do bruto,
  que deixa de ser a fonte de verdade.
- Relatório 06 §9.2 já sugere declarar o resíduo da calibração na Largura de Incerteza (uma
  classe "Aluno servido por estatística derivada"); este ticket não fez isso — o
  `usa_estatistica_derivada` avisa o Aluno, mas a probabilidade de aprovação ainda usa a mesma
  Largura de quem tem Edital oficial.
