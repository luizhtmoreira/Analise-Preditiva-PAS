# Defeitos pendentes conhecidos — `src/pas_intelligence/` e avaliação de modelos

Registro consolidado dos defeitos **documentados e ainda não corrigidos** na camada de
inteligência do PAS (`src/pas_intelligence/`, `scripts/baseline_avaliacao.py`, `docs/adr/`).
Não substitui os relatórios de ticket em `relatorios/` — cada entrada aponta para a fonte e
resume o que falta fazer. Objetivo: ponto único de partida para decidir o próximo ticket.

Convenção herdada do mapa `pdf-extraction`: cada entrada tem **Onde foi encontrado**, **O
defeito**, **O que falta fazer** e **Severidade** (impacto nos dados/produto, não esforço).

---

## 1. `test_guaranteed_scenario` codifica um contrato de domínio errado (e o piso da P2 não está documentado)

**Onde foi encontrado:** `pytest tests/test_pas_intelligence.py` —
`TestTargetCalculator::test_guaranteed_scenario` falha. Confirmado por `git stash` em
2026-07-26 que a falha **antecede** as mudanças do ticket 03 (34 passavam / 1 falhava antes,
37 passam / 1 falha depois). Não foi introduzida por aquele trabalho.

**Fato de domínio que resolve a leitura:** no PAS, **nota de prova pode ser negativa** — o
Escore Bruto desconta erro, tal como o Argumento pode ser negativo. O próprio código já
carrega isso: `target_calculator.py:125` trunca `p1_pred` em `max(-20.0, min(20.0, ...))`,
ou seja, P1 ∈ [−20, 20].

**O defeito:** o teste espera que um cenário garantido produza
`status == 'garantido'` e `p2_necessario == 0.0`. Mas o código
(`target_calculator.py:291-304`) trabalha com a faixa simétrica implícita **P2 ∈ [−100, 100]**:

```python
if   p2_necessario >  100:  status = 'impossivel'
elif p2_necessario < -100:  status = 'garantido';  p2_necessario = -100.0
elif p2_necessario >   80:  status = 'improvavel'
else:                        status = 'possivel'
```

Reproduzido com o cenário do próprio teste (histórico máximo, alvo baixo):

```
status = possivel  |  p2_necessario = -99.44  |  arg_pas3_necessario = -99.233
```

Sob P2 com sinal, `-99.44` significa literalmente *"você precisa de pelo menos −99,44 na
Parte 2"* — quase qualquer desempenho serve, mas **não é garantido**, porque ainda existe um
desempenho pior que isso dentro da faixa. Então `'possivel'` está correto e o truncamento em
`-100.0` (piso da faixa) é coerente. **Quem codifica o contrato errado é o teste**, ao presumir
que zero é o mínimo da prova.

**O que falta fazer:**

1. **Corrigir o teste**, não o código: o cenário "garantido" precisa de um alvo baixo o
   bastante para levar `p2_necessario` abaixo de `-100`, e a asserção passa a ser
   `p2_necessario == -100.0`.
2. **Verificar qual é o piso real da P2** — esta é a pergunta aberta de verdade. Os literais
   `100` / `-100` aparecem **só** nessas duas linhas, sem constante nomeada, sem docstring e
   sem fonte no Edital. Se a faixa real for mais estreita (a P2 tem número finito de itens, e
   o piso é `−N` para `N` itens), o ramo `'garantido'` é **código morto** e nenhum aluno jamais
   o alcança. Se for mais larga, o `'impossivel'` dispara cedo demais. Enquanto o número não
   tiver origem documentada, os quatro status repousam sobre uma faixa chutada.
3. Ao mexer, checar o consumidor: `api/services/gestao_service.py:265` compõe
   `eb_nec = path.p1_estimado + path.p2_necessario`, e um `-100.0` ali empurra o EB necessário
   para muito abaixo de zero antes de ir para `calculate_cohort_evolution_probability`.

**Severidade: baixa para o teste, média para o piso não documentado.** A falha do teste é uma
asserção errada e não afeta o aluno. O que afeta é a faixa `[−100, 100]` sem procedência: ela
decide, sozinha, quando o produto diz "impossível" e quando diz "garantido".

**Nota de comunicação (não é defeito de lógica):** mesmo correto, o texto
*"Meta alcançável! Você precisa de -99.4 pts na Parte 2"* é ruim de ler. Um aluno nessa
situação entende melhor "praticamente qualquer desempenho na Parte 2 mantém sua aprovação".
Melhoria de mensagem, sem urgência.

---

## 2. `ADR-0007` (baseline dos modelos v1) é inválido: features na ordem errada

**Onde foi encontrado:** ticket 03, ao inspecionar o booster para medir o formato nativo do
LightGBM. Ver `relatorios/03-formato-e-versionamento-do-artefato.md`, seção 2.2.

**O defeito:** os modelos foram treinados com

```
booster.feature_name() → ['EB_PAS1', 'Red_PAS1', 'EB_PAS2', 'Red_PAS2', 'Cresc_EB', 'Cresc_Red']
```

e `scripts/baseline_avaliacao.py:55` os alimenta com

```
FEATURE_COLS_BASE = ["EB_PAS1", "EB_PAS2", "Cresc_EB", "Media_EB", "Std_EB", "CV_EB"]
```

Só a posição 1 coincide; as outras cinco estão trocadas. O `CLAUDE.md` documenta o vetor
correto, o que confirma que o errado é o script de avaliação. O LightGBM aceitou em silêncio
porque recebeu um array NumPy — array não carrega nome de coluna, só posição.

Isso explica os números impossíveis do ADR: `R² = -83.4`, `MAPE = 1.25e+19`, `MaxErr = 31293`
num alvo que vai de 0 a 92. Não é desempenho ruim; é modelo alimentado com lixo.

**O que falta fazer:** ticket 07. Consertar `FEATURE_COLS_BASE`, rodar de novo, e emitir um
ADR corrigido — o ADR-0007 se declara "somente leitura" e diz que qualquer modelo retreinado
deve ser comparado contra ele, o que hoje é uma régua de borracha. Não presumir que a correção
inverte o ranking: `mlp`, `arg_final` e `linear` podem ser ruins de verdade.

**Severidade: alta.** É a linha de base contra a qual o mapa inteiro prometeu se comparar.
Enquanto estiver assim, qualquer modelo novo parece um triunfo contra ruído.

---

## 3. `p1_pas3_model` e `red_pas3_model` não carregam (causa raiz pendente)

**Onde foi encontrado:** ticket 03, seção 2.1. Já registrado no ADR-0007 em 2026-07-20 como
nota de rodapé (`⚠ Modelo não carregado — incompatibilidade de versão do sklearn`) sem
tratamento.

**O defeito:** ambos falham com `ModuleNotFoundError: No module named '_loss'` no ambiente
atual (`python 3.14.3`, `scikit-learn 1.9.0`). Foram serializados quando o `sklearn` ainda
tinha o módulo interno `_loss` em outro lugar. Os números seguem íntegros no arquivo; a receita
de remontagem é que aponta para o vazio.

**Estado atual — mitigado, não corrigido (2026-07-26):** a degradação deixou de ser silenciosa.
`target_calculator.py` agora registra o motivo em `self.model_load_error`, devolve
`fallback_reason` junto de `method`, grita em log `ERROR`, e levanta `ModelLoadError` com
`PAS_STRICT_MODELS=1`. O `except Exception: pass` de `api/services/gestao_service.py:268` virou
`logger.exception`. **Mas a calculadora reversa continua respondendo por média ponderada em vez
de ML** — a feature segue degradada, agora com aviso.

**O que falta fazer:** ticket 12 regera os artefatos sob o esquema decidido no ticket 03
(manifesto com versões, portão de carregamento no build). A partir daí, ligar
`PAS_STRICT_MODELS=1` em produção. Também revisar o contrato de features de
`predict_stable_components` (`target_calculator.py:96-119`): ele monta 11 colunas — 6 base mais
`delta_p1`, `delta_red`, `delta_p2`, `mean_p1`, `mean_red` — cuja ordem só está registrada num
**comentário**. É exatamente a fragilidade do defeito 2 esperando para se repetir; essa ordem
tem que passar a viver no `manifest.json`.

**Severidade: média.** A feature degrada em vez de mentir, e agora avisa. Mas a calculadora
reversa é um dos produtos anunciados e está sem ML há meses.
