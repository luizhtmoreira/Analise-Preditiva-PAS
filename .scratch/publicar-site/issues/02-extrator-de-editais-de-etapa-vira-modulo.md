# 02 — Extrator de Editais de Etapa vira módulo com teste

**What to build:** extrair média e desvio de um **Edital isolado de Etapa** passa a ser uma
operação reprodutível do repositório, com teste, em vez de um script descartável que roda à mão.

Hoje o extrator vive em `.scratch/publicar-site/medicao-passo-1/extrair_etapa.py`, escrito para
responder uma pergunta de medição e descartado depois. Ele precisa virar módulo porque **não é
operação de uma vez**: a calibração do Deslocamento (ticket 06) roda de novo a cada triênio novo, e
em 2026 o Edital real vai substituir as entradas derivadas.

Depois deste ticket, apontar o módulo para uma pasta de PDFs produz um `medias_desvios_etapa.csv`
com média e desvio por `(ano, etapa)` — Parte 2, Redação e a Parte 1 misturada — pronto para
alimentar as entradas derivadas do ticket 07.

**O que o Passo 1 já provou e que sustenta a promoção:** seis Editais, ~19,5 mil registros cada,
**zero falhas** no checksum embutido, e as notas batem em **99,63%** com o `resultado_final.csv`
para os Alunos que aparecem nos dois lados.

**O checksum é o próprio Edital.** Cada registro lista `inscrição, nome, EB parte 1, EB parte 2,
somatório, nota tipo D, nota de redação`, e `parte 1 + parte 2 = somatório`. Registro cuja extração
de texto saiu corrompida não fecha e é descartado — é o mesmo mecanismo do ticket 04 da extração,
e é o que dispensa conferência humana de 19 mil linhas.

**Armadilha de documento que precisa virar código.** "Retificação" no nome do arquivo **não diz** se
o Edital é parcial ou completo, e confiar nisso custou tempo:

- Edital 8 de 2023 (retificação): **827 registros** — parcial, não serve;
- Edital 7 de 2023: 19.505 registros — completo;
- Em 2022 foi ao contrário — o Edital **original** não trazia os escores brutos das Partes 1 e 2,
  que só apareceram na retificação.

O módulo tem que **contar os registros e recusar** um documento parcial, com mensagem dizendo
quantos achou, em vez de deixar a conferência para quem rodar.

**Segunda armadilha, já neutralizada mas que precisa continuar neutralizada:** os números saem do
PDF com espaço no meio (`2. 046`, `1 6.005`, `0 .220`). A normalização que remove espaço antes de
converter já existe no script; o checksum é a rede que prova que ela funcionou.

**Onde o módulo mora:** `src/pas_extraction/` é gitignored (lógica de extração não é pública)
enquanto `tests/test_pas_extraction_*.py` é rastreado. O módulo novo segue esse arranjo, que já é o
do resto da frente. Prior art de forma: `medias_desvios.py` e `notas_corte.py`; prior art de
fixture sintética: `fixtures.py`.

**Blocked by:** Nenhum — pode começar imediatamente.

**Status:** ready-for-agent

- [ ] Existe um módulo em `src/pas_extraction/` que lê um Edital isolado de Etapa e devolve os
      registros validados pelo checksum embutido, com diagnóstico dos descartes por motivo
- [ ] Um Edital parcial é **recusado** com mensagem nomeando a contagem encontrada; o caso real do
      Edital 8 de 2023 (827 registros) é o teste
- [ ] Números com espaço interno (`2. 046`, `1 6.005`) são normalizados, e o checksum prova isso
      numa fixture sintética
- [ ] A saída é um `medias_desvios_etapa.csv` com `(ano, etapa)`, `n`, e média/desvio de Parte 2,
      Redação e Parte 1 misturada
- [ ] Teste sobre fixture **sintética** — nenhuma linha de Aluno real entra em teste ou fixture
      rastreada
- [ ] Rodando sobre os 6 Editais que já estão em `data/pdfs`, reproduz os números do Passo 1
      (`.scratch/publicar-site/medicao-passo-1/editais_isolados.csv`)
- [ ] `pytest tests/` continua verde
