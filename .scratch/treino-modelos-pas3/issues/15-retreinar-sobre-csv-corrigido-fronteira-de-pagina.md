# 15 — Retreinar sobre o CSV corrigido (fronteira de página)

**Type:** task
**Status:** ready-for-agent
**Relatório:** `relatorios/15-retreinar-sobre-csv-corrigido-fronteira-de-pagina.md`

## Question

**Nada se decide aqui** — mesma cláusula do ticket 13. O mapa deste projeto está fechado
(`map.md`, 2026-07-28); este é um ticket de execução isolado, motivado por um defeito de dado
descoberto depois do fechamento, não por uma decisão nova de modelagem.

**O que aconteceu:** o `resultado_final.csv` que produziu `models/pas3/` (`manifest.json`,
`criado_em: 2026-07-29T02:32:50`, 64.298 linhas) foi extraído **antes** do ticket 15 do mapa
`pdf-extraction` (`.scratch/pdf-extraction/relatorios/15-fronteira-de-pagina-no-parser.md`), que
corrigiu um bug no parser de Resultado Final: o `pypdf` emitia o número da página no início do
texto de cada página, e quando um campo de registro caía exatamente na fronteira, o parser lia
esse número no lugar do valor real.

**Impacto medido no CSV** (relatório do ticket citado acima, §3-4):
- **380 registros de Aluno genuínos** que eram descartados silenciosamente (parse falhava em
  algum campo corrompido) voltam a aparecer — 0,57% do corpus de 66.693 registros pós-correção.
- **52 registros** têm algum campo de classificação (Sistema de Concorrência) corrigido — esse
  campo **não entra no vetor de features do modelo A3** (`CLAUDE.md`, "Canonical feature
  vector"), então essa parte do defeito não tocou o treino em si, só a dedução de cota.
- Os 380 registros recuperados, ao contrário, têm notas e Argumento Final válidos (conferido
  por amostragem no relatório do ticket) — são linhas de treino genuínas que faltaram, não uma
  correção de valor dentro de uma linha que já existia.

**Por que abrir ticket em vez de só rodar o comando:** o mapa fechado documenta que promover
modelo novo exige comparação lado a lado revisada pelo dono do produto antes de substituir o
artefato em produção (ticket 13, "Comparação lado a lado antes de promover") — a mesma barra
vale aqui, mesmo sendo um retreino "de manutenção", porque o comportamento do modelo em produção
muda.

## O que fazer

1. Republicar `resultado_final.csv` corrigido: `deploy/publicar_pacote.py dados` (ou o comando
   equivalente vigente — conferir `deploy/README.md`). Atualiza `deploy/ponteiro.json`.
2. Retreinar: `.venv/bin/python scripts/treinar_pipeline.py <resultado_final.csv corrigido>
   --saida <dir>`. O comando **recusa gravar** se o Portão 1 (critério de aceite do ticket 06 do
   mapa) não for batido — não passar `--forcar` sem entender por que reprovou.
3. Comparar lado a lado o pacote novo contra `models/pas3/` atual: RMSE em `A3`, viés, erro de
   decisão — mesmo formato da tabela do relatório do ticket 13. Diferença esperada é pequena
   (0,57% mais linhas de treino), então o objetivo aqui é confirmar "não piorou", não caçar
   ganho.
4. Se aprovado, promover: mover o pacote atual para `models/aposentados-<data>/` (mesmo padrão
   do ticket 13) e o novo para `models/pas3/`.
5. Atualizar `manifest.json`/`CLAUDE.md` se algum número citado (RMSE, largura de incerteza,
   contagem de registros) mudar o suficiente para ficar desatualizado.

**Blocked by:** nenhum — o CSV corrigido e o comando de treino já existem, prontos para rodar.

- [ ] `resultado_final.csv` corrigido republicado (`deploy/ponteiro.json` atualizado)
- [ ] Pipeline de treino roda sobre o CSV novo e bate o Portão 1 sem `--forcar`
- [ ] Comparação lado a lado (modelo atual vs. novo) revisada pelo dono do produto antes da
      promoção — mesmo critério do ticket 13
- [ ] `models/pas3/` atualizado, anterior preservado em `models/aposentados-<data>/`
- [ ] `CLAUDE.md`/`manifest.json` conferidos quanto a números que ficaram desatualizados
- [ ] Relatório em `relatorios/15-retreinar-sobre-csv-corrigido-fronteira-de-pagina.md`
