# 09 — Troca dos CSVs de Nota de Corte e da base populacional

**What to build:** o Preditor passa a comparar a previsão do Aluno com as Notas de Corte que a
frente de extração produziu, e não com a base ad-hoc antiga.

Hoje a API lê `data/notas_corte_pas.csv` — **2.307 linhas**, base ad-hoc, sem o triênio 2023-2025.
A frente de extração já produziu um `notas_corte.csv` com **5.225 linhas**, incluindo 2023-2025. A
troca nunca foi feita: **o modelo novo está sendo servido contra Notas de Corte velhas.**

Uma **Nota de Corte** é o menor Argumento Final entre os aprovados de um curso, num Sistema de
Concorrência, na última chamada. É o número contra o qual a probabilidade de aprovação é calculada —
se ele estiver errado, a chance na tela está errada por igual.

## Os dois esquemas são diferentes

O CSV novo usa nomes minúsculos (`trienio`, `sistema_nome`, `curso`, `turno`, `campus`,
`nota_corte`, `chamada`); o carregamento atual espera `Trienio`, `Sistema_Nome`, `Curso_Limpo`,
`Semestre`, `Min`. A tradução acontece no ponto único de carga que já existe (`load_resources`) —
não espalhe adaptação pelos serviços.

## ⚠ O CSV novo carrega PII

Ele tem colunas `inscricao` e `nome` de Alunos reais — é assim que o pipeline registra de quem veio
o corte. O arquivo é gitignored, mas **vai para dentro da imagem Docker** do ticket 08. A API serve
só o corte agregado: as duas colunas de PII são **descartadas na carga**, não embarcadas.

## Ordem interna que não dá para inverter

Antes de promover o `notas_corte.csv`, feche o **ticket 14 da frente de extração**
(`.scratch/pdf-extraction/issues/14-validacao-formato-classificacao.md` — validação de formato do
campo de classificação). Sem ele, cortes implausíveis passam. O caso conhecido é MEDICINA / Darcy
Ribeiro / Sistema Universal em 2020-2022 saindo com `199.162,872`. Um corte desses no Preditor
público vira uma probabilidade absurda na tela de um Aluno.

## Contagem: meça, não herde

O mapa registra "4.786 cortes". Medido em disco em 2026-07-29: **5.225 linhas**, das quais **4.986**
com `checksum_fecha == True` e 5.154 não parciais. **Nenhum dos três recortes dá 4.786.** Meça, e
escreva o critério do recorte escolhido (só as limpas ou todas) em vez de herdar o número.

## A base populacional tem a mesma família de sujeira, e o checksum pega tudo

`resultado_final.csv` tem 66.313 registros em 8 triênios. **510 linhas (0,77%)** têm nota com escala
corrompida — `eb_p2` chegando a 39.617. **Todas as 510 falham o `checksum_fecha`**, então o filtro é
único e resolve: `checksum_fecha == True` deixa **64.298 de 66.313**. A contaminação está só nos
cinco triênios mais antigos; os três recentes estão limpos.

O arquivo vive em `.scratch/pdf-extraction/saida-nova/resultado_final.csv`, **não** em `data/`.

**Blocked by:** ticket 14 da frente de extração
(`.scratch/pdf-extraction/issues/14-validacao-formato-classificacao.md`).

**Status:** ready-for-agent

- [ ] A API serve as Notas de Corte novas, com a tradução de esquema no ponto único de carga
- [ ] `inscricao` e `nome` são descartados na carga e não existem em memória nem na imagem
- [ ] Uma varredura de plausibilidade não acha nenhum corte fora da faixa observada de Argumento
      Final nos 8 triênios; o caso MEDICINA/Darcy/2020-2022 não aparece
- [ ] A contagem final de cortes promovidos é **medida** e o critério do recorte está escrito
- [ ] A base populacional é filtrada por `checksum_fecha == True` (64.298 de 66.313) onde quer que
      seja consumida
- [ ] O Preditor responde para o triênio 2023-2025, que não existia na base antiga
- [ ] `pytest tests/` continua verde
