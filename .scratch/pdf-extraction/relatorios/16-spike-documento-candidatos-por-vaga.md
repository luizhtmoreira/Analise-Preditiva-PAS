# Relatório — Ticket 16: Spike, documento de candidatos por vaga como fonte de N

**Ticket:** `.scratch/pdf-extraction/issues/16-spike-documento-candidatos-por-vaga.md`
**Status:** concluído — spike, sem código de produção
**Onde vivem os artefatos de investigação:** este relatório; nenhum script novo em
`src/pas_extraction/` (fora de escopo de um spike)

---

## 1. O que foi pedido

Responder, com evidência medida (não suposição), se o documento de **candidatos por vaga**
("Demanda de candidato por vaga") é a fonte independente de N que faltava para fechar o ponto
cego documentado no ticket 02 (`_buracos_por_sistema` infere N como `max(posições observadas)`
porque nenhum Edital de Resultado Final declara o total real de candidatos — se o registro
perdido for justo o de classificação N, o `max` encolhe junto e a checagem não vê buraco
nenhum).

## 2. Conclusão

**Não fecha o ponto cego do ticket 02.** O documento existe, está em `data/pdfs`, e reporta
candidatos na granularidade certa (por curso, campus, turno e Sistema de Concorrência) — mas o
número que ele publica (`Inscritos`) usa uma definição de "candidato" diferente e
sistematicamente maior que a do Resultado Final (`max(posições observadas)`), com diferenças de
dezenas a centenas de candidatos por curso. Uma técnica desenhada para pegar a perda de
exatamente 1 posição não tem uso para uma fonte cujo desvio típico já é de dezenas de posições
— o "sinal" que se quer detectar (1 registro faltando) está soterrado no "ruído" da própria
diferença de definição.

O ponto cego do ticket 02 continua de pé, sem fonte independente disponível — é limitação
permanente da técnica, como já estava documentado.

Ao mesmo tempo, o documento **é útil para outra coisa**: o ticket 19 (conteúdo de
vagas/candidatos na página pública de curso) não precisa da mesma definição de candidato que a
checagem de buracos — precisa só de um número de "quanta gente concorre por vaga", que é
exatamente o que `Inscritos`/`Demanda` já publicam, na granularidade certa. O spike libera o
ticket 19 para prosseguir, com essa ressalva de definição registrada no próprio ticket.

---

## 3. Evidência

### 3.1 Localização — item 1 do checklist

O documento **não estava** entre os PDFs já baixados quando o ticket foi aberto. Apareceu em
`data/pdfs/candidatos-por-vaga/` em 2026-08-08 (4 arquivos, confirmado por `find -newer
data/pdfs/INDICE.md`), obtido separadamente pelo Luiz — a mesma fonte que o próprio ticket já
citava ("documento exclusivo disponibilizado por eles").

Cobertura por triênio (identificada pelo cabeçalho `SUBPROGRAMA <ano>` de cada PDF):

| Arquivo | Subprograma/Triênio | Etapa |
|---|---|---|
| `PAS_21 - 3 Etapa - Demanda de candidato por vaga - Primeiro Semestre.pdf` | 2021 (2021/2023) | 3 |
| `PAS_22 - 3 _ Demanda de candidato por vaga.pdf` | 2022 (2022/2024) | 3 |
| `D0125082...pdf` (nome de hash) | 2023 (2023/2025) | 3 |
| `D72275C2...pdf` (nome de hash) | 2023 (2023/2025) | 3 |

Cobre só 2 dos 8 triênios presentes em `resultado_final.csv` hoje (2021/2023, 2022/2024,
2023/2025 parcialmente — os dois PDFs de 2023 parecem ser 1º/2º semestre, não dois triênios
diferentes). Os 5 triênios mais antigos (2016/2018 a 2020/2022) não têm esse documento em mãos
ainda. Fora de escopo deste spike buscar os que faltam — registrado aqui para quem abrir o
ticket 19 saber que a cobertura é parcial.

A pasta ainda não está listada em `data/pdfs/INDICE.md` (gerado antes dela existir) nem
categorizada por `scripts/organizar_pdfs.py` — ajuste de housekeeping, não faz parte do
checklist deste ticket, mas fica registrado para quem tocar o ticket 19.

### 3.2 Granularidade — item 2 do checklist

**Por (curso, campus, turno, Sistema de Concorrência)** — mais fino, inclusive, que o grão da
checagem de buracos do ticket 02 (que é por curso + Sistema, sem separar turno). Estrutura de
cada linha, confirmada extraindo o texto de todos os PDFs com `pypdf`:

- Uma linha por curso, sob um cabeçalho de seção `Campus .../ Diurno` ou `.../ Noturno` — cursos
  com os dois turnos (ex. Administração) aparecem em duas linhas distintas, uma por turno.
- 11 grupos de 3 números por linha: 5 pares (Sistema de Cotas Escolas Públicas × 2 faixas de
  renda, Sistema de Cotas para Negros × 2 faixas de renda, Sistema Universal) × (Deficientes,
  Geral) = 10 grupos, mais 1 grupo Total. Cada grupo é `(Vagas, Inscritos, Demanda)`.
- Isso bate com os 10 `classificacao_sistema_1..10` que `resultado_final.csv` já carrega (ticket
  01/06) — mesma partição de Sistema de Concorrência, mesma contagem de 10 categorias.

### 3.3 Comparabilidade com `max(posições observadas)` — item 3 do checklist

**Não comparável.** Medido cruzando `Inscritos` (coluna Total, última do grupo de 11) contra a
contagem real de linhas em `resultado_final.csv` para o triênio 2021/2023, por curso × turno
(`data/resultado_final.csv`, coluna `curso`/`turno`, filtrando `trienio=='2021/2023'` e
excluindo cursos "SUB JUDICE"):

| Curso | Turno | `Inscritos` (PDF) | Candidatos no Resultado Final | Diferença |
|---|---|---:|---:|---:|
| Administração (Bacharelado) | Diurno | 209 | 160 | +49 |
| Administração (Bacharelado) | Noturno | 76 | 49 | +27 |
| Agronomia (Bacharelado) | Diurno | 125 | 91 | +34 |
| Ciência da Computação (Bacharelado) | Diurno | 476 | 345 | +131 |

(86 de 87 cursos do triênio casaram por nome normalizado; a amostra completa confirma o mesmo
padrão — `Inscritos` maior que o total do Resultado Final na maioria dos casos, com desvios de
dezenas a mais de cem candidatos, não de 1 ou 2.)

**Por quê:** "Demanda de candidato por vaga" é um documento de **inscrição**, publicado antes
da correção da Etapa 3 — `Inscritos` conta quem se inscreveu para concorrer, não quem chegou ao
fim sem ser eliminado. `max(posições observadas)` no Resultado Final é o oposto: só conta quem
sobreviveu até ter uma classificação final publicada (exclui ausentes, eliminados, etc.). São
dois estágios diferentes do mesmo funil, com definição de "candidato" incompatível — não uma
questão de arredondamento ou pequena defasagem.

Uma verificação isolada por curso (sem separar turno) tinha dado um "acerto" aparente — 209 =
209 para Administração — mas era coincidência: o PDF separa por turno (209 Diurno + 76 Noturno
= 285 inscritos) e o total real por curso, somando os dois turnos, também soma 209 (160 + 49) —
os dois totais batendo em 209 é um acaso aritmético dessa combinação específica, desfeito assim
que a comparação é feita no grão certo (por turno). Nenhum outro curso testado repetiu esse
acaso.

---

## 4. Por que a limitação do ticket 02 continua de pé

Mesmo com o grão certo (curso × Sistema, até mais fino com turno), a diferença sistemática entre
`Inscritos` e o Resultado Final — dezenas a centenas de candidatos por curso — é ordens de
grandeza maior que o efeito que a checagem de buracos do ticket 02 precisa detectar: a perda de
exatamente 1 registro (o de posição N) dentro de um Sistema. Um comparador que já diverge por
30-130 não consegue distinguir "N real é X" de "N real é X-1" — não tem sensibilidade para o
caso que importa. Fechar esse ponto cego continuaria a exigir uma fonte que publique o total de
**classificados/não eliminados** por (curso, Sistema) — que, pelo que se sabe hoje, nenhum
Edital declara. Sem essa fonte, a limitação documentada em `validacao.py` e no relatório do
ticket 02 (§3.4) permanece válida.

## 5. Consequência para o ticket 19

O ticket 19 (vagas e candidatos por vaga na página pública) **não depende** dessa conclusão
negativa — ele só precisa de "quantos candidatos concorrem, por curso/Sistema", e é exatamente
isso que `Inscritos`/`Demanda` publicam, na granularidade que a página pública quer (`vagas: N,
candidatos: M, concorrência: M/N`). A ressalva a herdar: o número exibido representa
**inscrição**, não o total de candidatos que chegaram ao fim do processo — vale a pena que a
página deixe isso implícito no rótulo (ex. "candidatos inscritos por vaga", não "candidatos
classificados por vaga") para não confundir com a Nota de Corte, que é derivada de quem chegou
ao fim.

Cobertura parcial (3.1): o ticket 19, se for extrair todos os 8 triênios hoje presentes em
`resultado_final.csv`, só vai conseguir para 2021/2023 e 2022/2024 (e parte de 2023/2025) até
que os documentos dos triênios mais antigos sejam obtidos.

---

## 6. Escopo deliberadamente fora deste ticket

| Não feito aqui | Por quê |
|---|---|
| Extrator de produção para o documento | Ticket 19, que agora está desbloqueado por esta conclusão |
| Buscar os PDFs dos 5 triênios sem cobertura | Fora do escopo de um spike; registrado como lacuna conhecida (3.1) |
| Categorizar as 4 PDFs em `organizar_pdfs.py`/`INDICE.md` | Housekeeping, não bloqueia a resposta do spike |

---

## 7. Glossário — termos novos deste ticket

| Termo | Significado |
|---|---|
| **Demanda de candidato por vaga** | Documento publicado pela banca (Cebraspe) antes da correção da Etapa 3, com `Vagas`, `Inscritos` e `Demanda` (razão Inscritos/Vagas) por curso, campus, turno e Sistema de Concorrência. Reflete quem se inscreveu, não quem terminou o processo. |
| **`Inscritos`** | Coluna do documento de Demanda — candidatos que se inscreveram para concorrer, medido antes da eliminação/desistência. Não é o mesmo universo que `max(posições observadas)` do Resultado Final. |
