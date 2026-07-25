# 10 — Notas de Corte por curso e por Sistema de Concorrência

**What to build:** a Nota de Corte de cada curso deixa de ser um número único e passa a ser um
número por Sistema de Concorrência.

Hoje as Notas de Corte do projeto não distinguem sistema de concorrência, embora a informação
exista no Edital — um Aluno que concorre por Cota para Negros é comparado contra um corte
Universal que não é o dele. A informação já está no Resultado Final: as 10 classificações são os
sistemas, e o ticket 06 já deduziu o perfil de cada Aluno.

A Nota de Corte é, por definição no `CONTEXT.md`, o Argumento Final mínimo exigido para aprovação
num curso **na última chamada**. Daí a dependência da família de convocação: o Resultado Final diz
como cada Aluno se classificou em cada sistema, mas só a convocação diz até onde a chamada chegou.

**Algoritmo exato** (definido pelo dono do produto em 2026-07-24): para um curso e um Sistema de
Concorrência, identifica-se primeiro a maior chamada em que houve convocação naquele sistema (se
houve 4 chamadas, é a 4ª; se aquele sistema não teve convocado na 4ª, cai pra 3ª, e assim por
diante). A Nota de Corte é o **menor** Argumento Final entre os Alunos convocados naquele sistema
**nessa** chamada — se mais de um Aluno do mesmo sistema aparece na maior chamada, o corte é o da
menor nota entre eles, não a média nem a maior.

Saída em CSV, com proveniência, como as demais. Carregar isso em Supabase ou consumir no app está
fora de escopo — a decisão sobre o banco é trabalho posterior.

**Blocked by:**
- 06 — Dedução das Cotas Declaradas
- 08 — Rodada completa sobre os 77 Editais, determinística
- 09 — Família Convocação

**Status:** ready-for-agent

- [ ] Sai uma Nota de Corte por curso e por Sistema de Concorrência, não uma por curso
- [ ] Para cada (curso, sistema), a maior chamada com convocado naquele sistema é identificada primeiro
- [ ] O corte é o menor Argumento Final entre os convocados daquele sistema nessa maior chamada — não a média, não o maior
- [ ] Um curso/sistema sem convocado numa chamada mais recente cai corretamente para a chamada anterior mais recente que teve
- [ ] A saída é CSV com proveniência, sem carga em banco e sem alteração no app
- [ ] Um teste verifica o corte derivado de um curso com dado conhecido, incluindo o caso de empate/múltiplos Alunos do mesmo sistema na maior chamada
