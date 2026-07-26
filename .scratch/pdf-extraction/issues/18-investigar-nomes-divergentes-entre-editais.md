# 18 — Investigar nomes divergentes entre Editais

**What to build:** não é ticket de implementação — é uma investigação que classifica, um a um,
os 10 casos em que a mesma inscrição aparece com nomes diferentes em Editais distintos do mesmo
triênio, achado pela reconciliação cruzada do ticket 08
(`.scratch/pdf-extraction/relatorios/08-rodada-completa-deterministica.md`: "10 inscrições com
nome divergente entre Editais diferentes, em ~100 mil registros cruzados").

Esse achado nunca foi investigado a fundo — só registrado como "proporção baixa, compatível com
ruído de extração isolado". É candidato a ser inteiramente explicado pelo ticket 13 (nome
quebrado por espaço no meio da palavra): se um dos dois Editais tiver o nome corrompido por
aquele defeito e o outro não, a reconciliação por inscrição os veria como "divergentes" mesmo
sendo a mesma pessoa. Por isso este ticket só deve rodar **depois** do 13, para não gastar
esforço analisando um sintoma que o 13 já resolve.

**Blocked by:** 13 — Reparo de nome quebrado por espaço.

**Status:** ready-for-agent

- [ ] Os 10 casos (ou o conjunto atualizado após o corpus ser reprocessado com o ticket 13) são
      revistos comparando os dois nomes de cada par
- [ ] Cada caso é classificado: (a) resolvido pelo reparo do ticket 13, ou (b) causa raiz
      diferente (ex. erro de digitação real no Edital, grafia alterada entre triênios, etc.)
- [ ] Se sobrar algum caso de causa raiz distinta, documentado em `defeitos-pendentes.md` como
      item novo, com número próprio
- [ ] Se todos os casos forem explicados pelo ticket 13, o item 6 de `defeitos-pendentes.md` é
      fechado com essa conclusão (sem novo defeito a corrigir)
