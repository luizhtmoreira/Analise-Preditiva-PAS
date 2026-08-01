# 09 — Família Convocação

**What to build:** a segunda Família de Edital entra no pipeline: quem foi chamado, em que chamada
e em que Sistema de Concorrência. É o que falta para derivar Nota de Corte, cuja definição no
`CONTEXT.md` é o Argumento Final mínimo *na última chamada* — informação que o Resultado Final
sozinho não tem.

O classificador de schema declarado do ticket 01 já separa esta família. Duas diferenças em
relação ao Resultado Final:

**Modo de extração `layout`, não `plain`.** Contraintuitivo dado que `layout` produz mais números
partidos no fluxo, mas aqui o dado é colunar e depende do alinhamento visual para ser lido.

**Triênio, semestre e número da chamada lidos do conteúdo do Edital**, não de tabela hardcoded. A
tabela do `extrator_master.py` já está dessincronizada dos arquivos em disco — referencia PDFs que
não estão mais em `data/pdfs` — e essa dessincronização silenciosa é exatamente o que o pipeline
existe para eliminar.

Um detalhe do código antigo é aproveitável e vira constante compartilhada entre as duas famílias:
o `mapa_sistemas` de `extrator_master.py` numera os 10 Sistemas de Concorrência de 1 a 10, na
mesma ordem em que as classificações aparecem no Resultado Final. É o que liga a convocação às
cotas deduzidas no ticket 06.

CSV próprio da família — a granularidade aqui é uma convocação, não um Aluno — com as mesmas
colunas de proveniência.

**Blocked by:** 01 — Costura `extrair_edital` + classificador de família + CSV de Resultado Final.

**Status:** ready-for-agent

- [ ] A Família *Convocação* é reconhecida pelo classificador de schema declarado
- [ ] A extração usa modo `layout`, e o Resultado Final continua em `plain`
- [ ] Triênio, semestre e número da chamada são lidos do conteúdo do Edital, sem tabela hardcoded
- [ ] Sai um CSV próprio, com quem foi chamado, em que chamada e em que Sistema de Concorrência
- [ ] O `mapa_sistemas` é uma constante compartilhada entre as duas famílias, com os 10 sistemas na ordem em que as classificações aparecem
- [ ] Existe fixture de convocação gerada localmente (não commitada, ver ticket 01), e um teste que verifica a contagem de registros extraídos dela, pulando se a fixture não existir
