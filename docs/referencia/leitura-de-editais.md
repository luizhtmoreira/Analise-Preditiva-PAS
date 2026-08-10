# Leitura dos editais

`src/pas_extraction/` — o pipeline que transforma os PDFs oficiais do Cebraspe na base de dados do
produto. A leitura não técnica deste mesmo assunto está em
[De onde vêm os dados](../confianca/de-onde-vem-os-dados.md).

!!! warning "Este módulo não está no controle de versão"
    A pasta inteira está no `.gitignore` por decisão de privacidade — ela contém PDFs e planilhas
    com dados de alunos reais. Os testes (`tests/test_pas_extraction*.py`) são versionados
    normalmente e rodam sobre PDFs sintéticos gerados em tempo de execução.

    Existe um repositório privado de backup e uma rotina de sincronização obrigatória após
    qualquer alteração; veja o `CLAUDE.md` na raiz.

## Famílias de documento

A família de cada edital é decidida pelo **schema que o próprio documento declara** na primeira
página, canonizado — não pelo nome do arquivo. Validado contra os 77 PDFs reais: 64 convocações,
8 resultados finais, 5 tabelas de médias e desvios, zero desconhecidos e zero erros.

A ancoragem em termos estruturais faz um edital novo com redação institucional diferente ser
classificado corretamente **sem mudança de código** — foi o que aconteceu quando o Cebraspe passou
de "nome do candidato" para "nome da pessoa candidata" e de "nota final" para "nota provisória".

| Família | Volume | O que traz |
|---|---|---|
| Resultado Final | 8 editais | 9 notas, Argumento Final, 10 classificações, campus/curso/turno |
| Convocação | 64 editais | quem foi chamado, em que chamada, semestre e sistema |
| Médias e Desvios | 5 avulsos + 3 em cauda | média e desvio oficiais por etapa × prova × língua |

A tabela de médias e desvios é procurada nos dois lugares onde o Cebraspe a publica: edital avulso
(até 2020/2022) ou cauda do resultado final (a partir de 2021/2023). Nenhum triênio usa os dois
formatos.

## As seis camadas de validação

Cada registro carrega no CSV o resultado das próprias validações, para que o consumidor filtre por
confiança em vez de confiar cegamente.

1. **Checksum do Argumento Final** — recompõe o argumento a partir das 9 notas e da tabela oficial
   do mesmo edital. Um número verifica 12 campos. Tolerância `|delta| ≤ 0,005`, que não é folga
   arbitrária: é o arredondamento de três casas com que o Cebraspe publica todos os operandos,
   propagado na recomposição.
2. **Sequência de classificação `1..N`** sem buracos, por curso e por sistema. É a única camada que
   detecta um registro que o parser **nunca extraiu**.
3. **Ordem alfabética** dentro do curso — quebra indica registros colados.
4. **Formato numérico exato** (`^-?\d+\.\d{3}$`) — pega número partido por espaço. O reparo é
   tolerante *e* sinaliza: o valor é recuperado e o campo fica marcado.
5. **Fecho do reticulado de cotas** — padrão de cota impossível. Dos 512 padrões teóricos, apenas
   os 10 fechos válidos aparecem nos dados.
6. **Reconciliação cruzada entre editais** — mesma inscrição, mesmo nome. Verificação independente
   de qualquer fórmula.

!!! note "`checksum = None` nunca significa 'passou'"
    Quando a tabela oficial está incompleta, o edital sai **sem** checksum, e o campo fica nulo —
    distinto de `False`. Conferir contra uma tabela parcial mediria a falta da tabela, não a
    qualidade do registro.

## Diagnóstico por distribuição, não por taxa

O relatório de validação distingue deltas *concentrados* (indício de fórmula incompleta) de
*espalhados* (dado corrompido pontual). Isso não é estética: no protótipo, um checksum com 83,9%
de acerto teria descartado 200 de 1.261 registros perfeitamente válidos, e o que denunciou o
problema foi a **forma** da distribuição, não a taxa.

Regra absoluta do módulo: nenhum registro é descartado sem que o padrão da falha esteja explicado.

## Dados deduzidos

Duas informações são recuperadas sem estarem impressas:

- **Língua estrangeira por etapa**, inferida por qual das 27 combinações faz o checksum fechar.
  Cerca de 20% dos alunos trocam de língua entre etapas — tratá-la como fixa por aluno erraria em
  um quinto da base.
- **Perfil de cota declarado**, deduzido do padrão das 10 classificações publicadas.

Além disso, a **nota de corte por sistema de concorrência** é derivada cruzando o resultado final
com as convocações: 4.786 cortes, com chave de seis dimensões (triênio, semestre, campus, curso,
turno, sistema). Em 56% dos casos a chamada que define o corte de um sistema é anterior à maior
chamada do curso — o que é a prova de que "nota de corte do curso", sem o sistema, é um número
que não existe.

## Determinismo e proveniência

Duas execuções sobre a mesma entrada produzem CSVs idênticos byte a byte, verificado por
comparação direta. Empates têm regra fixa de desempate. Não há caminho absoluto de máquina.

Cada linha carrega arquivo de origem, número do edital, triênio e página.

## Fora de escopo

- **Alunos eliminados** — têm apenas duas notas e não formam o vetor de nove. Custo medido:
  cerca de 1.449 por edital.
- **Vagas e candidatos por vaga** — o documento existe e foi analisado, mas o extrator não foi
  construído. Cobre apenas os triênios recentes.
- **Download automático** dos editais novos: os PDFs entram manualmente.
