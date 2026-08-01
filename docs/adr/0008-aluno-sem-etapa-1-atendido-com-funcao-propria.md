# O Aluno sem Etapa 1 é atendido, com função própria, e a ausência é declarada

O PAS permite faltar à Etapa 1 e seguir no programa; quem falta à Etapa 2 fica impedido da Etapa 3, e quem falta à Etapa 3 não aparece no Resultado Final. Existe portanto exatamente uma classe de Aluno com Etapa Ausente — o **Aluno sem Etapa 1** — e ela é 8,7% do Resultado Final histórico e existe nas Escolas Parceiras. O produto **atende** essa classe: nenhum Aluno recebe recusa.

O Quanto Falta já a atende corretamente sem nenhuma mudança de modelo: com as três notas da Etapa 1 em `0,000`, `calculate_argument_etapa` produz o z de zero, que é exatamente o que o Cebraspe aplica de 2018/2020 em diante. É aritmética fiel ao regime, não aproximação.

A previsão do Argumento Final, não. Para esse Aluno o **Momentum** — a evolução da Etapa 1 para a Etapa 2, hipótese central do produto — é indefinido, não zero. A Volatilidade (CV) sobre `[0, eb_pas2]` devolve exatamente 100% para qualquer `eb_pas2`, que é a assinatura matemática de grandeza indefinida. Logo a previsão para essa classe é **função própria** (modelo separado ou roteamento de valor faltante — decidido por medição), e não um caso do mesmo modelo. A alternativa de servir todo mundo com um modelo que ignore a Etapa 1 foi rejeitada: descartaria o Momentum de 91% dos Alunos para acomodar 9%.

A ausência é **declarada, nunca inferida de notas zeradas**. O Edital de Resultado Final do PAS 3 pode imprimir `0,000` sem ambiguidade porque o Cebraspe sabe quem compareceu; o sistema não sabe. Nos Editais por etapa — a fonte das notas do Aluno vivo — a ausência não é um valor impresso, é um registro que não existe, e "não encontrado" tem causa conhecida de defeito: nome quebrado ou divergente entre Editais (tickets 13 e 18 do mapa `pdf-extraction`). Inferir ausência de um silêncio cuja causa mais provável é um bug seu é construir sobre o defeito.

## Considered Options

- **Excluir a classe do produto (recusa explícita na interface)**: descartado — joga fora a única coisa que se sabe com certeza sobre esse Aluno (o A1 fixo) e nega atendimento a 1 em cada 11.
- **Servir com o modelo atual, alimentado com zeros**: descartado — é o comportamento de hoje, por omissão: `emptyScores()` no Preditor e `?? 0` / `?? 6` na Gestão entregam Semáforo de Risco e Sugestão de curso a partir de um vetor que nenhum modelo viu no treino.
- **Um modelo único com features só da Etapa 2, servindo todos**: descartado pelo dono do produto — apagaria o Momentum, que é a hipótese que motivou o produto.
- **Inferir a ausência de `0/0/0`**: descartado — indistinguível de "não encontrado na fonte", cuja causa conhecida é defeito de casamento de nome.
- **Treinar num dataset misto para "ser justo" com a classe**: não descartado, mas **não decidido aqui** — é medição do ticket 10, não escolha de produto. Misturar dois regimes pode piorar o modelo dos 91% sem que ninguém perceba.
