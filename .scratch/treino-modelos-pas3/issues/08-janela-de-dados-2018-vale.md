# 08 — Janela de dados: os alunos desde 2018 ainda ajudam?

**Type:** task
**Status:** open
**Blocked by:** 02 (resolvido), 04, 06, 07

## Question

**A pergunta original do mapa.** Treinar com os 8 triênios (2016/2018 a 2023/2025) produz um
modelo melhor ou pior, para prever a Etapa 3 de um aluno de hoje, do que treinar só com os mais
recentes?

Não é uma pergunta de opinião — é medível dentro da régua do ticket 06. O que se mede:

- modelo treinado em **todos** os triênios;
- modelo treinado nos **N mais recentes**, varrendo N de 1 a 8;
- modelo treinado em todos, com **peso decrescente** por idade do triênio (o dado velho não é
  descartado nem tratado como igual ao novo);
- se o ticket 02 concluir que a fórmula mudou: modelo com os triênios antigos **corrigidos**
  para a fórmula atual, em vez de descartados.

A curva de erro contra N responde a pergunta diretamente. Se cair monotonicamente até N=8, dado
velho ajuda. Se tiver um mínimo em N=4, existe um horizonte de validade.

**O que os tickets 01 e 02 já resolveram, e que encolhe este ticket:**

- **Mudança de fórmula está descartada.** Pesos idênticos de 2016 a 2025, recuperados por
  regressão com resíduo máximo de 0,005. Não existe fronteira de regime por fórmula.
- **O déficit de 2018/2020 é coorte menor, não perda de dado** — 12.740 candidatos contra
  18.726 do triênio seguinte, com densidade de registro por página idêntica à dos vizinhos.
  Causa: calendário quebrado pela pandemia (Resultado Final publicado 644 dias após o anterior,
  e o seguinte apenas 154 dias depois).
- **O degrau de checksum era a regra da Etapa 1 ausente**, tratada pelo ticket 14 — corte por
  linha, não por triênio. A população limpa é de **60.013 linhas com 100% de checksum fechando
  nos 8 triênios**.

**Portanto a janela pode, em princípio, ir até 2016/2018** — nenhum obstáculo de qualidade de
dado sobrou. O que este ticket ainda precisa medir é se o dado antigo **ajuda**, que é outra
pergunta.

**Mapa de coortes pandêmicas** (ticket 02), para tratar como variável e não como surpresa:

| Triênio | Etapas em ano pandêmico |
|---|---|
| 2018/2020 | E3 |
| 2019/2021 | E2 + E3 |
| 2020/2022 | E1 + E2 |
| 2021/2023 | E1 |

**Achado que é o novo centro deste ticket:** o **EB da Etapa 3 varia ~35% entre triênios**,
enquanto o **Argumento Final é estável** — a normalização por média e desvio do ano absorve a
diferença de dificuldade da prova. Isso significa que "o padrão mudou desde 2018?" tem respostas
**diferentes conforme o alvo do ticket 04**: em EB há deriva grande e real; em Argumento Final,
possivelmente quase nenhuma. Medir a janela sem fixar o alvo antes produziria uma conclusão sem
sentido.

**Candidatos a quebra de regime que ainda precisam ser distinguidos**, porque cada um implica um
corte diferente:

1. **Pandemia.** As Etapas 3 de 2020, 2021 e 2022 foram afetadas (o PAS 3 de 2020 foi adiado).
   Isso atinge os triênios 2018/2020, 2019/2021 e 2020/2022 — e 2018/2020 tem **5.896 registros
   contra ~8.500 dos vizinhos, 30% a menos**. Se a quebra é pandêmica, ela é um *buraco* no
   meio da série, não uma fronteira: os triênios *anteriores* a ela podem continuar válidos, e
   cortar "tudo antes de 2021" jogaria fora dado bom junto.
2. **Mudança de fórmula ou normalização.** Veredito do ticket 02. Se for isso, é fronteira
   verdadeira — mas possivelmente corrigível em vez de cortável.
3. **Deriva gradual.** Nem buraco nem fronteira: a relação entre desempenho na Etapa 1/2 e na
   Etapa 3 muda devagar (mudança de currículo, de perfil de candidato, de dificuldade da prova).
   Aparece como degradação suave com a idade do dado, e o remédio é ponderação, não corte.

Medir a deriva de distribuição por triênio (média e dispersão de cada nota, e sobretudo a
**relação** entre Etapas 1/2 e a Etapa 3) separa os três.

- [ ] Curva de erro contra número de triênios de treino, dentro do esquema do ticket 06
- [ ] Comparação entre corte por janela e ponderação por idade
- [ ] Deriva de distribuição medida por triênio, com a pandemia isolada dos demais efeitos
- [ ] Os três candidatos a quebra de regime distinguidos por evidência, não por hipótese
- [ ] Resposta escrita e defendida: **usa 2018 ou não, e por quê** — com o custo em erro de
      cada alternativa
- [ ] Relatório em `relatorios/08-janela-de-dados-2018-vale.md`
