# 16 — Os dez Sistemas de Concorrência na tela

**What to build:** o Aluno escolhe qualquer um dos 10 Sistemas de Concorrência que o Edital
publica, e não só os 5 que a tela oferece hoje. Em especial, **Cota para Negros passa a existir**.

## O defeito

`MAPA_SISTEMAS` (`src/pas_extraction/constants.py:24`) tem os **10** sistemas do Edital, e o
`notas_corte.csv` traz corte publicado para os 10. A tela oferece **5**, na constante `COTAS`,
duplicada em dois arquivos:

- `landing-page/components/public/calculadora/CalculadoraPage.tsx:19`
- `landing-page/components/public/predict/PreditorPage.tsx:23`

Os 5 ausentes se dividem em dois casos que **não** têm a mesma gravidade.

### Caso 1 — Cota para Negros: ausência sem justificativa

É o **terceiro sistema mais comum** do dado, à frente de três das quatro cotas que estão na tela:

| sistema | linhas de corte | cursos×campus×turno cobertos | na tela? |
|---|---:|---:|:---:|
| Universal | 1.257 | 180 | sim |
| EP / Alta Renda / Não-PPI | 972 | 150 | sim (L10) |
| **Cota para Negros** | **892** | **130** | **não** |
| EP / Alta Renda / PPI | 794 | 137 | sim (L9) |
| EP / Baixa Renda / Não-PPI | 677 | 124 | sim (L2) |
| EP / Baixa Renda / PPI | 565 | 116 | sim (L1) |

Não é escassez de dado. E **o backend já a atende**: o docstring de `_resolver_sistema`
(`api/services/gestao_service.py:93`) nomeia "Cota para Negros" como um dos casos que o fuzzy
match resolve sem alias. É omissão do dropdown, não falta de capacidade. Não há ADR, comentário
nem relatório que a justifique — procurei.

### Caso 2 — as quatro variantes PcD: escassas, mas a tela já sabe lidar

Sistemas 4, 6, 8 e 10: **68 linhas de 5.225 (1,3%)**, cobrindo 29, 10, 4 e 3 cursos. Omitir tem
justificativa aparente (o Aluno escolheria e quase sempre não haveria corte), **mas essa
justificativa já não se sustenta**: o Preditor tem o estado `curso_alvo_sem_dados_cota`
(`PreditorPage.tsx:997`), que diz exatamente "a cota selecionada não tem dado para esse curso".
A degradação graciosa existe e está testada em produção pelas cotas atuais.

O custo de omitir é que o Aluno PcD **não tem como se declarar**. O custo de incluir é ver
"sem dados" com frequência — que é a verdade, dita.

## ⚠ O dano é o inverso do que a intuição diz — medido

A suposição natural é que o Aluno de Cota para Negros que marca "Sistema Universal" veria uma
barra mais alta e **subestimaria** a chance. **É o contrário.**

Medido em 530 pares curso×campus×turno×triênio que têm os dois cortes publicados:

| | valor |
|---|---:|
| Universal **maior** que Cota para Negros | 45,3% dos casos |
| diferença mediana (Universal − Cota para Negros) | **−6,6 pontos** |
| diferença média | −7,9 pontos |

O corte Universal é tipicamente **mais baixo**. A causa é estrutural: a cota tem menos vagas, e
o corte é o menor Argumento entre os admitidos — com menos admitidos, desce-se menos na
distribuição, então o corte sobe.

**Consequência:** hoje o Aluno de Cota para Negros que marca Universal recebe uma probabilidade
**otimista demais**. São ~6,6 pontos, contra uma Largura de Incerteza de 14,97 — cerca de 0,44
largura, que cai exatamente na faixa onde o modelo tem sinal de verdade (relatório 14 §4.2) e
onde 6,6 pontos **viram veredito trocado**. É o mesmo tipo de dano que o viés de +8,66 do modelo
antigo causava e que o ADR-0009 se propôs a eliminar: esperança que não existe, para quem está
em cima da linha.

Quem implementar: **não repita a suposição errada no texto da UI nem no commit.** O sentido é
otimista demais, não pessimista.

## ⚠ Pergunta para o dono do domínio — responder ANTES de codar

Os rótulos `L1`, `L2`, `L9`, `L10` da tela **podem não bater com o Edital do PAS**. Na
nomenclatura do SiSU / Lei 12.711, `L1` é escola pública de baixa renda **sem** recorte PPI (e
`L2` é a com PPI — invertido em relação à tela), e `L9`/`L10` são categorias **de PcD**, não as
de ampla escola pública.

O **mapeamento** está coerente e não é o que se questiona: como os sistemas são aninhados
(`src/pas_extraction/cotas.py:5` — ser ≤1,5 SM habilita as vagas de >1,5 SM), mandar "EP + PPI"
sem recorte de renda para o sistema de Alta Renda é pegar o balde mais amplo, que é o certo. O
que está em dúvida é só a **numeração L** exibida ao Aluno.

Se os rótulos estiverem errados, corrigi-los é parte deste ticket; se o Edital do PAS usa a
própria numeração e ela confere, não se toca. **Isto não se decide lendo o repositório** — o
Edital não está versionado aqui. Perguntar antes de implementar.

## As decisões de forma

- A constante sai dos dois componentes e vira **uma só**, em `landing-page/lib/`. Hoje são duas
  listas idênticas que podem divergir em silêncio — foi assim que a lacuna passou despercebida
  em duas telas de uma vez.
- Os 10 sistemas entram, incluindo os PcD. O estado "sem dados para esse curso nessa cota" já
  existe e é a resposta honesta quando não houver corte.
- **A Calculadora ganha o mesmo estado de "sem dados"** se ainda não o tiver — hoje só o Preditor
  o tem, e incluir cotas escassas sem ele deixaria a Calculadora mostrando corte vazio sem
  explicar por quê.
- O `COTA_ALIASES` (`gestao_service.py:81`) ganha as entradas que faltarem. "Cota para Negros"
  provavelmente dispensa alias (o fuzzy já resolve), mas **confirmar com teste**, não por
  inspeção: o próprio comentário acima do dicionário conta que o fallback silencioso do
  `.get(sistema, fallback)` já mascarou exatamente esse tipo de erro antes, resolvendo toda cota
  para Universal sem avisar ninguém.
- Perfil salvo (`profile.cota`, lido em `PreditorPage.tsx:686` e `CalculadoraPage.tsx:251`) pode
  conter valor fora da lista. Hoje o `<select>` cai no primeiro item em silêncio. Com a lista
  crescendo, verificar que um valor desconhecido não vira "Sistema Universal" calado.

## Status: needs-owner-answer

Não é `ready-for-agent`: a pergunta dos rótulos L precisa de resposta antes. O resto do ticket
está fechado e pode ser implementado no mesmo passo assim que ela chegar.

- [ ] Uma única constante de cotas em `landing-page/lib/`, consumida pela Calculadora e pelo Preditor
- [ ] Os 10 Sistemas de Concorrência do `MAPA_SISTEMAS` aparecem na tela
- [ ] Cota para Negros resolve para o corte certo — teste que **falharia** se ela caísse no
      fallback de Universal (não basta inspecionar: o fallback é silencioso por construção)
- [ ] Os rótulos L conferem com o Edital, ou foram corrigidos conforme a resposta do dono do domínio
- [ ] A Calculadora tem o estado "sem dados para esse curso nessa cota"
- [ ] Um `profile.cota` desconhecido não vira "Sistema Universal" em silêncio
- [ ] `pytest tests/`, `eslint` e `tsc --noEmit` verdes
