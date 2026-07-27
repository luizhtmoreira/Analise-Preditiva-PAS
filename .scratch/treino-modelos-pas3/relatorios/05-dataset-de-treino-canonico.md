# Relatório — Ticket 05: dataset de treino canônico

**Ticket:** `.scratch/treino-modelos-pas3/issues/05-dataset-de-treino-canonico.md`
**Status:** concluído
**Tipo:** execução — script determinístico + artefato de dado, código de produção novo
**Dado analisado:** `.scratch/pdf-extraction/saida-nova/resultado_final.csv` (66.313 registros,
8 triênios)
**Código novo:** `src/pas_intelligence/training_dataset.py`,
`scripts/build_training_dataset.py`, `tests/test_training_dataset.py`
**Privacidade:** só agregados e contagens neste relatório. O dataset gerado (`data/training/`)
fica fora do git (`data/` já está no `.gitignore`) e não carrega `nome` nem `inscricao` em
texto puro — ver seção 5.

---

## 1. O que foi construído

`python scripts/build_training_dataset.py` lê o `resultado_final.csv` e escreve
`data/training/pas3_dataset.parquet`: **64.298 linhas**, uma por aluno-triênio, com o alvo
canônico do ticket 04 (`A3`) e as features que saem de aritmética exata (`A1`, `A2`) já
calculadas. Semente: nenhuma — o processo é 100% determinístico (filtro + aritmética, sem
amostragem), então "semente fixa" do checklist é satisfeita por não haver passo aleatório a
semear.

```
$ python scripts/build_training_dataset.py
64298 linhas escritas em data/training/pas3_dataset.parquet
etapa_1_ausente: 4285 (6.66%)
inscricao_repetida_entre_trienios: 296 (0.46%)
trienio
2016/2018    8877
2017/2019    8874
2018/2020    5804
2019/2021    8392
2020/2022    7130
2021/2023    8019
2022/2024    8499
2023/2025    8703
```

Todos os oito números batem, linha a linha, com a tabela §7.2 do relatório 01 e com a
contagem `60.013 + 4.285` do relatório 14 — a implementação não descobriu nada novo sobre o
dado, ela **materializa** o que os tickets 01/02/04/14 já haviam decidido e medido.

---

## 2. Regra de inclusão/exclusão (ticket 01) — aplicada e conferida

```python
df = source[source["checksum_fecha"] == True]
```

Sem filtrar por `campos_formato_invalido`, como o relatório 01 recomenda. Descarte por
triênio, script vs. relatório 01 (idênticos):

| triênio | total bruto | descartadas | incluídas | % incluída |
|---|---:|---:|---:|---:|
| 2016/2018 | 9.611 | 734 | 8.877 | 92,36% |
| 2017/2019 | 9.852 | 978 | 8.874 | 90,07% |
| 2018/2020 | 5.896 | 92 | 5.804 | 98,44% |
| 2019/2021 | 8.505 | 113 | 8.392 | 98,67% |
| 2020/2022 | 7.228 | 98 | 7.130 | 98,64% |
| 2021/2023 | 8.019 | 0 | 8.019 | 100,00% |
| 2022/2024 | 8.499 | 0 | 8.499 | 100,00% |
| 2023/2025 | 8.703 | 0 | 8.703 | 100,00% |
| **TOTAL** | **66.313** | **2.015** | **64.298** | **96,96%** |

A tabela acima sai do próprio `scripts/build_training_dataset.py` (compara `trienio` do CSV bruto
contra o do dataset final), não foi copiada do relatório 01 — é o script publicando sua própria
contagem de descarte, como o checklist pede. O motivo do descarte (População A — Etapa 1 zerada
em regime antigo, vs. População B — corrupção grossa) não foi recalculado: é o mesmo
`checksum_fecha`, e o relatório 01 já decompôs o total em 1.446 + 569 por triênio. Não duplicar
essa medição evita que as duas contas divirjam por arredondamento silencioso; quem quiser o
motivo por linha volta ao relatório 01.

**Verificação de fechamento embutida no build:** para cada linha incluída, o script recompõe
`Argumento Final = 1·A1 + 2·A2 + 3·A3` a partir das notas e da língua gravada e compara com o
`argumento_final` impresso. Divergência acima de 0,01 (a tolerância de 0,005 do checksum mais a
folga de três arredondamentos independentes pesados por 1/2/3) faz o build **falhar**, não
seguir em frente com dado ruim. Nas 64.298 linhas reais, zero falharam — o que já era esperado,
porque a língua gravada nessas linhas é a que o próprio checksum usou para fechar (relatório 02,
glossário: "a língua gravada nas linhas que **falham** não é confiável" — aqui só sobrevivem as
que fecham).

---

## 3. Duplicata de inscrição entre triênios

**Medido no dataset final:** 296 linhas (0,46%) carregam `inscricao_repetida_entre_trienios =
True`, correspondendo a **144 alunos** que aparecem em mais de um triênio (296 ≈ 2×144, com uma
pequena sobra porque pelo menos um aluno aparece em mais de dois).

Isso é **quase** o número que o relatório 01 mediu sobre a base bruta — **146** inscrições
repetidas em 66.159 distintas (0,22%) — e a diferença de 2 se explica sozinha: o relatório 01
mediu antes do filtro de checksum; aqui, 2 dos 146 pares perderam um dos dois lados no filtro da
seção 2 e sobraram como ocorrência única, deixando de contar como repetição **dentro do
dataset entregue**.

**Veredito: proporção pequena demais para justificar ticket próprio, e a flag já resolve o
risco.** 0,46% das linhas é abaixo de qualquer ruído de split razoável. A flag existe para que o
ticket 06 (esquema de validação) decida, com o dado em mãos, se agrupa o split por
`id_pseudonimo` — sem precisar recalcular nada, e sem o dataset carregar `inscricao` para fazer
esse agrupamento.

---

## 4. Escala e unidade entre triênios

A pergunta do ticket era se `eb_p2`, `red` e `argumento_final` mudam de escala entre triênios de
um jeito que quebraria o treino. Como o ticket 04 já decidiu que o alvo é `A3` (Argumento da
Etapa 3, um z-score ponderado), a resposta correta é medir a **estabilidade do alvo**, não do
Escore Bruto — e ela já está estabelecida (relatório 02, §8): o EB varia ~35% de dificuldade
entre triênios, o Argumento não, porque a padronização por média/desvio do próprio ano absorve
a diferença. Conferido aqui, no dataset materializado:

| triênio | média(A3) | desvio(A3) | mín(A3) | máx(A3) |
|---|---:|---:|---:|---:|
| 2016/2018 | 0,355 | 9,187 | −18,880 | 33,986 |
| 2017/2019 | 0,513 | 9,167 | −20,189 | 40,748 |
| 2018/2020 | 0,018 | 9,155 | −21,588 | 34,765 |
| 2019/2021 | 0,001 | 9,103 | −18,329 | 37,910 |
| 2020/2022 | −0,006 | 9,053 | −19,575 | 37,323 |
| 2021/2023 | 0,000 | 9,071 | −20,551 | 35,825 |
| 2022/2024 | −0,001 | 9,083 | −21,651 | 33,581 |
| 2023/2025 | 0,000 | 9,146 | −21,552 | 34,458 |

Média ~0 e desvio ~9,1 em todos os oito, sem degrau nem tendência — a escala do alvo é a mesma
regime a regime. **Nenhuma descontinuidade a documentar.** O EB cru (`eb_p1_e*`, `eb_p2_e*`,
`red_e*`) continua no dataset em escala bruta e variável por ano — isso é esperado e é sinal a
ser tratado pelo ticket 09 (features), não defeito deste ticket.

---

## 5. Privacidade — `nome` fica de fora, `inscricao` vira hash de mão única

**Decisão: identificador pseudonimizado, não ausência total.** `id_pseudonimo` é
`sha256(inscricao)` truncado a 16 caracteres hex, sem sal, calculado uma única vez dentro de
`build_training_dataset` — `inscricao` nunca chega a ser escrita no `DataFrame` de saída.

**Por quê pseudonimizar em vez de simplesmente remover:** a seção 3 precisa de alguma forma de
agrupar aluno repetido, e o ticket 06 (esquema de validação) provavelmente vai querer a mesma
coisa para decidir o split. Calcular a flag `inscricao_repetida_entre_trienios` uma vez aqui e
depois jogar fora todo identificador funcionaria para *este* ticket, mas obrigaria o 06 a voltar
ao CSV bruto (com `nome` e `inscricao` de novo) se quisesse agrupar por aluno de um jeito
diferente do que esta flag prevê. O hash mantém essa porta aberta sem reintroduzir PII.

**Por que sem sal, e o que isso custa:** sem sal, o mesmo `id_pseudonimo` sai do script em
qualquer rodada — é o que permite ao ticket 06 comparar splits gerados em sessões diferentes. O
custo é que o hash é teoricamente reversível por força bruta, porque `inscricao` é um número de
~8 dígitos (espaço de busca pequeno para SHA-256). Na prática isso exige que alguém já tenha a
inscrição candidata para testar — o dataset não publica nada que ajude a *adivinhar* qual
inscrição testar primeiro (não tem nome, não tem campus+data suficientes para restringir a busca
a um universo pequeno). É uma pseudonimização fraca contra um atacante com acesso ao Cadastro do
Cebraspe, mas suficiente para o uso interno deste dataset — mesmo nível de exposição que um
índice de linha ordenado por inscrição teria.

**O que NÃO está no dataset, ponto final:** `nome`, `inscricao` em texto puro,
`classificacao_sistema_*` (cotas — não usadas pelo alvo nem pedidas pelo ticket), `campos_
formato_invalido`, `checksum_delta`/`checksum_fecha` (o filtro já foi aplicado; carregar a coluna
depois do filtro só serviria para reconstruir "por que essa linha entrou", que é debug, não
treino — quem precisar disso volta ao CSV fonte).

---

## 6. `lingua_e1/e2/e3` — presente e utilizável

As três colunas de língua por Etapa saem no dataset sem transformação, porque `A1`/`A2`/`A3` já
foram calculados usando exatamente essas colunas contra `OFFICIAL_STATS` (seção 2). Não há
"normalização de P1 ignorando a língua": a normalização por língua é o próprio mecanismo de
cálculo, não um passo separado.

`lingua_ambigua` (subproduto do checksum do `pdf-extraction`, ticket 01 deste mapa §7.4) também
sobrevive ao filtro: **4.498 das 64.298 linhas incluídas (7,0%)** têm mais de uma combinação de
língua fechando o checksum. Isso não invalida `A1`/`A2`/`A3` dessas linhas — o checksum valida o
**conjunto** das 9 notas mais o Argumento Final, e a divergência de 0,01 verificada no build
(seção 2) teria pego uma língua errada o suficiente para mudar o resultado. Fica marcado para
que o ticket 09 saiba que 7% das linhas têm uma fonte de ruído a mais nessa feature específica.

---

## 7. Formato e regeneração do artefato

**Formato: Parquet**, não `.joblib` nem CSV. Diferente do ticket 03 (que decide formato de
**modelo treinado**), este é um **dataset tabular** — Parquet é colunar, tipado (preserva `bool`
para `etapa_1_ausente`, sem o `"True"`/`"False"` como string que o CSV fonte carrega), comprime
melhor que CSV para 64 mil linhas × 26 colunas, e é o formato que `pandas`/`pyarrow` (já
dependência transitiva do projeto) leem nativamente sem parsing extra. Não é código executável
como `.joblib` — não herda a fragilidade de versão de biblioteca que motivou a Decisão 1 do
ticket 03.

**Caminho:** `data/training/pas3_dataset.parquet`. `data/` já está no `.gitignore` (linha
"Datasets e arquivos de dados brutos/temporários"), então o artefato nunca chega a ser
adicionável ao git por acidente — não foi preciso editar o `.gitignore`.

**Como regenerar:**

```bash
python scripts/build_training_dataset.py
```

Sem argumento, lê `.scratch/pdf-extraction/saida-nova/resultado_final.csv` e escreve no caminho
acima; `--source`/`--output` sobrescrevem os dois se o `resultado_final.csv` mudar de lugar (por
exemplo quando o mapa `pdf-extraction` gerar uma rodada nova).

**O que fica fora deste ticket, de propósito:** manifesto com hash do dado/commit/versões
(Decisão 5 do ticket 03) é escopo de **pacote de modelo treinado** (ticket 12), não de dataset de
treino. Se o ticket 12 quiser registrar de qual dataset um modelo saiu, o hash do
`pas3_dataset.parquet` é um `sha256sum` de um arquivo — não precisou de infraestrutura nova para
isso já estar disponível.

---

## 8. O que o ticket 05 decidiu, e o que ele deliberadamente não decidiu

**Decidido aqui:**
- Regra de inclusão (herdada do ticket 01, conferida bit a bit).
- Cálculo de `A1`, `A2`, `A3` com validação cruzada contra o `argumento_final` impresso —
  decisão nova deste ticket, não pedida explicitamente pelo checklist, mas necessária para não
  publicar um dataset com alvo errado por língua mal gravada.
- Pseudonimização de `inscricao` via hash sem sal (seção 5).
- Formato Parquet e caminho do artefato (seção 7).

**Não decidido aqui, de propósito:**
- **Se o split agrupa por `id_pseudonimo`.** A flag existe; a decisão é do ticket 06.
- **Se `etapa_1_ausente` vira dois datasets ou um com feature.** Ticket 14 já decidiu que não se
  descarta a classe; se o ticket 10 quiser medir "um modelo com faltante" contra "dois modelos
  separados", os dois recortes saem do mesmo Parquet por um filtro de uma coluna — não precisou
  de dois arquivos.
- **Quais colunas viram features do modelo.** Isso é o ticket 09. Este dataset é deliberadamente
  mais largo que o vetor final de features (guarda EB cru por Etapa, língua, flags) para que o
  09 possa medir contribuição marginal em vez de decidir por remoção prematura.

---

## 9. Limitações

- **O teste de recomposição do Argumento Final (seção 2) usa tolerância 0,01, não 0,005.** É
  mais folgada que o checksum original porque soma o arredondamento de três valores
  independentes (`A1`, `A2`, `A3` a 3 casas cada, pesados por 1/2/3). Na base real isso não
  importou — zero linhas ficaram entre 0,005 e 0,01 —, mas é uma tolerância desenhada, não
  medida contra um caso real que a exigisse.
- **`inscricao_repetida_entre_trienios` marca o par, não desambigua quem é "o mesmo aluno" com
  certeza.** A correspondência é por número de inscrição idêntico; se o mesmo aluno se
  reinscreveu com número diferente entre triênios (não deveria acontecer, mas não foi
  verificado), ele não é pego por esta flag. É o mesmo limite que o relatório 01 já carregava.
- **O hash sem sal (seção 5) não foi testado contra um modelo de ameaça formal.** A
  argumentação de que "não há como restringir o universo de busca" é qualitativa, não uma prova.
  Se este dataset algum dia sair da posse de quem já tem acesso ao `resultado_final.csv`, essa
  decisão precisa ser revisitada.
- **`lingua_ambigua` não foi propagada para uma incerteza por linha.** As 4.498 linhas marcadas
  entram no dataset com o mesmo peso das demais; se o ticket 11 (incerteza calibrada) quiser
  usar isso, a coluna já está lá, mas nada a consome ainda.

---

## 10. Glossário — termos novos deste relatório

- **Dataset de treino canônico** — o artefato único que os tickets 06–13 devem medir e treinar
  sobre, para que "o baseline melhorou" signifique sempre a mesma coisa.
- **`id_pseudonimo`** — hash SHA-256 (16 hex) de `inscricao`, sem sal, determinístico. Permite
  agrupar o mesmo aluno entre triênios sem guardar o número de inscrição em texto puro.
- **Argumento de Etapa recomposto** — `1·A1 + 2·A2 + 3·A3`, calculado no build a partir das
  notas e da língua gravadas, e comparado ao `argumento_final` impresso como verificação de
  qualidade — não é o checksum do `pdf-extraction`, é uma segunda checagem, específica deste
  dataset, que pega o caso em que a língua gravada é ambígua o bastante para mudar o resultado.

---

## 11. Onde continuar

- **Ticket 06 (esquema de validação):** decide se o split agrupa por `id_pseudonimo`
  (0,46% das linhas, 144 alunos) e usa `etapa_1_ausente` para estratificar, como o ticket 14
  já havia deixado registrado.
- **Ticket 07 (baseline honesto):** consome `a3` como alvo e `eb_p1_e1..e3`, `red_e1..e3`,
  `a1`, `a2` como candidatos a feature — sem decidir ainda o vetor final.
- **Ticket 08 (janela):** a seção 4 já responde, para o alvo `A3`, que não há degrau de escala
  entre 2016/2018 e 2023/2025 — resta medir dificuldade de prova/coorte pandêmica, não escala.
- **Ticket 09 (features):** o dataset guarda mais colunas do que qualquer vetor final vai usar,
  de propósito — inclui `lingua_ambigua` e o EB cru por Etapa para que a seleção de features
  meça em vez de presumir.
