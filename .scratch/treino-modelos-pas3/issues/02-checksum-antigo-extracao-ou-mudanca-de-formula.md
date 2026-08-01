# 02 — O checksum falha nos triênios antigos por extração ruim ou porque a fórmula mudou?

**Type:** research
**Status:** resolved (2026-07-26) — ver `relatorios/02-checksum-antigo-extracao-ou-mudanca-de-formula.md`
**Blocked by:** nenhum

## Question

`checksum_fecha=False` cai quase inteiro nos dois triênios mais antigos e desaparece nos três
mais recentes. **Isso é dado sujo ou é outro regime?**

```
2016/2018   734 / 9.611   (7,6%)   ←
2017/2019   978 / 9.852   (9,9%)   ←
2018/2020    92 / 5.896   (1,6%)
2019/2021   113 / 8.505   (1,3%)
2020/2022    98 / 7.228   (1,4%)
2021/2023     0 / 8.019   (0,0%)
2022/2024     0 / 8.499   (0,0%)
2023/2025     0 / 8.703   (0,0%)
```

Um degrau desses raramente é acaso. Duas explicações levam a decisões **opostas** sobre a
janela de dados do ticket 08:

- **(a) Extração.** Editais mais antigos têm layout pior e o parser erra mais. Então é ruído
  de leitura: as linhas boas de 2016/2018 continuam sendo dado legítimo e a janela pode ir até
  lá.
- **(b) Mudança de fórmula.** O checksum recalcula o Argumento Final com os pesos oficiais
  atuais (`PESO_P1=0.72`, `PESO_P2=8.28`, `PESO_REDACAO=1.00`) e compara com o valor impresso
  no Edital. Se o Cebraspe usava **outros pesos** ou outra normalização naqueles anos, o
  checksum falha em dado perfeitamente correto — e 2016/2018 e 2017/2019 passam a ser um
  regime diferente, não uma versão suja do mesmo regime.

**Por que este ticket vem antes de tudo:** é a pergunta original do Luiz ("o padrão mudou desde
2018?") na sua forma mais falsificável. Se for (b), a resposta sobre a janela já está meio dada
antes de treinar qualquer coisa.

Terceira coisa a checar de passagem, porque mora na mesma evidência: **2018/2020 tem 5.896
registros contra ~8.500 dos vizinhos — 30% a menos.** A Etapa 3 desse triênio caiu em 2020
(PAS 3 adiado pela pandemia); 2019/2021 e 2020/2022 também têm etapas pandêmicas. Isso é um
candidato independente a quebra de regime, e precisa ser separado do efeito de checksum.

- [x] Verificado nos Editais extraídos (ou em `medias_desvios.csv` / `src/pas_extraction/`) se
      os pesos ou a normalização do Argumento Final diferem entre triênios
- [x] Distribuição do `checksum_delta` nas linhas que falham, por triênio: erro **sistemático**
      (viés numa direção, sugerindo fórmula) ou **disperso** (sugerindo extração)?
- [x] Veredito entre (a) e (b), com a evidência que o sustenta
- [x] Explicado o déficit de 2.600 registros em 2018/2020: menos inscritos, mais eliminados,
      Edital parcial, ou perda de extração?
- [x] Listadas as etapas afetadas pela pandemia por triênio, para o ticket 08 tratar como
      variável e não como surpresa
- [x] Relatório em `relatorios/02-checksum-antigo-extracao-ou-mudanca-de-formula.md`
