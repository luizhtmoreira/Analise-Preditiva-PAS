# 06 — Dedução das Cotas Declaradas

**What to build:** o perfil de cotas de cada Aluno, deduzido do padrão de preenchimento das 10
classificações — outro dado que o Edital não imprime em lugar nenhum, mas que está lá
implicitamente.

As 10 classificações **são** os Sistemas de Concorrência. Quatro atributos binários — escola
pública, renda ≤1,5 salário mínimo per capita, PPI, PcD — geram os 9 sistemas de cota, e os
sistemas são **aninhados, não exclusivos**: ser ≤1,5 SM habilita a concorrer também às vagas de
>1,5 SM, ser PPI habilita as não-PPI, PcD idem. É a cascata de remanejamento da Lei 12.711. O
Aluno é ranqueado em todos os subsistemas que subsome, e seus atributos são os do subsistema
**mais específico** em que aparece.

O modelo, vindo do protótipo, com os índices na ordem em que as classificações aparecem no
Edital:

```python
# atributos exigidos por cada subsistema de Escola Pública
EP_ATTRS = {
    2: {"R", "PPI"},   3: {"R", "PPI", "PcD"},
    4: {"R"},          5: {"R", "PcD"},
    6: {"PPI"},        7: {"PPI", "PcD"},
    8: set(),          9: {"PcD"},
}
# um padrão válido é sempre o fecho para baixo desse reticulado:
def fecho(attrs): return {i for i, need in EP_ATTRS.items() if need <= attrs}
```

A validação decisiva que sustenta o modelo: apenas 8 padrões distintos ocorrem, de 2⁹ = 512
possíveis, e **todos os 8 são fecho para baixo válido — 0 violações em 1.843 registros**. Padrão
que não seja fecho é sinal de corrupção de extração e deve ser sinalizado como suspeito, não
descartado.

`Cota para Negros` nunca coocorre com subsistemas de Escola Pública: o Aluno opta por um sistema
ou por outro na inscrição.

Duas restrições de nomenclatura, ambas com razão de ser:

- O campo se chama **cota declarada**, nunca *cota elegível*. Para os 71% de Alunos que aparecem
  só no Universal é impossível distinguir quem não tem direito de quem tem e optou por não usar. O
  dado registra a opção, não a elegibilidade, e o nome precisa dizer isso.
- As cotas são registradas para **todos os Alunos não eliminados**, não só para os aprovados,
  porque os campos são ranking e não aprovação.

Colunas derivadas por Aluno, além das 10 classificações cruas: `sistema_negros`, `escola_publica`,
`renda_baixa`, `ppi`, `pcd` e `perfil_cota`.

**Blocked by:** 01 — Costura `extrair_edital` + classificador de família + CSV de Resultado Final.

**Status:** ready-for-agent

- [ ] As seis colunas derivadas são gravadas por Aluno, junto das 10 classificações cruas
- [ ] Os quatro atributos vêm do subsistema mais específico em que o Aluno aparece
- [ ] Todos os Alunos não eliminados recebem perfil de cota, não só os aprovados
- [ ] O campo é nomeado *cota declarada* em código, CSV e docs — nunca *cota elegível*
- [ ] Padrão que não seja fecho para baixo do reticulado é sinalizado como suspeito, e não descartado
- [ ] Um teste verifica o perfil deduzido de um Aluno com padrão conhecido
