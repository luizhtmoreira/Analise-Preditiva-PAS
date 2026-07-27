# 14 — Alunos com Etapa 1 ausente: fora do treino, ou caso que o produto precisa servir?

**Type:** grilling
**Status:** open
**Blocked by:** nenhum — os tickets 01 e 02 já entregaram toda a evidência

## Question

**5.768 alunos (8,7% da base) têm a Etapa 1 inteira zerada** — `0,000 / 0,000 / 0,000` em P1,
P2 e Redação. Não é defeito de extração: o ticket 02 confirmou no texto bruto do Edital que o
parser leu certo. O Edital publica zero mesmo.

Os dois tickets de research chegaram nesse grupo por caminhos independentes e ambos recomendam
excluí-lo. A recomendação é tecnicamente sólida — **mas a pergunta que falta é de produto, não
de dado**, e é por isso que isto é um ticket e não uma linha do ticket 05.

**Por que excluir é tecnicamente óbvio:**

- Um modelo que mapeia PAS 1 + PAS 2 → PAS 3 não tem entrada válida para esse aluno: metade
  das features é zero estrutural, não desempenho.
- A **Volatilidade (CV)** que roteia o ensemble atual vira constante 100% (`std/mean` sobre
  `[0, eb_pas2]`), então o mecanismo de roteamento colapsa para um único ramo em 8,7% da base.
- É a explicação do degrau de checksum que abriu este mapa: 94,7% e 96,6% dessas linhas falham
  o checksum em 2016/2018 e 2017/2019, contra 1,0–1,4% depois. Excluí-las leva a base a
  **60.013 linhas com 100% de checksum fechando nos 8 triênios**.

**A pergunta que fica em aberto — e que decide o escopo do produto:** *por que* a Etapa 1 é
zero? Duas leituras com consequências opostas:

- **(i) O aluno não fez o PAS 1.** Entrou no programa na Etapa 2, ou faltou. Então existe uma
  classe real de aluno que o produto pode encontrar — e treinar sem eles significa que o app
  **não sabe atender quem só fez o PAS 2**. Isso pode ser uma decisão consciente ("o produto
  exige as duas etapas"), mas não pode ser um acidente de limpeza de dados.
- **(ii) O aluno fez e tirou zero, ou foi eliminado.** Aí é dado degenerado de verdade, e
  excluir não fecha porta nenhuma.

Uma pista forte para (i): a **regra da Etapa 1 ausente** que o ticket 02 descobriu. O Argumento
Final impresso para esses alunos é sistematicamente **mais generoso** que o z-score de zero
(mediana +2,704 em 2016/2018 e +3,549 em 2017/2019) — o Cebraspe trata a Etapa 1 ausente com
alguma regra própria, não como "tirou zero". Instituição não compensa quem tirou zero; compensa
quem não fez. A mecânica exata dessa regra não foi reconstruída (limitação declarada no
relatório 02), e ela mudou: de 2018/2020 em diante o tratamento passa a ser literal.

- [ ] Determinado, com evidência do Edital, se Etapa 1 zerada significa ausência ou zero real
- [ ] Decidido se o produto precisa atender o aluno que só fez PAS 2 — pergunta ao dono do
      produto, não inferência a partir dos dados
- [ ] Se precisar: decidido como (modelo separado com features de uma etapa só? recusa
      explícita na interface?) e isso vira ticket próprio
- [ ] Se não precisar: exclusão confirmada, e a interface do app registra que esse aluno está
      fora do escopo em vez de receber uma previsão silenciosamente ruim
- [ ] Contagem final do dataset entregue ao ticket 05
- [ ] Relatório em `relatorios/14-alunos-com-etapa-1-ausente.md`
