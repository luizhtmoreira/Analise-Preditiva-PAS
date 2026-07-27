# 14 — Alunos com Etapa 1 ausente: fora do treino, ou caso que o produto precisa servir?

**Type:** grilling
**Status:** concluído — 2026-07-27
**Blocked by:** nenhum — os tickets 01 e 02 já entregaram toda a evidência

> **Resolvido.** É ausência, não zero real, e o produto **atende** essa classe. A previsão do
> Argumento Final exige função própria (o Momentum é indefinido para ela); o Quanto Falta já a
> atende corretamente hoje, por aritmética. A ausência passa a ser **declarada**, nunca inferida
> de notas zeradas. Treinar em todos vs. treinar nos 60.013 **não** foi decidido aqui — é medição
> do ticket 10.
> → [relatório](../relatorios/14-alunos-com-etapa-1-ausente.md) ·
> [ADR-0008](../../../docs/adr/0008-aluno-sem-etapa-1-atendido-com-funcao-propria.md)

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

- [x] Determinado se Etapa 1 zerada significa ausência ou zero real — **ausência**. Fechado sem
      arqueologia de Edital: a regra do PAS (pode faltar à Etapa 1; quem falta à 2 fica impedido
      da 3; quem falta à 3 não entra no Resultado Final) **prevê as três células** da tabela de
      etapas zeradas observada — 5.768 / 0 / 0.
- [x] Decidido se o produto precisa atender o Aluno sem Etapa 1 — **sim**, e a classe está no
      funil comercial (existe nas Escolas Parceiras), não só na estatística do Cebraspe.
- [x] Decidido como — **função própria** para a previsão, porque o Momentum é indefinido para ela;
      o Quanto Falta já a atende por aritmética exata. Um modelo com roteamento de faltante vs.
      dois modelos separados é medição do ticket 10. Modelo único só com features da Etapa 2 foi
      **rejeitado pelo dono do produto** (apagaria o Momentum de 91% para acomodar 9%).
- [x] Recusa na interface — **descartada**. Nenhum Aluno recebe recusa. Em vez disso: a ausência
      vira **declarada** (notas `Optional[float] = None`, campo próprio de ausência), separando
      "não fez" de "não informado" de "tirou zero".
- [x] Contagem final entregue ao ticket 05 — **64.298** linhas com `checksum_fecha`, das quais
      **60.013** com Etapa 1 presente e **4.285** com Etapa 1 Ausente. O ticket 05 materializa uma
      tabela só, com a coluna `etapa_1_ausente`, e **não deleta** os 4.285.
- [x] Relatório em `relatorios/14-alunos-com-etapa-1-ausente.md`
- [x] `CONTEXT.md`: **Etapa Ausente**, **Momentum**, **Aluno sem Etapa 1**; Volatilidade (CV)
      afiada para registrar que é cega à direção.
- [x] `docs/adr/0008-aluno-sem-etapa-1-atendido-com-funcao-propria.md`
- [x] Defeitos 4 e 5 abertos em `relatorios/defeitos-pendentes.md` (`Red_PAS1 = 6.0` inventado;
      roteador do ensemble cego à direção do Momentum).
