# 15 — Fronteira de página no parser de Resultado Final

**What to build:** um registro que é o último de uma página, cujo campo final (10ª
classificação) só é impresso no início da página seguinte, deixa de ter esse valor confundido
com o número da própria página injetado pelo `pypdf` no texto extraído.

Achado no ticket 06 (`.scratch/pdf-extraction/relatorios/06-deducao-das-cotas-declaradas.md`,
§3): o `pypdf` emite o número da página no início do texto de cada página. Quando o 22º campo
de um registro (a 10ª classificação) cai na página seguinte, `resultado_final._separar_registro`
lê o número da página no lugar do valor real — porque o parser trabalha sobre o blob já
concatenado, sem consciência de onde uma página termina e a próxima começa.

**Impacto medido:** 8 de 10 casos conhecidos no corpus real (66.313 registros) são pegos pela
checagem de fecho de cota e saem marcados com `cota_padrao_suspeito=True` (não silencioso). Os
outros 2 caem em padrões que continuam sendo fecho válido (ex. `{1,9,10}`, fecho de PcD) e ficam
invisíveis a essa camada — não dá para distinguir, só pelo fecho, um PcD genuíno de um número de
página que caiu num lugar plausível.

**Blocked by:** Nenhum — pode começar imediatamente. (Nota: toca a mesma área de código do
ticket 14 — `resultado_final._montar_registro`/`_separar_registro` — sem dependência lógica
entre os dois; vale sequenciar a implementação pra evitar conflito de merge.)

**Status:** concluído — ver `.scratch/pdf-extraction/relatorios/15-fronteira-de-pagina-no-parser.md`

- [x] `_separar_registro` (ou onde for a correção) tem consciência de fronteira de página, não
      só do blob de texto já concatenado
- [x] Os 8 casos conhecidos no corpus real passam a ter o valor correto na 10ª classificação em
      vez do número da página seguinte
- [x] Teste sintético reproduz um registro exatamente na borda de página (22º campo cai na
      página seguinte) e confirma que o valor correto é lido, não o número da página
- [x] Os 2 casos hoje invisíveis à checagem de fecho (padrão coincide com fecho válido) são
      reavaliados: se a correção de fronteira os resolve também, documentar; se não, manter
      como limitação conhecida e registrar isso em `defeitos-pendentes.md`
