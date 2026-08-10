# Motor de previsão

`src/pas_intelligence/` — o núcleo que transforma notas em previsão.

## Um modelo, mais aritmética

A decisão estruturante está no [ADR-0009](../adr/0009-alvo-canonico-argumento-da-etapa-3.md): o
sistema prevê **um único número**, o Argumento da Etapa 3. Tudo o mais é conta.

```
Argumento Final = A1 + 2·A2 + 3·Â3
```

Para um aluno que já fez PAS 1 e PAS 2, `A1` e `A2` são aritmética exata — as notas estão na mão
e as estatísticas oficiais daqueles anos são públicas. Apenas `Â3` é previsto. Disso decorre
também que `σ(Argumento Final) = 3 × σ(A3)`, exato, porque as duas primeiras parcelas têm
variância zero.

A consequência de produto é que **nenhum número na tela pode contradizer outro**: Argumento Final,
escore bruto equivalente, chance de aprovação e "quanto falta" saem todos da mesma fonte.

## Módulos

**`model_package.py`** — a única porta entre a API e o artefato treinado. Carrega `models/pas3/`,
calcula `A1` e `A2`, prevê `A3` e devolve o Argumento Final junto com a Largura de Incerteza da
classe do aluno. Reutiliza os construtores de features do treino, de modo que runtime e
treinamento não possam divergir; há teste que passa o mesmo aluno pelas duas portas e compara os
vetores.

**`validation.py`** — a régua do projeto ([ADR-0010](../adr/0010-validacao-deslizante-com-holdout-lacrado.md)):
validação deslizante de janela expansiva, 5 dobras, holdout 2023/2025 lacrado. Recebe uma
*fábrica* de modelo (modelo novo por dobra, nunca retreino sobre o anterior), exige semente
explícita e **recusa como feature qualquer coluna da Etapa 3**.

**`training_pipeline.py` / `training_dataset.py` / `dataset_pas3.py`** — do CSV ao pacote
treinado, com o portão de aceitação aplicado em código.

**`argument_calculator.py`** — a conversão oficial do edital, com os pesos do Cebraspe
(`PESO_P1=0,72`, `PESO_P2=8,28`, `PESO_REDACAO=1,00`).

**`target_calculator.py`** — o cálculo reverso: dada uma nota de corte, qual a nota necessária na
última etapa.

**`statistics.py`** — a chance de aprovação como `P(X > corte)`, com `X ~ N(previsão, largura²)`.
A largura é parâmetro **obrigatório e sem valor padrão**: ela vive no manifesto do pacote, por
classe de aluno, e muda a cada retreinamento ([ADR-0012](../adr/0012-largura-fixa-por-classe-em-vez-de-incerteza-por-aluno.md)).

**`pas_constants.py`** — `OFFICIAL_STATS`, as médias e desvios publicados pelo Cebraspe por
`(ano, etapa)`. Seu único consumidor é o gráfico temporal público; os cálculos de argumento,
probabilidade e meta recebem média e desvio como parâmetro, nunca leem daqui.

**`derivado_deploy.py`** — dono único da lista de colunas que podem sair do disco. Lido tanto por
quem publica quanto pelos serviços que consomem.

## O pacote de modelo

`models/pas3/` contém dois arquivos: `modelo_pas3.txt` (LightGBM em texto nativo) e
`manifest.json`.

O manifesto carrega a procedência completa, preenchida automaticamente: hash SHA-256 do CSV de
origem, commit e estado da árvore de trabalho, versões das bibliotecas, nomes e ordem das
features, métricas com o recorte que as produziu, e o bloco de incerteza com a cobertura empírica
verificada.

A **ordem das features é contrato**: o carregador recusa um pacote cuja ordem difira da declarada.

Para o aluno sem Etapa 1, as colunas derivadas da primeira etapa entram como **ausente nativo**,
nunca como zero literal
([ADR-0011](../adr/0011-lightgbm-unico-com-faltante-nativo-substitui-o-ensemble.md)).

## Retreinamento

Um comando vai do CSV ao pacote:

```bash
.venv/bin/python scripts/treinar_pipeline.py <resultado_final.csv> --saida <dir>
```

O pipeline confere o portão de aceitação sozinho e **recusa escrever qualquer coisa em disco** se
o critério não for atingido. A chave de força exige um motivo, que fica gravado no manifesto.
Duas execuções com a mesma semente produzem o mesmo arquivo de modelo, byte a byte.

## O que foi aposentado

O ensemble dinâmico de quatro modelos foi retirado no
[ADR-0011](../adr/0011-lightgbm-unico-com-faltante-nativo-substitui-o-ensemble.md): ele ganhava
0,10% do seu melhor componente isolado, dentro do ruído entre dobras. A previsão dupla de
Argumento Final e escore bruto foi retirada no
[ADR-0009](../adr/0009-alvo-canonico-argumento-da-etapa-3.md).

Os artefatos antigos permanecem em `models/aposentados-2026-07-28/` para permitir reversão.
`p1_pas3_model.joblib` e `red_pas3_model.joblib` ainda são referenciados pelo cálculo reverso, mas
**não carregam** sob a versão atual do scikit-learn.
