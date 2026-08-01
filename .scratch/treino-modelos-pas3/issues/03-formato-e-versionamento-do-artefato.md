# 03 — Formato, versionamento e promoção do artefato de modelo

**Type:** grilling
**Status:** closed
**Blocked by:** nenhum
**Relatório:** `relatorios/03-formato-e-versionamento-do-artefato.md`

## Question

Como um modelo treinado sai da máquina de quem treinou e chega na API em produção — de forma
que se saiba **qual** modelo está rodando, **que dado** o gerou, e **como voltar** para o
anterior?

Hoje: 10 arquivos `.joblib` em `models/`, gitignored, hospedados no Dropbox. Não há versão, não
há registro de qual dado ou hiperparâmetro gerou cada um, e a API assume que o arquivo certo
está no disco. O Luiz pediu explicitamente o **padrão de mercado, como um MLOps sênior faria**.

Duas perguntas que precisam de resposta conjunta, porque uma restringe a outra:

**Formato.** `joblib` é pickle: serializa o objeto Python vivo. Isso significa que carregar um
modelo antigo com uma versão nova de `scikit-learn`/`lightgbm` pode quebrar ou — pior — carregar
silenciosamente com comportamento alterado; e `pickle.load` executa código arbitrário, então
o artefato precisa ser tratado como executável confiável, não como dado. Alternativas: o formato
nativo do LightGBM (texto, estável entre versões), ONNX (portátil, inferência desacoplada do
framework de treino), ou manter joblib com a versão das libs fixada e registrada junto.

**Domicílio e promoção.** Dropbox não versiona nem tem noção de "modelo em produção". O
espectro vai de um esquema disciplinado de arquivos versionados + manifesto (barato, sem
dependência nova) até um registry de verdade (MLflow, ou o registry de um provedor). O critério
não é sofisticação — é o menor mecanismo que responde "o que está em produção agora e o que
gerou isso".

**Restrição que corta opções:** a stack do projeto é declaradamente gratuita
([[project_frontend_refactor]]), e `models/` está fora do git por ser IP do produto
([[project_parser_privacy]]) — não é só tamanho de arquivo. Qualquer solução precisa caber
nessas duas.

**Por que este ticket está no frontier desde já:** não depende de nenhuma decisão de modelagem
— pode rodar em paralelo com 01 e 02 — e a resposta define o formato que o ticket 12 (pipeline
de treino) tem que produzir.

- [x] Decidido o formato de serialização, com o motivo e o que se perde na escolha
- [x] Decidido onde o artefato mora e como a API o obtém em produção
- [x] Decidido o que acompanha cada artefato como metadado mínimo (versão de dado, commit,
      hiperparâmetros, métrica de holdout, versões das libs)
- [x] Decidido como se promove um modelo novo e como se reverte para o anterior
- [x] Confirmado que a escolha cabe na stack gratuita e na restrição de IP de `models/`
- [x] Relatório em `relatorios/03-formato-e-versionamento-do-artefato.md`

## Resolução (2026-07-26)

1. **Formato:** `.joblib`, com gatilho escrito — se o ticket 10 der "um LightGBM só, sem
   scaler", o ticket 12 emite texto nativo. O que ataca a fragilidade não é o formato: é
   manifesto + versões cravadas + falha barulhenta.
2. **Unidade:** o pacote inteiro de uma rodada de treino. Não existe meio pacote em produção.
3. **Domicílio:** repositório privado no Hugging Face Hub, separado do Space (100 GB grátis).
4. **Entrega:** assado na imagem no build, nunca baixado no boot (o Space hiberna em 48h).
5. **Manifesto:** dado (hash, nunca o dado), código (commit + árvore limpa), ambiente (versões
   exatas), modelos (com **nomes das features**), avaliação (com o recorte).
6. **Promoção:** ponteiro versionado no GitHub. Promover é commit, reverter é `git revert`.
7. **Portão:** carrega + versões batem + features batem, bloqueantes no build. Qualidade pior que
   produção bloqueia, com chave de força que fica gravada no manifesto.

Evidência que reordenou o ticket: `p1_pas3_model` e `red_pas3_model` **já não carregam**
(`ModuleNotFoundError: No module named '_loss'`) e `target_calculator.py:66` engole o erro,
respondendo por média ponderada em silêncio.

**Descoberta entregue ao ticket 07:** o ADR-0007 é inválido —
`scripts/baseline_avaliacao.py:55` alimenta os modelos com 5 das 6 features em posição errada.
