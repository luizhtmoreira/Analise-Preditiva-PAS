# 03 — CORS vindo do ambiente (relatório)

> Relatório escrito retroativamente: o ticket foi implementado e mergeado em duas sessões
> (`07a6c08`, `5ea71e5`, mergeado em `0fa79c9`) sem que nenhuma delas produzisse este documento.

## O que mudou

`api/main.py` deixa de cravar a lista de origens no código. `cors_origins_do_ambiente()` lê
`CORS_ALLOW_ORIGINS` (CSV) do ambiente e cai para o default de DEV
(`http://localhost:3000,http://localhost:3001`) quando a variável não está definida. O
`CORSMiddleware` passa a receber também `allow_origin_regex`, lido de `CORS_ALLOW_ORIGIN_REGEX`
com default `https://.*\.vercel\.app`.

## O bug que motivou o ticket

A lista antiga incluía `"https://*.vercel.app"` como item literal de `allow_origins`. O
`CORSMiddleware` do Starlette compara `Origin` por igualdade exata nessa lista — não trata `*`
como curinga. O wildcard nunca casava com nada; todo deploy de preview da Vercel ficava sem CORS,
silenciosamente (o navegador recusa a chamada antes dela sair, então nada aparece no log do
servidor — do lado do frontend isso lê como "API indisponível"). O mecanismo certo para padrão de
origem no Starlette é `allow_origin_regex`, não um item glob dentro de `allow_origins`.

`vetorpas.com.br` também não estava em lugar nenhum da lista antiga.

## Critérios de aceite do ticket — conferidos

- [x] Origens permitidas vêm de variável de ambiente, com default seguro de DEV quando ausente —
      `test_usa_defaults_de_dev_quando_variavel_ausente`.
- [x] `https://vetorpas.com.br` e `https://www.vetorpas.com.br` aceitos em PROD —
      `test_origem_de_producao_e_aceita`.
- [x] Preview da Vercel aceito por regex, não por texto literal com `*` —
      `test_regex_de_preview_da_vercel_casa_com_subdominio`.
- [x] Origem não listada é recusada — `test_origem_desconhecida_e_recusada`.
- [x] Teste cobrindo os três casos, novo — `tests/test_api_cors.py` (5 testes, nenhum prior art).
- [x] `pytest tests/` continua verde.

## Verificado

- `.venv/bin/python -m pytest tests/test_api_cors.py -q` — 5 passed.
- Suíte completa (`pytest tests/`) verde no momento do merge (`0fa79c9`).

## Fora do escopo deste ticket

- Nada é publicado por este ticket — ele só conserta a regra de CORS. A verificação num navegador
  de verdade contra um deploy real é o ticket 08/14.
