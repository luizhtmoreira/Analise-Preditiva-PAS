# 08 — API hospedada: Dockerfile, Space e o pacote assado na imagem

> **REABERTO em 2026-07-31 — o destino mudou, o trabalho não.** Ao publicar de verdade, o Hugging
> Face devolveu `402 Payment Required`: hospedar Docker Spaces passou a exigir assinatura PRO. A
> premissa "CPU Basic, gratuito" do ADR-0004 envelheceu sem ninguém notar. **A API vai para o
> Render** — decisão, porquês e alternativas rejeitadas no **ADR-0014**.
>
> **O que este ticket entregou continua valendo:** Dockerfile em duas etapas, `deploy/ponteiro.json`,
> `buscar_artefatos.py`, `publicar_pacote.py` (que já rodou — os artefatos estão no Domicílio
> Versionado), `requirements-api.txt`, CORS vindo do ambiente, e o bug do `libgomp1` resolvido.
>
> **O trabalho restante está em cinco sub-tickets**, em ordem de bloqueio:
>
> | | ticket | bloqueado por |
> |---|---|---|
> | `08a` | Derivado de Deploy: o dado que sai do disco não tem nome de Aluno | — |
> | `08b` | A imagem na forma do Render, entregue por um Repo de Deploy | — |
> | `08c` | O serviço no ar, em `api.vetorpas.com.br` | 08a, 08b |
> | `08d` | Boot Frio medido, e o frontend que sabe esperar | 08c |
> | `08e` | Keep-alive, como otimização descartável | 08d |
>
> **O ticket 14 depende de `08c`**, não do 08 inteiro — dá para publicar antes de `08d` e `08e`.
>
> Os critérios de aceite abaixo permanecem como o contrato do 08; quem os fecha agora são os
> sub-tickets. O único que nunca chegou a ser verificado — *o Preditor num navegador de verdade* —
> é o critério central do `08c`.

**What to build:** a API responde numa URL pública, e uma máquina limpa reproduz esse deploy sem
que ninguém copie arquivo à mão.

Hoje ela roda em `localhost:8000`. Não há Dockerfile e não há Space. E as três coisas de que a
imagem precisa não estão no git, porque `models/`, `data/` e `*.csv` são todos gitignored:

- `models/pas3/` — o pacote promovido (modelo LightGBM + manifesto);
- os CSVs de `data/` — Notas de Corte e banco populacional;
- `assets/` — só mais tarde, no B2B, para os templates whitelabel. **Fora desta rodada.**

**Esta é a dívida (a) do ticket 13 do treino.** O domicílio decidido no ticket 03 daquele mapa —
repositório privado no Hugging Face, artefato assado na imagem no build, promoção por commit de
ponteiro — **nunca foi construído**. Hoje o pacote existe só no disco do dono do produto: máquina
nova sobe sem modelo, e reverter para o modelo anterior é copiar arquivo de volta à mão.

**Por que Hugging Face e não Vercel:** a Vercel hospeda o Next.js. Modelo Python não roda lá
(ADR-0004).

**O carregamento já falha do jeito certo.** `PacoteDeModelo.carregar` confere no *startup* — não na
previsão — que as features do manifesto são as canônicas na ordem canônica, e que a Largura de
Incerteza está na escala `a3` e não já convertida. Um pacote errado derruba o *startup*, não uma
resposta de Aluno no meio da tarde. Isso não muda; o que muda é que o pacote passa a estar lá.

**O teste que só existe no navegador.** CORS não falha em teste de servidor — falha no navegador,
antes da requisição sair. O ticket 03 conserta a regra; **este ticket é onde ela é verificada de
verdade**, com o Preditor rodando num browser contra a URL pública.

**Blocked by:** 03 (CORS vindo do ambiente).

**Status:** ready-for-agent

- [ ] Existe um Dockerfile que constrói a API com o pacote de modelo e os CSVs dentro da imagem
- [ ] O Space privado no Hugging Face está de pé e `/health` responde numa URL pública
- [ ] O pacote de modelo é buscado no build a partir do domicílio versionado, não copiado do disco
      de ninguém
- [ ] Promover um modelo novo é um commit de ponteiro, e reverter é voltar o ponteiro — documentado
      em passos executáveis
- [ ] O Preditor funciona **num navegador** contra a URL pública, sem erro de CORS
- [ ] Uma máquina limpa (sem `models/`, sem `data/`) reproduz o deploy do zero, sem cópia manual de
      arquivo — verificado, não presumido
- [ ] Os templates de `assets/` **não** entram na imagem nesta rodada
