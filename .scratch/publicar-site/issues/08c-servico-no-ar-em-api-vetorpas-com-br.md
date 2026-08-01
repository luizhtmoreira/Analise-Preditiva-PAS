# 08c — O serviço no ar, em `api.vetorpas.com.br`

**What to build:** a API responde numa URL pública que é **sua**, e o Preditor funciona num
navegador de verdade contra ela.

Este é o passo que o ticket 08 nunca conseguiu executar — e o único critério de aceite dele que
ficou marcado como pendente, porque CORS não falha em teste de servidor: falha no navegador, antes
da requisição sair.

**A maior parte deste ticket é manual**, na conta do dono do produto: criar o serviço no Render a
partir do Repo de Deploy, cadastrar o secret de build, o CNAME `api` na Hostinger (o DNS está lá —
`cosmos/nova.dns-parking.com` — e a Vercel só ocupa o apex e o `www`), e as variáveis nos dois
lados. A entrega em código é o `deploy/README.md` em passos executáveis, não implementação.

**O endereço é `api.vetorpas.com.br`, nunca o `onrender.com`.** Gravar o nome da plataforma no
build do Next.js tornaria a saída cara: trocar de hospedagem exigiria redeployar a landing, e o
projeto na Vercel é **CLI-only, sem integração Git** — um deploy manual que ninguém roda há meses,
sob pressão, com a API fora. Com domínio próprio, sair do Render é um CNAME (ADR-0014). O Render dá
TLS gratuito em domínio próprio em todos os planos.

**Cuidado com a cota:** 750 horas de instância por **workspace** por mês. Não crie um segundo
serviço (staging, cron) sem contar — um serviço sempre acordado já consome ~730, e estourar suspende
tudo até virar o mês.

**Este ticket é o que destrava o ticket 14** (*publicação: `main` no ar contra a API hospedada*).
O 14 não espera pelo `08d` nem pelo `08e`.

**Blocked by:** 08a (Derivado de Deploy), 08b (imagem na forma do Render e Repo de Deploy).

**Status:** done — commit `29df5a0`

- [x] `/health` responde em `https://api.vetorpas.com.br`, com TLS válido
- [x] O **Preditor funciona num navegador de verdade** contra a URL pública, sem erro de CORS —
      testado no browser, não com `curl`
- [x] Nem `NEXT_PUBLIC_API_URL` nem `API_URL` contêm a string `onrender.com`
- [x] `/api/predict` devolve previsão real e `modelo_disponivel: true` a partir do pacote que veio
      do Domicílio Versionado
- [ ] Uma máquina limpa (sem `models/`, sem `data/`) reproduz o deploy do zero, sem cópia manual de
      arquivo — verificado, não presumido *(pendente — requer ambiente isolado, fora do escopo desta sessão)*
- [x] `deploy/README.md` leva alguém do zero ao serviço no ar sem adivinhar nenhum passo
- [x] Existe exatamente **um** serviço no workspace do Render
