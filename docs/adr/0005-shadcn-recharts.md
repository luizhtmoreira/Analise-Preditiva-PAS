# shadcn/ui + Recharts como stack de UI do dashboard

Após comparar quatro combinações via protótipo interativo (2026-06-06), escolhemos shadcn/ui com Tailwind CSS para componentes e Recharts para gráficos. As alternativas avaliadas foram: shadcn + Plotly (2º lugar), Mantine + @mantine/charts e Mantine + Plotly.

shadcn/ui ganhou pelo controle total sobre tokens de design — as cores institucionais da UnB (Azul `#003366`, Verde `#00843D`, Cyan `#00AEEF`) e o sistema de semáforo de risco são aplicados nativamente via CSS variables do Tailwind, sem brigar com estilos de terceiros. Recharts ganhou sobre Plotly porque os gráficos se integram ao visual do produto sem o overhead visual e o bundle (~3MB) do Plotly; animações e tooltips são customizáveis via Tailwind. Plotly ficou em 2º — pode ser considerado para gráficos estatísticos complexos (KDE overlay, distplot) se Recharts não cobrir bem a necessidade.

## Consequências

- O protótipo em `prototype/dashboard-ui/` pode ser deletado — respondeu a pergunta.
- Gráficos de distribuição com KDE (atualmente via `plotly.figure_factory`) precisarão ser reimplementados em Recharts ou via SVG customizado; avaliar caso a caso durante a implementação.
