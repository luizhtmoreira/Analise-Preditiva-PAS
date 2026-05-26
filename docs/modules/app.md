# Dashboard (Streamlit)

O módulo **app** é o ponto de contato entre as escolas e os dados do Vetor PAS. 

## `app/streamlit_app.py`

O script principal do dashboard serve como o orquestrador das interações. Ele é encarregado de:

1. **Autenticação e Multi-tenant**: Detecta a escola que está acessando e altera a paleta de cores, logotipos e referências visuais em tempo real.
2. **Upload de Turmas**: Disponibiliza a interface de _Drag and Drop_ para que a coordenação insira a base dos alunos.
3. **Visualização de Dados**:
    - Constrói o **Semáforo de Risco** que classifica os alunos entre _Verde_, _Amarelo_ ou _Vermelho_.
    - Renderiza gráficos comparativos utilizando as bibliotecas nativas do Streamlit integradas com pandas e plotly (quando aplicável).
4. **Chamada ao Motor Preditivo**: Recebe o *input* da interface, invoca as funções do `pas_intelligence` e exibe instantaneamente a nota predita.
5. **Comunicação com o Supabase**: Salva histórico de ações, gerencia permissões e puxa as tabelas estáticas das notas de corte.

!!! tip "Dica de Desenvolvimento"
    Ao adicionar novas abas (Tabs) no dashboard, mantenha o estado armazenado em `st.session_state` para evitar recarregamento desnecessário do motor de inferência a cada clique do usuário.
