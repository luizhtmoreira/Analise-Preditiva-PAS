# Relatórios em PDF

`src/pdf_generator.py` — a classe `PDFGenerator`, que injeta os dados calculados em modelos de PDF
whitelabel usando ReportLab.

## Abordagem

O motor de geração é neutro: ele recebe por parâmetro qual é a escola e qual o caminho do
logotipo, e injeta imagens e dados sobre o modelo base correspondente. O aluno recebe um documento
com a identidade da escola dele, sem marca do fornecedor.

Os modelos ficam em `assets/templates/`, fora do controle de versão — são ativo de produto
whitelabel, tratados do mesmo modo que os modelos treinados. O diretório é explicitamente excluído
das imagens do servidor.

## Geração em lote

Para uma turma inteira, o gerador itera sobre os alunos produzindo cada PDF em memória, comprime
tudo em um único arquivo ZIP e disponibiliza um download só.

## Estado atual

O módulo é consumido **apenas pela ferramenta interna Streamlit**. Ele ainda não foi portado para
a API, e portanto a coordenação ainda não emite relatórios pelo portal — a emissão é operada por
nós. É o item mais visível da fila de migração.
