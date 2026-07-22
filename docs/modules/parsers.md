# Extratores de Dados (Parsers)

Para alimentar o banco de dados do Vetor PAS de forma escalável e automática, foram criados os módulos de extração.

## `src/extract_pas1_pdf.py`

Gabaritos e resultados brutos do PAS 1 frequentemente são liberados pelo Cebraspe no formato PDF.
Este script é um *parser* customizado que:

1. Escaneia o texto blocando expressões regulares (Regex).
2. Isola o número de inscrição, nome do candidato e notas por matéria.
3. Trata os erros de leitura óptica comuns nestes documentos.
4. Exporta a base limpa para formato tabular (CSV).

## `src/extract_pas2_html.py`

Algumas notas finais ou boletins de desempenho do PAS 2 e 3 são publicados via portais da web estruturados em HTML.
Para essa finalidade, desenvolvemos:

- Extração utilizando bibliotecas de parsing HTML (como `BeautifulSoup`).
- Navegação pelo DOM (Document Object Model) da página oficial de resultados.
- Conversão limpa para a nossa estrutura de banco de dados no Supabase.

!!! warning "Mudanças no Cebraspe"
    Estes parsers são extremamente acoplados à formatação dos documentos de saída do Cebraspe. Caso a banca altere a fonte, estrutura da tabela ou o DOM, os parsers deverão ser ajustados imediatamente, caso contrário, gerarão exceções de Regex.
