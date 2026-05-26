# Gerador de Relatórios (PDF)

A entrega física (ou em arquivo digital) que o coordenador faz ao aluno é gerada pelo script `src/pdf_generator.py`.

## Renderização via ReportLab

O sistema utiliza a biblioteca **ReportLab** combinada com recursos visuais para criar documentos ricos e customizados.

### Abordagem Whitelabel

Como o Vetor PAS atende diversas instituições:

- O motor de geração é neutro.
- No momento da chamada do relatório, ele recebe via parâmetro qual a `escola` e o `caminho_logo`.
- Através da injeção de imagens sobre os templates base na pasta `assets/templates/`, o relatório de predição é re-estilizado (cores e logotipos) de forma imperceptível para o aluno final.

### Processamento em Lote

O script suporta a geração "assíncrona" ou em lote. Para uma escola que fez o *upload* de 200 alunos:
1. O sistema itera sobre os alunos e gera o PDF na memória.
2. Comprime (ZIP) todos os artefatos.
3. Disponibiliza um *download* único no Streamlit, garantindo eficiência computacional.
