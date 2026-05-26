import os
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

def criar_pdf_com_grid_numerado(arquivo_temporario, largura, altura, tamanho_grid_mm=5):
    """Gera um PDF contendo o grid e os números das coordenadas em milímetros."""
    c = canvas.Canvas(arquivo_temporario, pagesize=(largura, altura))
    tamanho_grid = tamanho_grid_mm * mm
    
    # Configurando o estilo da linha do grid
    c.setStrokeColorRGB(0.7, 0.7, 0.7) 
    c.setLineWidth(0.5)

    # Configurando a fonte dos números (tamanho 6 e cor vermelha)
    c.setFont("Helvetica", 6)
    c.setFillColorRGB(0.8, 0, 0) 

    # Desenhando o Eixo X (Linhas Verticais)
    x = 0
    coord_mm = 0
    while x <= largura:
        c.line(x, 0, x, altura)
        # Escreve o número na borda inferior e no meio da página
        c.drawString(x + 1, 2 * mm, str(coord_mm))
        c.drawString(x + 1, altura / 2, str(coord_mm)) 
        x += tamanho_grid
        coord_mm += tamanho_grid_mm

    # Desenhando o Eixo Y (Linhas Horizontais)
    y = 0
    coord_mm = 0
    while y <= altura:
        c.line(0, y, largura, y)
        # Escreve o número na borda esquerda e no meio da página
        c.drawString(2 * mm, y + 1, str(coord_mm))
        c.drawString(largura / 2, y + 1, str(coord_mm))
        y += tamanho_grid
        coord_mm += tamanho_grid_mm

    c.save()

def aplicar_grid_no_pdf(pdf_entrada, pdf_saida, tamanho_quadrado=5):
    """Lê o PDF original, sobrepõe o grid numerado e salva o novo PDF."""
    reader = PdfReader(pdf_entrada)
    writer = PdfWriter()

    for index, page in enumerate(reader.pages):
        largura = float(page.mediabox.width)
        altura = float(page.mediabox.height)
        
        arquivo_temp = f"temp_grid_pagina_{index}.pdf"
        
        # Chama a nova função que inclui os números
        criar_pdf_com_grid_numerado(arquivo_temp, largura, altura, tamanho_quadrado)
        
        grid_reader = PdfReader(arquivo_temp)
        pagina_grid = grid_reader.pages[0]
        
        page.merge_page(pagina_grid)
        writer.add_page(page)
        
        if os.path.exists(arquivo_temp):
            os.remove(arquivo_temp)
            
    with open(pdf_saida, "wb") as out_file:
        writer.write(out_file)
    
    print(f"Sucesso! PDF com coordenadas salvo como: {pdf_saida}")

# ==========================================
# Exemplo de Uso
# ==========================================
if __name__ == "__main__":
    # Já deixei o caminho correto que você usou no último teste!
    arquivo_input = "/Users/luizhenrique/Documents/repos/Analise-Preditiva-PAS/assets/templates/modelo_comparacao_impresso.pdf"
    arquivo_output = "template_com_coordenadas.pdf"
    
    aplicar_grid_no_pdf(arquivo_input, arquivo_output, tamanho_quadrado=5)