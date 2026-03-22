import os
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import sys

def criar_pdf_com_grid_numerado(arquivo_temporario, largura, altura, tamanho_grid_mm=5):
    """Gera um PDF contendo o grid e os números das coordenadas em milímetros."""
    c = canvas.Canvas(arquivo_temporario, pagesize=(largura, altura))
    tamanho_grid = tamanho_grid_mm * mm
    
    # IMPORTANTE: Aplicar exatamente a mesma transformação que o app real aplica
    # O arquivo actual tem: c.translate(0, 8.8)
    c.translate(0, 8.8)

    # Configurando a fonte dos números (tamanho 6)
    c.setFont("Helvetica", 6)

    # Desenhando o Eixo X (Linhas Verticais)
    x = 0
    coord_mm = 0
    # O loop vai um pouco além da largura original para compensar a translação, 
    # mas largura + translação é suficiente
    while x <= largura * 1.5:
        # Se for múltiplo de 10mm clarear para azul mais escuro ou verde
        if coord_mm % 10 == 0:
            c.setStrokeColorRGB(0.2, 0.4, 0.8) 
            c.setLineWidth(0.8)
            c.setFillColorRGB(0.2, 0.4, 0.8) 
        else:
            c.setStrokeColorRGB(0.7, 0.7, 0.7) 
            c.setLineWidth(0.5)
            c.setFillColorRGB(0.8, 0, 0) 

        c.line(x, -20 * mm, x, altura * 1.5)
        # Escreve o número na borda inferior e no meio
        c.drawString(x + 1, 5 * mm, str(coord_mm))
        c.drawString(x + 1, altura / 2, str(coord_mm)) 
        c.drawString(x + 1, altura - 20 * mm, str(coord_mm)) 
        
        x += tamanho_grid
        coord_mm += tamanho_grid_mm

    # Desenhando o Eixo Y (Linhas Horizontais)
    y = -20 * mm  # começar um pouco antes por causa do offset
    coord_mm = -20
    while y <= altura * 1.5:
        if coord_mm % 10 == 0:
            c.setStrokeColorRGB(0.2, 0.4, 0.8) 
            c.setLineWidth(0.8)
            c.setFillColorRGB(0.2, 0.4, 0.8) 
        else:
            c.setStrokeColorRGB(0.7, 0.7, 0.7) 
            c.setLineWidth(0.5)
            c.setFillColorRGB(0.8, 0, 0) 

        c.line(0, y, largura, y)
        # Escreve o número na borda esquerda e no meio da página
        c.drawString(5 * mm, y + 1.5, str(coord_mm))
        c.drawString(largura / 2, y + 1.5, str(coord_mm))
        c.drawString(largura - 20 * mm, y + 1.5, str(coord_mm))
        
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
        
        # Chama a nova função que inclui os números com o offset do app
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

if __name__ == "__main__":
    # Caminho base do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    arquivo_input = os.path.join(base_dir, "assets", "templates", "MODELO PAS-UNB (COMPARAÇÃO) IMPRESSO.pdf")
    arquivo_output = os.path.join(base_dir, "assets", "templates", "modelo_comparacao_grid_corrigido.pdf")
    
    if not os.path.exists(arquivo_input):
        print(f"Erro: Arquivo não encontrado - {arquivo_input}")
        sys.exit(1)
        
    aplicar_grid_no_pdf(arquivo_input, arquivo_output, tamanho_quadrado=5)
