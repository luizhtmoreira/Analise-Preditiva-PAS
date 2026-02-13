import io
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from pypdf import PdfReader, PdfWriter

class PDFGenerator:
    """
    Handles generation of student performance PDFs by overlaying text on a template.
    """
    
    def __init__(self):
        self.assets_dir = Path("assets")
        self.fonts_dir = self.assets_dir / "fonts"
        self.template_dir = self.assets_dir / "templates"
        self.template_path = self.template_dir / "MODELO PAS-UNB (ALUNOS) IMPRESSO.pdf"
        self.font_bold = "PublicSans-Bold"
        self.font_black = "PublicSans-Black"
        
        self._register_fonts()
        
    def _register_fonts(self):
        """Registers custom fonts if they exist."""
        try:
            # Check for font files
            bold_path = self.fonts_dir / "PublicSans-Bold.ttf"
            black_path = self.fonts_dir / "PublicSans-Black.ttf"
            
            if bold_path.exists():
                pdfmetrics.registerFont(TTFont(self.font_bold, str(bold_path)))
            else:
                print(f"Warning: {bold_path} not found. Using Helvetica-Bold.")
                self.font_bold = "Helvetica-Bold"
                
            if black_path.exists():
                pdfmetrics.registerFont(TTFont(self.font_black, str(black_path)))
            else:
                print(f"Warning: {black_path} not found. Using Helvetica-Bold.")
                self.font_black = "Helvetica-Bold"
                
        except Exception as e:
            print(f"Error registering fonts: {e}. Fallback to standard fonts.")
            self.font_bold = "Helvetica-Bold"
            self.font_black = "Helvetica-Bold"

    def generate_single_pdf(
        self, 
        data: Dict[str, Union[str, float]], 
        output_filename: str = None
    ) -> bytes:
        """
        Generates a single PDF in memory and returns the bytes.
        
        Args:
            data: Dictionary containing student info and scores.
            output_filename: Optional filename (not used for return check, just metadata if needed).
            
        Returns:
            bytes: The content of the final PDF.
        """
        # Create overlay canvas
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=landscape(A4))
        
        # Draw all content on the canvas
        # DEBUG: Print data keys to console to verify receipt
        print("DEBUG PDF DATA KEYS:", data.keys())
        print("DEBUG PDF DATA:", data)
        self._draw_content(c, data)
        
        c.save()
        packet.seek(0)
        
        # Merge with template
        try:
            if not self.template_path.exists():
                raise FileNotFoundError(f"Template not found at {self.template_path}")
                
            new_pdf = PdfReader(packet)
            existing_pdf = PdfReader(str(self.template_path))
            output = PdfWriter()
            
            # Assuming single page template for now
            page = existing_pdf.pages[0]
            if len(new_pdf.pages) > 0:
                page.merge_page(new_pdf.pages[0])
            
            output.add_page(page)
            
            output_stream = io.BytesIO()
            output.write(output_stream)
            output_stream.seek(0)
            return output_stream.getvalue()
            
        except Exception as e:
            print(f"Error merging PDF: {e}")
            return b""

    def generate_batch_zip(self, data_list: List[Dict[str, Union[str, float]]]) -> bytes:
        """
        Generates multiple PDFs and returns a ZIP file as bytes.
        
        Args:
            data_list: List of dictionaries with student data.
            
        Returns:
            bytes: ZIP file content.
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for student_data in data_list:
                # Use student name for filename, sanitize it
                name = str(student_data.get('aluno', 'aluno')).strip()
                safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
                filename = f"{safe_name}.pdf"
                
                pdf_bytes = self.generate_single_pdf(student_data)
                if pdf_bytes:
                    zip_file.writestr(filename, pdf_bytes)
                    
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def _draw_content(self, c: canvas.Canvas, data: Dict[str, Union[str, float]]):
        """Helper to draw text at specific coordinates."""
        
        # Common settings
        c.setFillColor(HexColor("#FFFFFF"))
        
        # --- HEADER INFO ---
        c.setFont(self.font_bold, 12)
        c.drawString(85, 505, str(data.get('aluno', '')))
        c.drawString(430, 505, str(data.get('curso', '')))
        
        # --- SCORES & CALCULATIONS ---
        # Coordinates based on user provided snippet and logical estimation from grid layout
        # (These will likely need calibration)
        
        c.setFont(self.font_bold, 10)
        
        # PAS 1
        # P1 (Part 1) and Redacao (Essay) are separate in snippet logic
        # User snippet: c.drawString(140, 341.9, str(dados['nota_pas1'])) -> Assuming 'nota_pas1' means Part 1?
        # Or Total PAS 1? Let's check keys requested by user:
        # "parte 2" and "redação" for stages 1 and 2
        
        # Using keys that will be mapped from inputs
        
        # --- PAS 1 ---
        c.drawString(140, 341.9, str(data.get('pas1_p1', ''))) # Parte 1
        c.drawString(140, 341.9 - 18, str(data.get('pas1_p2', ''))) # Parte 2
        c.drawString(140, 341.9 - 36, str(data.get('pas1_red', '')))  # Redação
        
        # --- PAS 2 ---
        c.drawString(275, 341.9, str(data.get('pas2_p1', '')))
        c.drawString(275, 341.9 - 18, str(data.get('pas2_p2', '')))
        c.drawString(275, 341.9 - 36, str(data.get('pas2_red', '')))
        
        # --- PAS 3 ---
        # Set color to Blue #184283 for PAS 3 data
        c.setFillColor(HexColor("#184283"))
        
        c.drawString(540, 110, f"{data.get('pas3_p1_est', '')}*")
        c.drawString(540, 93, str(data.get('pas3_p2_necessario', '')))
        c.drawString(540, 75, f"{data.get('pas3_red_est', '')}*")
        
        # --- RESULTS ---
        c.setFillColor(HexColor("#FFFFFF"))
        
        # Weighted Arguments & Accumulated
        # Ensure these keys exist in 'data' dict sent from streamlit_app.py
        # Adjusted X to 480 because 650 is off-page for A4 (width ~595)
        c.drawString(720, 390, str(data.get('arg_pond_1', '')))
        c.drawString(720, 348, str(data.get('arg_pond_2', '')))
        c.drawString(720, 303, str(data.get('arg_acumulado', '')))
        
        # Nota Corte and Arg Necessario (Multiplied by 3 as requested)
        c.drawString(160, 160, str(data.get('nota_corte', '-')))
        c.drawString(220, 143, str(data.get('arg_acumulado', '-'))) 
        
        # Calculate Arg Necessario * 3 for display if it's a number
        arg_nec = data.get('arg_necessario', '-')
        try:
            arg_nec_val = float(arg_nec)
            arg_nec_display = f"{arg_nec_val * 3:.3f}"
        except:
            arg_nec_display = str(arg_nec)
            
        c.drawString(250, 117, arg_nec_display)
        
        # --- PROBABILITY AND Z-SCORE ---
        # User requested: (400, 250) for probability and (400, 230) for Z-score
        #c.setFillColor(HexColor("#FFFFFF")) # Blue for statistical analysis
        #c.setFont(self.font_black, 14)
        #c.drawString(400, 250, f"Probabilidade: {data.get('probabilidade', '-')}")
        
        #c.setFont(self.font_bold, 12)
        #c.drawString(400, 230, f"{data.get('z_score', '-')}")
        
        # Restore color for subsequent items if any
        #c.setFillColor(HexColor("#FFFFFF"))
        
        # Extra fields requested:
        # "argumento acumulado"
        # "nota de corte" (pegar a nota de corte do 'curso' de 2022-2024)
        # "argumento necessário"
        
        # Positioning these might require a new block.
        # Let's estimate them at the bottom or side based on standard report layouts.
        # For now, I will stack them below PAS 3 row or in a "Resultados" box if visible in template.
        # Since I can't see the template, I'll place them below PAS 3 with clear labels.
        
        # c.drawString(250, 117, f"Arg. Necessário (Peso 3): {data.get('arg_necessario', '-')}")

