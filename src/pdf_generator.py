import io
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from pypdf import PdfReader, PdfWriter

class PDFGenerator:
    """
    Handles generation of student performance PDFs by overlaying text on a template.
    """
    
    def __init__(self):
        # Resolve assets path relative to this file
        base_dir = Path(__file__).resolve().parent.parent
        self.assets_dir = base_dir / "assets"
        
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
        output_filename: str = None,
        template_override: str = None
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
        
        # Resolve template path (suporte a whitelabel por escola)
        actual_template_path = self.template_path
        if template_override:
            base_dir = Path(__file__).resolve().parent.parent
            actual_template_path = (base_dir / template_override).resolve()

        if not actual_template_path.exists():
            raise FileNotFoundError(f"Template não encontrado em: {actual_template_path}")

        # Draw content on canvas
        self._draw_content(c, data, template_path=actual_template_path)

        c.save()
        packet.seek(0)

        # Lógica original comprovada: merge overlay sobre template
        new_pdf = PdfReader(packet)
        existing_pdf = PdfReader(str(actual_template_path))
        output = PdfWriter()

        page = existing_pdf.pages[0]
        if len(new_pdf.pages) > 0:
            page.merge_page(new_pdf.pages[0])

        output.add_page(page)

        output_stream = io.BytesIO()
        output.write(output_stream)
        output_stream.seek(0)
        return output_stream.getvalue()

    def generate_batch_zip(self, data_list: List[Dict[str, Union[str, float]]], template_override: str = None) -> bytes:
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
                # Use student name for filename — sanitize with stdlib (no extra deps)
                import unicodedata
                name = str(student_data.get('aluno', 'aluno')).strip()
                # Remove acentos e caracteres especiais para compatibilidade máxima de ZIP/Windows
                safe_name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
                safe_name = "".join([c for c in safe_name if c.isalnum() or c in (' ', '-', '_')]).strip()
                safe_name = safe_name.replace(" ", "_").upper()
                filename = f"{safe_name}.pdf"
                
                pdf_bytes = self.generate_single_pdf(student_data, template_override=template_override)
                if pdf_bytes:
                    zip_file.writestr(filename, pdf_bytes)
                    
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def generate_courses_pdf(
        self,
        courses_data: Dict[str, list],
        template_override: str = None
    ) -> bytes:
        """
        Generates a PDF with top 3 courses per semester overlaid on the CURSOS template.
        
        Args:
            courses_data: Dict with keys 'sem1' and 'sem2', each a list of up to 3 dicts
                          with keys 'curso' (str) and 'chance' (str, e.g. "85.3%").
            template_override: Optional path to template (relative to project root).
            
        Returns:
            bytes: The content of the final PDF.
        """
        # Resolve template
        if template_override:
            base_dir = Path(__file__).resolve().parent.parent
            template_path = (base_dir / template_override).resolve()
        else:
            template_path = self.template_dir / "MODELO PAS-UNB (CURSOS) IMPRESSO.pdf"

        if not template_path.exists():
            raise FileNotFoundError(f"Template de cursos não encontrado em: {template_path}")

        # Coordinates for each slot (curso_x, chance_x, y)
        coords_sem1 = [
            (190, 614, 415),  # 1º curso
            (190, 614, 380),  # 2º curso
            (190, 614, 346),  # 3º curso
        ]
        coords_sem2 = [
            (190, 614, 184),  # 1º curso
            (190, 614, 149),  # 2º curso
            (190, 614, 116),  # 3º curso
        ]

        # Create overlay canvas
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=landscape(A4))

        # Apply the same offset used for IMPRESSO templates
        if "IMPRESSO" in str(template_path).upper():
            c.translate(0, 8.8)

        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont(self.font_bold, 10)

        max_w = 550 - 190 # Largura máxima para o curso (começa em 190, não pode passar de 550)

        def draw_truncated_course(canvas_obj, x, y, text, max_width):
            """Desenha o texto truncando com '...' se exceder a largura."""
            if canvas_obj.stringWidth(text, self.font_bold, 10) <= max_width:
                canvas_obj.drawString(x, y, text)
            else:
                # Truncamento simples
                while canvas_obj.stringWidth(text + "...", self.font_bold, 10) > max_width and len(text) > 0:
                    text = text[:-1]
                canvas_obj.drawString(x, y, text + "...")

        # Draw 1º Semestre courses
        for i, slot in enumerate(coords_sem1):
            curso_x, chance_x, y = slot
            items = courses_data.get('sem1', [])
            if i < len(items):
                curso_nome = str(items[i].get('curso', ''))
                draw_truncated_course(c, curso_x, y, curso_nome, max_w)
                c.drawString(chance_x, y, str(items[i].get('chance', '')))

        # Draw 2º Semestre courses
        for i, slot in enumerate(coords_sem2):
            curso_x, chance_x, y = slot
            items = courses_data.get('sem2', [])
            if i < len(items):
                curso_nome = str(items[i].get('curso', ''))
                draw_truncated_course(c, curso_x, y, curso_nome, max_w)
                c.drawString(chance_x, y, str(items[i].get('chance', '')))

        c.save()
        packet.seek(0)

        # Merge overlay onto template (same proven logic)
        new_pdf = PdfReader(packet)
        existing_pdf = PdfReader(str(template_path))
        output = PdfWriter()

        page = existing_pdf.pages[0]
        if len(new_pdf.pages) > 0:
            page.merge_page(new_pdf.pages[0])

        output.add_page(page)

        output_stream = io.BytesIO()
        output.write(output_stream)
        output_stream.seek(0)
        return output_stream.getvalue()

    def generate_comparison_pdf(
        self,
        data: Dict[str, Union[str, float]],
        template_override: str = None
    ) -> bytes:
        """
        Generates a comparison PDF overlaying predicted vs real scores.
        
        Args:
            data: Dictionary containing:
                  eb_pas1, eb_pas2, eb_pas3_pred, arg_final_pred, eb_pas3_real, arg_final_real
            template_override: Optional path to template.
            
        Returns:
            bytes: The content of the final PDF.
        """
        if template_override:
            base_dir = Path(__file__).resolve().parent.parent
            template_path = (base_dir / template_override).resolve()
        else:
            template_path = self.template_dir / "MODELO PAS-UNB (COMPARAÇÃO) IMPRESSO.pdf"

        if not template_path.exists():
            raise FileNotFoundError(f"Template de comparação não encontrado em: {template_path}")

        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=landscape(A4))

        # Apply offset for IMPRESSO templates
        if "IMPRESSO" in str(template_path).upper():
            c.translate(0, 8.8)

        # Draw content
        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont(self.font_bold, 17)

        # Coordinates
        # EB PAS 1: (37.5, 95)
        # EB PAS 2: (97.5 ,95)
        # PROJETADO
        # EB PAS 3: (202.5, 152.5)
        # ARG. FINAL: (260, 152.5)
        # REAL
        # EB PAS 3: (202.5, 44)
        # ARG. FINAL: (260, 44)
        
        # Movidos mais 3 para esquerda (acumulado: 11 esquerda, 7 baixo das originais)
        
        c.drawString(27.5 * mm, 89 * mm, str(data.get('eb_pas1', '')))
        c.drawString(86.5 * mm, 89 * mm, str(data.get('eb_pas2', '')))
        
        c.drawString(193.5 * mm, 147.5 * mm, str(data.get('eb_pas3_pred', '')))
        c.drawString(251 * mm, 147.5 * mm, str(data.get('arg_final_pred', '')))
        
        c.drawString(193.5 * mm, 37 * mm, str(data.get('eb_pas3_real', '')))
        c.drawString(251 * mm, 37 * mm, str(data.get('arg_final_real', '')))

        c.save()
        packet.seek(0)

        new_pdf = PdfReader(packet)
        existing_pdf = PdfReader(str(template_path))
        output = PdfWriter()

        page = existing_pdf.pages[0]
        if len(new_pdf.pages) > 0:
            page.merge_page(new_pdf.pages[0])

        output.add_page(page)

        output_stream = io.BytesIO()
        output.write(output_stream)
        output_stream.seek(0)
        return output_stream.getvalue()

    def _draw_content(self, c: canvas.Canvas, data: Dict[str, Union[str, float]], template_path: str = ""):
        """Helper to draw text at specific coordinates."""
        
        # O texto dessas áreas superiores (Aluno, Curso, PAS 1 e 2) é sempre branco,
        # visto que o template novo Genérico também tem os banners azuis.
        base_color = "#FFFFFF" 
        if "IMPRESSO" in str(template_path).upper() or "DEFAULT" in str(template_path).upper():
            # Os templates IMPRESSO / GENERICO possuem arte e MediaBox levemente deslocados 
            # em relação ao DIGITAL. Para que as coordenadas originais continuem corretas, 
            # aplicamos um pequeno offset (x, y) no Canvas.
            c.translate(0, 8.8)
            
        c.setFillColor(HexColor(base_color))
        
        # --- HEADER INFO ---
        c.setFont(self.font_bold, 12)
        c.drawString(85, 506, str(data.get('aluno', '')))
        c.drawString(430, 506, str(data.get('curso', '')))
        c.setFont(self.font_bold, 10)
        
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
        c.drawString(140, 342.9, str(data.get('pas1_p1', ''))) # Parte 1
        c.drawString(140, 342.9 - 18, str(data.get('pas1_p2', ''))) # Parte 2
        c.drawString(140, 343.4 - 36, str(data.get('pas1_red', '')))  # Redação (+0.5)
        
        # --- PAS 2 ---
        c.drawString(275, 342.9, str(data.get('pas2_p1', '')))
        c.drawString(275, 342.9 - 18, str(data.get('pas2_p2', '')))
        c.drawString(275, 343.4 - 36, str(data.get('pas2_red', ''))) # Redação (+0.5)
        
        # --- PAS 3 ---
        # Set color to Blue #184283 for PAS 3 data
        c.setFillColor(HexColor("#184283"))
        
        c.drawString(550, 105, f"{data.get('pas3_p1_est', '')}*")
        c.drawString(550, 86, str(data.get('pas3_p2_necessario', '')))
        c.drawString(550, 66.5, f"{data.get('pas3_red_est', '')}*") # Redação (+0.5)
        
        # --- RESULTS ---
        c.setFillColor(HexColor("#FFFFFF"))
        
        # Weighted Arguments & Accumulated
        # Ensure these keys exist in 'data' dict sent from streamlit_app.py
        # Adjusted X to 480 because 650 is off-page for A4 (width ~595)
        c.drawString(720, 391, str(data.get('arg_pond_1', ''))) # Arg. Pond. 1 inc by 1
        c.drawString(720, 348, str(data.get('arg_pond_2', ''))) # EXCETO: Arg. Pond. 2
        c.drawString(720, 303, str(data.get('arg_acumulado', ''))) # EXCETO: Arg. Acumulado
        
        # Nota Corte and Arg Necessario (Multiplied by 3 as requested)
        c.drawString(160, 161, str(data.get('nota_corte', '-')))
        c.drawString(220, 143, str(data.get('arg_acumulado', '-'))) # EXCETO: Arg. Acumulado
        
        # Calculate Arg Necessario * 3 for display if it's a number
        arg_nec = data.get('arg_necessario', '-')
        try:
            arg_nec_val = float(arg_nec)
            arg_nec_display = f"{arg_nec_val * 3:.3f}"
        except:
            arg_nec_display = str(arg_nec)
            
        c.drawString(250, 119, arg_nec_display) # Arg. Necessário (+1)
        
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

