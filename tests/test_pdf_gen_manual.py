import sys
import os
from pathlib import Path

# Add src to path
project_root = Path("c:/Users/user/Documents/unb/Codigos/repositorios/Analise-Preditiva-PAS")
sys.path.insert(0, str(project_root / "src"))

from pdf_generator import PDFGenerator

def test_pdf_gen():
    print("Testing PDF Generation...")
    
    # Change CWD to project root so assets are found
    os.chdir(project_root)
    
    gen = PDFGenerator()
    
    data = {
        'aluno': 'Teste Verificacao',
        'curso': 'ENGENHARIA DE SOFTWARE - GAMA',
        'pas1_p1': '10.5',
        'pas1_p2': '50.0',
        'pas1_red': '8.5',
        'pas1_arg': '1.200',
        'pas2_p1': '12.0',
        'pas2_p2': '60.0',
        'pas2_red': '9.0',
        'pas2_arg': '1.500',
        'pas3_p1_est': '10.0*',
        'pas3_red_est': '8.0*',
        'arg_acumulado': '2.000',
        'nota_corte': '-1.500',
        'arg_necessario': '-0.500'
    }
    
    output_path = "test_output.pdf"
    
    try:
        pdf_bytes = gen.generate_single_pdf(data)
        if pdf_bytes:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"✅ PDF generated successfully: {output_path}")
            print(f"Size: {len(pdf_bytes)} bytes")
        else:
            print("❌ PDF generation returned empty bytes.")
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    test_pdf_gen()
