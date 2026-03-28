import sys
from pathlib import Path

# Adiciona a raiz do projeto ao PYTHONPATH para importar src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pdf_generator import PDFGenerator

def test_comparison_pdf():
    pdf_gen = PDFGenerator()
    
    comparacao_data = {
        'eb_pas1': "45.000",
        'eb_pas2': "55.000",
        'eb_pas3_pred': "60.000",
        'arg_final_pred': "120.000",
        'eb_pas3_real': "65.000",
        'arg_final_real': "125.000"
    }

    print("Generating comparison PDF...")
    template_override = "assets/templates/modelo_comparacao_impresso.pdf"
    
    try:
        pdf_bytes = pdf_gen.generate_comparison_pdf(comparacao_data, template_override=template_override)
        
        output_path = Path("/tmp/test_comparacao.pdf")
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
            
        print(f"Success! PDF saved to: {output_path}")
        
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    test_comparison_pdf()
