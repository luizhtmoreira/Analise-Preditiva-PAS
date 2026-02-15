
import pdfplumber
import pandas as pd
import re
import os

# --- Configuration ---
PDF_PATH = r"c:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\3089BE7E47EF3C07390C31C7506BA674B47B6248F1E990ED2648953612E61491.pdf"
MASTER_CSV_PATH = r"c:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\PAS_MESTRE_LIMPO_FINAL.csv"
OUTPUT_CSV_PATH = r"c:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\notas_corte_pas_final_BLINDADO.csv"
TRIENNIUM = "2023-2025"

def parse_cutoff_pdf(pdf_path):
    print(f"--- Parsing PDF: {os.path.basename(pdf_path)} ---")
    data = []
    current_course = None
    current_semester = "1º" 
    
    # Regex: "23117343 Alicia Cruz Porfirio 9"
    student_pattern = re.compile(r"^(\d{8})\s+(.+?)\s+(\d+)$")
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: 
                continue
                
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                
                # Detect Student
                match = student_pattern.match(line)
                if match:
                    if not current_course:
                        continue 
                        
                    inscricao = int(match.group(1))
                    nome = match.group(2)
                    sistema = int(match.group(3))
                    
                    data.append({
                        "Inscricao": inscricao,
                        "Nome": nome,
                        "Sistema": sistema,
                        "Curso": current_course,
                        "Semestre": current_semester
                    })
                    continue
                
                # Detect Course Header
                if (("BACHARELADO" in line) or ("LICENCIATURA" in line) or ("ENGENHARIA" in line) or ("TECNOL" in line)) and \
                   ("Nome do candidato" not in line) and \
                   (len(line) > 10):
                    current_course = line
                    
    print(f"Extracted {len(data)} student-course records.")
    return pd.DataFrame(data)

def calculate_stats():
    # 1. Parse PDF
    df_pdf = parse_cutoff_pdf(PDF_PATH)
    if df_pdf.empty:
        print("No data found in PDF.")
        return

    # 2. Filter System 1
    df_universal = df_pdf[df_pdf['Sistema'] == 1].copy()
    print(f"Filtered for Sistema 1 (Universal): {len(df_universal)} students.")
    
    # 3. Load Master
    df_master = pd.read_csv(MASTER_CSV_PATH)
    df_master['Inscricao'] = pd.to_numeric(df_master['Inscricao'], errors='coerce')
    
    # 4. Merge
    df_merged = pd.merge(df_universal, df_master[['Inscricao', 'Arg_Final']], on='Inscricao', how='inner')
    print(f"Matched with Master DB: {len(df_merged)} students.")
    
    if len(df_merged) < len(df_universal):
        print(f"WARNING: {len(df_universal) - len(df_merged)} mismatched students.")
    
    # 5. Stats
    stats = df_merged.groupby(['Curso', 'Semestre'])['Arg_Final'].agg(['min', 'max', 'mean', 'count']).reset_index()
    stats['Trienio'] = TRIENNIUM
    stats = stats.rename(columns={'min': 'Min', 'max': 'Max', 'mean': 'Media', 'count': 'N'})
    stats = stats[['Trienio', 'Semestre', 'Curso', 'Min', 'Max', 'Media', 'N']]
    
    # Rounding
    for col in ['Min', 'Max', 'Media']:
        stats[col] = stats[col].round(3)
    
    # 6. Save
    if os.path.exists(OUTPUT_CSV_PATH):
        df_existing = pd.read_csv(OUTPUT_CSV_PATH)
        df_existing = df_existing[df_existing['Trienio'] != TRIENNIUM]
        df_final = pd.concat([df_existing, stats], ignore_index=True)
    else:
        df_final = stats
        
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Updated {OUTPUT_CSV_PATH} with {len(stats)} courses.")

if __name__ == "__main__":
    calculate_stats()
