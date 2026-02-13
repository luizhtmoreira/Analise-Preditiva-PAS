import pandas as pd
from pypdf import PdfReader
import re

# Paths
PDF_PATH = 'data/Ed_8_2024_PAS_UnB_1_2024-2026_Res_final_tipo_D_redação.pdf'
CSV_PATH = 'data/banco_alunos_pas_final.csv'

def main():
    print("--- Starting PAS 1 Data Extraction (PDF) ---")

    # 1. Load CSV
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, dtype={'Inscricao': str, 'Ano_Trienio': str})
    
    # We only want to update students from the current triennium (PAS 2 students just added)
    # The requirement is: "Usando o número de inscrição dos alunos que acabaram de ser adicionados ao banco"
    # These students have Ano_Trienio = '2024-2026'
    target_mask = df['Ano_Trienio'] == '2024-2026'
    target_students = df[target_mask]
    
    print(f"Total students in CSV: {len(df)}")
    print(f"Target students (2024-2026) to update: {len(target_students)}")
    
    # 2. Extract PDF Text
    print(f"Reading PDF: {PDF_PATH}")
    try:
        reader = PdfReader(PDF_PATH)
    except FileNotFoundError:
        print("Error: PDF file not found.")
        return

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + " "
    
    # Normalize text: replace newlines with spaces to handle broken lines
    # Also handle multiple spaces
    normalized_text = full_text.replace('\n', ' ')
    normalized_text = re.sub(r'\s+', ' ', normalized_text)
    
    # 3. Parse Records
    # Separator is " / "
    # We usually have a start marker, but scanning the whole text string split by / is effective
    raw_records = normalized_text.split(' / ')
    
    extracted_data = {}
    first_student_pdf = None
    last_student_pdf = None
    count_pdf = 0
    
    print("Parsing PDF records...")
    
    for record in raw_records:
        # Each record: Inscricao, Nome, P1, P2, Soma, NotaD, Red
        # Example: 24141937, Aarao Matos, 0.000, 14.535, 14.535, 0.000, 6.759
        
        # Clean leading/trailing slashes or spaces
        clean_record = record.strip()
        if not clean_record:
            continue
            
        parts = [p.strip() for p in clean_record.split(',')]
        
        # We expect at least 7 parts. Sometimes names have commas, but usually not in this format.
        # Strict checking: index 0 should be numeric (Inscricao)
        if len(parts) >= 7 and parts[0].isdigit():
            inscricao = parts[0]
            nome = parts[1]
            try:
                p1 = float(parts[2])
                p2 = float(parts[3])
                red = float(parts[6])
                
                extracted_data[inscricao] = {
                    'P1_PAS1': p1,
                    'P2_PAS1': p2,
                    'Red_PAS1': red,
                    'Nome': nome # For reporting
                }
                
                if first_student_pdf is None:
                    first_student_pdf = nome
                last_student_pdf = nome
                count_pdf += 1
                
            except ValueError:
                # parsing error, skip
                continue

    print("\n--- PDF Extraction Report ---")
    print(f"First Student Read: {first_student_pdf}")
    print(f"Last Student Read:  {last_student_pdf}")
    print(f"Total Found in PDF: {count_pdf}")
    print("-----------------------------")

    # 4. Update CSV
    print("Updating CSV matching records...")
    
    matched_count = 0
    
    # Iterate over the target dataframe indices
    for idx in target_students.index:
        inscricao = str(target_students.at[idx, 'Inscricao'])
        
        if inscricao in extracted_data:
            data = extracted_data[inscricao]
            df.at[idx, 'P1_PAS1'] = data['P1_PAS1']
            df.at[idx, 'P2_PAS1'] = data['P2_PAS1']
            df.at[idx, 'Red_PAS1'] = data['Red_PAS1']
            matched_count += 1
            
    print(f"Updated {matched_count} records out of {len(target_students)} targets.")
    
    # 5. Save
    print("Saving updated CSV...")
    df.to_csv(CSV_PATH, index=False)
    print("Success! PAS 1 integration complete.")

if __name__ == "__main__":
    main()
