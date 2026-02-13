
import pypdf
import pandas as pd
import re
import os

def clean_value(val):
    if val is None: return 0.0
    val = val.strip().replace(',', '.')
    try:
        return float(val)
    except:
        return 0.0

def extract_from_pdf(pdf_path):
    print(f"Processing {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() + " "
    
    # Split by '/'
    parts = all_text.split('/')
    students = []
    
    for part in parts:
        # Expected: insc, nome, p1, p2, soma, type_d, red
        fields = [f.strip() for f in part.split(',')]
        if len(fields) < 7:
            continue
            
        insc = fields[0]
        # Check if insc is 8 digits
        if not re.match(r'^\d{8}$', insc):
            # Sometimes there is noise before the enrollment number
            match = re.search(r'(\d{8})', insc)
            if match:
                insc = match.group(1)
            else:
                continue
        
        nome = fields[1]
        p1 = clean_value(fields[2])
        p2 = clean_value(fields[3])
        red = clean_value(fields[len(fields)-1]) # Redacao is always the last field
        
        students.append({
            'Inscricao': int(insc),
            'Nome': nome,
            'P1': p1,
            'P2': p2,
            'Red': red
        })
    
    return pd.DataFrame(students)

def main():
    pas1_pdf = "data/Ed_7_PAS_1_2023_2025_Res_final_tipo_D_redação.pdf"
    pas2_pdf = "data/Ed_15_PAS_2_2023-2025_Res_final_tipo_D_redacao.pdf"
    master_csv = "data/banco_alunos_pas_final.csv"
    
    df1 = extract_from_pdf(pas1_pdf)
    df2 = extract_from_pdf(pas2_pdf)
    
    print(f"Extracted {len(df1)} students from PAS 1")
    print(f"Extracted {len(df2)} students from PAS 2")
    
    # Merge
    # We want a row per student. Some might only have PAS 1 or PAS 2.
    merged = pd.merge(df1, df2, on=['Inscricao'], how='outer', suffixes=('_P1', '_P2'))
    
    # Fill names
    merged['Nome'] = merged['Nome_P1'].fillna(merged['Nome_P2'])
    
    # Map to master CSV columns
    # Inscricao,Nome,P1_PAS1,P2_PAS1,Red_PAS1,P1_PAS2,P2_PAS2,Red_PAS2,P1_PAS3,P2_PAS3,Red_PAS3,Arg_Final,Ano_Trienio
    final_rows = pd.DataFrame()
    final_rows['Inscricao'] = merged['Inscricao']
    final_rows['Nome'] = merged['Nome']
    final_rows['P1_PAS1'] = merged['P1_P1'].fillna(0.0)
    final_rows['P2_PAS1'] = merged['P2_P1'].fillna(0.0)
    final_rows['Red_PAS1'] = merged['Red_P1'].fillna(0.0)
    final_rows['P1_PAS2'] = merged['P1_P2'].fillna(0.0)
    final_rows['P2_PAS2'] = merged['P2_P2'].fillna(0.0)
    final_rows['Red_PAS2'] = merged['Red_P2'].fillna(0.0)
    
    # PAS 3 and Arg_Final unknown yet
    final_rows['P1_PAS3'] = 0.0
    final_rows['P2_PAS3'] = 0.0
    final_rows['Red_PAS3'] = 0.0
    final_rows['Arg_Final'] = 0.0
    final_rows['Ano_Trienio'] = '2023-2025'
    
    # Append to master
    if os.path.exists(master_csv):
        master_df = pd.read_csv(master_csv)
        # Check if already exists to avoid duplicates
        if '2023-2025' in master_df['Ano_Trienio'].values:
            print("Warning: Data for 2023-2025 already in master CSV. Skipping append.")
        else:
            new_master = pd.concat([master_df, final_rows], ignore_index=True)
            new_master.to_csv(master_csv, index=False)
            print(f"Successfully appended {len(final_rows)} rows to {master_csv}")
    else:
        final_rows.to_csv(master_csv, index=False)
        print(f"Created {master_csv} with {len(final_rows)} rows")

if __name__ == "__main__":
    main()
