
import pypdf
import pandas as pd
import re
import os
import csv
import logging

# Suppress pdfplumber/pypdf logger noise
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)


def clean_name(name):
    """
    Cleans student name:
    - Removes newlines/carriage returns
    - Removes multiple spaces
    - Trims whitespace
    - UPPERCASE
    """
    if not name:
        return ""
    # Replace newlines with space
    name = name.replace('\n', ' ').replace('\r', ' ')
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    return name.strip().upper()

def clean_score(val):
    """
    Cleans score fields:
    - Handles 'N', '*', '-', empty strings as 0.0
    - Replaces comma with dot
    - Converts to float
    """
    if val is None:
        return 0.0
    
    # Textual representations of zero/missing
    if val.strip() in ['N', '*', '-', '', 'ELIMINADO', 'FALTOU']:
        return 0.0
        
    val = val.strip().replace(',', '.')
    
    # Remove any non-numeric chars except dot and minus
    val = re.sub(r'[^\d.-]', '', val)
    
    try:
        return float(val)
    except ValueError:
        return 0.0



def extract_pdf_data(pdf_path, stage_name):
    """
    Extracts data using pypdf for speed, with robust regex cleaning.
    """
    print(f"--- Extracting {stage_name} from {pdf_path} ---")
    
    students = []
    
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
        
    # Pattern specific to the PDF layout provided
    # Standard CESPE/Cebraspe layout usually separates records by '/' if it's a list
    
    records = full_text.split('/')
    
    for record in records:
        # Cleanup record
        record = record.strip()
        if not record: continue
        
        # Verify if it starts with enrollment number (ignoring potential scroll noise)
        # We look for the first 8-digit sequence
        # CHANGED: Allow comma after digits, not just space
        match = re.search(r'(\d{8})[\s,]+(.+)', record, re.DOTALL)
        if not match:
            continue
            
        inscricao = match.group(1)
        remainder = match.group(2)
        
        # The remainder contains Name and Scores, separated by commas
        fields = remainder.split(',')
        
        # Name is fields[0] (or part of it if comma separation is messy)
        # Generally fields[0] IS the name.
        
        name = clean_name(fields[0])
        
        if len(fields) < 3: # Need at least Name, P1, P2/Red
            continue
            
        try:
            p1 = clean_score(fields[1])
            p2 = clean_score(fields[2])
            red = clean_score(fields[-1]) # Last field is practically always Redacao
            
            students.append({
                'Inscricao': int(inscricao),
                'Nome': name,
                'P1': p1,
                'P2': p2,
                'Red': red
            })
        except Exception as e:
            # print(f"Skipping record {inscricao}: {e}")
            continue

    df = pd.DataFrame(students)
    # Remove duplicates if any (e.g. from page overlaps)
    df = df.drop_duplicates(subset=['Inscricao'])
    print(f"Found {len(df)} unique students in {stage_name}")
    return df

def main():
    base_path = "data"
    pas1_file = os.path.join(base_path, "Ed_7_PAS_1_2023_2025_Res_final_tipo_D_redação.pdf")
    pas2_file = os.path.join(base_path, "Ed_15_PAS_2_2023-2025_Res_final_tipo_D_redacao.pdf")
    master_csv = os.path.join(base_path, "banco_alunos_pas_final.csv")
    
    if not os.path.exists(master_csv):
        print("CRITICAL ERROR: Master CSV not found. Please restore backup first.")
        return

    # 1. VERIFY MASTER CSV STATE
    print("Verifying Master CSV...")
    df_master = pd.read_csv(master_csv)
    
    if '2023-2025' in df_master['Ano_Trienio'].astype(str).unique():
        print("ERROR: 2023-2025 data already exists in Master CSV.")
        print("Please restore the clean backup without this triennium.")
        return
        
    initial_count = len(df_master)
    print(f"Initial Row Count: {initial_count}")
    
    # 2. EXTRACT DATA
    df1 = extract_pdf_data(pas1_file, "PAS 1")
    df2 = extract_pdf_data(pas2_file, "PAS 2")
    
    # 3. MERGE
    print("Merging PAS 1 and PAS 2 Data...")
    merged = pd.merge(df1, df2, on='Inscricao', how='outer', suffixes=('_P1', '_P2'))
    
    # 4. PREPARE FINAL DATAFRAME
    # Resolving Name (prefer P2 name as it's more recent, fallback to P1)
    merged['Nome'] = merged['Nome_P2'].fillna(merged['Nome_P1'])
    
    # Create structure matching Master CSV
    # Columns: Inscricao,Nome,P1_PAS1,P2_PAS1,Red_PAS1,P1_PAS2,P2_PAS2,Red_PAS2,P1_PAS3,P2_PAS3,Red_PAS3,Arg_Final,Ano_Trienio
    
    final_df = pd.DataFrame()
    final_df['Inscricao'] = merged['Inscricao']
    final_df['Nome'] = merged['Nome']
    
    # Fill Scores (0.0 default)
    final_df['P1_PAS1'] = merged['P1_P1'].fillna(0.0)
    final_df['P2_PAS1'] = merged['P2_P1'].fillna(0.0)
    final_df['Red_PAS1'] = merged['Red_P1'].fillna(0.0)
    
    final_df['P1_PAS2'] = merged['P1_P2'].fillna(0.0)
    final_df['P2_PAS2'] = merged['P2_P2'].fillna(0.0)
    final_df['Red_PAS2'] = merged['Red_P2'].fillna(0.0)
    
    # Future/Unknown fields set to 0.0
    final_df['P1_PAS3'] = 0.0
    final_df['P2_PAS3'] = 0.0
    final_df['Red_PAS3'] = 0.0
    final_df['Arg_Final'] = 0.0
    
    final_df['Ano_Trienio'] = '2023-2025'
    
    # 5. VALIDATION BEFORE APPEND
    print("Validating New Data...")
    
    # Check for empty names
    empty_names = final_df[final_df['Nome'] == '']
    if not empty_names.empty:
        print(f"Warning: {len(empty_names)} rows have empty names. Filling with 'DESCONHECIDO'.")
        final_df.loc[final_df['Nome'] == '', 'Nome'] = 'DESCONHECIDO'
        
    # Check for NaN in scores
    score_cols = [c for c in final_df.columns if 'PAS' in c or 'Arg' in c]
    for col in score_cols:
        if final_df[col].isna().any():
            print(f"Fixing NaNs in {col}...")
            final_df[col] = final_df[col].fillna(0.0)
            
    # Check name formatting (newlines)
    bad_names = final_df[final_df['Nome'].str.contains(r'[\n\r]')]
    if not bad_names.empty:
        print(f"CRITICAL: Found {len(bad_names)} names with newlines! Cleaning...")
        final_df['Nome'] = final_df['Nome'].apply(clean_name)
        
    print(f"New Rows to Append: {len(final_df)}")
    
    # 6. FILTER DROPOUTS (2023-2025 ONLY)
    # As requested: remove those who have PAS 1 but all zeros in PAS 2
    print("Filtering out 2023-2025 dropouts (all zeros in PAS 2)...")
    dropout_mask = (final_df['P1_PAS2'] == 0.0) & (final_df['P2_PAS2'] == 0.0) & (final_df['Red_PAS2'] == 0.0)
    dropouts_count = dropout_mask.sum()
    final_df = final_df[~dropout_mask]
    print(f"Eliminados: {dropouts_count} alunos do triênio 2023-2025.")
    
    # 7. APPEND AND SAVE SAFELY
    print("Appending to Master CSV...")
    
    # Concatenate
    full_df = pd.concat([df_master, final_df], ignore_index=True)
    
    # Save
    full_df.to_csv(master_csv, index=False, quoting=csv.QUOTE_NONNUMERIC, float_format='%.3f')
    
    print("SUCCESS: Data ingestion and cleaning complete.")
    print(f"Final Row Count: {len(full_df)}")
    print(f"Total Added (Excluding Dropouts): {len(final_df)}")

if __name__ == "__main__":
    main()
