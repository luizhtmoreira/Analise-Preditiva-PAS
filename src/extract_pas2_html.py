import pandas as pd
from bs4 import BeautifulSoup
import os
import shutil

# Paths
HTML_PATH = 'data/Ed_12_PAS_2_2024-2026_Res_final_tipoD_red.html'
CSV_PATH = 'data/PAS_MESTRE_LIMPO_FINAL.csv'
BACKUP_PATH = 'data/PAS_MESTRE_LIMPO_FINAL_BACKUP.csv'

def main():
    print("--- Starting Data Extraction & Integration ---")

    # 1. Create Backup
    if os.path.exists(CSV_PATH):
        print(f"Creating backup of {CSV_PATH}...")
        shutil.copyfile(CSV_PATH, BACKUP_PATH)
        print(f"Backup created at {BACKUP_PATH}")
    else:
        print(f"Error: {CSV_PATH} not found.")
        return

    # 2. Parse HTML
    print(f"Reading HTML file: {HTML_PATH}")
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # The data is inside <p> tags. We look for tags containing " / ".
    extracted_data = []
    
    paragraphs = soup.find_all('p')
    print(f"Found {len(paragraphs)} paragraph tags. Scanning for data...")

    for p in paragraphs:
        text = p.get_text()
        if '/' in text and ',' in text:
            # Candidate for data line
            # Split by ' / ' to get individual records
            # Note: The file might end with a slash or have newlines, so strip.
            records = [r.strip() for r in text.split('/') if r.strip()]
            
            for record in records:
                # Each record: Inscricao, Nome, P1, P2, Soma, NotaD, Redacao
                # Example: 24141937, Aarao Matos de Araujo, 0.271, 10.569, 10.840, 0.000, 6.679
                parts = [part.strip() for part in record.split(',')]
                
                if len(parts) >= 7:
                    # Extract fields
                    inscricao = parts[0]
                    nome = parts[1]
                    p1_pas2 = parts[2]
                    p2_pas2 = parts[3]
                    red_pas2 = parts[6]
                    
                    extracted_data.append({
                        'Inscricao': inscricao,
                        'Nome': nome,
                        'P1_PAS2': float(p1_pas2),
                        'P2_PAS2': float(p2_pas2),
                        'Red_PAS2': float(red_pas2)
                    })

    total_extracted = len(extracted_data)
    print(f"Total students extracted: {total_extracted}")

    if total_extracted == 0:
        print("No data extracted. Aborting.")
        return

    # 3. Report
    first_student = extracted_data[0]['Nome']
    last_student = extracted_data[-1]['Nome']
    print("\n--- Extraction Report ---")
    print(f"First Student: {first_student}")
    print(f"Last Student:  {last_student}")
    print(f"Total Count:   {total_extracted}")
    print("-------------------------\n")

    # 4. Prepare DataFrame
    new_df = pd.DataFrame(extracted_data)
    
    # Add constant/default columns
    # Existing columns in CSV: 
    # Inscricao,Nome,P1_PAS1,P2_PAS1,Red_PAS1,P1_PAS2,P2_PAS2,Red_PAS2,P1_PAS3,P2_PAS3,Red_PAS3,Arg_Final,Ano_Trienio
    
    new_df['P1_PAS1'] = 0.00
    new_df['P2_PAS1'] = 0.00
    new_df['Red_PAS1'] = 0.00
    new_df['P1_PAS3'] = 0.00
    new_df['P2_PAS3'] = 0.00
    new_df['Red_PAS3'] = 0.00
    new_df['Arg_Final'] = 0.00
    new_df['Ano_Trienio'] = '2024-2026'
    
    # Reorder columns to match target CSV
    target_cols = [
        'Inscricao', 'Nome', 
        'P1_PAS1', 'P2_PAS1', 'Red_PAS1', 
        'P1_PAS2', 'P2_PAS2', 'Red_PAS2', 
        'P1_PAS3', 'P2_PAS3', 'Red_PAS3', 
        'Arg_Final', 'Ano_Trienio'
    ]
    
    new_df = new_df[target_cols]

    # 5. Append and Save
    print("Loading existing CSV...")
    # Load with low_memory=False to avoid dtypes warnings, or specify dtype if needed.
    # We load everything as is.
    master_df = pd.read_csv(CSV_PATH)
    
    print(f"Existing records: {len(master_df)}")
    
    print("Appending new data...")
    # Concatenate
    updated_df = pd.concat([master_df, new_df], ignore_index=True)
    
    print(f"New total records: {len(updated_df)}")
    
    print("Saving to CSV...")
    updated_df.to_csv(CSV_PATH, index=False)
    print("Success! Data integration complete.")

if __name__ == "__main__":
    main()
