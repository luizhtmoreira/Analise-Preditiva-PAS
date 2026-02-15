import pandas as pd
import os

students_to_find = [
    'Ana Clara Matos Ricardo',
    'Ana Luiza Silva Teixeirense Lima',
    'Mariana Souza Monteiro Oliveira',
    'Guilherme Pimenta Rodrigues',
    'Camila Alves Geraldo da Silva'
]

output_file = 'extracted_students.csv'
database_file = r'c:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\banco_alunos_pas_final.csv'

if os.path.exists(database_file):
    df = pd.read_csv(database_file)
    # Search for names (allowing for minor variations like case/accents if possible, but exact for now)
    found_df = df[df['Nome'].isin(students_to_find)]
    
    # If not all found, try substring search
    if len(found_df) < len(students_to_find):
        print(f"Exact match found {len(found_df)}/{len(students_to_find)}. Trying substring search...")
        mask = df['Nome'].apply(lambda x: any(s.lower() in str(x).lower() for s in students_to_find))
        found_df = df[mask]
    
    found_df.to_csv(output_file, index=False)
    print(f"Found {len(found_df)} students. Saved to {output_file}")
else:
    print(f"Database file not found at {database_file}")
