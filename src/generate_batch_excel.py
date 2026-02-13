
import pandas as pd
from pathlib import Path

# Data provided by the user
data_str = "23118417-administração,23112080-filosofia,23126994-audiovisual,23124597-serviço social,24213139-gestão de políticas públicas,23105528-música,23108707-agronomia,23116268-fisioterapia,23111319-fisioterapia,23101663-farmácia,23118429-saúde coletiva,23116297-computação(licenciatura),23110579-farmácia,23118867-engenharias,23111064-educação física,23124441-enfermagem,23112256-ciências contábeis,23108977-educação física,23116724-química,23114899-serviço social,23123545-turismo"

# Parse the pairs
mapping = {}
for pair in data_str.split(','):
    parts = pair.split('-')
    if len(parts) >= 2:
        insc = parts[0].strip()
        curso = "-".join(parts[1:]).strip()
        mapping[insc] = curso

# Load master dataset
csv_path = Path("data/banco_alunos_pas_final.csv")
df_master = pd.read_csv(csv_path)

# Ensure Inscricao is string for matching
df_master['Inscricao'] = df_master['Inscricao'].astype(str)

# Filter students
found_students = df_master[df_master['Inscricao'].isin(mapping.keys())].copy()

# Add the requested course
found_students['Curso'] = found_students['Inscricao'].map(mapping)

# Selected columns for Batch PDF generator (as defined in streamlit_app.py batch logic)
# Standard columns: Nome, Curso, P1_PAS1, P2_PAS1, Red_PAS1, P1_PAS2, P2_PAS2, Red_PAS2
final_columns = ['Nome', 'Curso', 'P1_PAS1', 'P2_PAS1', 'Red_PAS1', 'P1_PAS2', 'P2_PAS2', 'Red_PAS2']
df_output = found_students[final_columns]

# Save to Excel
output_path = Path("data/lista_alunos_lote.xlsx")
df_output.to_excel(output_path, index=False)

print(f"Generated Excel with {len(df_output)} students at {output_path}")
# Print missing registrations
missing = set(mapping.keys()) - set(found_students['Inscricao'])
if missing:
    print(f"Warning: Registration numbers not found: {missing}")
