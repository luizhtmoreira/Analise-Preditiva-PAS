
import pandas as pd

def flatten_course_name(row):
    return f"{row['Curso_Limpo']} ({row['Campus']} - {row['Turno']})"

try:
    df = pd.read_csv("data/notas_corte_pas.csv")
    
    # Simulation for 2020-2022, 2nd Sem, Universal
    ref_triennium = "2020-2022"
    semester_db = "2°"
    semester_int = 2
    cota_selecionada = "Sistema Universal"
    
    subset = df[
        (df['Trienio'] == ref_triennium) & 
        (df['Semestre'] == semester_db) &
        (df['Sistema_Nome'] == cota_selecionada)
    ].copy()
    
    subset['Combo_Nome'] = subset.apply(flatten_course_name, axis=1)
    
    print(f"Total rows for DIREITO before dedup:")
    direito_raw = subset[subset['Curso_Limpo'].str.contains("DIREITO", na=False)]
    print(direito_raw[['Chamada', 'Min', 'Combo_Nome']])
    
    # Extraction numeral logic
    subset['Chamada_Num'] = subset['Chamada'].str.extract('(\d+)').fillna(0).astype(int)
    
    # Dedup logic
    if semester_int == 1:
        clean = subset.sort_values(['Combo_Nome', 'Min'], ascending=[True, True]).drop_duplicates('Combo_Nome', keep='first')
    else:
        clean = subset.sort_values(['Combo_Nome', 'Chamada_Num'], ascending=[True, True]).drop_duplicates('Combo_Nome', keep='first')
        
    print(f"\nTotal rows for DIREITO after dedup:")
    print(clean[clean['Curso_Limpo'].str.contains("DIREITO", na=False)][['Chamada', 'Min', 'Combo_Nome']])
    
    if len(clean[clean['Curso_Limpo'].str.contains("DIREITO", na=False)]) > 1:
        print("\nDUPLICATES STILL EXIST! Investigating Combo_Nome values:")
        for idx, row in clean[clean['Curso_Limpo'].str.contains("DIREITO", na=False)].iterrows():
            print(f"'{row['Combo_Nome']}'")

except Exception as e:
    print(f"Error: {e}")
