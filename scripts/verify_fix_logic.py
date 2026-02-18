
import pandas as pd

def flatten_course_name(row):
    return f"{row['Curso_Limpo']} ({row['Campus']} - {row['Turno']})"

try:
    print("Loading data...")
    df_notas = pd.read_csv("data/notas_corte_pas.csv")
    
    # Simulation Parameters (Problematic Case)
    ref_triennium = "2020-2022"
    semester_db = "2°"
    semester_int = 2
    cota_selecionada = "Sistema Universal"
    
    print(f"\nTesting logic for: {ref_triennium} | {semester_db} | {cota_selecionada}")
    
    # 1. New Logic: Filter by Cota ONLY (No '1ª Chamada' filter)
    df_cota_atual = df_notas[
        (df_notas['Trienio'] == ref_triennium) & 
        (df_notas['Semestre'] == semester_db) &
        (df_notas['Sistema_Nome'] == cota_selecionada)
    ].sort_values(['Curso_Limpo', 'Campus', 'Turno'])
    
    print(f"Rows found after initial filter: {len(df_cota_atual)}")
    
    if len(df_cota_atual) == 0:
        print("CRITICAL FAIL: No data found even without '1ª Chamada' filter!")
        exit(1)
        
    # Generate Combo Names
    df_cota_atual['Combo_Nome'] = df_cota_atual.apply(flatten_course_name, axis=1)
    opcoes_lista = df_cota_atual['Combo_Nome'].unique().tolist()
    
    print(f"Unique Courses found: {len(opcoes_lista)}")
    
    # Test specific extraction for a few courses
    test_courses = opcoes_lista[:3]
    
    print("\n--- Testing Logic on Sample Courses ---")
    for combo in test_courses:
        df_base = df_cota_atual[df_cota_atual['Combo_Nome'] == combo]
        
        # Logic from App
        if semester_int == 1:
             df_chamadas = df_base.sort_values('Min', ascending=True)
        else:
             # 2nd Semester: Ascending by Chamada (string) to get "1ª", "2ª"...
             df_chamadas = df_base.sort_values('Chamada', ascending=True)
        
        if not df_chamadas.empty:
            best_call = df_chamadas.iloc[0]
            print(f"[SUCCESS] {combo}")
            print(f"   -> Selected Call: {best_call['Chamada']}")
            print(f"   -> Score: {best_call['Min']}")
            print(f"   -> Available Calls: {sorted(df_base['Chamada'].unique())}")
        else:
            print(f"[FAIL] {combo} - No calls found (Should be impossible here)")

    print("\nVerification Complete.")

except Exception as e:
    print(f"Result: ERROR - {e}")
