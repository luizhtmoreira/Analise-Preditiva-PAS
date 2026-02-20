
import pandas as pd

try:
    df = pd.read_csv("data/notas_corte_pas.csv")
    
    # Get all unique trienniums
    trienniums = sorted(df['Trienio'].unique())
    
    print(f"Checking {len(trienniums)} trienniums for 2nd Semester (2°) data:")
    print("-" * 60)
    print(f"{'Triennium':<15} | {'Rows (2° Sem)':<15} | {'Has 1ª Call?':<15} | {'Available Calls'}")
    print("-" * 60)
    
    for tri in trienniums:
        # Filter for this triennium and 2nd semester
        subset = df[
            (df['Trienio'] == tri) & 
            (df['Semestre'] == '2°')
        ]
        
        row_count = len(subset)
        
        if row_count > 0:
            calls = sorted(subset['Chamada'].unique())
            has_1a = '1ª' in calls
            calls_str = ", ".join(calls)
            print(f"{tri:<15} | {row_count:<15} | {str(has_1a):<15} | {calls_str}")
        else:
            print(f"{tri:<15} | {'0':<15} | {'N/A':<15} | -")

    print("-" * 60)

except Exception as e:
    print(f"Error: {e}")
