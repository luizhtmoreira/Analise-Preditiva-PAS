
import pandas as pd

try:
    df = pd.read_csv("data/notas_corte_pas.csv")
    
    # Filter for the problematic subset
    # Note: Semestre is "2°" (degree symbol) based on previous check
    subset = df[
        (df['Trienio'] == '2020-2022') & 
        (df['Semestre'] == '2°')
    ]
    
    print(f"Total rows for 2020-2022, 2° Semestre: {len(subset)}")
    
    if not subset.empty:
        print("Unique calls (Chamada) found:")
        print(subset['Chamada'].unique())
        
        # Check specifically for '1ª'
        has_1a = '1ª' in subset['Chamada'].values
        print(f"Has '1ª' call? {has_1a}")
    else:
        print("No data found for this subset.")

except Exception as e:
    print(f"Error: {e}")
