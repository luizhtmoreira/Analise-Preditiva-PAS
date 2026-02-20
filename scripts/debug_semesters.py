
import pandas as pd
import sys

try:
    df = pd.read_csv("data/notas_corte_pas.csv")
    print("Unique Semesters found in CSV:")
    print(df['Semestre'].unique())
    
    print("\nUnique Trienniums found in CSV:")
    print(df['Trienio'].unique())
    
    # Check specifically for 2nd semester variants
    print("\nChecking for '2°' match:")
    mask = df['Semestre'] == "2°"
    print(f"Rows matching '2°': {mask.sum()}")
    
    print("\nChecking for '2º' match:")
    mask = df['Semestre'] == "2º"
    print(f"Rows matching '2º': {mask.sum()}")

except Exception as e:
    print(f"Error: {e}")
