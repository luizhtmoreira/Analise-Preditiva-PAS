import pandas as pd
import unicodedata

def normalize(text):
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8').lower()

try:
    df = pd.read_csv('data/notas_corte_pas.csv')
    courses = sorted(df['Curso_Limpo'].dropna().unique())
    
    print("--- Searching for 'DIREITO' ---")
    for c in courses:
        if 'direito' in normalize(c):
            print(c)
            
    print("\n--- Searching for 'SERVIÇO SOCIAL' ---")
    for c in courses:
        if 'social' in normalize(c):
            print(c)
            
    print("\n--- Searching for 'ENFERMAGEM' ---")
    for c in courses:
        if 'enfermagem' in normalize(c):
            print(c)

except Exception as e:
    print(e)
