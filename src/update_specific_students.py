import pandas as pd

# Path
CSV_PATH = 'data/banco_alunos_pas_final.csv'

def main():
    print("--- Manual Student Update ---")
    df = pd.read_csv(CSV_PATH, dtype={'Inscricao': str})
    
    updates = [
        {'Inscricao': '24115085', 'Nome': 'Sofia Albino Barreto de Freitas Azevedo', 'P1_PAS1': 1.995, 'P2_PAS1': 28.926, 'Red_PAS1': 6.948},
        {'Inscricao': '24141937', 'Nome': 'Aarao Matos de Araujo', 'P1_PAS1': 0.000, 'P2_PAS1': 14.535, 'Red_PAS1': 6.759}
    ]
    
    for update in updates:
        mask = df['Inscricao'] == update['Inscricao']
        if mask.any():
            print(f"Updating Inscricao {update['Inscricao']}...")
            df.loc[mask, 'P1_PAS1'] = update['P1_PAS1']
            df.loc[mask, 'P2_PAS1'] = update['P2_PAS1']
            df.loc[mask, 'Red_PAS1'] = update['Red_PAS1']
        else:
            print(f"Inscricao {update['Inscricao']} not found. Adding as new record...")
            new_row = {
                'Inscricao': update['Inscricao'],
                'Nome': update['Nome'],
                'P1_PAS1': update['P1_PAS1'],
                'P2_PAS1': update['P2_PAS1'],
                'Red_PAS1': update['Red_PAS1'],
                'P1_PAS2': 0.0, 'P2_PAS2': 0.0, 'Red_PAS2': 0.0,
                'P1_PAS3': 0.0, 'P2_PAS3': 0.0, 'Red_PAS3': 0.0,
                'Arg_Final': 0.0, 'Ano_Trienio': '2024-2026'
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
    print("Saving updated CSV...")
    df.to_csv(CSV_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
