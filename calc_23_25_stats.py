import pandas as pd

def calc_stats():
    df = pd.read_csv('data/PAS_MESTRE_LIMPO_FINAL.csv')
    df_sub = df[df['Ano_Trienio'] == '2023-2025']
    
    print(f"Subprograma 2023-2025 (N={len(df_sub)})")
    
    parts = ['PAS1', 'PAS2']
    for p in parts:
        p1_col = f'P1_{p}'
        p2_col = f'P2_{p}'
        red_col = f'Red_{p}'
        
        # We need to filter out 0.0 because they represent students who didn't take that specific exam
        # (Since it was an outer join)
        df_p = df_sub[df_sub[p2_col] > 0]
        
        m1 = df_p[p1_col].mean()
        s1 = df_p[p1_col].std()
        m2 = df_p[p2_col].mean()
        s2 = df_p[p2_col].std()
        mr = df_p[red_col].mean()
        sr = df_p[red_col].std()
        
        print(f"\n{p}:")
        print(f"  mean_p1={m1:.4f}, std_p1={s1:.4f},")
        print(f"  mean_p2={m2:.4f}, std_p2={s2:.4f},")
        print(f"  mean_red={mr:.4f}, std_red={sr:.4f}")

if __name__ == "__main__":
    calc_stats()
