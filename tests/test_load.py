import pandas as pd
from pathlib import Path
import sys
import os

# Mock streamlit
class MockSt:
    def error(self, msg): print(f"ERROR: {msg}")
    def cache_data(self, func): return func

st = MockSt()

def load_course_stats(semester: int = 1, triennium=None, system: str = "Sistema Universal"):
    try:
        data_dir = Path("data")
        csv_path = data_dir / "notas_corte_pas.csv"
        
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            return None
        
        stats = pd.read_csv(csv_path)
        print(f"Loaded CSV. Columns: {stats.columns.tolist()}")
        
        rename_map = {}
        if 'Curso_Limpo' in stats.columns: rename_map['Curso_Limpo'] = 'Curso'
        if 'N_Banco' in stats.columns: rename_map['N_Banco'] = 'N'
        
        if rename_map:
            stats = stats.rename(columns=rename_map)
            print(f"Renamed. Columns: {stats.columns.tolist()}")

        if 'Semestre' in stats.columns:
            stats = stats[stats['Semestre'].astype(str).str.contains(str(semester))]
            print(f"Filtered by Semester {semester}. Rows: {len(stats)}")

        if 'Sistema_Nome' in stats.columns:
            stats = stats[stats['Sistema_Nome'] == system]
            print(f"Filtered by System {system}. Rows: {len(stats)}")

        if triennium:
            if 'Trienio' in stats.columns:
                stats = stats[stats['Trienio'] == triennium]
            elif 'Triênio' in stats.columns:
                stats = stats[stats['Triênio'] == triennium]
            print(f"Filtered by Triennium {triennium}. Rows: {len(stats)}")
        else:
            for col in ['Trienio', 'Triênio']:
                if col in stats.columns and not stats.empty:
                    recent_triennium = stats[col].max()
                    stats = stats[stats[col] == recent_triennium]
                    print(f"Fallback to recent triennium {recent_triennium} from col {col}. Rows: {len(stats)}")
                    break
        
        stats = stats.sort_values('Min', ascending=False).reset_index(drop=True)
        stats.index = stats.index + 1
        return stats
        
    except Exception as e:
        print(f"Exception: {e}")
        return None

df = load_course_stats(semester=1, triennium="2022-2024")
if df is not None:
    print(f"Final Success. Columns: {df.columns.tolist()}")
    if 'Curso' in df.columns:
        print("Column 'Curso' found!")
    else:
        print("Column 'Curso' NOT found!")
else:
    print("Failed to load.")
