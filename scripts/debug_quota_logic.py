
import pandas as pd
import unicodedata
from pathlib import Path

# --- MOCK find_best_match (copied from app) ---
def find_best_match(input_name: str, options_list: list[str]) -> str:
    if not isinstance(input_name, str) or not options_list:
        return str(input_name)
        
    normalized_input = unicodedata.normalize('NFKD', input_name).encode('ASCII', 'ignore').decode('utf-8').upper()
    
    # 1. Exact match
    for option in options_list:
        if normalized_input == unicodedata.normalize('NFKD', option).encode('ASCII', 'ignore').decode('utf-8').upper():
            return option
            
    # 2. Bidirectional substring
    matches = []
    for option in options_list:
        normalized_option = unicodedata.normalize('NFKD', option).encode('ASCII', 'ignore').decode('utf-8').upper()
        if normalized_input in normalized_option or normalized_option in normalized_input:
            matches.append(option)
    
    if matches:
        return max(matches, key=len)
        
    # 3. Token-based
    input_tokens = set(normalized_input.split())
    best_match = None
    max_score = 0.0
    
    for course in options_list:
        normalized_course = unicodedata.normalize('NFKD', course).encode('ASCII', 'ignore').decode('utf-8').upper()
        clean_course = normalized_course.replace('(', '').replace(')', '').replace('-', '')
        course_tokens = set(clean_course.split())
        
        common = input_tokens.intersection(course_tokens)
        if not input_tokens: continue
        
        # Token score logic
        # Using simplified version matching the app
        score = len(common) / len(input_tokens)
        
        if score > 0.6: # Relaxed cutoff for testing
             if score > max_score:
                 max_score = score
                 best_match = course
             elif score == max_score:
                 # Tie-break
                 pass # Simplified

    if best_match:
        return best_match

    return input_name

# --- REPRODUCE BATCH LOGIC ---

def main():
    print("Loading data...")
    try:
        df = pd.read_csv("data/notas_corte_pas.csv")
    except FileNotFoundError:
        print("Data file not found!")
        return

    # Simulate load_course_stats(system=None)
    # We'll filter for a specific triennium/semester to mimic the app
    # Checking CSV header to find valid triennium
    print("Columns:", df.columns)
    print("Available Trienniums:", df['Trienio'].unique())
    
    triennium = "2023-2025" # Example
    semester = "1°" # Assuming header uses 1°/2° or number? 
    # Checking first row content
    print("Sample row:", df.iloc[0].to_dict())
    
    # Filter
    df_filtered = df[df['Trienio'] == triennium].copy()
    # Assume semester filtering works (skip for now to ensure data exists)

    print(f"Filtered rows: {len(df_filtered)}")
    
    # Logic from app
    for col in ['Curso', 'Campus', 'Turno']:
        if col in df_filtered.columns:
            # Note: app uses 'Curso' but CSV header might be 'Curso_Limpo' or 'Curso'
            # CSV header check: 'Curso_Limpo' exists. 'Curso' might not?
            # App load_cutoff_data_global creates 'Curso_Limpo' = 'Curso' if missing.
            # But here we read raw CSV.
            # Let's check if 'Curso' exists in raw CSV.
            pass

    # Basic cleaning
    if 'Curso' not in df_filtered.columns and 'Curso_Limpo' in df_filtered.columns:
        df_filtered['Curso'] = df_filtered['Curso_Limpo']
            
    for col in ['Curso', 'Campus', 'Turno']:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].astype(str).str.strip()

    available_systems = []
    if 'Sistema_Nome' in df_filtered.columns:
        available_systems = df_filtered['Sistema_Nome'].unique().tolist()
        print("Available Systems:", available_systems)

    course_map = {}
    if all(col in df_filtered.columns for col in ['Curso', 'Campus', 'Turno']):
        df_filtered['Combo_Nome'] = df_filtered['Curso'] + " (" + df_filtered['Campus'] + " - " + df_filtered['Turno'] + ")"
        
        # Unique mapping
        df_ref_unique = df_filtered.sort_values('Min', ascending=True).drop_duplicates(['Combo_Nome', 'Sistema_Nome'], keep='first')
        
        course_map = dict(zip(zip(df_ref_unique['Combo_Nome'], df_ref_unique['Sistema_Nome']), df_ref_unique['Min']))
        
        print(f"Course Map size: {len(course_map)}")
        # Print some keys
        print("Sample Keys:", list(course_map.keys())[:5])
    else:
        print("Missing columns for Combo_Nome")
        return

    available_courses = sorted(list(set(k[0] for k in course_map.keys())))
    print(f"Available Courses: {len(available_courses)}")

    # --- TEST CASES ---
    test_cases = [
        {"raw_course": "administração", "raw_quota": "Sistema Universal"},
        {"raw_course": "administração", "raw_quota": "Cota para Negros"},
        {"raw_course": "direito", "raw_quota": "Universal"},
        {"raw_course": "direito", "raw_quota": "Negros"}, # Should fuzzy match Cota para Negros
        {"raw_course": "medicina", "raw_quota": "Escola Pública - Renda Baixa"}, # Complex match
    ]

    print("\n--- RUNNING TESTS ---")
    for tc in test_cases:
        raw_c = tc["raw_course"]
        raw_q = tc["raw_quota"]
        
        off_c = find_best_match(raw_c, available_courses)
        off_q = find_best_match(raw_q, available_systems)
        
        key = (off_c, off_q)
        score = course_map.get(key, "NOT FOUND")
        
        print(f"Input: Course='{raw_c}', Quota='{raw_q}'")
        print(f"Match: Course='{off_c}', Quota='{off_q}'")
        print(f"Score: {score}")
        
        if score == "NOT FOUND":
            # Check if key exists with Universal
            fb_score = course_map.get((off_c, "Sistema Universal"), "NOT FOUND")
            print(f"Fallback (Universal): {fb_score}")
        print("-" * 30)

if __name__ == "__main__":
    main()
