import sys
import os

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from streamlit_app import find_best_match

# Simulate the keys generated in streamlit_app.py
choices = [
    "DIREITO (BACHARELADO) - DIURNO (DARCY RIBEIRO)",
    "DIREITO (BACHARELADO) - NOTURNO (DARCY RIBEIRO)",
    "ENFERMAGEM (BACHARELADO) - DIURNO (DARCY RIBEIRO)",
    "ENFERMAGEM (BACHARELADO) - DIURNO (CEILÂNDIA)",
    "COMUNICAÇÃO SOCIAL - AUDIOVISUAL (BACHARELADO) - DIURNO (DARCY RIBEIRO)",
    "SERVIÇO SOCIAL (BACHARELADO) - DIURNO (DARCY RIBEIRO)",
    "SERVIÇO SOCIAL (BACHARELADO) - NOTURNO (DARCY RIBEIRO)"
]

tests = [
    ("direito noturno", "DIREITO (BACHARELADO) - NOTURNO (DARCY RIBEIRO)"),
    ("direito diurno", "DIREITO (BACHARELADO) - DIURNO (DARCY RIBEIRO)"),
    ("direito", "DIREITO (BACHARELADO) - DIURNO (DARCY RIBEIRO)"), # Should fallback to shortest or first?
    # Actually, for "direito", intersection with both is high.
    # "direito" (1 token). 
    # Option 1: {direito, bacharelado, diurno, darcy, ribeiro} -> 1/1 = 1.0
    # Option 2: {direito, bacharelado, noturno, darcy, ribeiro} -> 1/1 = 1.0
    # Returns first found or shortest? Logic says: if multiple candidates in substring, returns min(len).
    # But this is Token Intersection. 
    # Current logic: returns the FIRST one that exceeds score, OR keeps looking for BEST score?
    # Let's check find_best_match logic. It keeps best_token_score. 
    # If equal, it keeps the first one found.
    
    ("enfermagem ceilandia", "ENFERMAGEM (BACHARELADO) - DIURNO (CEILÂNDIA)"),
    ("enfermagem darcy", "ENFERMAGEM (BACHARELADO) - DIURNO (DARCY RIBEIRO)"),
    ("audiovisual", "COMUNICAÇÃO SOCIAL - AUDIOVISUAL (BACHARELADO) - DIURNO (DARCY RIBEIRO)")
]

print("--- Verifying Campus/Turno Fuzzy Matching ---")
all_passed = True
for query, expected in tests:
    result = find_best_match(query, choices)
    
    # Special handling for "direito" which is ambiguous
    if query == "direito":
        # Accept either diurno or noturno as valid match for generic query, 
        # BUT ideally it should be one of them.
        pass

    status = "✅ PASS" if result == expected else f"❌ FAIL (Got: {result})"
    print(f"Query: '{query}' -> {status}")
    
    if result != expected:
        # Allow "Direito" to likely match Diurno (first in list) or be ambiguous
        if query == "direito" and "DIREITO" in str(result):
             print(f"  (Accepting '{result}' for generic 'direito')")
        else:
             all_passed = False

if all_passed:
    print("\nAll tests passed!")
else:
    print("\nSome tests failed.")
