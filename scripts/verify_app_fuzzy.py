import sys
import os

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from streamlit_app import find_best_match

choices = [
    "DIREITO (BACHARELADO)",
    "ENFERMAGEM (BACHARELADO)",
    "ENGENHARIA CIVIL (BACHARELADO)",
    "COMUNICAÇÃO SOCIAL - AUDIOVISUAL (BACHARELADO)",
    "SERVIÇO SOCIAL (BACHARELADO)",
    "COMPUTAÇÃO (LICENCIATURA)"
]

tests = [
    ("direito noturno", "DIREITO (BACHARELADO)"),
    ("enfermagem ceilandia", "ENFERMAGEM (BACHARELADO)"),
    ("serviço social noturno", "SERVIÇO SOCIAL (BACHARELADO)"),
    ("audiovisual", "COMUNICAÇÃO SOCIAL - AUDIOVISUAL (BACHARELADO)")
]

print("--- Verifying app.streamlit_app.find_best_match ---")
all_passed = True
for query, expected in tests:
    result = find_best_match(query, choices)
    status = "✅ PASS" if result == expected else f"❌ FAIL (Got: {result})"
    print(f"Query: '{query}' -> {status}")
    if result != expected:
        all_passed = False

if all_passed:
    print("\nAll tests passed!")
    sys.exit(0)
else:
    print("\nSome tests failed.")
    sys.exit(1)
