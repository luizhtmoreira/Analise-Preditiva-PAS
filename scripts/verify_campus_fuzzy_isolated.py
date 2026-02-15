import difflib
import unicodedata

def find_best_match(query: str, choices: list[str], cutoff: float = 0.6) -> str:
    """
    Encontra a melhor correspondência para uma string em uma lista de opções.
    1. Tenta difflib (para typos).
    2. Tenta substring (para palavras-chave).
    3. Tenta interseção de tokens (para "direito noturno" -> "direito").
    """
    if not query or not choices:
        return query
        
    # 1. Tentativa DIFUSA (Typos, pequenas variações)
    matches = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff)
    if matches:
        print(f"  [DEBUG] Difflib matched: '{matches[0]}' with cutoff {cutoff}")
        return matches[0]
        
    print(f"  [DEBUG] Difflib failed for '{query}'")

    # Normalização para métodos 2 e 3
    try:
        query_norm = unicodedata.normalize('NFKD', str(query)).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
        print(f"  [DEBUG] Query Norm: '{query_norm}' Tokens: {set(query_norm.split())}")
        
        # 2. Tentativa SUBSTRING (Palavra-chave)
        # Ex: "audiovisual" in "COMUNICAÇÃO SOCIAL - AUDIOVISUAL"
        candidates = []
        for choice in choices:
            choice_norm = unicodedata.normalize('NFKD', str(choice)).encode('ASCII', 'ignore').decode('utf-8').lower()
            if query_norm in choice_norm:
                candidates.append(choice)
        
        if candidates:
            res = min(candidates, key=len)
            print(f"  [DEBUG] Substring matched: '{res}'")
            return res

        # 3. Tentativa INTERSEÇÃO DE TOKENS (Keywords soltas)
        # Remove pontuação para evitar (ceilandia) != ceilandia
        query_clean = query_norm.replace('(', ' ').replace(')', ' ').replace('-', ' ')
        query_tokens = set(query_clean.split())
        
        best_token_match = None
        best_token_score = 0.0
        
        for choice in choices:
            choice_norm = unicodedata.normalize('NFKD', str(choice)).encode('ASCII', 'ignore').decode('utf-8').lower()
            choice_clean = choice_norm.replace('(', ' ').replace(')', ' ').replace('-', ' ')
            choice_tokens = set(choice_clean.split())
            
            common = query_tokens.intersection(choice_tokens)
            if not common: continue
            
            score = len(common) / len(query_tokens)
            print(f"  [DEBUG] Choice: '{choice}' Score: {score:.2f} Common: {common}")
            
            if score > best_token_score:
                best_token_score = score
                best_token_match = choice
        
        if best_token_match and best_token_score >= 0.5:
            print(f"  [DEBUG] Token matched: '{best_token_match}' Score: {best_token_score}")
            return best_token_match
            
    except Exception as e:
        print(f"  [DEBUG] Exception: {e}")
        pass
    
    return None

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
    ("enfermagem ceilandia", "ENFERMAGEM (BACHARELADO) - DIURNO (CEILÂNDIA)"),
    ("enfermagem darcy", "ENFERMAGEM (BACHARELADO) - DIURNO (DARCY RIBEIRO)"),
    ("audiovisual", "COMUNICAÇÃO SOCIAL - AUDIOVISUAL (BACHARELADO) - DIURNO (DARCY RIBEIRO)")
]

print("--- Verifying Campus/Turno Fuzzy Matching (Isolated) ---")
all_passed = True
for query, expected in tests:
    result = find_best_match(query, choices)
    status = "✅ PASS" if result == expected else f"❌ FAIL (Got: {result})"
    print(f"Query: '{query}' -> {status}")
    if result != expected:
        all_passed = False

if all_passed:
    print("\nAll tests passed!")
else:
    print("\nSome tests failed.")
