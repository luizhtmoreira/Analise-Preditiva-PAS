import difflib
import unicodedata

def normalize(text):
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def find_best_match(query, choices, cutoff=0.6):
    print(f"--- Query: '{query}' ---")
    
    # 1. Difflib
    matches = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff)
    if matches:
        print(f"  [Difflib] Match: '{matches[0]}'")
        return matches[0]
    print(f"  [Difflib] No match (cutoff {cutoff})")

    # 2. Substring (Current logic)
    query_norm = normalize(query)
    candidates = []
    for choice in choices:
        choice_norm = normalize(choice)
        if query_norm in choice_norm:
            candidates.append(choice)
    if candidates:
        best = min(candidates, key=len)
        print(f"  [Substring] Match: '{best}'")
        return best
    print(f"  [Substring] No match")

    # 3. PROPOSED: Token Intersection
    best_token_match = None
    best_token_score = 0.0
    
    query_tokens = set(query_norm.split())
    
    for choice in choices:
        choice_norm = normalize(choice)
        choice_tokens = set(choice_norm.split())
        
        # Intersection
        common = query_tokens.intersection(choice_tokens)
        if not common: continue
        
        # Score strategy: Jaccard or simple inclusion?
        # Let's try simple overlap ratio relative to QUERY length (user intention)
        # If user typed "direito noturno", they want "direito". 
        # But "noturno" is ignored.
        
        score = len(common) / len(query_tokens)
        
        if score > best_token_score:
            best_token_score = score
            best_token_match = choice
            
    if best_token_match and best_token_score >= 0.5: # At least half the words match
        print(f"  [Token] Match: '{best_token_match}' (Score: {best_token_score:.2f})")
        return best_token_match
        
    print(f"  [Token] No match")
    return None

# Mock Data
choices = [
    "DIREITO (BACHARELADO)",
    "ENFERMAGEM (BACHARELADO)",
    "ENGENHARIA CIVIL (BACHARELADO)",
    "COMUNICAÇÃO SOCIAL - AUDIOVISUAL (BACHARELADO)",
    "SERVIÇO SOCIAL (BACHARELADO)",
    "COMPUTAÇÃO (LICENCIATURA)"
]

tests = [
    "direito noturno",
    "enfermagem ceilandia",
    "serviço social noturno",
    "audiovisual", # Check regression
    "computação"
]

for t in tests:
    find_best_match(t, choices)
    print("")
