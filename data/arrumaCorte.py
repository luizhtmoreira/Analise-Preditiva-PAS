import pandas as pd
import re

# 1. Carregar a sua tabela consolidada
df = pd.read_csv("notas_corte_pas_final.csv")

def separar_colunas(curso_raw):
    if pd.isna(curso_raw):
        return pd.Series(["DESCONHECIDO", "DARCY RIBEIRO", "DIURNO"])
    
    t = str(curso_raw).strip()
    
    # Valores padrão caso não encontre nada na string
    campus = "DARCY RIBEIRO"
    turno = "DIURNO"
    
    # 1. Identificar Campus
    if "CEILÂNDIA" in t.upper() or "FCE" in t.upper():
        campus = "CEILÂNDIA"
    elif "GAMA" in t.upper() or "FGA" in t.upper():
        campus = "GAMA"
    elif "PLANALTINA" in t.upper() or "FUP" in t.upper():
        campus = "PLANALTINA"
        
    # 2. Identificar Turno
    if "(NOTURNO)" in t.upper():
        turno = "NOTURNO"
    elif "(VESPERTINO)" in t.upper():
        turno = "VESPERTINO"
        
    # 3. Extrair Nome Base (Corta tudo o que vem depois do hífen ou de (DIURNO)/(NOTURNO))
    curso_limpo = re.split(r' - | \(DIURNO| \(NOTURNO| \(VESPERTINO', t)[0].strip()
    
    return pd.Series([curso_limpo, campus, turno])

# 2. Aplicar a função e criar as três novas colunas
print("Separando Campus e Turno...")
df[['Curso_Limpo', 'Campus', 'Turno']] = df['Curso'].apply(separar_colunas)

# 3. Reorganizar as colunas para ficar visualmente limpo
ordem = ['Trienio', 'Semestre', 'Curso', 'Curso_Limpo', 'Campus', 'Turno', 'Min', 'Max', 'Media', 'N']
df = df[ordem]

# 4. Salvar o arquivo
df.to_csv("notas_corte_pas_final.csv", index=False, encoding="utf-8")
print("✅ Tabela separada com sucesso!")