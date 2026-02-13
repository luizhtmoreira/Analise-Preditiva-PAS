import pandas as pd
import re

# 1. Carregue os dois bancos de dados
print("Carregando as tabelas...")
df_principal = pd.read_csv("banco_alunos_pas_final.csv") # Mude para o nome real do seu arquivo
df_faltantes = pd.read_csv("alunos_faltantes_pas2.csv")

# 2. Concatena os dois DataFrames colocando um debaixo do outro
print("Concatenando...")
df_final = pd.concat([df_principal, df_faltantes], ignore_index=True)

# 3. Garante que os vazios viram 0.00 (conforme seu padrão)
# Lista de colunas de notas que precisam ser 0.00 se estiverem vazias
colunas_notas = [
    "P1_PAS1", "P2_PAS1", "Red_PAS1", 
    "P1_PAS2", "P2_PAS2", "Red_PAS2", 
    "P1_PAS3", "P2_PAS3", "Red_PAS3", 
    "Arg_Final"
]

for col in colunas_notas:
    if col in df_final.columns:
        # Preenche os vazios (NaN) com 0.00
        df_final[col] = df_final[col].fillna(0.00)
        
        # Opcional: converte tudo para string temporariamente para arrumar formatação, 
        # removendo espaços acidentais no meio do número (ex: "29.38 9" -> "29.389")
        # e depois volta para float
        df_final[col] = df_final[col].astype(str).str.replace(' ', '', regex=False)
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0.00)

# 4. Salva o banco perfeito e definitivo
nome_arquivo_saida = "banco_alunos_pas_completo.csv"
df_final.to_csv(nome_arquivo_saida, index=False, encoding="utf-8")

print(f"Sucesso! O banco foi atualizado.")
print(f"Total de alunos no novo banco: {len(df_final)} linhas.")