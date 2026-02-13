import pandas as pd

# 1. Carregar os bancos
print("Carregando os dados...")
# Este deve ser o seu banco que já tem os dados do PAS 2 e a coluna Ano_Trienio
df_principal = pd.read_csv("banco_alunos_pas_completo.csv") 

# Este é o CSV que você acabou de extrair do PDF do PAS 1 (Triênio 2024-2026)
df_pas1_extraido = pd.read_csv("alunos_pas1_extraido.csv")

# 2. Padronização de segurança
# Garantir que a Inscrição seja tratada como texto para não haver erro de comparação
df_principal['Inscricao'] = df_principal['Inscricao'].astype(str)
df_pas1_extraido['Inscricao'] = df_pas1_extraido['Inscricao'].astype(str)

# 3. Criar o mapa de busca (Inscrição -> Notas)
# Isso transforma o CSV do PAS 1 em um dicionário de busca rápida
mapa_pas1 = df_pas1_extraido.set_index('Inscricao')

# 4. Aplicar a lógica da "Fonte da Verdade"
print("Cruzando dados para o triênio 2024-2026...")

# Criamos uma máscara para mexer APENAS nos alunos do novo triênio
mask_2024 = df_principal['Ano_Trienio'] == '2024-2026'

# Buscamos as notas no mapa. Se a inscrição não existir no PAS 1, o .fillna("0.00") resolve.
df_principal.loc[mask_2024, 'P1_PAS1'] = df_principal.loc[mask_2024, 'Inscricao'].map(mapa_pas1['P1_PAS1']).fillna("0.00")
df_principal.loc[mask_2024, 'P2_PAS1'] = df_principal.loc[mask_2024, 'Inscricao'].map(mapa_pas1['P2_PAS1']).fillna("0.00")
df_principal.loc[mask_2024, 'Red_PAS1'] = df_principal.loc[mask_2024, 'Inscricao'].map(mapa_pas1['Red_PAS1']).fillna("0.00")

# 5. Salvar o resultado final
df_principal.to_csv("banco_alunos_pas_final.csv", index=False)

print("\n--- RELATÓRIO DE SUCESSO ---")
print(f"Total de alunos no triênio 2024-2026: {mask_2024.sum()}")
print(f"Alunos que tiveram nota do PAS 1 encontrada: {df_principal.loc[mask_2024, 'P1_PAS1'].ne('0.00').sum()}")
print("Arquivo salvo como: banco_alunos_pas_final.csv")