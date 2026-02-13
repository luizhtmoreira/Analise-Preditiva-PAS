import pandas as pd

# Carregue o banco recém-concatenado (ajuste o nome se necessário)
df = pd.read_csv("banco_alunos_pas_completo.csv")

print("--- AUDITORIA DE INTEGRIDADE DO BANCO ---")

# 1. Checagem de Tamanho
print(f"\n1. Total de linhas: {len(df)}")
# Se você tinha ~71k e o extrator achou ~3.2k, o total aqui deve refletir a soma exata.

# 2. Checagem de Duplicatas (Chave Composta)
duplicatas = df.duplicated(subset=['Inscricao', 'Ano_Trienio']).sum()
print(f"2. Inscrições duplicadas no mesmo triênio: {duplicatas}")

# Bônus Analítico: Quantos alunos estão fazendo múltiplos triênios?
multiplos_trienios = df.duplicated(subset=['Nome'], keep=False)
qtd_alunos_multiplos = df[multiplos_trienios]['Nome'].nunique()
print(f" -> Curiosidade: Existem {qtd_alunos_multiplos} alunos com o mesmo nome fazendo mais de um triênio/inscrição no seu banco.")

# 3. Checagem de Valores Nulos 
nulos_totais = df.isnull().sum().sum()
print(f"3. Total de células vazias (NaN) no banco: {nulos_totais}")
if nulos_totais > 0:
    print(" ⚠️ ALERTA: O preenchimento com 0.00 falhou em alguma coluna.")

# 4. Checagem Estatística (Caçando lixo do PDF)
colunas_pas2 = ['P1_PAS2', 'P2_PAS2', 'Red_PAS2']
print("\n4. Resumo Estatístico das Notas do PAS 2:")
print(df[colunas_pas2].astype(float).describe())