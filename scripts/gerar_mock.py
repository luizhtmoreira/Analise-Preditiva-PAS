import pandas as pd
import numpy as np
import random

# Configurações
NUM_ALUNOS = 100
ARQUIVO_SAIDA = "dados_escola_padrao.xlsx"

# Listas de Opções Reais
UNIDADES = ['Asa Sul', 'Taguatinga', 'Lago Sul', 'Águas Claras']
TURMAS = ['3º A', '3º B', '3º C', '3º D']
COTAS = [
    'Sistema Universal', 
    'Cota para Negros', 
    'Escola Pública - Renda Baixa', 
    'Escola Pública - Renda Alta'
]
CURSOS = [
    'MEDICINA (BACHARELADO)', 
    'DIREITO (BACHARELADO)', 
    'ENGENHARIA DE SOFTWARE (BACHARELADO)',
    'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)',
    'PSICOLOGIA (BACHARELADO)',
    'RELAÇÕES INTERNACIONAIS (BACHARELADO)'
]

# Dados base
data = []

# --- 1. CASO DE TESTE: LUIZ HENRIQUE (Obrigatório) ---
# Garante que ele apareça com os dados que você quer validar
luiz = {
    'Nome': 'Luiz Henrique Moreira',
    'Inscricao': '20240001',
    'Unidade': 'Asa Sul',
    'Turma': '3º A',
    'Cota': 'Sistema Universal',  # AQUI ESTÁ A FIXAÇÃO
    'Curso_Alvo': 'ENGENHARIA DE SOFTWARE (BACHARELADO)',
    'Trienio': '2023-2025',
    # Notas boas para testar aprovação
    'P1_PAS1': 8.5, 'P2_PAS1': 45.0, 'Red_PAS1': 8.0,
    'P1_PAS2': 7.0, 'P2_PAS2': 55.0, 'Red_PAS2': 9.0
}
data.append(luiz)

# --- 2. GERAR MOCK PARA O RESTO ---
nomes_ficticios = [
    "Ana Silva", "Beatriz Costa", "Carlos Oliveira", "Daniel Santos", 
    "Eduarda Lima", "Felipe Souza", "Gabriela Rocha", "Hugo Almeida",
    "Isabela Dias", "João Pedro", "Karina Martins", "Lucas Pereira"
]

for i in range(NUM_ALUNOS):
    nome_base = random.choice(nomes_ficticios)
    sobrenome = f" {random.choice(['A.', 'B.', 'C.'])} {random.randint(100,999)}"
    
    # Distribuição de Cotas mais realista (70% Universal)
    cota_escolhida = np.random.choice(COTAS, p=[0.7, 0.1, 0.1, 0.1])
    
    aluno = {
        'Nome': f"{nome_base}{sobrenome}",
        'Inscricao': f"2024{1000+i}",
        'Unidade': random.choice(UNIDADES),
        'Turma': random.choice(TURMAS),
        'Cota': cota_escolhida, # Coluna Explicita
        'Curso_Alvo': random.choice(CURSOS),
        'Trienio': '2023-2025',
        # Notas Aleatórias
        'P1_PAS1': round(np.random.uniform(2, 9), 2),
        'P2_PAS1': round(np.random.uniform(15, 60), 2),
        'Red_PAS1': round(np.random.uniform(4, 9.5), 2),
        'P1_PAS2': round(np.random.uniform(2, 9), 2),
        'P2_PAS2': round(np.random.uniform(15, 70), 2),
        'Red_PAS2': round(np.random.uniform(5, 10), 2),
    }
    data.append(aluno)

# Criar DataFrame
df = pd.DataFrame(data)

# Salvar
df.to_excel(ARQUIVO_SAIDA, index=False)
print(f"✅ Arquivo '{ARQUIVO_SAIDA}' gerado com sucesso!")
print(f"📋 Total de alunos: {len(df)}")
print("🔍 Verifique se a coluna 'Cota' está preenchida corretamente.")