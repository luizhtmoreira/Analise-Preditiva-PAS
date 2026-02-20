import pandas as pd
import numpy as np
import os

# --- CONFIGURAÇÕES ---
ARQUIVO_BANCO = "data/banco_alunos_pas_final.csv"
ARQUIVO_SAIDA = "demo_final_reais_v3.xlsx"
TRIENIO_FIXO = "2022-2024"

# --- MAPA DOS ALUNOS SELECIONADOS (CORRIGIDO) ---
ALVOS = {
    # RELAÇÕES INTERNACIONAIS
    '22101385': ('Ana Clara (1º Lugar Rel. Int)', 'RELAÇÕES INTERNACIONAIS (BACHARELADO)'),
    '22109135': ('Beatriz Rel (Passou/Amarelo)', 'RELAÇÕES INTERNACIONAIS (BACHARELADO)'),

    # MEDICINA
    '22104427': ('Carlos Med (Verde/Aprovado)', 'MEDICINA (BACHARELADO)'),
    '22114275': ('Daniel Med (Amarelo - 1º Sem)', 'MEDICINA (BACHARELADO)'),
    '22117460': ('Eduardo Med (Amarelo - 2º Sem)', 'MEDICINA (BACHARELADO)'),
    '22109858': ('Fernanda Med (Quase - Por uma)', 'MEDICINA (BACHARELADO)'),
    '21185324': ('Gabriel Med (Vermelho)', 'MEDICINA (BACHARELADO)'),
    '22106201': ('Hugo Med (Vermelho)', 'MEDICINA (BACHARELADO)'),
    '22101529': ('Igor Med (Vermelho)', 'MEDICINA (BACHARELADO)'),

    # DIREITO
    '22202210': ('Julia Direito (Verde)', 'DIREITO (BACHARELADO)'),
    '22101316': ('Karina Direito (Passou 2º Sem)', 'DIREITO (BACHARELADO)'),
    '22107478': ('Lucas Direito (Amarelo/Não)', 'DIREITO (BACHARELADO)'),
    '22116932': ('Marcos Direito (Vermelho)', 'DIREITO (BACHARELADO)'),
    '22102094': ('Natalia Direito (Vermelho)', 'DIREITO (BACHARELADO)'),

    # CIÊNCIA DA COMPUTAÇÃO
    '22106667': ('Nathan CC (Verde)', 'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)'),
    '22103818': ('Olivia CC (Amarelo 2º Sem)', 'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)'),
    '22116975': ('Pedro CC (Amarelo)', 'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)'),
    '22122922': ('Quintino CC (Vermelho)', 'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)'),

    # ENGENHARIA DA COMPUTAÇÃO
    '22117007': ('Rafael EngComp (Amarelo 2º Sem)', 'ENGENHARIA DA COMPUTAÇÃO (BACHARELADO)'),
    '22118361': ('Sara EngComp (Amarelo)', 'ENGENHARIA DA COMPUTAÇÃO (BACHARELADO)'),
    '22106665': ('Tiago EngComp (Vermelho)', 'ENGENHARIA DA COMPUTAÇÃO (BACHARELADO)'),
    '22107906': ('Ursula EngComp (Vermelho)', 'ENGENHARIA DA COMPUTAÇÃO (BACHARELADO)'),

    # PSICOLOGIA (AGORA TODOS CORRETOS)
    '22123385': ('Vitoria Psi (Amarelo 2º Sem)', 'PSICOLOGIA (BACHARELADO)'),
    '22106706': ('William Psi (Amarelo)', 'PSICOLOGIA (BACHARELADO)'), 
    '22103002': ('Xavier Psi (Vermelho)', 'PSICOLOGIA (BACHARELADO)'),
    '22118411': ('Yara Psi (Vermelho)', 'PSICOLOGIA (BACHARELADO)'),
}

# --- 1. CARREGAR BANCO ---
if not os.path.exists(ARQUIVO_BANCO):
    print(f"❌ Erro: Arquivo {ARQUIVO_BANCO} não encontrado.")
    exit()

print(f"📂 Lendo banco de dados...")
df_banco = pd.read_csv(ARQUIVO_BANCO)

# Normaliza inscrição
if 'Inscricao' in df_banco.columns:
    df_banco['Inscricao'] = df_banco['Inscricao'].astype(str).str.replace('.0', '', regex=False).str.strip()
else:
    print("❌ Erro: Coluna 'Inscricao' não encontrada.")
    exit()

# --- 2. EXTRAÇÃO ---
print(f"🔍 Buscando {len(ALVOS)} alunos...")

dados_finais = []
encontrados = 0
nao_encontrados = []

for inscricao, (nome_display, curso_alvo) in ALVOS.items():
    aluno_row = df_banco[df_banco['Inscricao'] == inscricao]
    
    if not aluno_row.empty:
        dados = aluno_row.iloc[0]
        
        novo_aluno = {
            'Nome': nome_display,
            'Inscricao': inscricao + " ",
            'Curso_Alvo': curso_alvo,
            'Cota': 'Sistema Universal',
            'Trienio': TRIENIO_FIXO,
            'Unidade': np.random.choice(['Asa Sul', 'Taguatinga']),
            'Turma': np.random.choice(['3º A', '3º B', '3º C']),
            
            # --- EXTRAI NOTAS REAIS ---
            'P1_PAS1': dados.get('P1_PAS1', 0),
            'P2_PAS1': dados.get('P2_PAS1', 0),
            'Red_PAS1': dados.get('Red_PAS1', 0),
            'P1_PAS2': dados.get('P1_PAS2', 0),
            'P2_PAS2': dados.get('P2_PAS2', 0),
            'Red_PAS2': dados.get('Red_PAS2', 0),
        }
        
        dados_finais.append(novo_aluno)
        encontrados += 1
    else:
        nao_encontrados.append(inscricao)
        print(f"⚠️ Aluno não encontrado: {inscricao} ({nome_display})")

# --- 3. SALVAR ---
if dados_finais:
    df_saida = pd.DataFrame(dados_finais)
    df_saida.to_excel(ARQUIVO_SAIDA, index=False)
    
    print("\n" + "="*50)
    print(f"✅ SUCESSO! {encontrados} alunos extraídos.")
    print(f"📄 Arquivo gerado: {ARQUIVO_SAIDA}")
    if nao_encontrados:
        print(f"\n❌ Faltaram: {nao_encontrados}")
    print("="*50)
else:
    print("❌ Nenhum aluno foi encontrado.")