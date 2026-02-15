import pandas as pd
import os

# ================= CONFIGURAÇÕES =================
# Nome do arquivo de entrada (o que você disse que vai usar)
ARQUIVO_ENTRADA = "notas_corte_pas.csv"
# Nome do arquivo de saída (blindado)
ARQUIVO_SAIDA = "notas_corte_pas.csv"

def aplicar_blindagem():
    print(f"🛡️ Iniciando Blindagem Cronológica em: {ARQUIVO_ENTRADA}")
    
    if not os.path.exists(ARQUIVO_ENTRADA):
        print("❌ Erro: Arquivo de entrada não encontrado.")
        return

    # Carrega o CSV
    df = pd.read_csv(ARQUIVO_ENTRADA)
    
    # Cria coluna auxiliar para ordenar as chamadas corretamente (1ª, 2ª, 10ª...)
    # Extrai apenas os números da string 'Chamada'
    df['Chamada_Num'] = df['Chamada'].astype(str).str.extract(r'(\d+)').astype(int)
    
    # Garante que Sistema_ID é numérico
    df['Sistema_ID'] = pd.to_numeric(df['Sistema_ID'], errors='coerce')
    
    # Ordena os dados: Trienio -> Curso -> Sistema -> Chamada (Crescente)
    df = df.sort_values(by=['Trienio', 'Semestre', 'Curso_Limpo', 'Campus', 'Turno', 'Sistema_ID', 'Chamada_Num'])
    
    # Agrupa por turma (mesmo curso, mesmo ano, mesmo sistema)
    grupos = df.groupby(['Trienio', 'Semestre', 'Curso_Limpo', 'Campus', 'Turno', 'Sistema_ID'])
    
    total_correcoes = 0
    indices_alterados = []
    
    print(f"📊 Processando {len(grupos)} turmas...")
    
    for nome_grupo, grupo in grupos:
        # Pega os índices e as notas originais
        indices = grupo.index.tolist()
        notas = grupo['Min'].tolist()
        
        if len(notas) < 2:
            continue # Se só tem 1 chamada, não precisa blindar
            
        # LÓGICA DE BLINDAGEM (De trás para frente)
        # Se a Chamada N tem nota maior que a Chamada N-1, a N-1 sobe para igualar.
        # Ex: 2ª Chamada (120) > 1ª Chamada (80) -> 1ª vira 120.
        for i in range(len(notas) - 1, 0, -1):
            nota_atual = notas[i]      # Ex: 2ª Chamada
            nota_anterior = notas[i-1] # Ex: 1ª Chamada
            
            if nota_atual > nota_anterior:
                # Detectou inconsistência (nota subiu) -> Corrige a anterior
                notas[i-1] = nota_atual
                total_correcoes += 1
                indices_alterados.append(indices[i-1])
        
        # Atualiza as notas no DataFrame original
        df.loc[indices, 'Min'] = notas

    # Remove a coluna auxiliar e salva
    df = df.drop(columns=['Chamada_Num'])
    df.to_csv(ARQUIVO_SAIDA, index=False)
    
    print("-" * 50)
    print("✅ BLINDAGEM CONCLUÍDA!")
    print(f"Total de correções aplicadas: {total_correcoes}")
    print(f"Arquivo salvo como: {ARQUIVO_SAIDA}")
    print("-" * 50)

if __name__ == "__main__":
    aplicar_blindagem()