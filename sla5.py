import pandas as pd
import pdfplumber
import re
import os
import numpy as np

# ================= CONFIGURAÇÕES =================
# Ajuste os caminhos se necessário
ARQUIVO_CSV_ALVO = "notas_corte_pas.csv"
BANCO_ALUNOS = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\banco_alunos_pas_final.csv"
PASTA_PDFS = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\pdfs"

# =================================================

def carregar_dados():
    print("📂 Carregando arquivos...")
    if not os.path.exists(ARQUIVO_CSV_ALVO):
        print(f"❌ Erro: {ARQUIVO_CSV_ALVO} não encontrado.")
        return None, None
    
    df = pd.read_csv(ARQUIVO_CSV_ALVO)
    # Garante numérico
    for c in ['Min', 'Max', 'Media', 'N']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df_alunos = pd.read_csv(BANCO_ALUNOS, low_memory=False)
    df_alunos['Inscricao'] = df_alunos['Inscricao'].astype(str).str.strip()
    return df, df_alunos

def encontrar_outliers(df):
    # Regra de Outlier: Distância da Mínima para Média é desproporcional (>3x) à da Máxima
    # E a diferença absoluta é grande (> 40 pontos)
    df['Dist_Sup'] = df['Max'] - df['Media']
    df['Dist_Inf'] = df['Media'] - df['Min']
    
    mask = (df['N'] > 2) & (df['Dist_Inf'] > 3 * df['Dist_Sup']) & (df['Dist_Inf'] > 40)
    return df[mask]

def encontrar_pdf_correspondente(trienio, semestre, chamada):
    # Tenta achar o PDF na pasta baseado no nome
    # Normaliza termos para busca
    sem_str = "1" if "1" in str(semestre) else "2"
    cham_str = str(chamada).replace("ª", "").replace("º", "")
    
    candidatos = []
    for f in os.listdir(PASTA_PDFS):
        if not f.endswith(".pdf"): continue
        f_norm = f.upper()
        
        # Filtros heurísticos pelo nome do arquivo
        # Triênio (ex: 2020_2022 ou 2020-2022)
        tri_clean = trienio.replace("-", "_")
        if trienio not in f and tri_clean not in f: continue
        
        # Semestre
        if f"{sem_str}º_SEM" not in f_norm and f"{sem_str}_SEM" not in f_norm and f"{sem_str}SEM" not in f_norm:
             # Tente lógica inversa: se for 1º semestre, muitas vezes não tem nada escrito, mas se for 2º tem.
             if sem_str == "2" and "2" not in f_norm: continue
        
        # Chamada
        if f"{cham_str}ª_CHAMADA" in f_norm or f"{cham_str}_CHAMADA" in f_norm:
            candidatos.append(f)
            
    # Retorna o melhor candidato (o menor nome geralmente é o edital principal, ou o mais recente)
    if candidatos:
        return candidates[0] if len(candidatos) == 1 else candidates[-1] # Pega um deles
    return None

def extrair_notas_reais(pdf_name, curso_alvo, sistema_id, df_alunos):
    path = os.path.join(PASTA_PDFS, pdf_name)
    inscricoes_encontradas = []
    
    print(f"   📖 Lendo PDF: {pdf_name}...")
    try:
        with pdfplumber.open(path) as pdf:
            texto = ""
            for p in pdf.pages: texto += (p.extract_text() or "") + "\n"
            
            # Lógica simplificada de extração focada no curso alvo
            # Procura o bloco do curso
            lines = texto.split('\n')
            dentro_do_curso = False
            
            # Normaliza nome do curso para busca (remove parenteses se precisar)
            curso_clean = curso_alvo.split("(")[0].strip()
            
            for line in lines:
                l = re.sub(r'\s+', ' ', line.upper().strip())
                
                # Identifica início do curso (busca aproximada)
                if curso_clean in l and ("BACHARELADO" in l or "LICENCIATURA" in l):
                    dentro_do_curso = True
                    continue
                
                # Se achou outro curso, para
                if dentro_do_curso and re.match(r'^[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s\(\)\-\,\/\*]+$', l) and len(l) > 10:
                    if "SISTEMA" not in l and "CANDIDATO" not in l and curso_clean not in l:
                        dentro_do_curso = False
                        # Se já achamos gente, podemos parar para economizar tempo? 
                        # Melhor não, vai que o curso aparece quebrado.
                
                # Captura aluno se estiver dentro do bloco
                if dentro_do_curso:
                    m = re.match(r'^(\d{8})\s+(.+?)\s+(\d{1,2})$', l)
                    if m:
                        insc = m.group(1)
                        sis = m.group(3)
                        # Verifica se é o sistema que queremos corrigir
                        if str(sis) == str(sistema_id):
                            inscricoes_encontradas.append(insc)
                            
    except Exception as e:
        print(f"   ❌ Erro ao ler PDF: {e}")
        return []

    # Busca notas no banco
    if inscricoes_encontradas:
        alunos = df_alunos[df_alunos['Inscricao'].isin(inscricoes_encontradas)]
        notas = pd.to_numeric(alunos['Arg_Final'], errors='coerce').dropna()
        notas = notas[notas != 0] # Tira zeros
        return sorted(notas.tolist())
    return []

# ================= MAIN =================
df, df_alunos = carregar_dados()
if df is not None:
    outliers = encontrar_outliers(df)
    
    print(f"\n🚨 Encontrados {len(outliers)} cursos com notas suspeitas.")
    print("Iniciando cirurgia reparadora...\n")
    
    correcoes_feitas = 0
    
    for idx, row in outliers.iterrows():
        print(f"🔧 Corrigindo: {row['Trienio']} | {row['Curso_Limpo']} | {row['Sistema_Nome']}")
        print(f"   Nota Atual (Suspeita): {row['Min']} (Média: {row['Media']})")
        
        # 1. Achar PDF manualmente se a busca automática falhar
        # (Aqui usamos uma busca genérica na pasta toda se não achar específico)
        pdf_candidato = None
        termo_busca = row['Trienio'].replace("-", "_")
        if "1" in str(row['Semestre']): termo_busca += "_1"
        
        # Varredura simples nos arquivos para achar um match
        arquivos = os.listdir(PASTA_PDFS)
        matches = [f for f in arquivos if row['Trienio'] in f and (f"_{row['Chamada']}_" in f or f" {row['Chamada']}ª" in f)]
        
        if not matches:
            # Tenta ser mais flexível
            matches = [f for f in arquivos if row['Trienio'] in f and "Chamada" in f]
            
        if matches:
            # Prioriza o que tem o semestre certo
            sem_tag = "2º" if "2" in str(row['Semestre']) else "1º" # Padrão comum
            matches_sem = [f for f in matches if sem_tag in f]
            pdf_target = matches_sem[0] if matches_sem else matches[0]
            
            # Extrai notas reais
            notas_reais = extrair_notas_reais(pdf_target, row['Curso_Limpo'], row['Sistema_ID'], df_alunos)
            
            if len(notas_reais) > 1:
                # A MÁGICA: Remove a menor nota (o outlier) e recalcula
                nota_ruim = notas_reais[0]
                notas_boas = notas_reais[1:] # Pega da segunda em diante
                
                novo_min = min(notas_boas)
                novo_max = max(notas_boas)
                nova_media = sum(notas_boas) / len(notas_boas)
                novo_n = len(notas_boas)
                
                print(f"   ✅ Notas encontradas: {notas_reais}")
                print(f"   ✂️ Removendo a menor ({nota_ruim}). Nova Mínima: {novo_min}")
                
                # Atualiza no DataFrame
                df.at[idx, 'Min'] = round(novo_min, 3)
                df.at[idx, 'Max'] = round(novo_max, 3)
                df.at[idx, 'Media'] = round(nova_media, 3)
                df.at[idx, 'N'] = novo_n
                correcoes_feitas += 1
            else:
                print("   ⚠️ Não foi possível encontrar notas suficientes para recalcular.")
        else:
            print("   ⚠️ PDF correspondente não encontrado automaticamente.")
            
    if correcoes_feitas > 0:
        # Remove colunas temp
        df = df.drop(columns=['Dist_Sup', 'Dist_Inf'])
        df.to_csv(ARQUIVO_CSV_ALVO, index=False)
        print(f"\n🎉 Sucesso! {correcoes_feitas} correções aplicadas e arquivo salvo.")
    else:
        print("\nNenhuma correção foi aplicada (talvez falha ao achar PDFs).")