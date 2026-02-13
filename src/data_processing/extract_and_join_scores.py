
import pypdf
import pandas as pd
import re
import os
from pathlib import Path

def extract_and_join_scores(data_dir: str = "data"):
    """
    1. Lê a tabela MESTRA de notas (Ed_38) -> {Inscricao: Argumento}
    2. Lê os editais de CONVOCAÇÃO -> {Inscricao: (Curso, Semestre, Sistema, Trienio)}
    3. Faz o JOIN, filtra Sistema Universal e gera estatísticas.
    Uses pypdf for speed.
    """
    data_path = Path(data_dir)
    
    # --- 1. Mapeamento de Arquivos ---
    score_file = None
    convocation_files = []
    
    for f in data_path.glob("*.pdf"):
        if "Res_final_não_eliminados" in f.name:
            score_file = f
        else:
            convocation_files.append(f)
            
    if not score_file:
        print("ERRO CRÍTICO: Arquivo de Notas (Res_final_não_eliminados) não encontrado!")
        return

    print(f"Arquivo Mestre de Notas: {score_file.name}")
    print(f"Arquivos de Convocação: {len(convocation_files)}")

    # --- 2. Extração de Notas (Ed_38) ---
    print("\n>>> Extraindo Notas do Mestre (pypdf)...")
    scores_data = []
    
    try:
        reader = pypdf.PdfReader(score_file)
        print(f"   Total páginas: {len(reader.pages)}")
        
        full_text = ""
        # Concatena sem quebras para evitar problemas de split no meio de linhas
        # Mas cuidado com a memória? 242 paginas é tranquilo (~1MB text).
        for i, page in enumerate(reader.pages):
            if i % 50 == 0: print(f"   Processando página {i}...")
            t = page.extract_text()
            if t: full_text += " " + t
            
        # O separador de alunos parece ser ' / '
        # Ex: ... - / 22115152, Carina ...
        
        # Normaliza quebras de linha que o pypdf pode ter inserido no meio do texto
        full_text = full_text.replace('\n', ' ')
        
        students = full_text.split(' / ')
        print(f"   Total blocos detectados: {len(students)}")
        
        for student_block in students:
            # Limpeza basica
            block = student_block.strip()
            if not block: continue
            
            # Tenta extrair Inscrição (primeiro token numerico seguido de virgula ou espaco)
            # Regex: (Inicio)(Inscricao), (Nome) ...
            # Ex: 22117900, Danielly ...
            
            # Às vezes o split deixa residuos no inicio.
            # Procura padrao: Digitos(7-9) seguido de virgula
            match_insc = re.search(r"(\d{7,9}),\s*([A-Za-z\s]+),", block)
            if not match_insc:
                continue
                
            inscricao = match_insc.group(1)
            # nome = match_insc.group(2)
            
            # Busca todas as notas (floats com 3 casas decimais tipicos do CESPE/UnB)
            # Ex: 50.266, -10.403
            floats = re.findall(r"(-?\d+\.\d{3})", block)
            
            # Esperamos 10 notas: 
            # E1(P1, P2, Red), E2(P1, P2, Red), E3(P1, P2, Red), ArgumentoFinal
            if len(floats) >= 10:
                # O Argumento Final é tipicamente o ultimo da série de notas
                # ou o 10º (índice 9)
                arg_str = floats[9] # Pega o décimo valor
                
                try:
                    arg_final = float(arg_str)
                    scores_data.append({'Inscricao': inscricao, 'Argumento': arg_final})
                except: pass
            elif len(floats) > 0:
                # Fallback: Se tiver menos (ex: eliminado em alguma etapa mas apareceu aqui?), 
                # pega o ultimo disponivel? Não, arriscado.
                # Se for "Não eliminado", deve ter todas as notas.
                pass

    except Exception as e:
        print(f"   ERRO ao ler Mestre: {e}")
        return

    df_scores = pd.DataFrame(scores_data).drop_duplicates(subset='Inscricao')
    print(f"   Notas extraídas: {len(df_scores)} alunos.")
    
    if len(df_scores) < 100:
        print("   [ALERTA] Poucas notas extraídas. Regex pode estar falhando.")
        if len(df_scores) > 0:
            print(f"   Exemplo: {df_scores.iloc[0]}")

    # --- 3. Extração de Convocações (Cursos) ---
    print("\n>>> Processando Convocações (pypdf)...")
    convocation_data = []
    
    FILE_SEMESTER_MAP = {
        "CC84118B320255BB662477667A06EE58345E110B7CE556B60E0AC4BCA5138AF3.pdf": "2º",
        "Ed_39_2024_PAS_3_2022-2024_Conv_RA_1ª_Chamada.pdf": "1º",
        "Ed_42_2024_PAS_3_2022-2024_Conv_RA_2ª_Chamada.pdf": "1º",
        "Ed_46_2024_PAS_3_2022-2024_Conv_RA_3ª_Chamada.pdf": "1º",
        "Ed_52_2024_PAS_3_2022-2024_Conv_RA_4a_Chamada.pdf": "1º",
        "Ed_60_2024_PAS_3_2022-2024_Conv_RA_1ª_Chamada_2º_sem.pdf": "2º",
        "Ed_63_2024_PAS_3_2022-2024_Conv_RA_2ª_Chamada_2º_sem.pdf": "2º",
        "Ed_66_2024_PAS_3_2022-2024_Conv_RA_3ª_Chamada_2º_Sem.pdf": "2º"
    }

    course_cache = None
    
    for pdf_file in convocation_files:
        semestre = FILE_SEMESTER_MAP.get(pdf_file.name, None)
        if not semestre:
            lower = pdf_file.name.lower()
            semestre = "2º" if "2º" in lower or "2sem" in lower or "2°" in lower else "1º"
            
        trienio = "2022-2024" 
        print(f"   Lendo {pdf_file.name} | Sem: {semestre}")
        
        try:
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if not text: continue
                lines = text.split('\n')
                
                for line in lines:
                    # Curso
                    if ("BACHARELADO" in line or "LICENCIATURA" in line) and "SISTEMA" not in line:
                        course_cache = line.strip()
                        continue
                    
                    # Aluno: INSCRIÇÃO NOME SISTEMA
                    # Match: Inicio com Digitos, Fim com Digito 1 (Universal)
                    # Ex: 21100123 JOAO DA SILVA 1
                    
                    match = re.search(r"^(\d{7,9})\s+(.+?)\s+(\d+)(\s+\d+)?$", line.strip())
                    if match:
                        insc = match.group(1)
                        sis = match.group(3)
                        
                        if sis == '1': # Universal
                            if course_cache:
                                convocation_data.append({
                                    'Inscricao': insc,
                                    'Curso': course_cache,
                                    'Semestre': semestre,
                                    'Trienio': trienio
                                })
        except Exception as e:
            print(f"   Erro em {pdf_file.name}: {e}")

    df_conv = pd.DataFrame(convocation_data)
    print(f"\nTotal Convocações (Universal): {len(df_conv)}")
    
    # --- 4. Join ---
    if df_conv.empty or df_scores.empty:
        print("Dados insuficientes para Cruzamento.")
        return

    print("\n>>> Cruzando Dados...")
    merged = pd.merge(df_conv, df_scores, on='Inscricao', how='inner')
    print(f"Total pareado: {len(merged)}")
    
    if len(merged) == 0:
        print("ERRO: Join retornou vazio.")
        # Debug ids
        print(f"Ex Scores: {df_scores['Inscricao'].head().tolist()}")
        print(f"Ex Conv: {df_conv['Inscricao'].head().tolist()}")
        return

    # --- 5. Export ---
    print("\n>>> Consolidando...")
    summary = merged.groupby(['Trienio', 'Semestre', 'Curso'])['Argumento'].agg(
        Min='min', Max='max', Media='mean', N='count'
    ).reset_index()
    
    summary[['Min', 'Max', 'Media']] = summary[['Min', 'Max', 'Media']].round(3)
    
    print("\n--- SANITY CHECK (MEDICINA) ---")
    print(summary[summary['Curso'].str.contains("MEDICINA", case=False)])
    
    out_file = data_path / "notas_corte_PAS_consolidado_v2.csv"
    summary.to_csv(out_file, index=False)
    print(f"\nSucesso: {out_file}")

if __name__ == "__main__":
    extract_and_join_scores()
