import pdfplumber
import pandas as pd
import re
import os
import logging
import unicodedata
import time

# ================= CONFIGURAÇÕES =================
# Desativa logs técnicos chatos
logging.getLogger("pdfminer").setLevel(logging.ERROR)

pasta_pdfs = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\pdfs"
banco_alunos_path = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\banco_alunos_pas_final.csv"

# Arquivo FINAL (Só com matches perfeitos)
arquivo_saida = "notas_corte_pas_COMPLETO_FINAL.csv"
# Arquivo de CONTROLE (Para você conferir o que foi ignorado)
arquivo_auditoria = "auditoria_ignorados.csv"

# COLE AQUI SUA LISTA COMPLETA DE ARQUIVOS
arquivos_para_processar = [
    {"arquivo": "CC84118B320255BB662477667A06EE58345E110B7CE556B60E0AC4BCA5138AF3.pdf", "trienio": "2022-2024", "semestre": "2°", "chamada": "4ª"},
    {"arquivo": "Ed_31_PAS_3_2019_2021_Conv_Registro_1ª_Chamada.pdf", "trienio": "2019-2021", "semestre": "1°", "chamada": "1ª"},
    {"arquivo": "Ed_31_PAS_3_2020_2022_Conv_RA_1ª_Chamada.pdf", "trienio": "2020-2022", "semestre": "1°", "chamada": "1ª"},
    {"arquivo": "Ed_36_PAS_3_2019_2021_Conv_Registro_2ª_Chamada.pdf", "trienio": "2019-2021", "semestre": "1°", "chamada": "2ª"},
    {"arquivo": "Ed_36_PAS_3_2020_2022_Conv_RA_2ª_Chamada.pdf", "trienio": "2020-2022", "semestre": "1°", "chamada": "2ª"},
    {"arquivo": "Ed_39_2024_PAS_3_2022-2024_Conv_RA_1ª_Chamada.pdf", "trienio": "2022-2024", "semestre": "1°", "chamada": "1ª"},
    {"arquivo": "Ed_39_PAS_3_2020_2022_Conv_RA_3ª_Chamada.pdf", "trienio": "2020-2022", "semestre": "1°", "chamada": "3ª"},
    {"arquivo": "Ed_40_PAS_3_2019_2021_Conv_Registro_3ª_Chamada.pdf", "trienio": "2019-2021", "semestre": "1°", "chamada": "3ª"},
    {"arquivo": "Ed_42_2024_PAS_3_2022-2024_Conv_RA_2ª_Chamada.pdf", "trienio": "2022-2024", "semestre": "1°", "chamada": "2ª"},
    {"arquivo": "Ed_42_PAS_3_2020_2022_Conv_RA_4ª_Chamada.pdf", "trienio": "2020-2022", "semestre": "1°", "chamada": "4ª"},
    {"arquivo": "Ed_43_PAS_3_2019_2021_Conv_Registro_4ª_Chamada.pdf", "trienio": "2019-2021", "semestre": "1°", "chamada": "4ª"},
    {"arquivo": "Ed_44_PAS_3_2020_2022_Rel_Final_RA_4ª_Chamada.pdf", "trienio": "2020-2022", "semestre": "1°", "chamada": "4ª"},
    {"arquivo": "Ed_46_2024_PAS_3_2022-2024_Conv_RA_3ª_Chamada.pdf", "trienio": "2022-2024", "semestre": "1°", "chamada": "3ª"},
    {"arquivo": "Ed_47_PAS_3_2019_2021_Conv_RA_1ª_Chamada_2º_Semestre.pdf", "trienio": "2019-2021", "semestre": "2°", "chamada": "1ª"},
    {"arquivo": "Ed_48_PAS_3_2020_2022_Conv_RA_2ª_Chamada_2º_Semestre.pdf", "trienio": "2020-2022", "semestre": "2°", "chamada": "2ª"},
    {"arquivo": "Ed_51_PAS_3_2019_2021_Conv_RA_2ª_Chamada_2º_Semestre.pdf", "trienio": "2019-2021", "semestre": "2°", "chamada": "2ª"},
    {"arquivo": "Ed_51_PAS_3_2020_2022_Conv_RA_3ª_Chamada_2º_Semestre.pdf", "trienio": "2020-2022", "semestre": "2°", "chamada": "3ª"},
    {"arquivo": "Ed_52_2024_PAS_3_2022-2024_Conv_RA_4a_Chamada.pdf", "trienio": "2022-2024", "semestre": "1°", "chamada": "4ª"},
    {"arquivo": "Ed_54_PAS_3_2019_2021_Conv_RA_3ª_Chamada_2º_S.pdf", "trienio": "2019-2021", "semestre": "2°", "chamada": "3ª"},
    {"arquivo": "Ed_54_PAS_3_2020_2022_Conv_RA_4ª_Chamada_2º_Semestre.pdf", "trienio": "2020-2022", "semestre": "2°", "chamada": "4ª"},
    {"arquivo": "Ed_57_PAS_3_2019_2021_Conv_RA_4ª_Chamada_2º_Semestre.pdf", "trienio": "2019-2021", "semestre": "2°", "chamada": "4ª"},
    {"arquivo": "Ed_57_PAS_3_2020_2022_Conv_RA_5ª_Chamada_2º_Semestre.pdf", "trienio": "2020-2022", "semestre": "2°", "chamada": "5ª"},
    {"arquivo": "Ed_60_2024_PAS_3_2022-2024_Conv_RA_1ª_Chamada_2º_sem.pdf", "trienio": "2022-2024", "semestre": "2°", "chamada": "1ª"},
    {"arquivo": "Ed_63_2024_PAS_3_2022-2024_Conv_RA_2ª_Chamada_2º_sem.pdf", "trienio": "2022-2024", "semestre": "2°", "chamada": "2ª"},
    {"arquivo": "Ed_63_PAS_3_2019_2021_Conv_RA_6ª_Chamada_2º_Semestre.pdf", "trienio": "2019-2021", "semestre": "2°", "chamada": "6ª"},
    {"arquivo": "PAS_3_2019_2021_Ed_60_Conv_RA_5ª_Chamada_2º_Semestre_v2 1.pdf", "trienio": "2019-2021", "semestre": "2°", "chamada": "5ª"},
    {"arquivo": "Ed_66_2024_PAS_3_2022-2024_Conv_RA_3ª_Chamada_2º_Sem.pdf", "trienio": "2022-2024", "semestre": "2°", "chamada": "3ª"},
    {"arquivo": "Ed_52_PAS_3_2021_2023_Conv_RA_5ª_Chamada_2°_se.pdf", "trienio": "2021-2023", "semestre": "2°", "chamada": "5ª"},
    {"arquivo": "Ed_48_PAS_3_2021_2023_Conv_RA_4ª_Chamada_2º_Sem.pdf", "trienio": "2021-2023", "semestre": "2°", "chamada": "4ª"},
    {"arquivo": "Ed_45_PAS_3_2021_2023_Conv_RA_3ª_Chamada_2º_Sem.pdf", "trienio": "2021-2023", "semestre": "2°", "chamada": "3ª"},
    {"arquivo": "Ed_42_PAS_3_2021_2023_Conv_RA_2ª_Chamada_2°_sem.pdf", "trienio": "2021-2023", "semestre": "2°", "chamada": "2ª"},
    {"arquivo": "Ed_37_PAS_3_2021_2023_Conv_RA_1ª_Chamada_2°_sem.pdf", "trienio": "2021-2023", "semestre": "2°", "chamada": "1ª"},
    {"arquivo": "Ed_32_PAS_3_2021_2023_Conv_RA_2ª_Chamada.pdf", "trienio": "2021-2023", "semestre": "1°", "chamada": "2ª"},
    {"arquivo": "Ed_28_PAS_3_2021_2023_Conv_RA_1ª_Chamada.pdf", "trienio": "2021-2023", "semestre": "1°", "chamada": "1ª"},
    {"arquivo": "3089BE7E47EF3C07390C31C7506BA674B47B6248F1E990ED2648953612E61491.pdf", "trienio": "2023-2025", "semestre": "1°", "chamada": "1°"}
]

mapa_sistemas = {str(i): f"Sistema {i}" for i in range(1, 11)}
# =================================================

print("⏳ Carregando e padronizando banco de alunos (57k+ registros)...")
try:
    df_alunos = pd.read_csv(banco_alunos_path, low_memory=False)
    
    # === A LIMPEZA QUE FUNCIONOU NO DIAGNÓSTICO ===
    # 1. Converte para string
    # 2. Remove espaços
    # 3. Remove o sufixo .0 (ex: "12345.0" vira "12345")
    df_alunos['Inscricao'] = df_alunos['Inscricao'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    
    print(f"✅ Banco carregado. {len(df_alunos)} alunos prontos para match.")
except Exception as e:
    print(f"❌ Erro crítico ao abrir banco de alunos: {e}")
    exit()

resultados_validos = []
auditoria_ignorados = []

inicio_total = time.time()
print(f"\n🚀 Iniciando processamento de {len(arquivos_para_processar)} arquivos.\n")

for i, config in enumerate(arquivos_para_processar):
    nome_arquivo = config["arquivo"]
    pdf_path = os.path.join(pasta_pdfs, nome_arquivo)
    
    # Metadados seguros
    trienio = config.get("trienio", "N/A")
    semestre = config.get("semestre", "N/A")
    chamada = config.get("chamada", "N/A")

    if not os.path.exists(pdf_path):
        print(f"[{i+1}/{len(arquivos_para_processar)}] ⚠️ ARQUIVO NÃO ENCONTRADO: {nome_arquivo}")
        continue
        
    print(f"[{i+1}/{len(arquivos_para_processar)}] Lendo: {nome_arquivo}...")
    
    dicionario_cursos = {}
    campus_atual = "DARCY RIBEIRO" 
    turno_atual = "DIURNO"
    curso_atual = None
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                texto_pagina = page.extract_text()
                if not texto_pagina: continue
                
                linhas = texto_pagina.split('\n')
                
                for linha_raw in linhas:
                    # 1. Normalização (UNICODE + TRATAMENTO DE TEXTO)
                    l = unicodedata.normalize('NFKC', linha_raw.upper().strip())
                    l = l.replace('–', '-').replace('_', '')
                    l = re.sub(r'\s+', ' ', l)
                    if not l: continue

                    # 2. Ignora Sub Judice
                    if "SUB JUDICE" in l:
                        curso_atual = None
                        continue

                    # 3. Detecta Campus
                    m_campus = re.search(r'(?:1\.1\.\d+)?.*?CAMPUS\s+(.*)', l)
                    if m_campus and ("DARCY" in l or "PLANALTINA" in l or "CEILÂNDIA" in l or "GAMA" in l):
                        txt = m_campus.group(1)
                        if "CEILÂNDIA" in txt or "FCE" in txt: campus_atual = "CEILÂNDIA"
                        elif "GAMA" in txt or "FGA" in txt: campus_atual = "GAMA"
                        elif "PLANALTINA" in txt or "FUP" in txt: campus_atual = "PLANALTINA"
                        else: campus_atual = "DARCY RIBEIRO"
                        
                        turno_atual = "NOTURNO" if "NOTURNO" in txt else "VESPERTINO" if "VESPERTINO" in txt else "DIURNO"
                        curso_atual = None
                        continue

                    # 4. Concatenação (Arruma quebra de linha em nomes de cursos)
                    if l.startswith("(") and curso_atual:
                        if not any(x in l for x in ["SISTEMA", "COTAS"]):
                            curso_atual += " " + l
                        continue

                    # 5. Detecção de Curso (Híbrida e Robusta)
                    is_curso = False
                    if any(k in l for k in ["BACHARELADO", "LICENCIATURA", "ENGENHARIA", "FÍSICA", "QUÍMICA", "HISTÓRIA", "LÍNGUA", "COMPUTAÇÃO"]):
                        is_curso = True
                    elif re.match(r'^[^0-9]+$', l) and len(l) > 10:
                        is_curso = True
                    
                    if is_curso:
                        termos_proibidos = ["SISTEMA", "CANDIDATO", "EDITAL", "UNB", "INSCRIÇÃO", "UNIVERSIDADE", "NOME DO", "CAMPUS", "CEBRASPE", "CHAMADA", "AGENDA", "DATA PROVÁVEL"]
                        if not any(b in l for b in termos_proibidos):
                            curso_atual = l
                            continue

                    # 6. Captura de Aluno
                    m_aluno = re.search(r'(\d{7,8})\s+(.+?)\s+(\d{1,2})$', l)
                    if m_aluno:
                        if curso_atual:
                            inscricao = m_aluno.group(1)
                            sistema = m_aluno.group(3)
                            
                            chave = (curso_atual, campus_atual, turno_atual, sistema)
                            if chave not in dicionario_cursos: dicionario_cursos[chave] = []
                            dicionario_cursos[chave].append(inscricao)

        # === PROCESSAMENTO E FILTRAGEM ===
        if dicionario_cursos:
            for (curso, campus, turno, sistema), lista_inscricoes in dicionario_cursos.items():
                
                # Busca no Banco (Match)
                alunos = df_alunos[df_alunos['Inscricao'].isin(lista_inscricoes)]
                
                notas_validas = pd.DataFrame()
                if not alunos.empty and 'Arg_Final' in alunos.columns:
                    notas = pd.to_numeric(alunos['Arg_Final'], errors='coerce').dropna()
                    notas_validas = notas[notas != 0] # Filtra zeros
                
                # DECISÃO: SALVAR OU AUDITAR?
                if not notas_validas.empty:
                    # ✅ SUCESSO: Tem nota, vai para o CSV final
                    resultados_validos.append({
                        "Trienio": trienio, "Semestre": semestre, "Chamada": chamada,
                        "Curso_Limpo": curso, "Campus": campus, "Turno": turno,
                        "Sistema_ID": sistema, 
                        "Sistema_Nome": mapa_sistemas.get(sistema, f"Sistema {sistema}"),
                        "Min": round(notas_validas.min(), 3),
                        "Max": round(notas_validas.max(), 3), 
                        "Media": round(notas_validas.mean(), 3),
                        "N_Banco": len(notas_validas), 
                        "N_PDF": len(lista_inscricoes)
                    })
                else:
                    # ⚠️ ALERTA: Leu no PDF mas não achou no banco (ou nota era zero)
                    auditoria_ignorados.append({
                        "Arquivo": nome_arquivo,
                        "Curso": curso,
                        "Alunos_PDF": len(lista_inscricoes),
                        "Motivo": "Alunos não encontrados no banco ou sem nota válida"
                    })

    except Exception as e:
        print(f"❌ ERRO no arquivo {nome_arquivo}: {e}")

# ================= SALVAMENTO FINAL =================
tempo_total = time.time() - inicio_total
print(f"\n🏁 Processamento finalizado em {tempo_total:.1f} segundos.")

if resultados_validos:
    df_final = pd.DataFrame(resultados_validos)
    df_final['Sistema_Int'] = pd.to_numeric(df_final['Sistema_ID'], errors='coerce').fillna(99)
    df_final = df_final.sort_values(
        by=["Trienio", "Semestre", "Curso_Limpo", "Chamada", "Sistema_Int"], 
        ascending=[False, True, True, True, True]
    ).drop(columns=['Sistema_Int'])
    
    df_final.to_csv(arquivo_saida, index=False)
    print(f"\n✅ SUCESSO ABSOLUTO! {len(df_final)} linhas de corte salvas em '{arquivo_saida}'.")
else:
    print("\n⚠️ Nenhum dado válido gerado.")

if auditoria_ignorados:
    df_audit = pd.DataFrame(auditoria_ignorados)
    df_audit.to_csv(arquivo_auditoria, index=False)
    print(f"ℹ️ {len(df_audit)} turmas foram ignoradas (sem match no banco). Detalhes em '{arquivo_auditoria}'.")