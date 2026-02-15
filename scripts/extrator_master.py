import pdfplumber
import pandas as pd
import re
import os
import logging

# Desativa logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ================= CONFIGURAÇÕES =================
pasta_pdfs = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\pdfs"
banco_alunos_path = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\banco_alunos_pas_final.csv"

mapa_sistemas = {
    '1': 'Universal', '2': 'Cota para Negros', '3': 'EP / Baixa Renda / PPI',
    '4': 'EP / Baixa Renda / PPI / PCD', '5': 'EP / Baixa Renda / Não-PPI',
    '6': 'EP / Baixa Renda / Não-PPI / PCD', '7': 'EP / Alta Renda / PPI',
    '8': 'EP / Alta Renda / PPI / PCD', '9': 'EP / Alta Renda / Não-PPI',
    '10': 'EP / Alta Renda / Não-PPI / PCD'
}

# COLOQUE AQUI SUA LISTA FINAL
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
# =================================================

print("Carregando banco geral de alunos...")
df_alunos = pd.read_csv(banco_alunos_path, low_memory=False)
df_alunos['Inscricao'] = df_alunos['Inscricao'].astype(str).str.strip()

resultados_mestres = []

for config in arquivos_para_processar:
    pdf_path = os.path.join(pasta_pdfs, config["arquivo"])
    trienio = config["trienio"]
    semestre = config["semestre"]
    chamada = config.get("chamada", "1ª")
    
    if not os.path.exists(pdf_path):
        print(f"⚠️ ARQUIVO NÃO ENCONTRADO: {config['arquivo']}")
        continue
        
    print(f"\n🚀 Lendo: {config['arquivo'][:30]}... ({trienio} | {semestre} | {chamada})")
    
    dicionario_cursos = {}
    campus_atual, turno_atual, curso_atual = "DARCY RIBEIRO", "DIURNO", None
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texto_completo = ""
            for page in pdf.pages: texto_completo += (page.extract_text() or "") + "\n"
            
            for linha in texto_completo.split('\n'):
                l = re.sub(r'\s+', ' ', linha.upper().strip())
                if not l or "SUB JUDICE" in l: continue

                # 1. Âncora Campus
                m_campus = re.search(r'1\.1\.\d+.*?CAMPUS\s+(.*)', l)
                if m_campus:
                    txt = m_campus.group(1)
                    campus_atual = "CEILÂNDIA" if "CEILÂNDIA" in txt else "GAMA" if "GAMA" in txt else "PLANALTINA" if "PLANALTINA" in txt else "DARCY RIBEIRO"
                    turno_atual = "NOTURNO" if "NOTURNO" in txt else "VESPERTINO" if "VESPERTINO" in txt else "DIURNO"
                    curso_atual = None 
                    continue

                # 2. Identificação Curso
                if re.match(r'^[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s\(\)\-\,\/\*]+$', l) and len(l) > 8:
                    if not any(b in l for b in ["SISTEMA", "CANDIDATO", "EDITAL", "UNB", "INSCRIÇÃO"]):
                        curso_atual = "ENGENHARIAS (BACHARELADOS)" if ("(BACHARELADOS)**" in l and campus_atual == "GAMA") else l
                        continue

                # 3. Âncora Aluno
                m_aluno = re.match(r'^(\d{8})\s+(.+?)\s+(\d{1,2})$', l)
                if m_aluno and curso_atual:
                    inscricao = m_aluno.group(1)
                    sistema_num = m_aluno.group(3)
                    
                    chave = (curso_atual, campus_atual, turno_atual, sistema_num)
                    if chave not in dicionario_cursos: dicionario_cursos[chave] = []
                    dicionario_cursos[chave].append(inscricao)

        # CRUZAMENTO E CÁLCULOS
        if dicionario_cursos:
            print(f"   ✅ Extração OK. Calculando notas...")
            for (curso, campus, turno, sistema), lista_inscricoes in dicionario_cursos.items():
                alunos = df_alunos[df_alunos['Inscricao'].isin(lista_inscricoes)]
                
                if not alunos.empty and 'Arg_Final' in alunos.columns:
                    notas = pd.to_numeric(alunos['Arg_Final'], errors='coerce').dropna()
                    
                    # --- AQUI ESTÁ A CORREÇÃO MÁGICA ---
                    # Filtra notas iguais a 0.0 (consideradas erro de cadastro)
                    # Mantém apenas notas diferentes de zero.
                    notas_validas = notas[notas != 0]
                    
                    # Se sobrar alguma nota válida, usamos ela. Se todas forem 0, usamos 0 mesmo.
                    notas_finais = notas_validas if not notas_validas.empty else notas
                    
                    if not notas_finais.empty:
                        resultados_mestres.append({
                            "Trienio": trienio,
                            "Semestre": semestre,
                            "Chamada": chamada,
                            "Curso_Limpo": curso,
                            "Campus": campus,
                            "Turno": turno,
                            "Sistema_ID": sistema,
                            "Sistema_Nome": mapa_sistemas.get(sistema, f"Sistema {sistema}"),
                            "Min": round(notas_finais.min(), 3),  # O menor valor VÁLIDO achado
                            "Max": round(notas_finais.max(), 3),
                            "Media": round(notas_finais.mean(), 3),
                            "N": len(notas_finais)
                        })
    except Exception as e:
         print(f"❌ ERRO: {e}")

# Salva ordenado
if resultados_mestres:
    df_final = pd.DataFrame(resultados_mestres)
    df_final['Sistema_Int'] = pd.to_numeric(df_final['Sistema_ID'])
    df_final = df_final.sort_values(
        by=["Trienio", "Semestre", "Curso_Limpo", "Chamada", "Sistema_Int"], 
        ascending=[False, True, True, True, True]
    ).drop(columns=['Sistema_Int'])
    
    df_final.to_csv("notas_corte_pas_final_BLINDADO.csv", index=False)
    print(f"\n🎉 TABELA PERFEITA GERADA! {len(df_final)} linhas salvas.")
else:
    print("\n⚠️ Nada gerado.")