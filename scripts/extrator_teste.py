import pdfplumber
import pandas as pd
import re
import os
import logging
import unicodedata
import numpy as np

# ================= CONFIGURAÇÕES =================
logging.getLogger("pdfminer").setLevel(logging.ERROR)

pasta_pdfs = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\pdfs"
banco_alunos_path = r"C:\Users\user\Documents\unb\Codigos\repositorios\Analise-Preditiva-PAS\data\banco_alunos_pas_final.csv"
arquivo_saida = "notas_corte_pas_SISTEMA_1_2.csv"

# GAP MÁXIMO (Para remover outliers isolados)
# Se a diferença entre o penúltimo e o último for maior que 30, o último é descartado.
GAP_MAXIMO = 30.0 

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

# MAPA DE SISTEMAS ATUALIZADO
mapa_sistemas = {
    '1': 'Sistema Universal', 
    '2': 'Cota para Negros'  # <--- Nome ajustado conforme solicitado
}
# =================================================

print("⏳ Carregando banco de alunos...")
try:
    df_alunos = pd.read_csv(banco_alunos_path, low_memory=False)
    # Limpeza vital da chave primária
    df_alunos['Inscricao'] = df_alunos['Inscricao'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
except Exception as e:
    print(f"❌ Erro crítico: {e}")
    exit()

resultados = []
log_outliers = []

print(f"\n🚀 Iniciando processamento EXCLUSIVO para Sistemas 1 e 2...")

for config in arquivos_para_processar:
    nome_arquivo = config["arquivo"]
    pdf_path = os.path.join(pasta_pdfs, nome_arquivo)
    
    trienio = config.get("trienio", "N/A")
    semestre = config.get("semestre", "N/A")
    chamada = config.get("chamada", "N/A")

    if not os.path.exists(pdf_path):
        print(f"⚠️ Arquivo não encontrado: {nome_arquivo}")
        continue

    dicionario_cursos = {}
    campus_atual, turno_atual, curso_atual = "DARCY RIBEIRO", "DIURNO", None
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                texto_pagina = page.extract_text()
                if not texto_pagina: continue
                
                linhas = texto_pagina.split('\n')
                for linha_raw in linhas:
                    l = unicodedata.normalize('NFKC', linha_raw.upper().strip())
                    l = l.replace('–', '-').replace('_', '')
                    l = re.sub(r'\s+', ' ', l)
                    
                    if not l: continue
                    if "SUB JUDICE" in l: 
                        curso_atual = None
                        continue

                    # Regex Campus
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

                    # Concatenação de Quebra de Linha
                    if l.startswith("(") and curso_atual:
                        if not any(x in l for x in ["SISTEMA", "COTAS"]): curso_atual += " " + l
                        continue

                    # Regex Curso
                    is_curso = False
                    if any(k in l for k in ["BACHARELADO", "LICENCIATURA", "ENGENHARIA", "FÍSICA", "QUÍMICA", "HISTÓRIA", "LÍNGUA", "COMPUTAÇÃO"]): is_curso = True
                    elif re.match(r'^[^0-9]+$', l) and len(l) > 10: is_curso = True
                    
                    if is_curso:
                        termos_proibidos = ["SISTEMA", "CANDIDATO", "EDITAL", "UNB", "INSCRIÇÃO", "UNIVERSIDADE", "NOME DO", "CAMPUS", "CEBRASPE", "CHAMADA", "AGENDA"]
                        if not any(b in l for b in termos_proibidos):
                            curso_atual = l
                            continue

                    # Regex Aluno
                    m_aluno = re.search(r'(\d{7,8})\s+(.+?)\s+(\d{1,2})$', l)
                    if m_aluno and curso_atual:
                        inscricao = m_aluno.group(1)
                        sistema = m_aluno.group(3)
                        
                        # === FILTRO DE B2B: SÓ SISTEMA 1 E 2 ===
                        if sistema not in ['1', '2']:
                            continue
                        # =======================================

                        chave = (curso_atual, campus_atual, turno_atual, sistema)
                        if chave not in dicionario_cursos: dicionario_cursos[chave] = []
                        dicionario_cursos[chave].append(inscricao)

        # === PROCESSAMENTO DE NOTAS E LIMPEZA ===
        if dicionario_cursos:
            for (curso, campus, turno, sistema), lista_inscricoes in dicionario_cursos.items():
                
                alunos = df_alunos[df_alunos['Inscricao'].isin(lista_inscricoes)]
                
                if not alunos.empty and 'Arg_Final' in alunos.columns:
                    notas_series = pd.to_numeric(alunos['Arg_Final'], errors='coerce').dropna()
                    notas_series = notas_series[notas_series != 0] # Remove zeros
                    
                    if notas_series.empty: continue
                    
                    # Ordena: [Menor, Segundo_Menor, ..., Maior]
                    notas_sorted = np.sort(notas_series.values)
                    
                    min_original = notas_sorted[0]
                    n_banco = len(notas_sorted)
                    
                    min_final = min_original
                    
                    # === REGRA DE VIZINHANÇA (ANTI-OUTLIER) ===
                    # Se tiver mais de 1 aluno e o gap for gigante, usa o 2º menor.
                    if n_banco > 1:
                        segundo_menor = notas_sorted[1]
                        gap = segundo_menor - min_original
                        
                        if gap > GAP_MAXIMO:
                            min_final = segundo_menor
                            log_outliers.append({
                                "Curso": curso, "Sistema": sistema, 
                                "Min_Original": min_original, "Substituido_Por": segundo_menor,
                                "Gap": gap
                            })
                    
                    resultados.append({
                        "Trienio": trienio, "Semestre": semestre, "Chamada": chamada,
                        "Curso_Limpo": curso, "Campus": campus, "Turno": turno,
                        "Sistema_ID": sistema, 
                        "Sistema_Nome": mapa_sistemas.get(sistema, f"Sistema {sistema}"),
                        "Min": round(min_final, 3), 
                        "Max": round(notas_sorted[-1], 3), 
                        "Media": round(notas_sorted.mean(), 3),
                        "N_Banco": n_banco
                    })

    except Exception as e:
        print(f"❌ Erro em {nome_arquivo}: {e}")

# === PÓS-PROCESSAMENTO ===
if resultados:
    df = pd.DataFrame(resultados)
    
    # Ordenação
    def get_chamada_num(x):
        nums = re.findall(r'\d+', str(x))
        return int(nums[0]) if nums else 1
    
    df['Chamada_Int'] = df['Chamada'].apply(get_chamada_num)
    cols_grupo = ['Trienio', 'Semestre', 'Curso_Limpo', 'Campus', 'Turno', 'Sistema_ID']
    df = df.sort_values(by=cols_grupo + ['Chamada_Int'])

    # Correção em Cascata (Chamada 2 nunca > Chamada 1)
    print("\n📉 Aplicando blindagem lógica (Chamada N <= Chamada N-1)...")
    df['Min'] = df.groupby(cols_grupo)['Min'].transform('cummin')
    
    # Salvar
    cols_output = [c for c in df.columns if c not in ['Chamada_Int']]
    df_output = df[cols_output]
    
    df_output.to_csv(arquivo_saida, index=False)
    
    print(f"\n✅ SUCESSO! Arquivo '{arquivo_saida}' gerado.")
    print(f"   -> Filtro: Apenas Sistemas 1 e 2.")
    print(f"   -> Outliers removidos: {len(log_outliers)}")

else:
    print("⚠️ Nada gerado. Verifique se os PDFs contêm alunos dos Sistemas 1 e 2.")