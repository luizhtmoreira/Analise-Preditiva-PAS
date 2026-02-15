
import pdfplumber
import pandas as pd
import re
import os
from pathlib import Path

def extract_cutoff_scores(data_dir: str = "data"):
    """
    Extrai notas de corte e estatísticas do PAS/UnB a partir de arquivos PDF.
    
    Regras:
    - Lê todos os PDFs em data_dir.
    - Extrai Triênio do nome do arquivo.
    - Extrai Semestre do cabeçalho ou nome do arquivo.
    - Filtra Sistema Universal (Sistema/Subsistema == 1).
    - Agrupa por Triênio > Semestre > Curso.
    - Calcula Min, Max, Média, N.
    """
    
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado em {data_dir}")
        return

    all_students = []

    print(f"Encontrados {len(pdf_files)} arquivos PDF.")

    for pdf_file in pdf_files:
        print(f"\nProcessando: {pdf_file.name}")
        
        # 1. Extração de Metadados do Nome do Arquivo
        
        # Triênio (Ex: 2021-2023, 2022-2024)
        trienio_match = re.search(r"20\d{2}[-_]20\d{2}", pdf_file.name)
        if trienio_match:
            trienio = trienio_match.group(0).replace("_", "-")
        else:
            print(f"  [ALERTA] Triênio não identificado no arquivo: {pdf_file.name}")
            continue

        # Semestre (Ex: 2º_sem, 2ª_Chamada_2°_sem)
        # Se contiver "2°", "2º", "2o" ou "2sem", assume 2º Semestre. Caso contrário, 1º.
        # Mas atenção: alguns arquivos de 2ª chamada podem ser do 1º semestre. 
        # O padrão parece ser: se tem "2°_sem" ou similar, é 2º. Se não, é 1º?
        # Vamos refinar lendo o cabeçalho.
        
        filename_lower = pdf_file.name.lower()
        if "2°_sem" in filename_lower or "2º_sem" in filename_lower or "2sem" in filename_lower or "2o_sem" in filename_lower:
            semestre_preliminar = "2º"
        else:
            semestre_preliminar = "1º"
            
        print(f"  > Triênio: {trienio} | Semestre (Preliminar): {semestre_preliminar}")

        # 2. Leitura do PDF
        with pdfplumber.open(pdf_file) as pdf:
            semestre_confirmado = None
            
            for page in pdf.pages:
                text = page.extract_text()
                
                # Tenta confirmar semestre no cabeçalho da primeira página
                if semestre_confirmado is None:
                    if "2º SEMESTRE" in text.upper() or "SEGUNDO SEMESTRE" in text.upper():
                        semestre_confirmado = "2º"
                    elif "1º SEMESTRE" in text.upper() or "PRIMEIRO SEMESTRE" in text.upper():
                        semestre_confirmado = "1º"
                
                # Extração das tabelas
                # A estrutura típica é: Inscrição, Nome, Arg. Final, Sistema, Subsistema...
                tables = page.extract_table()
                
                if tables:
                    for row in tables:
                        # Limpa linhas vazias ou cabeçalhos repetidos
                        if not row or row[0] == "Inscrição" or row[0] is None:
                            continue
                            
                        # Heurística para identificar linhas de dados de alunos
                        # Geralmente: [Inscricao, Nome, Argumento, Sistema, ...]
                        # Mas o layout pode variar. Vamos tentar identificar pelo padrão.
                        
                        # Remove None e limpa espaços
                        clean_row = [str(c).strip() if c else "" for c in row]
                        
                        # Verifica se é uma linha de curso (Ex: "Campus Darcy Ribeiro...")
                        # Se for, atualiza o curso atual.
                        # Mas o pdfplumber as vezes quebra tabelas. 
                        # Vamos assumir que a linha de curso tem apenas 1 coluna preenchida ou padrão específico?
                        # Melhor: Tentar identificar o padrão de linha de ALUNO.
                        # Aluno tem Inscrição (número), Nome (texto), Argumento (float), Sistema (int/str)
                        
                        # Exemplo de linha de aluno:
                        # ['21105533', 'Alessandra Viana', '55.123', 'Sistema Universal', '1', ...]
                        
                        # Padrão Regex para Inscrição (8 dígitos costuma ser)
                        # Mas vamos ser flexíveis.
                        
                        # Problema: O nome do curso geralmente vem em uma linha separada ANTES dos alunos.
                        # Precisamos rastrear o "Curso Atual".
                        
                        pass # A lógica de extração linha a linha é complexa com pdfplumber direto em tabelas mistas.
                        # Vamos tentar extrair linhas de texto com layout=True para pegar o curso?
                        
                        # ABORDAGEM HÍBRIDA MAIS ROBUSTA:
                        # Identificar linhas que começam com texto de curso (Ex: "Curso:")
                        # Identificar linhas de aluno.
            
            # REINICIANDO LÓGICA DE LEITURA PARA SER MAIS ROBUSTA COM TEXTO + REGEX
            # Tabelas falham se o cabeçalho não for detectado.
            
            current_course = None
            current_semester = semestre_confirmado if semestre_confirmado else semestre_preliminar
            
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            
            # Divide em linhas
            lines = full_text.split('\n')
            
            for line in lines:
                # 1. Identificação de Curso (Geralmente em caixa alta, ou contém "DIURNO"/"NOTURNO")
                # Ex: "MEDICINA (BACHARELADO) - DIURNO"
                # Ex: "ENGENHARIA DE SOFTWARE (BACHARELADO) - DIURNO"
                # Padrão: Começa com texto, contém turno, não tem números no início (inscricao)
                
                # É difícil garantir que é um curso só pelo texto.
                # Mas no arquivo do PAS, os cursos aparecem como cabeçalhos de seção.
                
                # Vamos tentar identificar linhas de ALUNO primeiro.
                # Padrão: Num Inscrição (aprox 8 digitos) + Nome + Nota + Sistema
                
                # Regex para linha de aluno (aproximado):
                # ^(\d{7,9})\s+(.+?)\s+([-\d\.,]+)\s+(\d+)\s+(\d+)$  <- Formato tabular simples?
                # Na verdade, Argumento Final pode ser negativo.
                
                # Formato esperado: INSCRIÇÃO NOME ARGUMENTO SISTEMA
                # Ex: 17000001 NOME DO ALUNO 123.456 1
                
                # Melhor estratégia para "Curso": A linha anterior ao primeiro aluno de um bloco?
                # Ou procurar linhas que contêm "(BACHARELADO)" ou "(LICENCIATURA)"?
                if "(BACHARELADO)" in line or "(LICENCIATURA)" in line:
                    # Candidato a ser curso.
                    # Remove "Campus..." se tiver?
                    # O curso costuma ser a linha toda.
                    clean_line = line.strip()
                    # Ignorar se for cabeçalho de página
                    if "SISTEMA" not in clean_line and "ARGUMENTO" not in clean_line:
                        current_course = clean_line
                        # print(f"    Curso detectado: {current_course}")
                        continue

                # Identificação de Aluno
                # Tenta capturar: Inscricao, Nome, Argumento, Sistema
                # Regex robusto:
                # \d+ -> Inscrição
                # .+ -> Nome
                # -?\d+[.,]\d+ -> Argumento (pode ser negativo, decimal com . ou ,)
                # \d+ -> Sistema (Universa = 1)
                
                # Exemplo: 23100123 JOAO DA SILVA 50.125 1
                
                student_match = re.search(r"^(\d{5,10})\s+(.+?)\s+(-?\d+[\.,]\d+)\s+(\d+)$", line.strip())
                
                # As vezes o sistema não é o ultimo campo. Vamos tentar pegar o argumento como float.
                # O Sistema Universal é o código "1".
                
                if student_match:
                    inscricao = student_match.group(1)
                    nome = student_match.group(2)
                    arg_str = student_match.group(3).replace(',', '.')
                    sistema = student_match.group(4)
                    
                    try:
                        arg_final = float(arg_str)
                    except:
                        continue
                        
                    # CRITÉRIO CRUCIAL: SISTEMA 1 (UNIVERSAL)
                    if sistema == '1':
                        if current_course:
                            all_students.append({
                                'Trienio': trienio,
                                'Semestre': current_semester,
                                'Curso': current_course,
                                'Argumento': arg_final
                            })
                        else:
                            # Se achou aluno mas não curso, algo está errado na ordem de leitura
                            pass 

    # 3. Consolidação e Cálculos
    df_raw = pd.DataFrame(all_students)
    
    if df_raw.empty:
        print("Nenhum dado extraído do Sistema Universal (1). Verifique os arquivos.")
        return

    print(f"\nTotal de registros extraídos (Sistema Universal): {len(df_raw)}")
    
    # Agrupamento
    # Trienio, Semestre, Curso -> Min, Max, Media, N
    
    summary = df_raw.groupby(['Trienio', 'Semestre', 'Curso'])['Argumento'].agg(
        Min='min',
        Max='max',
        Media='mean',
        N='count'
    ).reset_index()
    
    # Arredondamento
    summary['Min'] = summary['Min'].round(3)
    summary['Max'] = summary['Max'].round(3)
    summary['Media'] = summary['Media'].round(3)
    
    # 4. Teste de Sanidade (Medicina)
    print("\n--- TESTE DE SANIDADE (MEDICINA) ---")
    medicina_recente = summary[summary['Curso'].str.contains("MEDICINA", case=False)]
    if not medicina_recente.empty:
        print(medicina_recente.sort_values('Trienio', ascending=False).head(5))
    else:
        print("ALERTA: Medicina não encontrada nos dados processados!")

    # 5. Exportação
    output_file = Path(data_dir) / "notas_corte_pas_final_BLINDADO.csv"
    summary.to_csv(output_file, index=False)
    print(f"\nArquivo consolidado gerado: {output_file}")
    
    # Gera arquivos separados por semestre para compatibilidade com app atual (se necessário)
    # Mas o usuário pediu "CSV Único". Vamos manter o único.

if __name__ == "__main__":
    extract_cutoff_scores()
