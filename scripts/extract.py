import pdfplumber
import pandas as pd

pdf_path = "8BF9D771C58F383321E81D054B720A9E75CF911E7A921DF6E017670779B74EEF.pdf"
start_name = "Nicole Gomes do Nascimento"

# Colunas alvo do seu banco
columns = [
    "Inscricao", "Nome", "P1_PAS1", "P2_PAS1", "Red_PAS1", 
    "P1_PAS2", "P2_PAS2", "Red_PAS2", "P1_PAS3", "P2_PAS3", 
    "Red_PAS3", "Arg_Final", "Ano_Trienio"
]

data = []
extracting = False
buffer_text = ""

print("Lendo o PDF...")
with pdfplumber.open(pdf_path) as pdf:
    # Como o corte no HTML foi perto da pág 254, lemos a partir da 250 por segurança
    for i in range(250, len(pdf.pages)):
        page = pdf.pages[i]
        text = page.extract_text()
        if text:
            # Substituir quebras de linha por espaço para unir os registros de alunos cortados ao meio
            text = text.replace('\n', ' ')
            buffer_text += text + " "

# Dividir os alunos usando o separador padrão do Cebraspe
students_raw = buffer_text.split('/')

print("Estruturando os dados...")
for student in students_raw:
    # Separar os dados do aluno por vírgula e limpar os espaços em branco
    parts = [p.strip() for p in student.split(',')]
    
    # Validação: garantir que o registro tem pelo menos Inscrição, Nome e as 5 notas brutas/finais
    if len(parts) >= 7:
        inscricao = parts[0]
        nome = parts[1]
        
        # Ativar a extração apenas quando bater na âncora da Nicole
        if not extracting:
            if start_name.lower() in nome.lower():
                extracting = True
            else:
                continue
                
        # Extrair estritamente as notas do PAS 2
        p1_pas2 = parts[2]
        p2_pas2 = parts[3]
        red_pas2 = parts[6] # O índice 6 é a nota final da prova de redação
        
        data.append({
            "Inscricao": inscricao,
            "Nome": nome,
            "P1_PAS1": "",
            "P2_PAS1": "",
            "Red_PAS1": "",
            "P1_PAS2": p1_pas2,
            "P2_PAS2": p2_pas2,
            "Red_PAS2": red_pas2,
            "P1_PAS3": "",
            "P2_PAS3": "",
            "Red_PAS3": "",
            "Arg_Final": "",
            "Ano_Trienio": "2024-2026"
        })

df = pd.DataFrame(data, columns=columns)
df.to_csv("alunos_faltantes_pas2.csv", index=False, encoding="utf-8")
print(f"Extração concluída com sucesso! {len(df)} alunos processados de 'Nicole' em diante e salvos em 'alunos_faltantes_pas2.csv'.")