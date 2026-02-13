import pandas as pd
import numpy as np

# Data from CSV search
students = [
    {"aluno": "Ana Clara Matos Ricardo", "p1_1": 1.14, "p2_1": 40.612, "red_1": 6.857, "p1_2": 7.317, "p2_2": 49.579, "red_2": 7.525},
    {"aluno": "Ana Luiza Silva Teixeirense Lima", "p1_1": 3.42, "p2_1": 36.623, "red_1": 7.0, "p1_2": 8.943, "p2_2": 41.585, "red_2": 8.704},
    {"aluno": "Camila Alves Geraldo da Silva", "p1_1": 5.415, "p2_1": 27.359, "red_1": 7.463, "p1_2": 1.897, "p2_2": 36.856, "red_2": 7.064},
    {"aluno": "Guilherme Pimenta Rodrigues", "p1_1": 0.0, "p2_1": 0.0, "red_1": 0.0, "p1_2": 2.439, "p2_2": 37.519, "red_2": 5.786},
]

rows = []
for s in students:
    rows.append({
        "Nome": s["aluno"], # Changed to match typical user header or internal name
        "Curso": "", # To be filled by user
        "P1_PAS1": s["p1_1"],
        "P2_PAS1": s["p2_1"],
        "Red_PAS1": s["red_1"],
        "P1_PAS2": s["p1_2"],
        "P2_PAS2": s["p2_2"],
        "Red_PAS2": s["red_2"]
    })

df = pd.DataFrame(rows)
df.to_excel("data/alunos_para_pdf_conferencia.xlsx", index=False)
print("File created: data/alunos_para_pdf_conferencia.xlsx")
