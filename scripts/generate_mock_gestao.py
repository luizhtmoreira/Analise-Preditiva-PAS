"""
Mock data generator for testing the "Gestão de Ativos" page.
Creates a realistic student dataset with Unidade, Turma, and Curso_Alvo columns.
"""
import pandas as pd
import numpy as np

def generate_mock_gestao_ativos(n: int = 40) -> pd.DataFrame:
    """Generates mock student data for the Gestão de Ativos page testing."""
    np.random.seed(42)
    
    nomes = [
        "Ana Silva", "Bruno Costa", "Carla Mendes", "Daniel Souza", "Elena Ferreira",
        "Felipe Santos", "Gabriela Lima", "Henrique Oliveira", "Isabela Rocha", "João Pereira",
        "Karine Almeida", "Lucas Ribeiro", "Marina Araújo", "Nicolas Barbosa", "Olívia Cardoso",
        "Pedro Martins", "Renata Gomes", "Samuel Nascimento", "Tatiana Lopes", "Vinícius Correia",
        "Alice Moreira", "Bernardo Castro", "Camila Duarte", "Diego Pinto", "Elisa Monteiro",
        "Fábio Ramos", "Giovanna Teixeira", "Hugo Nunes", "Inês Dias", "Jorge Campos",
        "Larissa Melo", "Matheus Farias", "Natália Cruz", "Otávio Vieira", "Patrícia Cunha",
        "Rafael Borges", "Sofia Carvalho", "Thiago Freitas", "Vanessa Machado", "William Azevedo"
    ]
    
    unidades = np.random.choice(["Asa Sul", "Taguatinga", "Lago Sul"], n, p=[0.4, 0.35, 0.25])
    turmas = np.random.choice(["3º A", "3º B", "3º C"], n, p=[0.35, 0.35, 0.30])
    
    cursos_alvo = np.random.choice([
        "MEDICINA (BACHARELADO)", "DIREITO (BACHARELADO)", "ENGENHARIA CIVIL (BACHARELADO)", 
        "ADMINISTRAÇÃO (BACHARELADO)", "PSICOLOGIA (BACHARELADO)", "CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)",
        "ARQUITETURA E URBANISMO (BACHARELADO)", "ENFERMAGEM (BACHARELADO)", 
        "FARMÁCIA (BACHARELADO)", "ODONTOLOGIA (BACHARELADO)"
    ], n, p=[0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05])
    
    # Notas realistas (ranges baseados em dados reais)
    p1_pas1 = np.random.uniform(-5, 15, n)
    p2_pas1 = np.random.uniform(-10, 55, n)
    red_pas1 = np.random.uniform(2, 10, n)
    
    tendencia = np.random.choice([-1, 0, 1], n, p=[0.25, 0.35, 0.40])
    variacao = np.random.uniform(2, 8, n) * tendencia
    
    p1_pas2 = np.clip(p1_pas1 + variacao * 0.15, -10, 18)
    p2_pas2 = np.clip(p2_pas1 + variacao, -20, 70)
    red_pas2 = np.clip(red_pas1 + np.random.uniform(-1, 2, n), 0, 10)
    
    df = pd.DataFrame({
        'Inscricao': [f"2024{i:04d}" for i in range(n)],
        'Nome': nomes[:n],
        'Unidade': unidades,
        'Turma': turmas,
        'Curso_Alvo': cursos_alvo,
        'P1_PAS1': p1_pas1.round(3),
        'P2_PAS1': p2_pas1.round(3),
        'Red_PAS1': red_pas1.round(3),
        'P1_PAS2': p1_pas2.round(3),
        'P2_PAS2': p2_pas2.round(3),
        'Red_PAS2': red_pas2.round(3),
    })
    
    return df


if __name__ == "__main__":
    df = generate_mock_gestao_ativos()
    output_path = "data/mock_gestao_ativos.csv"
    df.to_csv(output_path, index=False)
    print(f"Mock data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df.head()}")
