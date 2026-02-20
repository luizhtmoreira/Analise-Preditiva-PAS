import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_path = "models/meta_model.joblib"

model = joblib.load(model_path)
print(f"Model type: {type(model)}")

# Features explicitly mapped from the training pipeline used in streamlit_app.py
features = [
    "Escore Bruto PAS 1",
    "Redação PAS 1",
    "Escore Bruto PAS 2",
    "Redação PAS 2",
    "Crescimento (Delta) Escore Bruto",
    "Crescimento (Delta) Redação",
    "Taxa Relativa Crescimento EB",
    "Taxa Relativa Crescimento Redação",
    "Média Histórica EB",
    "Tendência Direcional EB (-1, 0, 1)"
]

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    plt.figure(figsize=(10, 6))
    
    # Custom colors: green if importance > 0.1, lightblue otherwise
    colors = ['#2ca02c' if val > 0.1 else '#1f77b4' for val in sorted_importances]
    
    bars = plt.barh(sorted_features, sorted_importances, color=colors)
    plt.title('Feature Importance do Metamodelo (Random Forest)\nComo o sistema escolhe o melhor modelo para o aluno', fontsize=14, pad=15)
    plt.xlabel('Importância Relativa na Decisão do Metamodelo', fontsize=12)
    plt.ylabel('Features / Variáveis do Aluno', fontsize=12)
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 ha='left', va='center', fontsize=10)

    # Adjust limits to fit the labels
    plt.xlim(0, max(importances) + 0.03)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('metamodelo_feature_importance.png', dpi=300)
    print("Grafico salvo como metamodelo_feature_importance.png")
else:
    print("O metamodelo carregado não possui 'feature_importances_'. Ele pode ser linear ou MLP.")
