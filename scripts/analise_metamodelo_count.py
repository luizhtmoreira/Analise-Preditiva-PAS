import pandas as pd
import joblib
import numpy as np
from collections import Counter
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print('Loading models...')
model_path = 'models/meta_model.joblib'
scaler_path = 'models/meta_scaler.joblib'
meta_model = joblib.load(model_path)
meta_scaler = joblib.load(scaler_path)
print('Models loaded.')

print('Loading data...')
df = pd.read_csv('data/banco_alunos_pas_final.csv', engine='python', on_bad_lines='skip', encoding='utf-8')
print('Data loaded.')

# Filter out students with Arg_Final == 0 (triennium 2024-2026)
initial_count = len(df)
df['Arg_Final_Num'] = pd.to_numeric(df['Arg_Final'], errors='coerce')
df = df[df['Arg_Final_Num'] != 0.0].copy()
print(f'Filtered data: {len(df)} rows remaining (removed {initial_count - len(df)} rows with Arg_Final == 0).')

print('Preparing features...')
df['Red_PAS1'] = pd.to_numeric(df['Red_PAS1'], errors='coerce').fillna(0)
df['Red_PAS2'] = pd.to_numeric(df['Red_PAS2'], errors='coerce').fillna(0)
df['P1_PAS1'] = pd.to_numeric(df['P1_PAS1'], errors='coerce').fillna(0)
df['P2_PAS1'] = pd.to_numeric(df['P2_PAS1'], errors='coerce').fillna(0)
df['P1_PAS2'] = pd.to_numeric(df['P1_PAS2'], errors='coerce').fillna(0)
df['P2_PAS2'] = pd.to_numeric(df['P2_PAS2'], errors='coerce').fillna(0)

df['EB_PAS1'] = df['P1_PAS1'] + df['P2_PAS1']
df['EB_PAS2'] = df['P1_PAS2'] + df['P2_PAS2']
df['Cresc_EB'] = df['EB_PAS2'] - df['EB_PAS1']
df['Cresc_Red'] = df['Red_PAS2'] - df['Red_PAS1']

df['Taxa_Rel_EB'] = abs(df['Cresc_EB']) / (abs(df['EB_PAS1']) + 0.01)
df['Taxa_Rel_Red'] = abs(df['Cresc_Red']) / (abs(df['Red_PAS1']) + 0.01)
df['Media_EB'] = (df['EB_PAS1'] + df['EB_PAS2']) / 2

# Efficient vectorization instead of apply
df['Tendencia_EB'] = np.where(df['Cresc_EB'] > 0, 1, np.where(df['Cresc_EB'] < 0, -1, 0))
print('Features prepared.')

features = [
    'EB_PAS1', 'Red_PAS1', 'EB_PAS2', 'Red_PAS2',
    'Cresc_EB', 'Cresc_Red', 'Taxa_Rel_EB', 'Taxa_Rel_Red',
    'Media_EB', 'Tendencia_EB'
]

X_meta = df[features].values
X_meta_scaled = meta_scaler.transform(X_meta)
print('Data scaled.')

print('Generating predictions...')
predictions = meta_model.predict(X_meta_scaled)
print('Predictions completed.')

LABEL_TO_MODEL = {0: 'LightGBM', 1: 'Random Forest', 2: 'Regressão Linear', 3: 'MLP'}
mapped_predictions = [LABEL_TO_MODEL.get(p, 'Desconhecido') for p in predictions]

counts = Counter(mapped_predictions)
print('\n=== Contagem de Modelos Favoritos (excluindo Arg_Final = 0) ===')
for model, count in counts.items():
    print(f'{model}: {count} alunos')
print('=====================================')

# Generate a bar chart
models = list(counts.keys())
frequencies = list(counts.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(models, frequencies, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
plt.title('Frequência de Seleção dos Modelos Base pelo Metamodelo (Excluindo Arg_Final=0)')
plt.xlabel('Modelo Base')
plt.ylabel('Número de Alunos')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(frequencies) * 0.01), int(yval), ha='center', va='bottom', fontsize=10)

chart_path = 'metamodelo_model_counts.png'
plt.tight_layout()
plt.savefig(chart_path)
print(f'Grafico salvo como {chart_path}')
