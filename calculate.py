import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

LABEL_TO_MODEL = {0: 'lgbm', 1: 'rf', 2: 'linear', 3: 'mlp'}

df = pd.read_csv('data/banco_alunos_pas_final.csv', low_memory=False)
df_2023 = df[df['Ano_Trienio'] == '2023-2025'].copy()

for col in ['P1_PAS1', 'P2_PAS1', 'Red_PAS1', 'P1_PAS2', 'P2_PAS2', 'Red_PAS2', 'P1_PAS3', 'P2_PAS3', 'Arg_Final']:
    df_2023[col] = pd.to_numeric(df_2023[col], errors='coerce')

df_2023['eb_pas1'] = df_2023['P1_PAS1'].fillna(0) + df_2023['P2_PAS1'].fillna(0)
df_2023['eb_pas2'] = df_2023['P1_PAS2'].fillna(0) + df_2023['P2_PAS2'].fillna(0)
df_2023['c_eb'] = df_2023['eb_pas2'] - df_2023['eb_pas1']
df_2023['c_red'] = df_2023['Red_PAS2'].fillna(0) - df_2023['Red_PAS1'].fillna(0)

df_2023['eb_pas3_real'] = df_2023['P1_PAS3'].fillna(0) + df_2023['P2_PAS3'].fillna(0)
df_2023['arg_final_real'] = df_2023['Arg_Final'].fillna(np.nan)

df_valid_eb = df_2023[df_2023['eb_pas3_real'] > 0].copy()
df_valid_arg = df_2023[pd.notna(df_2023['arg_final_real']) & (df_2023['arg_final_real'] != 0)].copy()

try:
    m_lgbm = joblib.load('models/modelo_lgbm.joblib')
    m_rf = joblib.load('models/modelo_rf.joblib')
    m_linear = joblib.load('models/modelo_linear.joblib')
    m_mlp = joblib.load('models/modelo_mlp.joblib')
    scaler = joblib.load('models/scaler.joblib')
    meta_model = joblib.load('models/meta_model.joblib')
    meta_scaler = joblib.load('models/meta_scaler.joblib')
    arg_model = joblib.load('models/modelo_arg_final.joblib')

    # Vectorized EB PAS3
    eb_p1 = df_valid_eb['eb_pas1'].values
    red_p1 = df_valid_eb['Red_PAS1'].fillna(0).values
    eb_p2 = df_valid_eb['eb_pas2'].values
    red_p2 = df_valid_eb['Red_PAS2'].fillna(0).values
    c_eb = df_valid_eb['c_eb'].values
    c_red = df_valid_eb['c_red'].values

    # features: [eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red]
    features = np.column_stack((eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red))
    features_scaled = scaler.transform(features)

    preds_lgbm = m_lgbm.predict(features)
    preds_rf = m_rf.predict(features)
    preds_linear = m_linear.predict(features_scaled)
    preds_mlp = m_mlp.predict(features_scaled)

    # meta_features: [eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red, abs(c_eb)/(abs(eb_p1)+0.01), abs(c_red)/(abs(red_p1)+0.01), (eb_p1+eb_p2)/2, sign(c_eb)]
    meta_f7 = np.abs(c_eb) / (np.abs(eb_p1) + 0.01)
    meta_f8 = np.abs(c_red) / (np.abs(red_p1) + 0.01)
    meta_f9 = (eb_p1 + eb_p2) / 2.0
    meta_f10 = np.where(c_eb > 0, 1, np.where(c_eb < 0, -1, 0))

    meta_features = np.column_stack((eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red, meta_f7, meta_f8, meta_f9, meta_f10))
    meta_features_scaled = meta_scaler.transform(meta_features)

    best_model_labels = meta_model.predict(meta_features_scaled)
    
    # 0: lgbm, 1: rf, 2: linear, 3: mlp
    all_preds = np.column_stack((preds_lgbm, preds_rf, preds_linear, preds_mlp))
    final_preds_eb = np.array([all_preds[i, best_model_labels[i]] for i in range(len(best_model_labels))])

    mae_eb_ensemble = np.mean(np.abs(final_preds_eb - df_valid_eb['eb_pas3_real'].values))

    # Vectorized Arg Final
    eb_p1_a = df_valid_arg['eb_pas1'].values
    red_p1_a = df_valid_arg['Red_PAS1'].fillna(0).values
    eb_p2_a = df_valid_arg['eb_pas2'].values
    red_p2_a = df_valid_arg['Red_PAS2'].fillna(0).values
    c_eb_a = df_valid_arg['c_eb'].values
    c_red_a = df_valid_arg['c_red'].values

    features_a = np.column_stack((eb_p1_a, red_p1_a, eb_p2_a, red_p2_a, c_eb_a, c_red_a))
    final_preds_arg = arg_model.predict(features_a)

    mae_arg = np.mean(np.abs(final_preds_arg - df_valid_arg['arg_final_real'].values))

    print(f"Alunos válidos (EB): {len(df_valid_eb)}")
    print(f"Alunos válidos (Arg Final): {len(df_valid_arg)}")
    print(f"Erro Médio Absoluto (MAE) Escore Bruto Previsto (Ensemble): {mae_eb_ensemble:.4f}")
    print(f"Erro Médio Absoluto (MAE) Argumento Final Previsto: {mae_arg:.4f}")

except Exception as e:
    import traceback
    traceback.print_exc()
