import joblib
import os
import numpy as np

scalers = [
    'meta_scaler.joblib',
    'scaler.joblib',
    'scaler_v1.joblib'
]

models = [
    'meta_model.joblib',
    'modelo_arg_final.joblib',
    'modelo_lgbm.joblib',
    'modelo_lgbm_v1.joblib',
    'modelo_linear.joblib',
    'modelo_mlp.joblib',
    'modelo_rf.joblib',
    'p1_pas3_model.joblib',
    'red_pas3_model.joblib'
]

print("=== DOSSIÊ TÉCNICO ===\n")
print("## 1. Scalers\n")

for s in scalers:
    try:
        path = os.path.join('models', s)
        obj = joblib.load(path)
        print(f"### {s}")
        print(f"- **Tipo:** `{type(obj).__name__}`")
        
        attrs = []
        if hasattr(obj, 'mean_'): attrs.append('mean_')
        if hasattr(obj, 'scale_'): attrs.append('scale_')
        if hasattr(obj, 'var_'): attrs.append('var_')
        if hasattr(obj, 'data_min_'): attrs.append('data_min_')
        if hasattr(obj, 'data_max_'): attrs.append('data_max_')
        
        print(f"- **Atributos Principais:** {', '.join(attrs) if attrs else 'Nenhum encontrado'}")
        
        n_features = "Desconhecido"
        if hasattr(obj, 'n_features_in_'):
            n_features = obj.n_features_in_
        elif hasattr(obj, 'mean_') and obj.mean_ is not None:
            n_features = len(obj.mean_)
            
        print(f"- **Número de Features (N):** {n_features}\n")
    except Exception as e:
        print(f"Erro ao ler {s}: {e}\n")


print("## 2. Modelos Base\n")

key_params = ['max_depth', 'n_estimators', 'hidden_layer_sizes', 'learning_rate', 
              'C', 'penalty', 'alpha', 'num_leaves', 'max_iter', 'fit_intercept',
              'n_jobs', 'random_state', 'activation', 'solver']

for m in models:
    try:
        path = os.path.join('models', m)
        obj = joblib.load(path)
        print(f"### {m}")
        print(f"- **Algoritmo/Classe:** `{type(obj).__name__}`")
        
        params = obj.get_params()
        # Find intersection of key_params and available params
        model_params = {k: v for k, v in params.items() if k in key_params and v is not None}
        # If we couldn't find matches, just pick first 3-5
        if not model_params:
            for k in list(params.keys())[:5]:
                model_params[k] = params[k]
                
        print("- **Principais Hiperparâmetros:**")
        for k, v in model_params.items():
            print(f"  - `{k}`: {v}")
            
        # Importance
        print("- **Variáveis com maior peso/importância:**")
        if hasattr(obj, 'feature_importances_'):
            importances = obj.feature_importances_
            if hasattr(obj, 'feature_name_'):
                # LGBM stores feature names
                features = obj.feature_name_
            elif hasattr(obj, 'feature_names_in_'):
                features = obj.feature_names_in_
            else:
                features = [f"Feature_{i}" for i in range(len(importances))]
            
            # top 3
            indices = np.argsort(importances)[::-1][:3]
            for i in indices:
                print(f"  - `{features[i]}`: {importances[i]:.4f}")
        elif hasattr(obj, 'coef_'):
            coefs = obj.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0] # Take first class or row if it's 2D
                
            if hasattr(obj, 'feature_names_in_'):
                features = obj.feature_names_in_
            else:
                features = [f"Feature_{i}" for i in range(len(coefs))]
                
            indices = np.argsort(np.abs(coefs))[::-1][:3]
            for i in indices:
                print(f"  - `{features[i]}`: {np.abs(coefs[i]):.4f} (coef absoluto)")
        else:
            print("  - *Modelo não suporta ou não expõe feature_importances_ / coef_ diretamente.*")
        
        print("\n")
    except Exception as e:
        print(f"Erro ao ler {m}: {e}\n")
