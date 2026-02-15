"""
PAS Intelligence - Dashboard Streamlit

Dashboard para coordenadores pedagógicos interagirem com o sistema de
inteligência de dados PAS/UnB.

Funcionalidades:
1. Upload de CSV com dados da turma
2. Visualização de "Top Riscos" (Semáforo Vermelho/Amarelo/Verde)
3. Preditor de notas PAS 3 usando modelo LightGBM treinado
4. Comparação de grupos (Teste A/B)

Execução:
    python -m streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Adiciona src ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US')
    except:
        pass

import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from typing import Optional, Tuple # type: ignore
import joblib # type: ignore
import unicodedata


# Imports do pacote pas_intelligence
try:
    from pas_intelligence.ab_testing import compare_groups # type: ignore
    from pas_intelligence.argument_calculator import (
        HistoricalStats,
        calculate_argument_final,
        calculate_argument_etapa,
    )
    import pdf_generator
    import importlib
    importlib.reload(pdf_generator)
    from pdf_generator import PDFGenerator
except ImportError as e:
    st.error(f"⚠️ Módulo pas_intelligence não encontrado: {e}")
    st.stop()

# Import adicional para calculadora de meta
try:
    import importlib
    import pas_intelligence.target_calculator
    importlib.reload(pas_intelligence.target_calculator)
    from pas_intelligence.target_calculator import TargetCalculator # type: ignore
    
    # Import para estatísticas avançadas
    import pas_intelligence.statistics
    importlib.reload(pas_intelligence.statistics)
    from pas_intelligence.statistics import (
        calculate_approval_probability,
        calculate_cohort_evolution_probability
    )
except ImportError:
    TargetCalculator = None  # type: ignore
    calculate_approval_probability = None # type: ignore
    calculate_cohort_evolution_probability = None # type: ignore

# Estatísticas históricas projetadas para cálculo do Argumento Final
# Valores baseados em estatísticas típicas do PAS (triênio 2022-2024)
STATS_PAS1 = HistoricalStats(
    mean_p1=3.5, std_p1=2.5,
    mean_p2=20.0, std_p2=10.0,
    mean_red=5.5, std_red=2.0,
)
STATS_PAS2 = HistoricalStats(
    mean_p1=4.0, std_p1=2.5,
    mean_p2=22.0, std_p2=10.5,
    mean_red=6.5, std_red=2.0,
)
STATS_PAS3 = HistoricalStats(
    mean_p1=4.0, std_p1=2.0,
    mean_p2=25.0, std_p2=12.0,
    mean_red=6.5, std_red=1.8,
)


# =============================================================================
# ESTATÍSTICAS DE CURSOS PARA RECOMENDAÇÃO
# =============================================================================
ARG_FINAL_MAE = 13.49 # Erro médio do modelo para cálculos de probabilidade

@st.cache_data
def find_best_course_match(input_name, course_list):
    """
    Finds the best match for input_name in course_list using substring matching.
    Returns the official course name if found, otherwise returns inputs name.
    """
    if not input_name or input_name == "nan": return "Não informado"
    
    normalized_input = unicodedata.normalize('NFKD', input_name).encode('ASCII', 'ignore').decode('utf-8').upper()
    
    # 1. Exact match (case insensitive)
    for course in course_list:
        if normalized_input == unicodedata.normalize('NFKD', course).encode('ASCII', 'ignore').decode('utf-8').upper():
            return course
            
    # 2. Substring match
    matches = []
    for course in course_list:
        normalized_course = unicodedata.normalize('NFKD', course).encode('ASCII', 'ignore').decode('utf-8').upper()
        if normalized_input in normalized_course:
            matches.append(course)
    
    # Return the shortest match (likely the most specific/correct root) or the first one
    if matches:
        # Sort by length to prefer concise matches or just pick first
        # Example: "Direito" matches "Direito (Diurno)" and "Direito (Noturno)"
        # Use first for now or specific logic? User asked for "contains"
        return matches[0] 
        
    return input_name
def load_course_stats(semester: int = 1, triennium: Optional[str] = None):
    """
    Carrega estatísticas de nota de corte por curso do triênio especificado.
    Lê de CSVs pré-processados para carregamento instantâneo.
    
    Args:
        semester: 1 para 1º semestre, 2 para 2º semestre
        triennium: String do triênio (ex: "2022-2024"). Se None, usa o mais recente.
    """
    try:
        data_dir = Path(__file__).parent.parent / "data"
        
        # Arquivo de Notas de Corte Final
        csv_path = data_dir / "notas_corte_pas_final_BLINDADO.csv"
        
        if not csv_path.exists():
            st.error(f"⚠️ Arquivo não encontrado: {csv_path}")
            return None
        
        # Carrega CSV encontrado
        stats = pd.read_csv(csv_path)
        
        # Filtra pelo semestre selecionado
        sem_str = "1º" if semester == 1 else "2º"
        stats = stats[stats['Semestre'] == sem_str]

        # Filtra pelo Triênio (Se especificado)
        if triennium:
            stats = stats[stats['Trienio'] == triennium]
        else:
            # Fallback: Pega o triênio mais recente disponível no CSV se não especificado
            if 'Trienio' in stats.columns and not stats.empty:
                recent_triennium = stats['Trienio'].max()
                stats = stats[stats['Trienio'] == recent_triennium]
        

        

        
        # Cria Ranking (Reset Index)
        stats = stats.sort_values('Min', ascending=False).reset_index(drop=True)
        stats.index = stats.index + 1 # Ranking 1-based
        
        return stats
        
    except Exception as e:
        st.error(f"Erro detalhado (load_course_stats): {e}")
        return None


@st.cache_data
def load_cohort_data():
    """Calcula ou carrega dados históricos para análise de coorte."""
    try:
        data_dir = Path(__file__).parent.parent / "data"
        csv_path = data_dir / "banco_alunos_pas_final.csv"
        
        if not csv_path.exists():
            return pd.DataFrame()
            
        # Carrega apenas colunas necessárias para otimizar
        cols_to_load = ['P1_PAS1', 'P2_PAS1', 'P1_PAS2', 'P2_PAS2', 'P1_PAS3', 'P2_PAS3', 'Arg_Final']
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols_to_load)
        
        # Renomeia para o padrão esperado pelo backend de estatísticas
        if 'Arg_Final' in df.columns:
            df = df.rename(columns={'Arg_Final': 'ARG_FINAL_REAL'})
        
        # FILTRO CRÍTICO: Remove alunos sem Argumento Final (como o triênio atual 2023-2025)
        # Isso evita que o "Reality Check" compare com alunos que ainda não terminaram o curso.
        if 'ARG_FINAL_REAL' in df.columns:
            df = df[df['ARG_FINAL_REAL'] != 0]

        # Cria colunas de EB se não existirem
        if 'P1_PAS1' in df.columns and 'P2_PAS1' in df.columns:
            df['EB_PAS1'] = df['P1_PAS1'] + df['P2_PAS1']
        if 'P1_PAS2' in df.columns and 'P2_PAS2' in df.columns:
            df['EB_PAS2'] = df['P1_PAS2'] + df['P2_PAS2']
            
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados de coorte: {e}")
        return pd.DataFrame()


def get_closest_courses(arg_previsto: float, n: int = 5, semester: int = 1, triennium: Optional[str] = None) -> pd.DataFrame:
    """
    Retorna os N cursos com nota de corte mais próxima do argumento previsto.
    
    Args:
        arg_previsto: Argumento final previsto
        n: Número de cursos a retornar
        semester: 1 para 1º semestre, 2 para 2º semestre
        triennium: Triênio de referência
    """
    stats = load_course_stats(semester=semester, triennium=triennium)
    if stats is None or stats.empty:
        return pd.DataFrame()
    

    
    # Calcula diferença absoluta do min para o argumento previsto
    stats = stats.copy()
    stats['Diferenca'] = abs(stats['Min'] - arg_previsto)
    
    # Classifica status
    def get_status(row):
        if arg_previsto >= row['Min'] + 10:
            return '🟢 Seguro'
        elif arg_previsto >= row['Min']:
            return '🟡 Competitivo'
        else:
            return '🔴 Arriscado'
    
    stats['Status'] = stats.apply(get_status, axis=1)
    
    # Ordena por proximidade
    closest = stats.nsmallest(n, 'Diferenca')
    return closest[['Curso', 'Min', 'Max', 'Media', 'N', 'Diferenca', 'Status']]


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="VETOR PAS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    .stMetric { padding: 15px; border-radius: 10px; }
    .risk-high { background-color: #FFCDD2; }
    .risk-medium { background-color: #FFF9C4; }
    .risk-low { background-color: #C8E6C9; }
    .main-header { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #1565C0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CARREGAMENTO DOS MODELOS TREINADOS (ENSEMBLE + META-MODELO)
# =============================================================================

@st.cache_resource
def load_models():
    """Carrega todos os modelos treinados para ensemble e o meta-modelo seletor."""
    
    # Tenta múltiplos caminhos possíveis
    possible_paths = [
        Path(__file__).resolve().parent.parent / "models",  # Relativo ao app/
        Path.cwd() / "models",  # Relativo ao diretório de execução
    ]
    
    models = {
        'lgbm': None,
        'rf': None,
        'linear': None,
        'mlp': None,
    }
    scaler = None
    meta_model = None
    meta_scaler = None
    arg_final_model = None
    
    for models_dir in possible_paths:
        lgbm_path = models_dir / "modelo_lgbm.joblib"
        
        if lgbm_path.exists():
            try:
                models['lgbm'] = joblib.load(lgbm_path)
                
                rf_path = models_dir / "modelo_rf.joblib"
                if rf_path.exists():
                    models['rf'] = joblib.load(rf_path)
                
                linear_path = models_dir / "modelo_linear.joblib"
                if linear_path.exists():
                    models['linear'] = joblib.load(linear_path)
                
                mlp_path = models_dir / "modelo_mlp.joblib"
                if mlp_path.exists():
                    models['mlp'] = joblib.load(mlp_path)
                
                scaler_path = models_dir / "scaler.joblib"
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                
                meta_model_path = models_dir / "meta_model.joblib"
                if meta_model_path.exists():
                    meta_model = joblib.load(meta_model_path)
                
                meta_scaler_path = models_dir / "meta_scaler.joblib"
                if meta_scaler_path.exists():
                    meta_scaler = joblib.load(meta_scaler_path)
                
                arg_final_path = models_dir / "modelo_arg_final.joblib"
                if arg_final_path.exists():
                    arg_final_model = joblib.load(arg_final_path)
                
                return models, scaler, meta_model, meta_scaler, arg_final_model
            except Exception as e:
                print(f"Erro ao carregar modelos de {models_dir}: {e}")
                continue
    
    return models, None, None, None, None


# Carrega modelos no início
MODELS, SCALER, META_MODEL, META_SCALER, ARG_FINAL_MODEL = load_models()

# Mapeamento de labels do meta-modelo
LABEL_TO_MODEL = {0: 'lgbm', 1: 'rf', 2: 'linear', 3: 'mlp'}
MODEL_NAMES = {
    'lgbm': '🚀 LightGBM',
    'rf': '🌲 Random Forest',
    'linear': '📈 Regressão Linear',
    'mlp': '🧠 Rede Neural MLP',
}

# MAE de cada modelo para EB_PAS3 (do notebook)
MODEL_MAE = {
    'lgbm': 6.8123,
    'rf': 6.9965,
    'linear': 6.9371,
    'mlp': 6.8423,
}

# Pesos inversos ao MAE (modelo com menor erro tem mais peso)
total_inverse_mae = sum(1/mae for mae in MODEL_MAE.values())
MODEL_WEIGHTS = {name: (1/mae)/total_inverse_mae for name, mae in MODEL_MAE.items()}

# Função global para formatação numérica consistente (sempre ponto decimal)
def fmt(val, decimals=2):
    """Formata número com ponto decimal, independente do locale."""
    return f"{val:.{decimals}f}".replace(",", ".")





# =============================================================================
# ESTATÍSTICAS POR TRIÊNIO (Régua Histórica)
# =============================================================================

TRIENNIUM_STATS = {
    "2023-2025": {
        "PAS1": HistoricalStats(mean_p1=2.2175, std_p1=2.4766, mean_p2=23.8314, std_p2=12.3387, mean_red=6.0345, std_red=2.4790),
        "PAS2": HistoricalStats(mean_p1=3.1496, std_p1=3.2475, mean_p2=25.3101, std_p2=14.2913, mean_red=6.1569, std_red=2.4728),
        "PAS3": HistoricalStats(mean_p1=3.8200, std_p1=2.1000, mean_p2=33.7400, std_p2=14.5000, mean_red=7.6500, std_red=1.8500), # Projeção
    },
    "2022-2024": {
        "PAS1": HistoricalStats(mean_p1=3.6037, std_p1=3.0053, mean_p2=20.7094, std_p2=13.5819, mean_red=5.8878, std_red=2.7796),
        "PAS2": HistoricalStats(mean_p1=3.7393, std_p1=2.2378, mean_p2=30.3477, std_p2=13.2532, mean_red=6.9370, std_red=1.9723),
        "PAS3": HistoricalStats(mean_p1=3.7679, std_p1=2.1778, mean_p2=32.0862, std_p2=14.1289, mean_red=7.5791, std_red=1.7304),
    },
    "2021-2023": {
        "PAS1": HistoricalStats(mean_p1=4.3730, std_p1=3.2775, mean_p2=21.8058, std_p2=12.4484, mean_red=5.9836, std_red=2.9086),
        "PAS2": HistoricalStats(mean_p1=4.8611, std_p1=2.6549, mean_p2=22.1923, std_p2=11.8326, mean_red=7.5055, std_red=1.6451),
        "PAS3": HistoricalStats(mean_p1=3.8569, std_p1=1.9469, mean_p2=27.2585, std_p2=12.9242, mean_red=6.8934, std_red=1.9844),
    },
    "2020-2022": {
        "PAS1": HistoricalStats(mean_p1=2.3277, std_p1=2.4701, mean_p2=24.7838, std_p2=13.3673, mean_red=5.7425, std_red=2.6371),
        "PAS2": HistoricalStats(mean_p1=3.3276, std_p1=2.1757, mean_p2=25.3493, std_p2=11.9121, mean_red=7.1249, std_red=1.8389),
        "PAS3": HistoricalStats(mean_p1=3.3614, std_p1=1.8490, mean_p2=26.3846, std_p2=13.1469, mean_red=7.4822, std_red=1.7520),
    }
}


# 2. PAS 3: Projeção Dinâmica (O Futuro)
# Agora calculada dinamicamente com base no perfil do triênio do aluno.
# A lógica de seleção está dentro do fluxo da aplicação (abaixo).
STATS_PAS3_TREND = HistoricalStats(
    mean_p1=3.82, std_p1=2.1, 
    mean_p2=33.74, std_p2=14.5, 
    mean_red=7.65, std_red=1.85
)




# =============================================================================
# FUNÇÕES AUXILIARES - CORRIGIDAS
# =============================================================================

def classify_risk(eb_pas1: float, eb_pas2: float) -> Tuple[str, str, str]:
    """
    Classifica o risco de um aluno baseado em seu histórico.
    
    CORREÇÃO: Prioriza a TENDÊNCIA (subida/descida) sobre volatilidade.
    - Nota subindo = BOM (baixo risco)
    - Nota descendo = RUIM (médio/alto risco)
    - Nota muito baixa = sempre alto risco
    """
    # Calcula tendência (variação absoluta e percentual)
    trend = eb_pas2 - eb_pas1
    trend_pct = (trend / eb_pas1 * 100) if eb_pas1 > 0 else 0
    mean_score = (eb_pas1 + eb_pas2) / 2
    
    # === LÓGICA CORRIGIDA ===
    
    # 1. Notas muito baixas = sempre alto risco
    if eb_pas2 < 20:
        return "🔴 Alto Risco", "high", f"Nota PAS2 muito baixa ({eb_pas2:.1f})"
    
    # 2. Queda significativa = alto risco
    if trend < -5:
        return "🔴 Alto Risco", "high", f"Queda de {abs(trend):.1f} pontos"
    
    # 3. Queda moderada = médio risco
    if trend < -2:
        return "🟡 Médio Risco", "medium", f"Queda de {abs(trend):.1f} pontos"
    
    # 4. Nota baixa mesmo estável = médio risco
    if mean_score < 30:
        return "🟡 Médio Risco", "medium", f"Média baixa ({mean_score:.1f})"
    
    # 5. Estável ou subindo = baixo risco
    if trend >= 0:
        if trend > 5:
            return "🟢 Baixo Risco", "low", f"Subiu {trend:.1f} pontos! 📈"
        else:
            return "🟢 Baixo Risco", "low", "Desempenho estável"
    
    # Default: estável
    return "🟢 Baixo Risco", "low", "Desempenho estável"


def predict_eb_pas3(features: np.ndarray) -> float:
    """Prediz o Escore Bruto do PAS 3 usando o modelo LightGBM."""
    if MODELS['lgbm'] is None:
        raise ValueError("Modelo não carregado")
    
    # O modelo foi treinado sem scaler para LightGBM
    prediction = MODELS['lgbm'].predict(features)
    return float(prediction[0])


def load_sample_data(include_pas3: bool = False) -> pd.DataFrame:
    """Carrega dados de exemplo para demonstração."""
    np.random.seed(42)
    n = 30
    
    # Cria dados mais realistas com tendências variadas
    p1_pas1 = np.random.uniform(4, 10, n)
    p2_pas1 = np.random.uniform(15, 40, n)
    
    # Alguns alunos sobem, alguns descem
    tendencia = np.random.choice([-1, 0, 1], n, p=[0.25, 0.35, 0.40])
    variacao = np.random.uniform(3, 10, n) * tendencia
    
    p1_pas2 = np.clip(p1_pas1 + variacao * 0.1, 0, 15)
    p2_pas2 = np.clip(p2_pas1 + variacao, 5, 55)
    
    df_data = {
        'Inscricao': [f"2024{i:04d}" for i in range(n)],
        'Nome': [f"Aluno {i+1}" for i in range(n)],
        'P1_PAS1': p1_pas1.round(2),
        'P2_PAS1': p2_pas1.round(2),
        'Red_PAS1': np.random.uniform(4, 10, n).round(2),
        'P1_PAS2': p1_pas2.round(2),
        'P2_PAS2': p2_pas2.round(2),
        'Red_PAS2': np.random.uniform(5, 10, n).round(2),
        'Turma': np.random.choice(['A', 'B'], n),
    }

    if include_pas3:
        # Gera dados do PAS 3 seguindo a tendência
        tendencia_pas3 = np.random.choice([-1, 0, 1], n, p=[0.2, 0.4, 0.4])
        variacao_pas3 = np.random.uniform(2, 8, n) * tendencia_pas3
        
        p1_pas3 = np.clip(p1_pas2 + variacao_pas3 * 0.1, 0, 15)
        p2_pas3 = np.clip(p2_pas2 + variacao_pas3, 5, 60)
        
        df_data.update({
            'P1_PAS3': p1_pas3.round(2),
            'P2_PAS3': p2_pas3.round(2),
            'Red_PAS3': np.random.uniform(5, 10, n).round(2),
        })
    
    return pd.DataFrame(df_data)


# =============================================================================
# SIDEBAR - NAVEGAÇÃO
# =============================================================================

st.sidebar.markdown("# 🎓 VETOR PAS")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Ir para:",
    ["📊 Análise Temporal", "🚦 Semáforo de Risco", "🔮 Preditor PAS 3", "🏫 Análise da Escola", "📈 Comparação Entre Grupos", "📄 Gerador de PDF"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**🚀 Modelo de Projeção Ativo**
Calculando metas com base na **Tendência de Crescimento (Regressão)** para o PAS 3 e notas reais para o passado.
""")

st.sidebar.markdown("### ℹ️ Sobre")
st.sidebar.markdown("""
Sistema de inteligência para análise 
preditiva do PAS/UnB.

**Versão:** 1.1.0  
**Autor:** Luiz Henrique Tomaz Moreira
""")

# Status do modelo
models_loaded = sum(1 for m in MODELS.values() if m is not None)
if models_loaded > 0:
    st.sidebar.success(f"✅ {models_loaded}/4 modelos carregados")
else:
    st.sidebar.warning("⚠️ Nenhum modelo disponível")


# =============================================================================
# ESTADO DA SESSÃO
# =============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None


# =============================================================================
# PÁGINA 1: UPLOAD & ANÁLISE
# =============================================================================



if page == "📊 Análise Temporal":
    st.markdown('<p class="main-header">📊 Análise Temporal</p>', unsafe_allow_html=True)
    
    # Seletor de Modo de Análise
    analysis_mode = st.radio(
        "Modo de Análise:",
        ["Triênio Atual (Em Andamento)", "Triênios Concluídos (Histórico)"],
        horizontal=True,
        help="Escolha 'Triênio Atual' para turmas que ainda não fizeram o PAS 3. Escolha 'Triênios Concluídos' para analisar o ciclo completo (PAS 1, 2 e 3)."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Faça upload do arquivo da turma (CSV ou Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="O arquivo deve conter colunas: Nome, P1_PAS1, P2_PAS1, Red_PAS1, P1_PAS2, P2_PAS2, Red_PAS2" + 
                 (" e P1_PAS3, P2_PAS3, Red_PAS3" if analysis_mode == "Triênios Concluídos (Histórico)" else "")
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(f"✅ Arquivo carregado: {len(st.session_state.df)} alunos")
            except Exception as e:
                st.error(f"❌ Erro ao ler arquivo: {e}")
    
    with col2:
        if st.button("📥 Usar Dados de Exemplo"):
            # Carrega dados de exemplo baseado no modo selecionado
            include_pas3 = (analysis_mode == "Triênios Concluídos (Histórico)")
            st.session_state.df = load_sample_data(include_pas3=include_pas3)
            st.success("✅ Dados de exemplo carregados!")
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Garante que a coluna 'Turma' seja a última se existir
        if 'Turma' in df.columns:
            cols = [c for c in df.columns if c != 'Turma'] + ['Turma']
            df = df[cols]
        
        st.markdown("### 📋 Prévia dos Dados")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Estatísticas gerais
        st.markdown("### 📈 Estatísticas Gerais")
        
        # Definição de colunas necessárias baseada no modo
        required_cols = ['P1_PAS1', 'P2_PAS1', 'P1_PAS2', 'P2_PAS2']
        if analysis_mode == "Triênios Concluídos (Histórico)":
            required_cols.extend(['P1_PAS3', 'P2_PAS3'])
            
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Colunas faltando para o modo '{analysis_mode}': {', '.join(missing_cols)}")
            st.info("""
            📋 **Colunas necessárias:**
            - P1_PAS1, P2_PAS1, Red_PAS1 (notas do PAS 1)
            - P1_PAS2, P2_PAS2, Red_PAS2 (notas do PAS 2)
            """ + ("- P1_PAS3, P2_PAS3, Red_PAS3 (notas do PAS 3)" if analysis_mode == "Triênios Concluídos (Histórico)" else "") + """
            
            💡 Use **Dados de Exemplo** para testar o sistema.
            """)
        else:
            # Cálculos de Escore Bruto
            df['EB_PAS1'] = df['P1_PAS1'] + df['P2_PAS1']
            df['EB_PAS2'] = df['P1_PAS2'] + df['P2_PAS2']
            
            cols_metrics = st.columns(4 if analysis_mode == "Triênio Atual (Em Andamento)" else 5)
            
            with cols_metrics[0]:
                st.metric("Total de Alunos", len(df))
            with cols_metrics[1]:
                st.metric("Média EB PAS 1", f"{df['EB_PAS1'].mean():.2f}")
            with cols_metrics[2]:
                st.metric("Média EB PAS 2", f"{df['EB_PAS2'].mean():.2f}")
            
            if analysis_mode == "Triênio Atual (Em Andamento)":
                with cols_metrics[3]:
                    trend = df['EB_PAS2'].mean() - df['EB_PAS1'].mean()
                    st.metric("Tendência (P1 → P2)", f"{trend:+.2f}", delta=f"{trend:+.2f}")
                
                # Gráfico de distribuição (Apenas PAS 1 e 2)
                fig = px.histogram(
                    df.melt(value_vars=['EB_PAS1', 'EB_PAS2'], var_name='Etapa', value_name='Escore Bruto'),
                    x='Escore Bruto',
                    color='Etapa',
                    barmode='overlay',
                    title='Distribuição de Escores Brutos (PAS 1 vs PAS 2)',
                    opacity=0.7,
                )
                st.plotly_chart(fig, use_container_width=True)

            else: # Triênios Concluídos
                df['EB_PAS3'] = df['P1_PAS3'] + df['P2_PAS3']
                
                with cols_metrics[3]:
                    st.metric("Média EB PAS 3", f"{df['EB_PAS3'].mean():.2f}")
                
                with cols_metrics[4]:
                    trend_total = df['EB_PAS3'].mean() - df['EB_PAS2'].mean()
                    st.metric("Tendência (P2 → P3)", f"{trend_total:+.2f}", delta=f"{trend_total:+.2f}")
                
                # Gráfico de distribuição (PAS 1, 2 e 3)
                fig = px.histogram(
                    df.melt(value_vars=['EB_PAS1', 'EB_PAS2', 'EB_PAS3'], var_name='Etapa', value_name='Escore Bruto'),
                    x='Escore Bruto',
                    color='Etapa',
                    barmode='overlay',
                    title='Distribuição de Escores Brutos (Ciclo Completo)',
                    opacity=0.6,
                    color_discrete_map={'EB_PAS1': '#EF553B', 'EB_PAS2': '#00CC96', 'EB_PAS3': '#AB63FA'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de Evolução Média
                st.markdown("### 📉 Evolução da Média da Turma")
                means = pd.DataFrame({
                    'Etapa': ['PAS 1', 'PAS 2', 'PAS 3'],
                    'Média': [df['EB_PAS1'].mean(), df['EB_PAS2'].mean(), df['EB_PAS3'].mean()]
                })
                fig_line = px.line(means, x='Etapa', y='Média', markers=True, title='Trajetória de Desempenho (Média)')
                st.plotly_chart(fig_line, use_container_width=True)


# =============================================================================
# PÁGINA 2: SEMÁFORO DE RISCO (CORRIGIDO)
# =============================================================================

elif page == "🚦 Semáforo de Risco":
    st.markdown('<p class="main-header">🚦 Semáforo de Risco</p>', unsafe_allow_html=True)
    
    st.info("""
    📌 **Lógica do Semáforo:**
    - 🔴 **Alto Risco**: Nota muito baixa (<20) OU queda >5 pontos
    - 🟡 **Médio Risco**: Queda moderada (2-5 pontos) OU média baixa (<30)
    - 🟢 **Baixo Risco**: Estável ou subindo
    """)
    
    if st.session_state.df is None:
        st.warning("⚠️ Primeiro faça upload dos dados na página 'Upload & Análise'")
        st.stop()
    
    df = st.session_state.df.copy()
    
    # Verifica colunas necessárias
    required_cols = ['P1_PAS1', 'P2_PAS1', 'P1_PAS2', 'P2_PAS2']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Colunas faltando: {', '.join(missing_cols)}")
        st.info("💡 Faça upload de um arquivo com as colunas P1_PAS1, P2_PAS1, P1_PAS2, P2_PAS2 ou use **Dados de Exemplo**.")
        st.stop()
    
    # Calcula EB se não existir
    if 'EB_PAS1' not in df.columns:
        df['EB_PAS1'] = df['P1_PAS1'] + df['P2_PAS1']
        df['EB_PAS2'] = df['P1_PAS2'] + df['P2_PAS2']
    
    # Calcula tendência
    df['Tendência'] = df['EB_PAS2'] - df['EB_PAS1']
    
    # Classifica risco (LÓGICA CORRIGIDA)
    risk_data = df.apply(
        lambda row: classify_risk(row['EB_PAS1'], row['EB_PAS2']),
        axis=1,
        result_type='expand'
    )
    df['Risco'] = risk_data[0]
    df['Risco_Level'] = risk_data[1]
    df['Motivo'] = risk_data[2]
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    high_risk = (df['Risco_Level'] == 'high').sum() # type: ignore
    medium_risk = (df['Risco_Level'] == 'medium').sum() # type: ignore
    low_risk = (df['Risco_Level'] == 'low').sum() # type: ignore
    
    with col1:
        st.markdown("### 🔴 Alto Risco")
        st.markdown(f"<h1 style='color: #D32F2F;'>{high_risk}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("### 🟡 Médio Risco")
        st.markdown(f"<h1 style='color: #FFA000;'>{medium_risk}</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown("### 🟢 Baixo Risco")
        st.markdown(f"<h1 style='color: #388E3C;'>{low_risk}</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filtro por risco
    risk_filter = st.selectbox(
        "Filtrar por nível de risco:",
        ["Todos", "🔴 Alto Risco", "🟡 Médio Risco", "🟢 Baixo Risco"]
    )
    
    if risk_filter != "Todos":
        df_filtered = df[df['Risco'] == risk_filter]
    else:
        df_filtered = df
    
    # Tabela com tendência
    st.markdown("### 📋 Lista de Alunos")
    
    display_cols = ['Nome', 'EB_PAS1', 'EB_PAS2', 'Tendência', 'Risco', 'Motivo']
    available_cols = [c for c in display_cols if c in df_filtered.columns]
    
    # Ordena: alto risco primeiro
    order = {'high': 0, 'medium': 1, 'low': 2}
    df_filtered['order'] = df_filtered['Risco_Level'].map(order)
    df_sorted = df_filtered.sort_values('order')
    
    st.dataframe(
        df_sorted[available_cols],
        use_container_width=True,
    )


# =============================================================================
# PÁGINA 3: PREDITOR PAS 3 (USANDO MODELO ML)
# =============================================================================

elif page == "🔮 Preditor PAS 3":
    st.markdown('<p class="main-header">🔮 Preditor de Argumento Final</p>', unsafe_allow_html=True)
    
    models_loaded = sum(1 for m in MODELS.values() if m is not None)
    if models_loaded == 0:
        st.error("❌ Nenhum modelo carregado. Verifique se os arquivos .joblib existem em models/")
        st.stop()

    # --- CARREGAMENTO DO BANCO DE DADOS PADRONIZADO ---
    data_dir = Path(__file__).parent.parent / "data"
    ARQUIVO_DADOS = data_dir / "notas_corte_pas_final_BLINDADO.csv"
    
    try:
        @st.cache_data
        def load_data_preditor():
            if not ARQUIVO_DADOS.exists():
                return None
            df = pd.read_csv(ARQUIVO_DADOS)
            df['Min'] = pd.to_numeric(df['Min'], errors='coerce')
            return df
            
        df_notas = load_data_preditor()
        if df_notas is None:
            st.error(f"❌ Arquivo '{ARQUIVO_DADOS.name}' não encontrado na pasta data/.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar banco de dados: {e}")
        st.stop()

    # --- CONFIGURAÇÃO (GLOBAL) ---
    st.markdown("### ⚙️ Configuração do Candidato")
    
    col_sem, col_tri, col_cota = st.columns([1, 1, 2])
    
    with col_sem:
        st.markdown("**📅 Semestre**")
        semester_option = st.radio(
            "Semestre", ["1º Semestre", "2º Semestre"], 
            label_visibility="collapsed", horizontal=True
        )
        semester_db = "1°" if semester_option == "1º Semestre" else "2°"
        semester_int = 1 if semester_option == "1º Semestre" else 2

    with col_tri:
        st.markdown("**🎓 Triênio**")
        ciclo_aluno = st.selectbox(
            "Triênio", list(TRIENNIUM_STATS.keys()), 
            label_visibility="collapsed"
        )
        stats_ciclo = TRIENNIUM_STATS[ciclo_aluno]
        # Lógica de referência (Ano Anterior)
        try:
            start_year, end_year = map(int, ciclo_aluno.split('-'))
            ref_triennium = f"{start_year - 1}-{end_year - 1}"
        except:
            ref_triennium = "2022-2024"

    with col_cota:
        st.markdown("**🏷️ Sistema de Concorrência (Cota)**")
        # Lista de cotas ordenada com Universal no topo
        lista_cotas = sorted(df_notas['Sistema_Nome'].unique().astype(str).tolist())
        if 'Universal' in lista_cotas:
            lista_cotas.insert(0, lista_cotas.pop(lista_cotas.index('Universal')))
        
        cota_selecionada = st.selectbox("Cota", lista_cotas, label_visibility="collapsed")

    st.caption(f"ℹ️ Referência: **{ref_triennium}** | Cota: **{cota_selecionada}**")

    # --- ABAS ---
    tab_diagnostico, tab_estrategia = st.tabs(["🔮 Diagnóstico Realista", "🎯 Calculadora de Estratégia"])

    # =========================================================================
    # ABA 1: DIAGNÓSTICO (ESTILO ORIGINAL RESTAURADO)
    # =========================================================================
    with tab_diagnostico:
        st.markdown("> **Previsão baseada em IA:** Insira suas notas para ver sua projeção.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📝 Notas do PAS 1")
            p1_pas1 = st.number_input("P1 PAS 1 (Língua Estrangeira)", -20.0, 20.0, value=None, step=0.001, key="p1_1", format="%.3f")
            p2_pas1 = st.number_input("P2 PAS 1 (Conhecimentos)", -100.0, 100.0, value=None, step=0.001, key="p2_1", format="%.3f")
            red_pas1 = st.number_input("Redação PAS 1", 0.0, 10.0, value=None, step=0.001, key="r_1", format="%.3f")
        with col2:
            st.markdown("### 📝 Notas do PAS 2")
            p1_pas2 = st.number_input("P1 PAS 2", -20.0, 20.0, value=None, step=0.001, key="p1_2", format="%.3f")
            p2_pas2 = st.number_input("P2 PAS 2", -100.0, 100.0, value=None, step=0.001, key="p2_2", format="%.3f")
            red_pas2 = st.number_input("Redação PAS 2", 0.0, 10.0, value=None, step=0.001, key="r_2", format="%.3f")
        
        missing_data = any(v is None for v in [p1_pas1, p2_pas1, red_pas1, p1_pas2, p2_pas2, red_pas2])
        
        if not missing_data and st.button("🚀 Calcular Projeção", type="primary"):
            try:
                # Cálculo Original
                eb_pas1, eb_pas2 = p1_pas1 + p2_pas1, p1_pas2 + p2_pas2
                cresc_eb, cresc_red = eb_pas2 - eb_pas1, red_pas2 - red_pas1
                
                features = np.array([[eb_pas1, red_pas1, eb_pas2, red_pas2, cresc_eb, cresc_red]])
                features_scaled = SCALER.transform(features) if SCALER else features
                
                # Predições de cada modelo para ensemble
                predictions = {}
                if MODELS['lgbm']: predictions['lgbm'] = float(MODELS['lgbm'].predict(features)[0])
                if MODELS['rf']: predictions['rf'] = float(MODELS['rf'].predict(features)[0])
                if MODELS['linear']: predictions['linear'] = float(MODELS['linear'].predict(features_scaled)[0])
                if MODELS['mlp']: predictions['mlp'] = float(MODELS['mlp'].predict(features_scaled)[0])

                # Determina o melhor modelo para este perfil (Meta-Modelo)
                recommended_model = 'lgbm' # Default
                if META_MODEL and META_SCALER:
                    meta_features = np.array([[
                        eb_pas1, red_pas1, eb_pas2, red_pas2,
                        cresc_eb, cresc_red,
                        abs(cresc_eb)/(abs(eb_pas1)+0.01), abs(cresc_red)/(abs(red_pas1)+0.01),
                        (eb_pas1+eb_pas2)/2, 1 if cresc_eb > 0 else (-1 if cresc_eb < 0 else 0)
                    ]])
                    best_model_label = META_MODEL.predict(META_SCALER.transform(meta_features))[0]
                    recommended_model = LABEL_TO_MODEL.get(best_model_label, 'lgbm')
                
                # Predição do Argumento Final
                arg_final_pred = float(ARG_FINAL_MODEL.predict(features)[0]) if ARG_FINAL_MODEL else 0.0
                
                st.session_state.prediction_results = {
                    'predictions': predictions,
                    'recommended_model': recommended_model,
                    'arg_final_pred': arg_final_pred,
                    'eb_pas1': eb_pas1, 'eb_pas2': eb_pas2,
                    'red_pas1': red_pas1, 'red_pas2': red_pas2,
                    'p1_pas1': p1_pas1, 'p2_pas1': p2_pas1,
                    'p1_pas2': p1_pas2, 'p2_pas2': p2_pas2,
                }
            except Exception as e:
                st.error(f"Erro: {e}")

        # --- EXIBIÇÃO DE RESULTADOS (LAYOUT ORIGINAL) ---
        if 'prediction_results' in st.session_state and st.session_state.prediction_results:
            res = st.session_state.prediction_results
            arg_final_pred = res['arg_final_pred']
            recommended_model = res['recommended_model']
            recommended_eb = res['predictions'].get(recommended_model, 0.0) 
            
            st.markdown("---")
            st.markdown("### 🔢 Previsões do Modelo")
            
            c_eb, c_arg = st.columns(2)
            c_eb.metric("EB PAS 3 Previsto", f"{recommended_eb:.3f}", help=f"Modelo selecionado pelo meta-modelo: {recommended_model.upper()}")
            c_arg.metric("Argumento Final Previsto", f"{arg_final_pred:.3f}", delta=f"± {ARG_FINAL_MAE:.2f}")
            
            st.markdown("---")
            st.markdown("#### 🎛️ Ajuste de Cenário")
            arg_ajustado = st.slider(
                "Simule variações no seu Argumento Final:",
                min_value=float(arg_final_pred - ARG_FINAL_MAE),
                max_value=float(arg_final_pred + ARG_FINAL_MAE),
                value=float(arg_final_pred),
                step=0.1, format="%.3f"
            )
            
            # --- ANÁLISE DE PROBABILIDADE (ORIGINAL + COTA) ---
            st.markdown(f"#### 🎓 Análise de Probabilidade ({semester_option})")
            
            # 1. Filtra Dados pela COTA SELECIONADA
            df_cota_atual = df_notas[
                (df_notas['Trienio'] == ref_triennium) & 
                (df_notas['Semestre'] == semester_db) &
                (df_notas['Sistema_Nome'] == cota_selecionada) &
                (df_notas['Chamada'] == '1ª')
            ].sort_values(['Curso_Limpo', 'Campus', 'Turno'])
            
            # Cria uma lista de objetos/dicionários para o selectbox para garantir unicidade
            # Unimos Nome + Campus + Turno para a chave única
            df_cota_atual['Combo_Nome'] = df_cota_atual['Curso_Limpo'] + " (" + df_cota_atual['Campus'] + " - " + df_cota_atual['Turno'] + ")"
            opcoes_lista = df_cota_atual['Combo_Nome'].tolist()
            
            # Cria dicionário com a última chamada de cada curso
            ultimas_chamadas = {}
            for combo in opcoes_lista:
                curso_info = df_cota_atual[df_cota_atual['Combo_Nome'] == combo].iloc[0]
                df_chamadas = df_notas[
                    (df_notas['Trienio'] == ref_triennium) & 
                    (df_notas['Semestre'] == semester_db) &
                    (df_notas['Curso_Limpo'] == curso_info['Curso_Limpo']) &
                    (df_notas['Campus'] == curso_info['Campus']) &
                    (df_notas['Turno'] == curso_info['Turno']) &
                    (df_notas['Sistema_Nome'] == cota_selecionada)
                ].sort_values('Chamada', ascending=False)
                
                if not df_chamadas.empty:
                    ultimas_chamadas[combo] = {
                        'nota': df_chamadas.iloc[0]['Min'],
                        'chamada': df_chamadas.iloc[0]['Chamada']
                    }
                else:
                    ultimas_chamadas[combo] = {
                        'nota': curso_info['Min'],
                        'chamada': '1ª'
                    }
            
            # Seletor de Curso
            curso_combo_sel = st.selectbox(
                "Selecione um curso de interesse:", 
                ["Selecione..."] + opcoes_lista,
                format_func=lambda x: x if x == "Selecione..." else f"{x} [Corte ({ultimas_chamadas[x]['chamada']}): {ultimas_chamadas[x]['nota']:.3f}]"
            )
            
            if curso_combo_sel != "Selecione...":
                # Extrai os dados do curso selecionado via Combo_Nome
                row_sel = df_cota_atual[df_cota_atual['Combo_Nome'] == curso_combo_sel].iloc[0]
                curso_selecionado = row_sel['Curso_Limpo']
                campus_sel = row_sel['Campus']
                turno_sel = row_sel['Turno']
                
                # Busca a última chamada disponível para este curso
                df_chamadas_curso = df_notas[
                    (df_notas['Trienio'] == ref_triennium) & 
                    (df_notas['Semestre'] == semester_db) &
                    (df_notas['Curso_Limpo'] == curso_selecionado) &
                    (df_notas['Campus'] == campus_sel) &
                    (df_notas['Turno'] == turno_sel) &
                    (df_notas['Sistema_Nome'] == cota_selecionada)
                ].sort_values('Chamada', ascending=False)
                
                if not df_chamadas_curso.empty:
                    ultima_chamada = df_chamadas_curso.iloc[0]
                    nota_corte = ultima_chamada['Min']
                    chamada_ref = ultima_chamada['Chamada']
                else:
                    nota_corte = row_sel['Min']
                    chamada_ref = '1ª'
                
                # CÁLCULO ORIGINAL DE PROBABILIDADE (Mantido!)
                if calculate_approval_probability:
                    prob = calculate_approval_probability(arg_ajustado, nota_corte, rmse=ARG_FINAL_MAE)
                    
                    # CORES ORIGINAIS
                    color = "#4CAF50" if prob >= 0.8 else "#FFC107" if prob >= 0.3 else "#F44336"
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
                        <h2 style="margin:0;">{prob*100:.1f}% de Chance</h2>
                        <p style="margin:5px 0 0 0;">Curso: {curso_selecionado} | Campus: {campus_sel} | Turno: {turno_sel}</p>
                        <p style="font-size: 0.9em;">Cota: {cota_selecionada} | Corte ({chamada_ref}): {nota_corte:.3f} | Simulação: {arg_ajustado:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    color = "#999999"  # Default color if probability can't be calculated
                
                # --- NOVO: HISTÓRICO DE CHAMADAS (O que você pediu) ---
                st.markdown("##### 📉 Histórico de Chamadas (Lista de Espera)")
                # Busca todas as chamadas deste curso/campus/turno/cota
                df_hist = df_notas[
                    (df_notas['Trienio'] == ref_triennium) & 
                    (df_notas['Semestre'] == semester_db) &
                    (df_notas['Curso_Limpo'] == curso_selecionado) &
                    (df_notas['Campus'] == campus_sel) &
                    (df_notas['Turno'] == turno_sel) &
                    (df_notas['Sistema_Nome'] == cota_selecionada)
                ].sort_values('Chamada')
                
                if len(df_hist) > 1:
                    # Exibe tabela limpa
                    st.dataframe(
                        df_hist[['Chamada', 'Campus', 'Turno', 'Min']].rename(columns={'Min': 'Nota de Corte'}).style.format({'Nota de Corte': '{:.3f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.caption("Apenas 1ª chamada registrada para este período.")

            # --- LISTA AUTOMÁTICA (Restaurada e Filtrada pela Cota) ---
            st.markdown(f"#### 🏫 Cursos ao seu alcance (Top 10 no Sistema de Concorrência)")
            
            # Recalcula probabilidades para TODOS os cursos da cota
            if not df_cota_atual.empty and calculate_approval_probability:
                df_cota_atual['Chance %'] = df_cota_atual['Min'].apply(
                    lambda x: calculate_approval_probability(arg_ajustado, x, rmse=ARG_FINAL_MAE) * 100
                )
                
                # Ordena pela maior chance, mas remove os 100% fáceis demais se quiser focar nos "próximos"
                # Na versão original, mostrávamos os mais próximos (distância) ou maior chance. 
                # Vou usar Distância Absoluta para mostrar o "Radar" (o que está perto da nota dele)
                df_cota_atual['Dist'] = abs(df_cota_atual['Min'] - arg_ajustado)
                closest = df_cota_atual.sort_values('Dist').head(10)
                
                st.dataframe(
                    closest[['Curso_Limpo', 'Campus', 'Turno', 'Min', 'Chance %']].rename(columns={'Curso_Limpo': 'Curso', 'Min': 'Corte'}).style.format({'Corte': '{:.3f}', 'Chance %': '{:.1f}%'}),
                    use_container_width=True,
                    hide_index=True
                )

    # =========================================================================
    # ABA 2: CALCULADORA (MANTER ORIGINAL COM FILTRO DE COTA)
    # =========================================================================
    with tab_estrategia:
        st.markdown("> **Engenharia Reversa:** Defina onde quer chegar e descubra quanto precisa tirar.")
        
        if 'prediction_results' in st.session_state and TargetCalculator:
            res = st.session_state.prediction_results
            # Prepara notas (usa valores reais de P1 e P2)
            notas_validas = {
                'P1_PAS1': res['p1_pas1'], 'P2_PAS1': res['p2_pas1'], 'Red_PAS1': res['red_pas1'],
                'P1_PAS2': res['p1_pas2'], 'P2_PAS2': res['p2_pas2'], 'Red_PAS2': res['red_pas2']
            }
            calc = TargetCalculator()
            
            st.markdown(f"### 🎯 Meta ({semester_option} | {cota_selecionada})")
            
            # Filtro para Dropdown (Mesma lógica da aba 1)
            df_estrat = df_notas[
                (df_notas['Trienio'] == ref_triennium) & 
                (df_notas['Semestre'] == semester_db) &
                (df_notas['Sistema_Nome'] == cota_selecionada) &
                (df_notas['Chamada'] == '1ª')
            ].sort_values(['Curso_Limpo', 'Campus', 'Turno'])
            
            if not df_estrat.empty:
                # Dropdown com nomes únicos (Curso + Campus + Turno)
                df_estrat['Combo_Nome'] = df_estrat['Curso_Limpo'] + " (" + df_estrat['Campus'] + " - " + df_estrat['Turno'] + ")"
                opcoes_meta = df_estrat['Combo_Nome'].tolist()
                notas_meta = dict(zip(df_estrat['Combo_Nome'], df_estrat['Min']))
                
                curso_alvo_combo = st.selectbox(
                    "Curso Objetivo:", opcoes_meta, 
                    format_func=lambda x: f"{x} (Corte: {notas_meta[x]:.3f})"
                )
                
                nota_alvo = notas_meta[curso_alvo_combo]

                # --- NOVO: SIMULAÇÃO DE DESEMPENHO (OVERRIDES) ---
                with st.expander("🛠️ Customizar Estimativas (Parte 1 e Redação)", expanded=False):
                    st.caption("Ajuste suas próprias expectativas ou use as projeções automáticas da IA.")
                    
                    # Busca predições baseadas nos modelos (p1_pas3_model e red_pas3_model)
                    previsao_ia = calc.predict_stable_components(notas_validas)
                    p1_ia = float(previsao_ia['p1_pred'])
                    red_ia = float(previsao_ia['red_pred'])
                    metodo_ia = previsao_ia.get('method', 'algoritmo')

                    col_ov1, col_ov2 = st.columns(2)
                    with col_ov1:
                        p1_ov = st.number_input(
                            f"Expectativa P1 (Est. IA: {p1_ia:.2f})", 
                            -20.0, 20.0, p1_ia, 0.5,
                            help=f"Baseado no modelo {metodo_ia.upper()}"
                        )
                    with col_ov2:
                        red_ov = st.number_input(
                            f"Expectativa Redação (Est. IA: {red_ia:.2f})", 
                            0.0, 10.0, red_ia, 0.1,
                            help=f"Baseado no modelo {metodo_ia.upper()}"
                        )
                
                if st.button("🔢 Calcular Caminho", type="primary"):
                    stats_p3 = STATS_PAS3_TREND if ciclo_aluno == "2023-2025" else stats_ciclo["PAS3"]
                    
                    # Usa os overrides do slider
                    result = calc.calculate_required_score(
                        notas_validas, nota_alvo,
                        stats_ciclo["PAS1"], stats_ciclo["PAS2"], stats_p3,
                        p1_override=p1_ov,
                        red_override=red_ov
                    )
                    
                    # Exibição Original
                    cor = "success" if result.status in ['possivel', 'garantido'] else "error"
                    icon = "🎉" if result.status == 'garantido' else ("✅" if result.status == 'possivel' else "⚠️")
                    
                    getattr(st, cor)(f"{icon} {result.mensagem}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P1 PAS 3 (Est.)", f"{result.p1_estimado:.3f}")
                    c2.metric("Redação (Est.)", f"{result.red_estimada:.3f}")
                    c3.metric("P2 NECESSÁRIA", f"{result.p2_necessario:.3f}")
                    
                    # --- REALITY CHECK (COHORTE) ---
                    if calculate_cohort_evolution_probability:
                        try:
                            df_hist_cohort = load_cohort_data()
                            if not df_hist_cohort.empty:
                                aluno_dados = {'eb_pas1': res['eb_pas1'], 'eb_pas2': res['eb_pas2']}
                                # Usa EB PAS 3 calculado (P1 + P2 necessária)
                                eb_pas3_necessario = result.p1_estimado + result.p2_necessario
                                prob_hist, amostra = calculate_cohort_evolution_probability(aluno_dados, eb_pas3_necessario, df_hist_cohort)
                                
                                if amostra > 0:
                                    st.markdown(f"""
                                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; margin-top: 15px;">
                                        <p style="margin:0; color: #31333F; font-weight: bold;">📊 Reality Check (Estatística Real)</p>
                                        <p style="margin:5px 0 0 0; color: #31333F;">
                                            De <b>{amostra}</b> alunos com desempenho similar ao seu no PAS 1 e 2, 
                                            <b>{prob_hist:.1f}%</b> conseguiram alcançar um EB PAS 3 de <b>{eb_pas3_necessario:.2f}</b> ou superior.
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.caption(f"ℹ️ Reality Check: Não encontramos alunos históricos suficientemente parecidos (Amostra: {amostra}).")
                        except Exception as e:
                            st.caption(f"Reality Check indisponível: {e}")
            else:
                st.warning("Sem dados para esta cota.")
        else:
            st.warning("Preencha as notas na aba Diagnóstico primeiro.")


# =============================================================================
# PÁGINA 5: ANÁLISE DA ESCOLA (NOVA)
# =============================================================================

elif page == "🏫 Análise da Escola":
    st.markdown('<p class="main-header">🏫 Análise da Escola vs População Geral</p>', unsafe_allow_html=True)
    
    st.markdown("""
    > **Compare o desempenho dos alunos da sua escola com a média geral do PAS/UnB.**
    > 
    > Faça upload de um arquivo Excel (.xlsx) contendo os **nomes dos alunos** da sua escola.
    """)
    
    # Carrega dataset completo
    @st.cache_data
    def load_pas_data():
        try:
            return pd.read_csv(Path(__file__).parent.parent / "data" / "banco_alunos_pas_final.csv")
        except:
            return pd.read_csv("data/banco_alunos_pas_final.csv")
    
    df_geral = load_pas_data()
    
    # Upload do arquivo da escola
    uploaded_file = st.file_uploader(
        "📤 Upload da lista de alunos da escola (Excel)",
        type=['xlsx', 'xls'],
        help="O arquivo deve ter uma coluna 'Nome' com os nomes dos alunos."
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📥 Usar Exemplo de Escola"):
            example_path = Path(__file__).parent.parent / "data" / "exemplo_escola_1000_alunos.xlsx"
            if not example_path.exists():
                # Fallback para o caso de estar rodando na raiz
                example_path = Path("data/exemplo_escola_1000_alunos.xlsx")
                
            if example_path.exists():
                try:
                    escola_exemplo = pd.read_excel(example_path)
                    st.session_state.escola_df = escola_exemplo
                    st.success("✅ Carregado: 1000 alunos de exemplo")
                except Exception as e:
                    st.error(f"Erro ao ler arquivo de exemplo: {e}")
            else:
                st.error("❌ Arquivo de exemplo não encontrado. Por favor, execute o script 'scripts/generate_sample_school.py' primeiro.")
    
    if uploaded_file is not None:
        try:
            st.session_state.escola_df = pd.read_excel(uploaded_file)
            st.success(f"✅ Arquivo carregado: {len(st.session_state.escola_df)} nomes")
        except Exception as e:
            st.error(f"❌ Erro ao ler arquivo: {e}")
    
    # Processa se houver dados da escola
    if 'escola_df' in st.session_state and st.session_state.escola_df is not None:
        escola_nomes = st.session_state.escola_df
        
        st.markdown("---")
        st.markdown("### 📋 Prévia dos nomes")
        st.dataframe(escola_nomes.head(10), use_container_width=True)
        
        # Seleciona triênio - Ordem inversa (mais recente primeiro)
        trienios = sorted(df_geral['Ano_Trienio'].unique(), reverse=True)
        trienio_sel = st.selectbox(
            "Selecione o triênio para comparação:",
            trienios,
            index=0
        )
        
        df_trienio = df_geral[df_geral['Ano_Trienio'] == trienio_sel]
        
        if st.button("🔍 Analisar Escola vs População", type="primary"):
            # Encontra os nomes na base geral
            if 'Nome' in escola_nomes.columns:
                nomes_escola = escola_nomes['Nome'].str.strip().str.upper()
                df_trienio_upper = df_trienio.copy()
                df_trienio_upper['Nome_Upper'] = df_trienio['Nome'].str.strip().str.upper()
                
                # Match por nome (inclui homônimos)
                df_escola = df_trienio_upper[df_trienio_upper['Nome_Upper'].isin(nomes_escola)]
                
                n_encontrados = len(df_escola)
                n_total = len(escola_nomes)
                
                st.markdown("---")
                st.markdown("### 📊 Resultados da Análise")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nomes enviados", n_total)
                with col2:
                    st.metric("Encontrados no PAS", n_encontrados)
                with col3:
                    taxa = (n_encontrados / n_total * 100) if n_total > 0 else 0
                    st.metric("Taxa de match", f"{taxa:.1f}%")
                
                if n_encontrados < 5:
                    st.warning("⚠️ Poucos alunos encontrados. Verifique se os nomes estão corretos.")
                    st.stop()
                
                # Calcula Escore Bruto por etapa
                df_escola['EB_PAS1'] = df_escola['P1_PAS1'] + df_escola['P2_PAS1']
                df_escola['EB_PAS2'] = df_escola['P1_PAS2'] + df_escola['P2_PAS2']
                df_escola['EB_PAS3'] = df_escola['P1_PAS3'] + df_escola['P2_PAS3']
                df_escola['EB_Total'] = df_escola['EB_PAS1'] + df_escola['EB_PAS2'] + df_escola['EB_PAS3']
                
                df_trienio['EB_PAS1'] = df_trienio['P1_PAS1'] + df_trienio['P2_PAS1']
                df_trienio['EB_PAS2'] = df_trienio['P1_PAS2'] + df_trienio['P2_PAS2']
                df_trienio['EB_PAS3'] = df_trienio['P1_PAS3'] + df_trienio['P2_PAS3']
                df_trienio['EB_Total'] = df_trienio['EB_PAS1'] + df_trienio['EB_PAS2'] + df_trienio['EB_PAS3']
                
                # ======================
                # COMPARAÇÃO DE ESCORE BRUTO POR ETAPA
                # ======================
                st.markdown("---")
                st.markdown("### 📊 Comparação de Escore Bruto por Etapa")
                
                # Médias por etapa
                eb_escola_1 = df_escola['EB_PAS1'].mean()
                eb_escola_2 = df_escola['EB_PAS2'].mean()
                eb_escola_3 = df_escola['EB_PAS3'].mean()
                
                eb_geral_1 = df_trienio['EB_PAS1'].mean()
                eb_geral_2 = df_trienio['EB_PAS2'].mean()
                eb_geral_3 = df_trienio['EB_PAS3'].mean()
                
                # Cards com as médias
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    diff1 = eb_escola_1 - eb_geral_1
                    st.metric(
                        "📘 PAS 1",
                        f"{eb_escola_1:.1f}",
                        delta=f"{diff1:+.1f} vs média geral ({eb_geral_1:.1f})"
                    )
                
                with col2:
                    diff2 = eb_escola_2 - eb_geral_2
                    st.metric(
                        "📗 PAS 2",
                        f"{eb_escola_2:.1f}",
                        delta=f"{diff2:+.1f} vs média geral ({eb_geral_2:.1f})"
                    )
                
                with col3:
                    diff3 = eb_escola_3 - eb_geral_3
                    st.metric(
                        "📙 PAS 3",
                        f"{eb_escola_3:.1f}",
                        delta=f"{diff3:+.1f} vs média geral ({eb_geral_3:.1f})"
                    )
                
                # Gráfico de barras agrupadas
                fig_eb = go.Figure()
                
                fig_eb.add_trace(go.Bar(
                    x=['PAS 1', 'PAS 2', 'PAS 3'],
                    y=[eb_escola_1, eb_escola_2, eb_escola_3],
                    name='Sua Escola',
                    marker_color='#1E88E5',
                    text=[f'{eb_escola_1:.1f}', f'{eb_escola_2:.1f}', f'{eb_escola_3:.1f}'],
                    textposition='outside'
                ))
                
                fig_eb.add_trace(go.Bar(
                    x=['PAS 1', 'PAS 2', 'PAS 3'],
                    y=[eb_geral_1, eb_geral_2, eb_geral_3],
                    name='Média Geral',
                    marker_color='#90A4AE',
                    text=[f'{eb_geral_1:.1f}', f'{eb_geral_2:.1f}', f'{eb_geral_3:.1f}'],
                    textposition='outside'
                ))
                
                fig_eb.update_layout(
                    title="Escore Bruto Médio por Etapa (P1 + P2)",
                    yaxis_title="Escore Bruto",
                    barmode='group',
                    height=400,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02) # type: ignore
                )
                
                st.plotly_chart(fig_eb, use_container_width=True)
                
                # Comparação de Argumento Final
                media_escola = df_escola['Arg_Final'].mean()
                media_geral = df_trienio['Arg_Final'].mean()
                std_escola = df_escola['Arg_Final'].std()
                std_geral = df_trienio['Arg_Final'].std()
                diff = media_escola - media_geral
                
                st.markdown("---")
                st.markdown("### 📈 Comparação de Argumento Final")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🏫 Sua Escola")
                    st.metric("Média", f"{media_escola:.2f}")
                    st.caption(f"n = {n_encontrados}, σ = {std_escola:.2f}")
                    
                with col2:
                    st.markdown("#### 🌍 População Geral")
                    st.metric("Média", f"{media_geral:.2f}")
                    st.caption(f"n = {len(df_trienio)}, σ = {std_geral:.2f}")
                    
                with col3:
                    diff = media_escola - media_geral
                    st.markdown("#### 📊 Diferença")
                    color = "green" if diff > 0 else "red"
                    st.metric("Sua escola está", f"{diff:+.2f}", delta=f"{diff:+.2f}")
                
                # Teste estatístico
                try:
                    result = compare_groups(
                        group_a=df_escola['Arg_Final'].values,
                        group_b=df_trienio['Arg_Final'].values,
                        group_a_name="Sua Escola",
                        group_b_name="População Geral",
                        metric_name="Argumento Final"
                    )
                    
                    st.markdown("---")
                    st.markdown("### 🔬 Análise Estatística")
                    
                    p_val_display = f"{result['p_value']:.4f}" if result['p_value'] >= 0.0001 else "< 0.0001"
                    
                    if result['statistically_significant']:
                        if diff > 0:
                            st.success(f"✅ Sua escola está **estatisticamente acima** da média geral! (p = {p_val_display})")
                        else:
                            st.error(f"⚠️ Sua escola está **estatisticamente abaixo** da média geral. (p = {p_val_display})")
                    else:
                        st.info(f"ℹ️ Não há diferença estatisticamente significativa. (p = {p_val_display})")
                    
                    st.caption(f"Tamanho do efeito (Cohen's d): {result['effect_size']:.2f} - {result['interpretation']}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Não foi possível realizar teste estatístico: {e}")
                
                # ======================
                # VISUALIZAÇÕES DIDÁTICAS
                # ======================
                st.markdown("---")
                st.markdown("### 📊 Resumo Visual")
                
                # 2. Resumo textual didático
                st.markdown("---")
                st.markdown("### 📝 Resumo em Linguagem Simples")
                
                if diff > 0:
                    emoji = "🎉"
                    cor = "green"
                    texto_pos = "ACIMA"
                else:
                    emoji = "📉"
                    cor = "red"
                    texto_pos = "ABAIXO"
                
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {cor};">
                    <h2 style="margin-top: 0;">{emoji} Sua Escola está {abs(diff):.1f} pontos {texto_pos} da média geral</h2>
                    <p style="font-size: 18px;">
                        <strong>O que isso significa?</strong>
                    </p>
                    <ul style="font-size: 16px;">
                        <li>✅ A <strong>média da sua escola</strong> no Argumento Final é <strong>{media_escola:.1f}</strong></li>
                        <li>📊 A <strong>média geral do PAS</strong> (todos os candidatos) é <strong>{media_geral:.1f}</strong></li>
                        <li>📈 Isso representa uma diferença de <strong>{diff:+.1f} pontos</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # 3. Ranking percentual
                percentil_escola = (df_trienio['Arg_Final'] < media_escola).mean() * 100
                
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <h3>🏆 Posição da sua escola</h3>
                    <p style="font-size: 18px;">
                        A média da sua escola supera <strong>{percentil_escola:.0f}%</strong> dos candidatos do PAS.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 4. Histograma com destaque (avançado mas visual)
                st.markdown("---")
                st.markdown("### 📈 Onde sua escola se posiciona")
                
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=df_trienio['Arg_Final'],
                    name='Todos os candidatos',
                    marker_color='#90A4AE',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                # Linha vertical para média da escola
                fig_hist.add_vline(
                    x=media_escola,
                    line_dash="dash",
                    line_color="#1E88E5",
                    line_width=3,
                    annotation_text=f"Sua Escola: {media_escola:.1f}",
                    annotation_position="top"
                )
                
                # Linha vertical para média geral
                fig_hist.add_vline(
                    x=media_geral,
                    line_dash="dot",
                    line_color="#666",
                    line_width=2,
                    annotation_text=f"Média Geral: {media_geral:.1f}",
                    annotation_position="bottom"
                )
                
                fig_hist.update_layout(
                    title="Distribuição dos Argumentos Finais (todos os candidatos)",
                    xaxis_title="Argumento Final",
                    yaxis_title="Quantidade de candidatos",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.caption("""
                📌 **Como ler este gráfico:** As barras mostram quantos candidatos obtiveram cada faixa de nota.
                A **linha azul tracejada** mostra onde está a média da sua escola.
                A **linha cinza pontilhada** mostra a média geral.
                """)
                
                # ============================================
                # HISTOGRAMAS POR ETAPA (PAS 1, 2, 3)
                # ============================================
                st.markdown("---")
                st.markdown("### 📊 Distribuição por Etapa do PAS")
                
                # PAS 1
                fig_pas1 = go.Figure()
                fig_pas1.add_trace(go.Histogram(
                    x=df_trienio['EB_PAS1'],
                    name='Todos os candidatos',
                    marker_color='#90A4AE',
                    opacity=0.7,
                    nbinsx=25
                ))
                fig_pas1.add_vline(x=eb_escola_1, line_dash="dash", line_color="#1E88E5", line_width=3,
                    annotation_text=f"Sua Escola: {eb_escola_1:.1f}", annotation_position="top")
                fig_pas1.add_vline(x=eb_geral_1, line_dash="dot", line_color="#666", line_width=2,
                    annotation_text=f"Média Geral: {eb_geral_1:.1f}", annotation_position="bottom")
                fig_pas1.update_layout(
                    title="📘 Distribuição Escore Bruto - PAS 1",
                    xaxis_title="Escore Bruto (P1 + P2)",
                    yaxis_title="Quantidade de candidatos",
                    showlegend=False, height=350
                )
                st.plotly_chart(fig_pas1, use_container_width=True)
                
                # PAS 2
                fig_pas2 = go.Figure()
                fig_pas2.add_trace(go.Histogram(
                    x=df_trienio['EB_PAS2'],
                    name='Todos os candidatos',
                    marker_color='#90A4AE',
                    opacity=0.7,
                    nbinsx=25
                ))
                fig_pas2.add_vline(x=eb_escola_2, line_dash="dash", line_color="#43A047", line_width=3,
                    annotation_text=f"Sua Escola: {eb_escola_2:.1f}", annotation_position="top")
                fig_pas2.add_vline(x=eb_geral_2, line_dash="dot", line_color="#666", line_width=2,
                    annotation_text=f"Média Geral: {eb_geral_2:.1f}", annotation_position="bottom")
                fig_pas2.update_layout(
                    title="📗 Distribuição Escore Bruto - PAS 2",
                    xaxis_title="Escore Bruto (P1 + P2)",
                    yaxis_title="Quantidade de candidatos",
                    showlegend=False, height=350
                )
                st.plotly_chart(fig_pas2, use_container_width=True)
                
                # PAS 3
                fig_pas3 = go.Figure()
                fig_pas3.add_trace(go.Histogram(
                    x=df_trienio['EB_PAS3'],
                    name='Todos os candidatos',
                    marker_color='#90A4AE',
                    opacity=0.7,
                    nbinsx=25
                ))
                fig_pas3.add_vline(x=eb_escola_3, line_dash="dash", line_color="#FB8C00", line_width=3,
                    annotation_text=f"Sua Escola: {eb_escola_3:.1f}", annotation_position="top")
                fig_pas3.add_vline(x=eb_geral_3, line_dash="dot", line_color="#666", line_width=2,
                    annotation_text=f"Média Geral: {eb_geral_3:.1f}", annotation_position="bottom")
                fig_pas3.update_layout(
                    title="📙 Distribuição Escore Bruto - PAS 3",
                    xaxis_title="Escore Bruto (P1 + P2)",
                    yaxis_title="Quantidade de candidatos",
                    showlegend=False, height=350
                )
                st.plotly_chart(fig_pas3, use_container_width=True)
                
                st.caption("📌 **Como ler:** A linha colorida tracejada mostra a média da sua escola. A linha cinza pontilhada mostra a média geral.")
                
            else:
                st.error("❌ O arquivo não tem uma coluna 'Nome'. Verifique o formato.")


# =============================================================================
# PÁGINA 6: COMPARAÇÃO ENTRE GRUPOS (Teste A/B)
# =============================================================================

elif page == "📈 Comparação Entre Grupos":
    st.markdown('<p class="main-header">📈 Comparação Entre Grupos (Teste A/B)</p>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Primeiro faça upload dos dados na página 'Análise Temporal' ou use Dados de Exemplo.")
        st.stop()
        
    df = st.session_state.df.copy()
    
    # Identifica colunas numéricas e categóricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not num_cols:
        st.error("❌ Não foram encontradas colunas numéricas para comparação.")
        st.stop()
        
    st.markdown("""
    > **Ferramenta de Validação Estatística**: Compare dois grupos de alunos para verificar se existe uma diferença significativa entre eles.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Métrica")
        # Prioriza EB_PAS2 se existir
        def_idx = num_cols.index('EB_PAS2') if 'EB_PAS2' in num_cols else 0
        metric = st.selectbox("Selecione a nota para comparar:", num_cols, index=def_idx)
        
    with col2:
        st.markdown("### 👥 Agrupamento")
        group_col = st.selectbox("Selecione a coluna para dividir os grupos:", cat_cols if cat_cols else ["Manual"])
        
    if group_col != "Manual":
        unique_vals = [str(v) for v in df[group_col].unique() if pd.notna(v)]
        if len(unique_vals) < 2:
            st.warning(f"A coluna '{group_col}' possui apenas um valor ({unique_vals[0] if unique_vals else 'Nenhum'}). Selecione outra coluna ou use seleção manual.")
            st.stop()
            
        col_a, col_b = st.columns(2)
        with col_a:
            val_a = st.selectbox(f"Grupo A ({group_col}):", unique_vals, index=0)
        with col_b:
            val_b = st.selectbox(f"Grupo B ({group_col}):", unique_vals, index=1 if len(unique_vals) > 1 else 0)
            
        group_a = df[df[group_col].astype(str) == val_a][metric].dropna().values
        group_b = df[df[group_col].astype(str) == val_b][metric].dropna().values
        name_a = f"{group_col}: {val_a}"
        name_b = f"{group_col}: {val_b}"
    else:
        st.info("Funcionalidade de seleção manual em desenvolvimento. Por favor, use uma coluna de agrupamento (ex: Turma, Sexo, etc).")
        st.stop()
        
    if st.button("📊 Realizar Teste Estatístico", type="primary"):
        if len(group_a) < 2 or len(group_b) < 2:
            st.error("❌ Grupos insuficientes para teste estatístico (mínimo 2 alunos por grupo).")
        else:
            try:
                result = compare_groups(
                    group_a=group_a,
                    group_b=group_b,
                    group_a_name=name_a,
                    group_b_name=name_b,
                    metric_name=metric
                )
                
                st.markdown("---")
                st.markdown(f"### 🔬 Resultado: {name_a} vs {name_b}")
                
                # Cards de Resumo
                ca, cb, cd = st.columns(3)
                ca.metric(f"Média {val_a}", f"{result['group_a_mean']:.2f}")
                cb.metric(f"Média {val_b}", f"{result['group_b_mean']:.2f}")
                cd.metric("Diferença", f"{result['difference']:+.2f}", delta=f"{result['difference']:+.2f}")
                
                # Interpretação
                if result['statistically_significant']:
                    st.success(f"✅ **Diferença Estatisticamente Significante!**")
                else:
                    st.info(f"ℹ️ **Diferença NÃO Significante.**")
                    
                st.markdown(f"> {result['interpretation']}")
                
                # Detalhes técnicos
                with st.expander("📈 Detalhes Técnicos (Estatística)"):
                    st.write(f"**Valor-p:** {result['p_value']:.4f}")
                    st.write(f"**Estatística t:** {result['t_statistic']:.4f}")
                    st.write(f"**Tamanho do Efeito (Cohen's d):** {result['effect_size']:.2f} ({result['effect_interpretation']})")
                    st.write(f"**Amostras**: nA={result['group_a_n']}, nB={result['group_b_n']}")
                
                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Box(y=group_a, name=name_a, marker_color='#1E88E5'))
                fig.add_trace(go.Box(y=group_b, name=name_b, marker_color='#FB8C00'))
                fig.update_layout(title=f"Comparação de Distribuição - {metric}", yaxis_title=metric)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erro ao realizar teste: {e}")

# =============================================================================
# PÁGINA 7: GERADOR DE PDF
# =============================================================================

elif page == "📄 Gerador de PDF":
    st.markdown('<p class="main-header">📄 Gerador de Relatórios PDF</p>', unsafe_allow_html=True)
    st.markdown("""
    > **Gera um PDF estilizado com sua projeção e metas.**
    > Use a calculadora na aba anterior para estimar seus valores ou preencha manualmente abaixo.
    """)
    
    tab_manual, tab_batch = st.tabs(["✍️ Manual", "📦 Em Lote (Escola)"])
    
    pdf_gen = PDFGenerator()
    
    with tab_manual:
        st.markdown("### Preenchimento Manual (Automático)")
        
        # --- SELEÇÃO DE CURSO (Igual à aba Estratégia) ---
        target_semester = 1 # Definido pelo usuário: 1º Semestre
        ref_triennium_pdf = "2022-2024" # Definido pelo usuário
        
        df_cursos_pdf = load_course_stats(semester=target_semester, triennium=ref_triennium_pdf)
        
        if df_cursos_pdf is not None:
            cursos_lista = df_cursos_pdf['Curso'].unique().tolist()
            course_scores_pdf = dict(zip(df_cursos_pdf['Curso'], df_cursos_pdf['Min']))
            
            def fmt_course_pdf(nome):
                return f"{nome} (Corte: {course_scores_pdf.get(nome, 0):.3f})"
                
            selected_course_name = st.selectbox(
                "Curso Pretendido (Ref. 2022-2024 - 1º Semestre)", 
                cursos_lista,
                format_func=fmt_course_pdf
            )
            nota_corte_val = course_scores_pdf.get(selected_course_name, 0.0)
        else:
            st.error("Erro ao carregar lista de cursos.")
            selected_course_name = ""
            nota_corte_val = 0.0

        with st.form("pdf_manual_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                aluno = st.text_input("Nome do Aluno", "Estudante")
                # curso = st.text_input("Curso Pretendido") # Substituído pelo selectbox acima
                
                st.markdown("#### 📝 Notas PAS 1")
                p1_pas1 = st.number_input("PAS 1 - P1 (Língua)", 0.0, 20.0, 0.0, step=0.001, format="%.3f")
                p2_pas1 = st.number_input("PAS 1 - P2 (Gerais)", 0.0, 100.0, 0.0, step=0.001, format="%.3f")
                red_pas1 = st.number_input("PAS 1 - Redação", 0.0, 10.0, 0.0, step=0.001, format="%.3f")
                
            with col2:
                # Spacer
                st.write("") 
                st.write("")
                
                st.markdown("#### 📝 Notas PAS 2")
                p1_pas2 = st.number_input("PAS 2 - P1 (Língua)", 0.0, 20.0, 0.0, step=0.001, format="%.3f")
                p2_pas2 = st.number_input("PAS 2 - P2 (Gerais)", 0.0, 100.0, 0.0, step=0.001, format="%.3f")
                red_pas2 = st.number_input("PAS 2 - Redação", 0.0, 10.0, 0.0, step=0.001, format="%.3f")
                
            st.info(f"Nota de Corte Selecionada: **{nota_corte_val:.3f}** (Calculada automaticamente)")

            submitted = st.form_submit_button("Gerar PDF 📄", type="primary")
            
        if submitted:
            # 1. Carrega Estatísticas para Cálculo (Usando 2023-2025 como base atual)
            stats_base = TRIENNIUM_STATS["2023-2025"]
            
            # 2. Calcula Argumentos PAS 1 e 2
            arg_pas1 = calculate_argument_etapa(p1_pas1, p2_pas1, red_pas1, stats_base["PAS1"])
            arg_pas2 = calculate_argument_etapa(p1_pas2, p2_pas2, red_pas2, stats_base["PAS2"])
            arg_acumulado = arg_pas1 + 2 * arg_pas2
            
            # 3. Calcula Meta (Reverse Prediction)
            calc = TargetCalculator()
            
            notas_input = {
                'P1_PAS1': p1_pas1, 'P2_PAS1': p2_pas1, 'Red_PAS1': red_pas1,
                'P1_PAS2': p1_pas2, 'P2_PAS2': p2_pas2, 'Red_PAS2': red_pas2,
            }
            
            # Usa projeção de tendência para PAS 3 (ou stats_base["PAS3"] se preferir média histórica)
            stats_pas3_proj = STATS_PAS3_TREND 
            
            result = calc.calculate_required_score(
                notas_input, nota_corte_val,
                stats_base["PAS1"], stats_base["PAS2"], stats_pas3_proj
            )
            
            # 4. Calcula Z-score e Probabilidade para o PDF usando o Modelo ML (Fonte da Verdade)
            eb_p1 = p1_pas1 + p2_pas1
            eb_p2 = p1_pas2 + p2_pas2
            c_eb = eb_p2 - eb_p1
            c_red = red_pas2 - red_pas1
            
            features_manual = np.array([[eb_p1, red_pas1, eb_p2, red_pas2, c_eb, c_red]])
            
            # Predição do Modelo AI
            arg_final_pred_ml = 0.0
            if ARG_FINAL_MODEL:
                arg_final_pred_ml = float(ARG_FINAL_MODEL.predict(features_manual)[0])
            else:
                # Fallback para cálculo manual se modelo não carregar
                arg3_pred_realista = calculate_argument_etapa(result.p1_estimado, stats_pas3_proj.mean_p2, result.red_estimada, stats_pas3_proj)
                arg_final_pred_ml = arg_pas1 + 2*arg_pas2 + 3*arg3_pred_realista

            z_score_val = (arg_final_pred_ml - nota_corte_val) / ARG_FINAL_MAE
            prob_pdf = 0.0
            if calculate_approval_probability:
                prob_pdf = calculate_approval_probability(arg_final_pred_ml, nota_corte_val, rmse=ARG_FINAL_MAE)

            # REALITY CHECK (COHORTE)
            reality_check_str = "-"
            if calculate_cohort_evolution_probability:
                df_hist = load_cohort_data()
                aluno_dados = {'eb_pas1': eb_p1, 'eb_pas2': eb_p2}
                prob_hist, amostra = calculate_cohort_evolution_probability(aluno_dados, nota_corte_val, df_hist)
                if amostra > 0:
                    reality_check_str = f"Em {amostra} alunos: {prob_hist:.1f}% aprovação"
            
            # 5. Prepara Dados para o PDF
            data = {
                'aluno': aluno, 
                'curso': selected_course_name,
                # Notas Brutas
                'pas1_p1': f"{p1_pas1:.3f}", 'pas1_p2': f"{p2_pas1:.3f}", 'pas1_red': f"{red_pas1:.3f}", 
                'pas1_arg': f"{arg_pas1:.3f}",
                'pas2_p1': f"{p1_pas2:.3f}", 'pas2_p2': f"{p2_pas2:.3f}", 'pas2_red': f"{red_pas2:.3f}", 
                'pas2_arg': f"{arg_pas2:.3f}",
                # Argumentos Ponderados e Acumulado
                'arg_pond_1': f"{arg_pas1:.3f}", 
                'arg_pond_2': f"{arg_pas2 * 2:.3f}",
                'arg_acumulado': f"{arg_acumulado:.3f}",
                # PAS 3 (Estimativas e Meta)
                'pas3_p1_est': f"{result.p1_estimado:.3f}", 
                'pas3_red_est': f"{result.red_estimada:.3f}",
                'pas3_p2_necessario': f"{result.p2_necessario:.3f}",
                'nota_corte': f"{nota_corte_val:.3f}",
                'arg_necessario': f"{result.arg_pas3_necessario:.3f}",
                # Estatísticas de Aprovação
                'probabilidade': f"{prob_pdf * 100:.1f}%",
                'z_score': reality_check_str # Substituído pelo Reality Check
            }
            
            try:
                pdf_bytes = pdf_gen.generate_single_pdf(data)
                st.success("PDF Gerado com Sucesso!")
                st.download_button(
                    label="📥 Baixar PDF",
                    data=pdf_bytes,
                    file_name=f"Relatorio_PAS_{aluno.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")

    with tab_batch:
        st.markdown("### Processamento em Lote")
        st.info("Faça upload de uma planilha com colunas: **Nome, Curso, P1_PAS1, P2_PAS1, Red_PAS1, P1_PAS2, P2_PAS2, Red_PAS2**.")
        
        st.markdown("#### ⚙️ Configuração do Lote")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            batch_semester = st.radio("Semestre de Ingresso (para corte)", [1, 2], index=0, horizontal=True, key="batch_sem")
        with col_b2:
            batch_ref_triennium = st.selectbox("Triênio de Referência (Corte)", list(TRIENNIUM_STATS.keys()), index=1, key="batch_tri")

        uploaded_batch = st.file_uploader("Upload de Arquivo de Dados", type=['csv', 'xlsx'])
        
        if uploaded_batch:
            if st.button("Gerar PDFs em Lote"):
                try:
                    df_batch = pd.read_csv(uploaded_batch) if uploaded_batch.name.endswith('.csv') else pd.read_excel(uploaded_batch)
                    
                    # Normaliza colunas para minúsculo
                    df_batch.columns = df_batch.columns.str.lower().str.replace(' ', '_')
                    
                    # Carrega notas de corte
                    df_cursos_ref = load_course_stats(semester=batch_semester, triennium=batch_ref_triennium)
                    
                    if df_cursos_ref is not None:
                        course_map = dict(zip(df_cursos_ref['Curso'], df_cursos_ref['Min']))
                        available_courses = list(course_map.keys())
                    else:
                        course_map = {}
                        available_courses = []
                    
                    # Stats base para cálculo (2023-2025)
                    stats_base = TRIENNIUM_STATS["2023-2025"]
                    calc = TargetCalculator()
                    
                    processed_data = []
                    
                    progress_bar = st.progress(0)
                    total_rows = len(df_batch)
                    
                    for idx, row in df_batch.iterrows():
                        # Extrai dados (com fallbacks seguros)
                        aluno_name = str(row.get('nome', row.get('aluno', 'Estudante')))
                        raw_course_name = str(row.get('curso', 'Não informado'))
                        
                        # --- Fuzzy Match Logic ---
                        # Tenta encontrar o nome oficial do curso
                        official_course_name = find_best_course_match(raw_course_name, available_courses)
                        nota_corte = course_map.get(official_course_name, 0.0)
                        
                        # Feedback no console/UI (opcional, pode poluir se muitos)
                        # if raw_course_name != official_course_name:
                        #    print(f"Mapped '{raw_course_name}' to '{official_course_name}'")

                        # Notas PAS 1
                        p1_1 = float(row.get('p1_pas1', 0))
                        p2_1 = float(row.get('p2_pas1', 0))
                        red_1 = float(row.get('red_pas1', 0))
                        
                        # Notas PAS 2
                        p1_2 = float(row.get('p1_pas2', 0))
                        p2_2 = float(row.get('p2_pas2', 0))
                        red_2 = float(row.get('red_pas2', 0))
                        
                        # Cálculos
                        arg1 = calculate_argument_etapa(p1_1, p2_1, red_1, stats_base["PAS1"])
                        arg2 = calculate_argument_etapa(p1_2, p2_2, red_2, stats_base["PAS2"])
                        arg_acum = arg1 + 2 * arg2
                        
                        # Projeção PAS 3
                        notas_input = {
                            'P1_PAS1': p1_1, 'P2_PAS1': p2_1, 'Red_PAS1': red_1,
                            'P1_PAS2': p1_2, 'P2_PAS2': p2_2, 'Red_PAS2': red_2,
                        }
                        
                        # Usa projeção de tendência para PAS 3
                        result = calc.calculate_required_score(
                            notas_input, nota_corte,
                            stats_base["PAS1"], stats_base["PAS2"], STATS_PAS3_TREND
                        )
                        
                        # --- Estatísticas de Aprovação Usando Modelo AI ---
                        eb_b1 = p1_1 + p2_1
                        eb_b2 = p1_2 + p2_2
                        cb_eb = eb_b2 - eb_b1
                        cb_red = red_2 - red_1
                        
                        feat_b = np.array([[eb_b1, red_1, eb_b2, red_2, cb_eb, cb_red]])
                        
                        arg_final_batch_pred = 0.0
                        if ARG_FINAL_MODEL:
                            arg_final_batch_pred = float(ARG_FINAL_MODEL.predict(feat_b)[0])
                        else:
                            # Fallback
                            arg3_p = calculate_argument_etapa(result.p1_estimado, stats_pas3_proj.mean_p2, result.red_estimada, stats_pas3_proj)
                            arg_final_batch_pred = 1*arg1 + 2*arg2 + 3*arg3_p
                        
                        z_score_batch = (arg_final_batch_pred - nota_corte) / ARG_FINAL_MAE
                        prob_batch = 0.0
                        if calculate_approval_probability:
                            prob_batch = calculate_approval_probability(arg_final_batch_pred, nota_corte, rmse=ARG_FINAL_MAE)

                        # Monta dict final
                        student_data = {
                            'aluno': aluno_name,
                            'curso': official_course_name, # Usa o nome oficial encontrado
                            'pas1_p1': f"{p1_1:.3f}", 'pas1_p2': f"{p2_1:.3f}", 'pas1_red': f"{red_1:.3f}",
                            'pas1_arg': f"{arg1:.3f}",
                            'pas2_p1': f"{p1_2:.3f}", 'pas2_p2': f"{p2_2:.3f}", 'pas2_red': f"{red_2:.3f}",
                            'pas2_arg': f"{arg2:.3f}",
                            'arg_pond_1': f"{arg1:.3f}",
                            'arg_pond_2': f"{arg2*2:.3f}",
                            'arg_acumulado': f"{arg_acum:.3f}",
                            'pas3_p1_est': f"{result.p1_estimado:.3f}",
                            'pas3_red_est': f"{result.red_estimada:.3f}",
                            'pas3_p2_necessario': f"{result.p2_necessario:.3f}",
                            'nota_corte': f"{nota_corte:.3f}",
                            'arg_necessario': f"{result.arg_pas3_necessario:.3f}",
                            'probabilidade': f"{prob_batch * 100:.1f}%",
                            'z_score': f"{z_score_batch:+.2f}"
                        }
                        processed_data.append(student_data)
                        progress_bar.progress((idx + 1) / total_rows)
                    
                    zip_buffer = pdf_gen.generate_batch_zip(processed_data)
                    
                    st.success(f"✅ Processamento concluído: {len(processed_data)} arquivos gerados.")
                    st.download_button(
                        label="📦 Baixar Arquivos (ZIP)",
                        data=zip_buffer,
                        file_name="relatorios_pas_batch.zip",
                        mime="application/zip"
                    )
                    
                except Exception as e:
                    st.error(f"Erro no processamento em lote: {e}")
