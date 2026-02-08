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

# Imports do pacote pas_intelligence
try:
    from pas_intelligence.ab_testing import compare_groups # type: ignore
    from pas_intelligence.argument_calculator import (
        HistoricalStats,
        calculate_argument_final,
    )
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

@st.cache_data
def load_course_stats(semester: int = 1):
    """
    Carrega estatísticas de nota de corte por curso do triênio 2022-2024.
    Lê de CSVs pré-processados para carregamento instantâneo.
    
    Args:
        semester: 1 para 1º semestre, 2 para 2º semestre
    """
    try:
        data_dir = Path(__file__).parent.parent / "data"
        csv_path = data_dir / "notas_corte_PAS_consolidado_v2.csv"
        
        if not csv_path.exists():
            return None
        
        # Carrega CSV consolidado
        stats = pd.read_csv(csv_path)
        
        # Filtra pelo semestre selecionado
        sem_str = "1º" if semester == 1 else "2º"
        stats = stats[stats['Semestre'] == sem_str]
        
        # Corrige nome truncado do curso de Engenharias
        stats['Curso'] = stats['Curso'].replace(
            '(BACHARELADOS)**',
            'ENGENHARIAS – AEROESPACIAL / AUTOMOTIVA / ELETRÔNICA / ENERGIA / SOFTWARE (BACHARELADOS)'
        )
        
        # Remove curso sob judice
        stats = stats[~stats['Curso'].str.contains('JUDICE', case=False, na=False)]
        
        # Cria Ranking (Reset Index)
        stats = stats.sort_values('Min', ascending=False).reset_index(drop=True)
        stats.index = stats.index + 1 # Ranking 1-based
        
        return stats
        
    except Exception as e:
        return None


@st.cache_data
def load_cohort_data():
    """Calcula ou carrega dados históricos para análise de coorte."""
    try:
        data_dir = Path(__file__).parent.parent / "data"
        csv_path = data_dir / "PAS_MESTRE_LIMPO_FINAL.csv"
        
        if not csv_path.exists():
            return pd.DataFrame()
            
        # Carrega apenas colunas necessárias para otimizar
        cols_to_load = ['P1_PAS1', 'P2_PAS1', 'P1_PAS2', 'P2_PAS2', 'Arg_Final']
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


def get_closest_courses(arg_previsto: float, n: int = 5, semester: int = 1) -> pd.DataFrame:
    """
    Retorna os N cursos com nota de corte mais próxima do argumento previsto.
    
    Args:
        arg_previsto: Argumento final previsto
        n: Número de cursos a retornar
        semester: 1 para 1º semestre, 2 para 2º semestre
    """
    stats = load_course_stats(semester=semester)
    if stats is None or stats.empty:
        return pd.DataFrame()
    
    # Remove curso sob judice (garantia adicional)
    stats = stats[~stats['Curso'].str.contains('JUDICE', case=False, na=False)]
    
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
    page_title="PAS Intelligence",
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


def load_sample_data() -> pd.DataFrame:
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
    
    return pd.DataFrame({
        'Inscricao': [f"2024{i:04d}" for i in range(n)],
        'Nome': [f"Aluno {i+1}" for i in range(n)],
        'P1_PAS1': p1_pas1.round(2),
        'P2_PAS1': p2_pas1.round(2),
        'Red_PAS1': np.random.uniform(4, 10, n).round(2),
        'P1_PAS2': p1_pas2.round(2),
        'P2_PAS2': p2_pas2.round(2),
        'Red_PAS2': np.random.uniform(5, 10, n).round(2),
        'Turma': np.random.choice(['A', 'B'], n),
    })


# =============================================================================
# SIDEBAR - NAVEGAÇÃO
# =============================================================================

st.sidebar.markdown("# 🎓 PAS Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegação",
    ["📊 Análise Temporal", "🚦 Semáforo de Risco", "🔮 Preditor PAS 3", "🏫 Análise da Escola", "📈 Comparação Entre Grupos"],
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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Faça upload do arquivo da turma (CSV ou Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="O arquivo deve conter colunas: Nome, P1_PAS1, P2_PAS1, Red_PAS1, P1_PAS2, P2_PAS2, Red_PAS2"
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
            st.session_state.df = load_sample_data()
            st.success("✅ Dados de exemplo carregados!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("### 📋 Prévia dos Dados")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Estatísticas gerais
        st.markdown("### 📈 Estatísticas Gerais")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Verifica colunas necessárias
        required_cols = ['P1_PAS1', 'P2_PAS1', 'P1_PAS2', 'P2_PAS2']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Colunas faltando: {', '.join(missing_cols)}")
            st.info("""
            📋 **Colunas necessárias:**
            - P1_PAS1, P2_PAS1, Red_PAS1 (notas do PAS 1)
            - P1_PAS2, P2_PAS2, Red_PAS2 (notas do PAS 2)
            
            💡 Use **Dados de Exemplo** para testar o sistema.
            """)
            with col1:
                st.metric("Total de Alunos", len(df))
            st.caption(f"Colunas encontradas: {', '.join(df.columns.tolist())}")
        else:
            df['EB_PAS1'] = df['P1_PAS1'] + df['P2_PAS1']
            df['EB_PAS2'] = df['P1_PAS2'] + df['P2_PAS2']
            
            with col1:
                st.metric("Total de Alunos", len(df))
            with col2:
                st.metric("Média EB PAS 1", f"{df['EB_PAS1'].mean():.2f}")
            with col3:
                st.metric("Média EB PAS 2", f"{df['EB_PAS2'].mean():.2f}")
            with col4:
                trend = df['EB_PAS2'].mean() - df['EB_PAS1'].mean()
                st.metric("Tendência Média", f"{trend:+.2f}", delta=f"{trend:+.2f}")
            
            # Gráfico de distribuição
            fig = px.histogram(
                df.melt(value_vars=['EB_PAS1', 'EB_PAS2'], var_name='Etapa', value_name='Escore Bruto'),
                x='Escore Bruto',
                color='Etapa',
                barmode='overlay',
                title='Distribuição de Escores Brutos',
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)


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

    # Toggle de Semestre (GLOBAL para ambas as abas)
    st.markdown("### 📅 Semestre de Ingresso")
    semester_option = st.radio(
        "Selecione para qual semestre você está concorrendo:",
        options=["1º Semestre", "2º Semestre"],
        index=0,
        horizontal=True,
        key="global_semester_toggle"
    )
    semester = 1 if semester_option == "1º Semestre" else 2
    
    # Criação das Abas
    tab_diagnostico, tab_estrategia = st.tabs(["🔮 Diagnóstico Realista", "🎯 Calculadora de Estratégia"])

    # =========================================================================
    # ABA 1: DIAGNÓSTICO REALISTA
    # =========================================================================
    with tab_diagnostico:
        st.markdown("""
        > **Previsão baseada em Inteligência Artificial:** Insira suas notas acumuladas para ver sua projeção.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Notas do PAS 1")
            p1_pas1 = st.number_input("P1 PAS 1 (Língua Estrangeira)", 0.0, 20.0, value=None, step=0.001, format="%.3f", key="pred_p1_1")
            p2_pas1 = st.number_input("P2 PAS 1 (Demais Disciplinas)", 0.0, 100.0, value=None, step=0.001, format="%.3f", key="pred_p2_1")
            red_pas1 = st.number_input("Redação PAS 1", 0.0, 10.0, value=None, step=0.001, format="%.3f", key="pred_r_1")
            
        with col2:
            st.markdown("### 📝 Notas do PAS 2")
            p1_pas2 = st.number_input("P1 PAS 2 (Língua Estrangeira)", 0.0, 20.0, value=None, step=0.001, format="%.3f", key="pred_p1_2")
            p2_pas2 = st.number_input("P2 PAS 2 (Demais Disciplinas)", 0.0, 100.0, value=None, step=0.001, format="%.3f", key="pred_p2_2")
            red_pas2 = st.number_input("Redação PAS 2", 0.0, 10.0, value=None, step=0.001, format="%.3f", key="pred_r_2")
        
        # Validação de preenchimento
        missing_data = any(v is None for v in [p1_pas1, p2_pas1, red_pas1, p1_pas2, p2_pas2, red_pas2])
        
        if not missing_data:
            # Calcula Escores Brutos
            eb_pas1 = p1_pas1 + p2_pas1
            eb_pas2 = p1_pas2 + p2_pas2
            
            st.markdown("---")
            # st.markdown(f"**Escore Bruto PAS 1:** {eb_pas1:.3f} | **Escore Bruto PAS 2:** {eb_pas2:.3f}")
        else:
            st.warning("⚠️ Insira todas as notas acima para habilitar a predição.")
            st.stop()
        
        # Botão de Predição
        if st.button("🚀 Calcular Projeção", type="primary"):
            try:
                # Features EXATAS que o modelo espera
                cresc_eb = eb_pas2 - eb_pas1
                cresc_red = red_pas2 - red_pas1
                
                features = np.array([[
                    eb_pas1, red_pas1, eb_pas2, red_pas2, cresc_eb, cresc_red
                ]])
                
                features_scaled = SCALER.transform(features) if SCALER else features
                
                # Predições de cada modelo para ensemble
                predictions = {}
                if MODELS['lgbm']: predictions['lgbm'] = float(MODELS['lgbm'].predict(features)[0])
                if MODELS['rf']: predictions['rf'] = float(MODELS['rf'].predict(features)[0])
                if MODELS['linear']: predictions['linear'] = float(MODELS['linear'].predict(features_scaled)[0])
                if MODELS['mlp']: predictions['mlp'] = float(MODELS['mlp'].predict(features_scaled)[0])
                
                if len(predictions) > 0:
                    # Meta-modelo Select
                    recommended_model = 'lgbm'
                    if META_MODEL and META_SCALER:
                        meta_features = np.array([[
                            eb_pas1, red_pas1, eb_pas2, red_pas2,
                            cresc_eb, cresc_red,
                            abs(cresc_eb)/(eb_pas1+0.01), abs(cresc_red)/(red_pas1+0.01),
                            (eb_pas1+eb_pas2)/2, 1 if cresc_eb > 0 else (-1 if cresc_eb < 0 else 0)
                        ]])
                        best_model_label = META_MODEL.predict(META_SCALER.transform(meta_features))[0]
                        recommended_model = LABEL_TO_MODEL.get(best_model_label, 'lgbm')
                    
                    # Argumento Final
                    arg_final_pred = None
                    if ARG_FINAL_MODEL:
                        arg_final_pred = float(ARG_FINAL_MODEL.predict(features)[0])
                    
                    st.session_state.prediction_results = {
                        'predictions': predictions,
                        'recommended_model': recommended_model,
                        'arg_final_pred': arg_final_pred,
                        'eb_pas1': eb_pas1, 'eb_pas2': eb_pas2,
                        'red_pas1': red_pas1, 'red_pas2': red_pas2,
                    }
                else:
                    st.error("❌ Erro: Modelos não carregados corretamente.")
                    
            except Exception as e:
                st.error(f"❌ Erro na predição: {e}")

        # Exibição dos Resultados (Se existirem)
        if 'prediction_results' in st.session_state and st.session_state.prediction_results:
            results = st.session_state.prediction_results
            arg_final_pred = results['arg_final_pred']
            predictions = results['predictions']
            recommended_model = results['recommended_model']
            ARG_FINAL_MAE = 13.49
            
            # --- DISPLAY ESCORE BRUTO PAS 3 PREVISTO (RESTAURADO) ---
            st.markdown("---")
            st.markdown("### 🔢 Previsões do Modelo")
            
            c_eb, c_arg = st.columns(2)
            
            # 1. EB PAS 3
            recommended_eb_pred = predictions.get(recommended_model, 0.0)
            model_mae = MODEL_MAE.get(recommended_model, 6.8) # Default error if not found
            
            with c_eb:
                st.metric(
                    "EB PAS 3 Previsto",
                    f"{recommended_eb_pred:.3f}",
                    help=f"Previsão gerada pelo modelo {recommended_model.upper()}. Intervalo de confiança (erro médio): ± {model_mae:.2f} pontos."
                )
            
            # 2. Argumento Final (já existente)
            with c_arg:
                if arg_final_pred is not None:
                    st.metric(
                        "Argumento Final Previsto",
                        f"{arg_final_pred:.3f}",
                        delta=f"± {ARG_FINAL_MAE:.2f}",
                        help="Nota final que será usada para classificação no curso."
                    )
            
            if arg_final_pred is not None:
                st.markdown("---")
                
                # Slider de Cenário
                st.markdown("#### 🎛️ Ajuste de Cenário")
                arg_ajustado = st.slider(
                    "Simule variações no seu Argumento Final (dentro do intervalo de confiança):",
                    min_value=float(arg_final_pred - ARG_FINAL_MAE),
                    max_value=float(arg_final_pred + ARG_FINAL_MAE),
                    value=float(arg_final_pred),
                    format="%.3f",
                    step=0.1
                )
                
                # Input de Curso para Probabilidade
                st.markdown(f"#### 🎓 Análise de Probabilidade ({semester_option})")
                
                # Carrega cursos com SEMESTRE DINÂMICO
                df_cursos = load_course_stats(semester=semester) 
                if df_cursos is not None:
                    cursos_lista = df_cursos['Curso'].unique().tolist()
                    
                    # Cria mapa para exibir nota no dropdown
                    course_scores = dict(zip(df_cursos['Curso'], df_cursos['Min']))
                    
                    def fmt_course(nome):
                        if nome == "Selecione...": return nome
                        return f"{nome} (Nota: {course_scores.get(nome, 0):.3f})"
                    
                    curso_selecionado = st.selectbox(
                        "Selecione um curso de interesse para ver sua chance:", 
                        ["Selecione..."] + cursos_lista,
                        format_func=fmt_course
                    )
                    
                    if curso_selecionado != "Selecione...":
                        curso_stats = df_cursos[df_cursos['Curso'] == curso_selecionado].iloc[0]
                        nota_corte = curso_stats['Min']
                        
                        # Probabilidade
                        if calculate_approval_probability:
                            prob = calculate_approval_probability(arg_ajustado, nota_corte, rmse=ARG_FINAL_MAE)
                            
                            # Card Visual
                            color = "#4CAF50" if prob >= 0.8 else "#FFC107" if prob >= 0.3 else "#F44336"
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
                                <h2 style="margin:0;">{prob*100:.1f}% de Chance</h2>
                                <p style="margin:5px 0 0 0;">Nota de Corte: {nota_corte:.3f} | Sua Simulação: {arg_ajustado:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Módulo de estatísticas não carregado.")
                            
                    # Tabela de Cursos Próximos com Probabilidade
                    st.markdown(f"#### 🏫 Cursos ao seu alcance no {semester_option}")
                    closest = get_closest_courses(arg_ajustado, n=10, semester=semester)
                    
                    if not closest.empty and calculate_approval_probability:
                        closest['Chance %'] = closest['Min'].apply(
                            lambda x: calculate_approval_probability(arg_ajustado, x, rmse=ARG_FINAL_MAE) * 100
                        )
                        st.dataframe(
                            closest[['Curso', 'Min', 'Chance %', 'Status']].style.format({'Min': '{:.3f}', 'Chance %': '{:.1f}%'}),
                            use_container_width=True
                        )

    # =========================================================================
    # ABA 2: CALCULADORA DE ESTRATÉGIA
    # =========================================================================
    with tab_estrategia:
        st.markdown("""
        > **Engenharia Reversa:** Defina onde quer chegar e descubra quanto precisa tirar.
        """)
        
        # =========================================================================
        # CONFIGURAÇÃO DE CICLO (Exclusivo da Estratégia)
        # =========================================================================
        st.markdown("### ⚙️ Configuração do Subprograma")
        # Seleção de Ciclo (Impacta o cálculo de z-score do passado)
        ciclo_aluno = st.selectbox(
            "Em qual Subprograma (Triênio) você está?",
            options=list(TRIENNIUM_STATS.keys()),
            index=1, # Default 2022-2024
            help="Isso garante que o sistema use a 'régua' correta das médias do ano em que você fez o PAS 1 e 2."
        )
        stats_ciclo = TRIENNIUM_STATS[ciclo_aluno]
        
        if TargetCalculator and 'prediction_results' in st.session_state:
            # Reusa dados inputados na aba 1 se disponíveis
            res = st.session_state.prediction_results
            notas_input = {
                'P1_PAS1': res.get('eb_pas1', 0)/2,
                'P2_PAS1': res.get('eb_pas1', 0)/2,
                'Red_PAS1': res.get('red_pas1', 0),
                'P1_PAS2': res.get('eb_pas2', 0)/2,
                'P2_PAS2': res.get('eb_pas2', 0),
                'Red_PAS2': res.get('red_pas2', 0),
            } 
            # Vamos usar os inputs diretos dos widgets que estão no escopo global desta pagina
            notas_validas = {
                'P1_PAS1': p1_pas1, 'P2_PAS1': p2_pas1, 'Red_PAS1': red_pas1,
                'P1_PAS2': p1_pas2, 'P2_PAS2': p2_pas2, 'Red_PAS2': red_pas2
            }
            
            if not missing_data:
                calc = TargetCalculator()
                
                # Seleção de Curso Alvo
                st.markdown(f"### 🎯 Meta ({semester_option})")
                df_cursos_estrat = load_course_stats(semester=semester)
                if df_cursos_estrat is not None:
                    cursos_lista = df_cursos_estrat['Curso'].unique().tolist()
                    
                    # Cria mapa para exibir nota no dropdown
                    course_scores_estrat = dict(zip(df_cursos_estrat['Curso'], df_cursos_estrat['Min']))
                    
                    def fmt_course_estrat(nome):
                        return f"{nome} (Nota: {course_scores_estrat.get(nome, 0):.3f})"

                    curso_alvo_nome = st.selectbox(
                        "Curso Objetivo:", 
                        cursos_lista, 
                        key="target_course",
                        format_func=fmt_course_estrat
                    )
                    
                    meta_arg = df_cursos_estrat[df_cursos_estrat['Curso'] == curso_alvo_nome]['Min'].values[0]
                    st.info(f"Nota de Corte Alvo no **{semester_option}**: **{meta_arg:.3f}**")
                    
                    # ESTRATÉGIA DINÂMICA
                    # Define a projeção do PAS 3 baseada no subprograma escolhido
                    if ciclo_aluno == "2023-2025":
                        # Para o ciclo atual, usa a projeção de tendência pura
                        stats_pas3_proj = STATS_PAS3_TREND
                    else:
                        # Para ciclos passados, usa a média real daquele PAS 3
                        stats_pas3_proj = stats_ciclo["PAS3"]

                    # Botão Principal de Cálculo
                    if st.button("🔢 Calcular Caminho", type="primary"):
                        result_reverso = calc.calculate_required_score(
                            notas_validas, meta_arg,
                            stats_ciclo["PAS1"], stats_ciclo["PAS2"], stats_pas3_proj
                        )
                        st.session_state.strategy_result = result_reverso
                        st.session_state.strategy_active = True
                        st.session_state.simulacao_ativa = False # Reset simulação ao recalcular do zero
                    

                    # Exibe Resultado (Persistente)
                    if st.session_state.get('strategy_active') and 'strategy_result' in st.session_state:
                        result_reverso = st.session_state.strategy_result
                        
                        if result_reverso.status == 'possivel':
                            cor_msg = "success"
                            icon = "✅"
                        elif result_reverso.status == 'garantido':
                            cor_msg = "success"
                            icon = "🎉"
                        else:
                            cor_msg = "error"
                            icon = "⚠️"
                            
                        # --- Início do Ajuste de Cenário ---
                        st.markdown("---")
                        with st.expander("🛠️ Ajuste de Cenário (Personalizar Previsões)", expanded=st.session_state.get('simulacao_ativa', False)):
                            st.info("O modelo estima sua nota de P1 e Redação com base no histórico. Se você discorda, ajuste abaixo:")
                            c_sim1, c_sim2 = st.columns(2)
                            
                            # Inputs com precisão de 3 casas
                            p1_val = float(result_reverso.p1_estimado)
                            red_val = float(result_reverso.red_estimada)
                            
                            p1_override = c_sim1.number_input(
                                "Estimativa P1 PAS 3", 
                                0.0, 20.0, 
                                value=p1_val, 
                                step=0.001, format="%.3f",
                                help="Personalize quanto você acha que vai tirar na P1 (Língua Estrangeira)."
                            )
                            red_override = c_sim2.number_input(
                                "Estimativa Redação PAS 3", 
                                0.0, 10.0, 
                                value=red_val,
                                step=0.001, format="%.3f",
                                help="Personalize quanto você acha que vai tirar na Redação."
                            )
                            
                            if st.button("🔄 Recalcular com meu Cenário"):
                                st.session_state.simulacao_ativa = True
                                # Recalcula usando os valores INPUTADOS pelo usuário
                                notas_com_override = notas_validas.copy()
                                notas_com_override['P1_PAS3_Override'] = p1_override
                                notas_com_override['Red_PAS3_Override'] = red_override
                                
                                # Recalcula usando os valores e stats corretos
                                new_result = calc.calculate_required_score(
                                    notas_com_override, meta_arg,
                                    stats_ciclo["PAS1"], stats_ciclo["PAS2"], stats_pas3_proj,
                                    p1_override=p1_override,
                                    red_override=red_override
                                )
                                st.session_state.strategy_result = new_result
                                st.rerun()

                        getattr(st, cor_msg)(f"{icon} {result_reverso.mensagem}")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("P1 PAS 3 (Est.)", f"{result_reverso.p1_estimado:.3f}", help="Valor utilizado no cálculo.")
                        c2.metric("Redação (Est.)", f"{result_reverso.red_estimada:.3f}", help="Valor utilizado no cálculo.")
                        c3.metric("P2 PAS 3 NECESSÁRIA", f"{result_reverso.p2_necessario:.3f}", delta="Meta" if not st.session_state.get('simulacao_ativa') else "Meta Ajustada")
                        
                        # REALITY CHECK (COHORTE)
                        st.markdown("### 📊 Reality Check (Base Histórica)")
                        if calculate_cohort_evolution_probability:
                            df_hist = load_cohort_data()
                            
                            # Dados do aluno atual para busca
                            aluno_atual_dados = {
                                'eb_pas1': eb_pas1,
                                'eb_pas2': eb_pas2
                            }
                            
                            prob_hist, amostra = calculate_cohort_evolution_probability(
                                aluno_atual_dados, meta_arg, df_hist
                            )
                            
                            if amostra > 0:
                                st.warning(f"""
                                **Análise de Coorte:** De {amostra} alunos com desempenho semelhante ao seu no PAS 1 e 2 nos últimos anos,
                                **{prob_hist:.1f}%** conseguiram atingir essa nota final.
                                """)
                            else:
                                st.info("Dados históricos insuficientes para perfil similar.")
                        
            else:
                st.warning("Preencha as notas na aba Diagnóstico primeiro.")
        else:
            st.warning("Calculadora não disponível.")
            

    
    

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
            return pd.read_csv(Path(__file__).parent.parent / "data" / "PAS_MESTRE_LIMPO_FINAL.csv")
        except:
            return pd.read_csv("data/PAS_MESTRE_LIMPO_FINAL.csv")
    
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
            try:
                escola_exemplo = pd.read_excel(Path(__file__).parent.parent / "data" / "exemplo_escola_1000_alunos.xlsx")
                st.session_state.escola_df = escola_exemplo
                st.success("✅ Carregado: 1000 alunos de exemplo")
            except Exception as e:
                st.error(f"Erro: {e}")
    
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

