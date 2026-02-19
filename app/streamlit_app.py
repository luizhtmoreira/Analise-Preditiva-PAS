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
import time
from supabase import create_client, Client
import os
import zipfile
import requests
import shutil


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
    # Inicializa o gerador de PDF
    pdf_gen = PDFGenerator()
except ImportError as e:
    st.error(f":material/warning: Módulo pas_intelligence não encontrado: {e}")
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
            
    # 2. Bidirectional substring match
    matches = []
    for course in course_list:
        normalized_course = unicodedata.normalize('NFKD', course).encode('ASCII', 'ignore').decode('utf-8').upper()
        # Input inside course OR course inside input (bidirectional)
        if normalized_input in normalized_course or normalized_course in normalized_input:
            matches.append(course)
    
    if matches:
        return max(matches, key=len)
        
    # 3. Token-based Match (Bag of Words) - Para "Direito Noturno" vs "Direito (Brasília - Noturno)"
    input_tokens = set(normalized_input.split())
    best_match = None
    max_score = 0.0
    
    for course in course_list:
        normalized_course = unicodedata.normalize('NFKD', course).encode('ASCII', 'ignore').decode('utf-8').upper()
        course_tokens = set(normalized_course.replace('(', '').replace(')', '').replace('-', '').split())
        
        # Interseção
        common = input_tokens.intersection(course_tokens)
        
        if not input_tokens: continue
        
        # Score: % dos tokens do input que estão no curso
        score = len(common) / len(input_tokens)
        
        # Se score alto e curso contém algo a mais (contexto), ok.
        # Ex: Input "Direito" (1 token), Match "Direito" (1/1=1.0). Matches "Direito Noturno" (1/1=1.0).
        # Precisamos desempatar. Preferir o que tem tamanho mais próximo? Ou o que tem mais tokens em comum?
        
        if score > 0.8: # Pelo menos 80% das palavras do input estão no curso alvo
             if score > max_score:
                 max_score = score
                 best_match = course
             elif score == max_score:
                 # Desempate: Menor diferença de tamanho (mais conciso/exato)
                 if abs(len(normalized_course) - len(normalized_input)) < abs(len(unicodedata.normalize('NFKD', best_match).encode('ASCII', 'ignore').decode('utf-8').upper()) - len(normalized_input)):
                     best_match = course

    if best_match:
        return best_match

    return input_name
def load_course_stats(semester: int = 1, triennium: Optional[str] = None, system: Optional[str] = "Sistema Universal"):
    """
    Carrega estatísticas de nota de corte por curso do triênio especificado.
    Lê de CSVs pré-processados para carregamento instantâneo.
    
    Args:
        semester: 1 para 1º semestre, 2 para 2º semestre
        triennium: String do triênio (ex: "2022-2024"). Se None, usa o mais recente.
        system: Nome do sistema de concorrência. Se None, retorna todos. Default: "Sistema Universal".
    """
    try:
        data_dir = Path(__file__).parent.parent / "data"
        
        # Arquivo de Notas de Corte Final
        csv_path = data_dir / "notas_corte_pas.csv"
        
        if not csv_path.exists():
            st.error(f":material/warning: Arquivo não encontrado: {csv_path}")
            return None
        
        # Carrega CSV encontrado
        stats = pd.read_csv(csv_path)
        
        # Padroniza nomes de colunas para compatibilidade
        rename_map = {}
        if 'Curso_Limpo' in stats.columns: rename_map['Curso_Limpo'] = 'Curso'
        if 'N_Banco' in stats.columns: rename_map['N_Banco'] = 'N'
        
        if rename_map:
            stats = stats.rename(columns=rename_map)

        # Filtra pelo semestre selecionado (formato CSV: '1°' ou '2°')
        # Ajuste para garantir que o símbolo ° bata (pode ser UTF-8 ou outro)
        if 'Semestre' in stats.columns:
            stats = stats[stats['Semestre'].astype(str).str.contains(str(semester))]

        # Filtra por sistema
        if 'Sistema_Nome' in stats.columns and system is not None:
            stats = stats[stats['Sistema_Nome'] == system]

        # Filtra por triênio
        if triennium:
            if 'Trienio' in stats.columns:
                stats = stats[stats['Trienio'] == triennium]
            elif 'Triênio' in stats.columns:
                stats = stats[stats['Triênio'] == triennium]
        else:
            # Fallback: Pega o triênio mais recente disponível no CSV se não especificado
            for col in ['Trienio', 'Triênio']:
                if col in stats.columns and not stats.empty:
                    recent_triennium = stats[col].max()
                    stats = stats[stats[col] == recent_triennium]
                    break
        

        

        
        
        # --- LÓGICA DE CORTE POR SEMESTRE (REFINADO PAS_UNB) ---
        # Limpeza para evitar duplicatas por espaços (Sync Step 540)
        for col in ['Curso', 'Campus', 'Turno', 'Chamada']:
            if col in stats.columns:
                stats[col] = stats[col].astype(str).str.strip()
        
        # Extrai numeral da chamada para ordenação precisa
        if 'Chamada' in stats.columns:
            stats['Chamada_Num'] = stats['Chamada'].str.extract('(\d+)').fillna(0).astype(int)
        else:
            stats['Chamada_Num'] = 1

        if semester == 1:
            # 1º Semestre: Prioridade para ÚLTIMA CHAMADA (Menor Nota = Piso de entrada)
            # Ordenamos por Min crescente e pegamos o primeiro (o menor)
            subset_cols = ['Curso', 'Campus', 'Turno']
            if 'Sistema_Nome' in stats.columns:
                subset_cols.append('Sistema_Nome')
                
            stats = stats.sort_values(['Curso', 'Campus', 'Turno', 'Min'], ascending=[True, True, True, True]).drop_duplicates(
                subset=subset_cols, keep='first'
            )
        else:
            # 2º Semestre: Prioridade para PRIMEIRA CHAMADA DISPONÍVEL (Maior Nota = Corte inicial)
            # Ordenamos por Chamada_Num (1ª, 2ª...) e pegamos o primeiro disponível
            subset_cols = ['Curso', 'Campus', 'Turno']
            if 'Sistema_Nome' in stats.columns:
                subset_cols.append('Sistema_Nome')

            stats = stats.sort_values(['Curso', 'Campus', 'Turno', 'Chamada_Num'], ascending=[True, True, True, True]).drop_duplicates(
                subset=subset_cols, keep='first'
            )

        # Cria Ranking (Reset Index) após a deduplicação
        # Re-ordena por nota para o Ranking final do PDF/Dashboard
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


import difflib
import unicodedata

def find_best_match(query: str, choices: list[str], cutoff: float = 0.6) -> str:
    """
    Encontra a melhor correspondência para uma string em uma lista de opções.
    1. Tenta difflib (para typos).
    2. Tenta substring (para palavras-chave).
    3. Tenta interseção de tokens (para "direito noturno" -> "direito").
    """
    if not query or not choices:
        return query
        
    # 1. Tentativa DIFUSA (Typos, pequenas variações)
    matches = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
        
    # Normalização para métodos 2 e 3
    try:
        query_norm = unicodedata.normalize('NFKD', str(query)).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
        
        # 2. Tentativa SUBSTRING (Palavra-chave)
        # Ex: "audiovisual" in "COMUNICAÇÃO SOCIAL - AUDIOVISUAL"
        candidates = []
        for choice in choices:
            choice_norm = unicodedata.normalize('NFKD', str(choice)).encode('ASCII', 'ignore').decode('utf-8').lower()
            if query_norm in choice_norm:
                candidates.append(choice)
        
        if candidates:
            return min(candidates, key=len)

        # 3. Tentativa INTERSEÇÃO DE TOKENS (Keywords soltas)
        # Útil para: "direito noturno" -> "DIREITO (BACHARELADO)"
        # Se 50% ou mais das palavras buscadas existirem no alvo, consideramos Match.
        # Remove pontuação para evitar (ceilandia) != ceilandia
        query_clean = query_norm.replace('(', ' ').replace(')', ' ').replace('-', ' ')
        query_tokens = set(query_clean.split())
        
        best_token_match = None
        best_token_score = 0.0
        
        for choice in choices:
            choice_norm = unicodedata.normalize('NFKD', str(choice)).encode('ASCII', 'ignore').decode('utf-8').lower()
            choice_clean = choice_norm.replace('(', ' ').replace(')', ' ').replace('-', ' ')
            choice_tokens = set(choice_clean.split())
            
            common = query_tokens.intersection(choice_tokens)
            if not common: continue
            
            score = len(common) / len(query_tokens)
            
            if score > best_token_score:
                best_token_score = score
                best_token_match = choice
        
        if best_token_match and best_token_score >= 0.5:
            return best_token_match
            
    except Exception:
        pass
    
    return query


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="VETOR PAS",
    page_icon=":material/school:",
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
# SISTEMA DE LOGIN E SEGURANÇA (SUPABASE)
# =============================================================================

@st.cache_resource
def init_connection():
    try:
        if "supabase" not in st.secrets:
            st.error("Seção [supabase] não encontrada no secrets.toml")
            return None
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Erro ao inicializar Supabase: {e}")
        return None

supabase = init_connection()

def check_login():
    # Inicializa estado
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = ''

    # Se NÃO estiver logado, mostra tela de bloqueio
    if not st.session_state['logged_in']:
        # --- CSS PERSONALIZADO PARA LOGIN ---
        st.markdown("""
            <style>
            /* Hover effect for the login button */
            div.stButton > button:first-child:hover {
                background-color: #007bff !important;
                color: white !important;
                border-color: #007bff !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Centraliza o login
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("## 🔒 Acesso Restrito - Coordenação")
            email = st.text_input("Email Corporativo")
            password = st.text_input("Senha", type="password")
            
            if st.button("Entrar no Sistema", use_container_width=True):
                if supabase:
                    try:
                        session = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state['logged_in'] = True
                        st.session_state['user_email'] = session.user.email
                        st.toast("Login realizado com sucesso!", icon="✅")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error("❌ Email ou senha incorretos.")
                else:
                    st.error("Erro de conexão com Supabase. Verifique secrets.toml")
        
        # Footer Branding
        st.markdown(
            """
            <div style="text-align: center; margin-top: 50px; color: #666; font-size: 0.85rem;">
                🔒 Ambiente Seguro | <b>Vetor PAS</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.stop() # 🛑 PARA TUDO AQUI SE NÃO ESTIVER LOGADO

# Executa o bloqueio imediatamente
check_login()


def download_models():
    url = 'https://github.com/luizhtmoreira/Analise-Preditiva-PAS/releases/download/v1.0/models.zip'
    
    base_path = Path(__file__).resolve().parent
    models_dir = base_path / "models"
    zip_path = base_path / "models_temp.zip"
    
    # --- TRAVA DE SEGURANÇA REATIVADA ---
    # Se o arquivo principal já existe, não faz nada (pula o download)
    if (models_dir / "modelo_lgbm.joblib").exists():
        return

    # Se chegou aqui, é porque não tem os arquivos. Cria a pasta.
    if not models_dir.exists():
        os.makedirs(models_dir, exist_ok=True)

    try:
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        status_text.text("⏳ Baixando arquivos do sistema...")
        
        response = requests.get(url, stream=True, timeout=600)
        if response.status_code != 200:
            status_text.error(f"❌ Erro HTTP: {response.status_code}")
            st.stop()
            
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        wrote = 0
        
        with open(zip_path, 'wb') as f:
            for data in response.iter_content(block_size):
                wrote = wrote + len(data)
                f.write(data)
                if total_size > 0:
                    percent = int((wrote / total_size) * 100)
                    if percent > 100: percent = 100
                    progress_bar.progress(percent)
        
        status_text.text("📦 Finalizando instalação...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
            
        nested = models_dir / "models"
        if nested.exists():
            import shutil
            for file in os.listdir(nested):
                shutil.move(str(nested / file), str(models_dir / file))
            os.rmdir(nested)

        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        progress_bar.empty()
        status_text.empty() # Limpa a mensagem de sucesso para não ficar na tela
        st.rerun() # Reinicia para carregar os modelos
            
    except Exception as e:
        st.error(f"Erro: {e}")
        st.stop()

# Chama a função
download_models()

# =============================================================================
# CARREGAMENTO DOS MODELOS TREINADOS (ENSEMBLE + META-MODELO)
# =============================================================================

@st.cache_resource
def load_models():
    # Garante que busca na pasta correta onde o script está
    base_path = Path(__file__).resolve().parent
    models_dir = base_path / "models"
    
    # Inicializa tudo como None para evitar o NameError
    models = {'lgbm': None, 'rf': None, 'linear': None, 'mlp': None}
    scaler = None
    meta_model = None
    meta_scaler = None
    arg_final_model = None

    if not models_dir.exists():
        return models, scaler, meta_model, meta_scaler, arg_final_model

    try:
        # Carregamento seguro
        if (models_dir / "modelo_lgbm.joblib").exists():
            models['lgbm'] = joblib.load(models_dir / "modelo_lgbm.joblib")
        if (models_dir / "modelo_rf.joblib").exists():
            models['rf'] = joblib.load(models_dir / "modelo_rf.joblib")
        if (models_dir / "modelo_linear.joblib").exists():
            models['linear'] = joblib.load(models_dir / "modelo_linear.joblib")
        if (models_dir / "modelo_mlp.joblib").exists():
            models['mlp'] = joblib.load(models_dir / "modelo_mlp.joblib")
        if (models_dir / "scaler.joblib").exists():
            scaler = joblib.load(models_dir / "scaler.joblib")
        if (models_dir / "meta_model.joblib").exists():
            meta_model = joblib.load(models_dir / "meta_model.joblib")
        if (models_dir / "meta_scaler.joblib").exists():
            meta_scaler = joblib.load(models_dir / "meta_scaler.joblib")
        if (models_dir / "modelo_arg_final.joblib").exists():
            arg_final_model = joblib.load(models_dir / "modelo_arg_final.joblib")
            
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        
    return models, scaler, meta_model, meta_scaler, arg_final_model

# --- ESTA PARTE É CRUCIAL: A ATRIBUIÇÃO ---
# Chamamos a função e garantimos que as variáveis globais existam
MODELS, SCALER, META_MODEL, META_SCALER, ARG_FINAL_MODEL = load_models()

# --- CONFIGURAÇÕES DOS MODELOS (Adicione logo abaixo do load_models) ---

# Mapeamento: O Meta-Modelo devolve um número (0-3), precisamos traduzir para texto
LABEL_TO_MODEL = {0: 'lgbm', 1: 'rf', 2: 'linear', 3: 'mlp'}

# Erro Médio Absoluto (MAE) de cada modelo (para exibir a margem de erro)
MODEL_MAE = {
    'lgbm': 6.8123,
    'rf': 6.9965,
    'linear': 6.9371,
    'mlp': 6.8423,
}

# Nomes bonitos para exibir na tela (opcional, mas bom ter)
MODEL_NAMES = {
    'lgbm': ':material/rocket_launch: LightGBM (Gradient Boosting)',
    'rf': ':material/forest: Random Forest',
    'linear': ':material/trending_up: Regressão Linear',
    'mlp': ':material/psychology: Rede Neural MLP',
}

# =============================================================================
# ESTATÍSTICAS POR TRIÊNIO (Régua Histórica)
# =============================================================================

TRIENNIUM_STATS = {
    "2024-2026": {
        "PAS1": HistoricalStats(mean_p1=2.2175, std_p1=2.4766, mean_p2=23.8314, std_p2=12.3387, mean_red=6.0345, std_red=2.4790), # PAS 1 já ocorreu (placeholder 23-25 por enquanto)
        "PAS2": HistoricalStats(mean_p1=3.1496, std_p1=3.2475, mean_p2=25.3101, std_p2=14.2913, mean_red=6.1569, std_red=2.4728), # PAS 2 já ocorreu (placeholder 23-25 por enquanto)
        "PAS3": None, # FUTURO - Será definido dinamicamente (Trend ou Imitar 23-25)
    },
    "2023-2025": {
        "PAS1": HistoricalStats(mean_p1=2.2175, std_p1=2.4766, mean_p2=23.8314, std_p2=12.3387, mean_red=6.0345, std_red=2.4790),
        "PAS2": HistoricalStats(mean_p1=3.1496, std_p1=3.2475, mean_p2=25.3101, std_p2=14.2913, mean_red=6.1569, std_red=2.4728),
        "PAS3": HistoricalStats(mean_p1=3.8200, std_p1=2.1000, mean_p2=33.7400, std_p2=14.5000, mean_red=7.6500, std_red=1.8500), # Histórico Real Consolidados
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
            return "🟢 Baixo Risco", "low", f"Subiu {trend:.1f} pontos! :material/trending_up:"
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
    """Carrega dados de exemplo para demonstração, incluindo alunos específicos se o arquivo existir."""
    try:
        # Tenta carregar alunos específicos do PAS 3 / Gestão de Ativos
        data_dir = Path(__file__).parent.parent / "data"
        specific_path = data_dir / "alunos_especificos.csv"
        
        if specific_path.exists():
            df_spec = pd.read_csv(specific_path)
            # Se já temos os específicos, retornamos eles (podemos adicionar mocks depois)
            # Mas vamos combinar com mocks para ter volume
        else:
            df_spec = pd.DataFrame()
    except Exception:
        df_spec = pd.DataFrame()

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
    
    df_mock = pd.DataFrame({
        'Inscricao': [f"2024{i:04d}" for i in range(n)],
        'Nome': [f"Aluno {i+1}" for i in range(n)],
        'P1_PAS1': p1_pas1.round(2),
        'P2_PAS1': p2_pas1.round(2),
        'Red_PAS1': np.random.uniform(4, 10, n).round(2),
        'P1_PAS2': p1_pas2.round(2),
        'P2_PAS2': p2_pas2.round(2),
        'Red_PAS2': np.random.uniform(5, 10, n).round(2),
        'Turma': np.random.choice(['3º A', '3º B'], n),
        'Unidade': np.random.choice(['Asa Sul', 'Taguatinga', 'Lago Sul'], n, p=[0.4, 0.35, 0.25]),
        'Curso_Alvo': np.random.choice([
            'MEDICINA (BACHARELADO)', 'DIREITO (BACHARELADO)', 'ENGENHARIA CIVIL (BACHARELADO)',
            'ADMINISTRAÇÃO (BACHARELADO)', 'PSICOLOGIA (BACHARELADO)', 'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)',
        ], n, p=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15]),
        'Cota': np.nan # Inicializa com NaN para ser preenchido pela lógica principal
    })

    if include_pas3:
        # Gera dados do PAS 3 seguindo a tendência
        tendencia_pas3 = np.random.choice([-1, 0, 1], n, p=[0.2, 0.4, 0.4])
        variacao_pas3 = np.random.uniform(2, 8, n) * tendencia_pas3
        
        df_mock['P1_PAS3'] = np.clip(p1_pas2 + variacao_pas3 * 0.1, 0, 15).round(2)
        df_mock['P2_PAS3'] = np.clip(p2_pas2 + variacao_pas3, 5, 60).round(2)
        df_mock['Red_PAS3'] = np.random.uniform(5, 10, n).round(2)
    
    # Combina específicos com mocks
    if not df_spec.empty:
        # Garante que as colunas batem para o concat
        # Preenche P1/P2/Red_PAS3 com 0 se não existirem nos específicos
        for col in ['P1_PAS3', 'P2_PAS3', 'Red_PAS3']:
            if col not in df_spec.columns:
                df_spec[col] = 0.0
                
        df_final = pd.concat([df_spec, df_mock], ignore_index=True)
        return df_final
    
    return df_mock


# =============================================================================
# SIDEBAR - NAVEGAÇÃO
# =============================================================================

# Logo do Colégio Ideal (Centralizado)
logo_path = Path(__file__).parent.parent / "assets" / "templates" / "logo_ideal.png"
if logo_path.exists():
    # Usa colunas para centralizar
    c1, c2, c3 = st.sidebar.columns([1, 4, 1])
    with c2:
        st.image(str(logo_path), use_container_width=True)
else:
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h2 style="color: #003366;">COLÉGIO IDEAL</h2>
            <p style="font-size: 0.8em; color: gray;">SISTEMA DE GESTÃO ESTRATÉGICA</p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.sidebar.markdown("---")


# (Bloco de info do usuário movido para o final da sidebar)

# Dicionário de Páginas (ID -> Label com Ícone Material)
PAGES = {
    "temporal": ":material/analytics: Análise Temporal",
    "ativos": ":material/business_center: Gestão de Ativos",
    "preditor": ":material/model_training: Preditor PAS 3", 
    "escola": ":material/domain: Análise da Escola",
    "comparacao": ":material/trending_up: Comparação Entre Grupos",
    "pdf": ":material/description: Gerador de PDF"
}

selection = st.sidebar.radio(
    "Navegação",
    list(PAGES.values()),
    label_visibility="collapsed"
)

# Encontra a chave (ID) baseada na seleção do usuário
page = next(key for key, value in PAGES.items() if value == selection)

st.sidebar.markdown("---")

# --- INFO DO USUÁRIO (Final da Sidebar) ---
if st.session_state.get('logged_in'):
    st.sidebar.caption(f"👤 Logado como: {st.session_state['user_email']}")
    if st.sidebar.button("Sair (Logout)"):
        if supabase:
            supabase.auth.sign_out()
        st.session_state['logged_in'] = False
        st.rerun()

# Espaçador e Footer com CSS para não quebrar linha
st.sidebar.markdown(
    """
    <div style="white-space: nowrap; font-size: 0.8rem; color: gray;">
        🔒 Ambiente Seguro | Desenvolvido por Vetor PAS
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# ESTADO DA SESSÃO
# =============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None

# --- CARREGAMENTO GLOBAL DO BANCO DE DADOS (CORTES) ---
data_dir_global = Path(__file__).parent.parent / "data"
ARQUIVO_DADOS_GLOBAL = data_dir_global / "notas_corte_pas.csv"

# =============================================================================
# CARREGAMENTO GLOBAL DE ALUNOS (SUPABASE)
# =============================================================================
@st.cache_data(ttl=60)
def buscar_alunos_nuvem_global():
    """Busca a tabela mestra do Supabase e formata para o App."""
    if not supabase: return None
    try:
        # Select * all
        response = supabase.table("tabela_mestra").select("*").execute()
        data = response.data
        
        if not data: return None
        
        df_cloud = pd.DataFrame(data)
        
        # Mapeamento Banco -> App
        rename_map = {
            'nome': 'Nome', 
            'inscricao': 'Inscricao',
            'turma': 'Turma', 
            'unidade': 'Unidade',
            'curso_alvo': 'Curso_Alvo', 
            'cota': 'Sistema_Nome', 
            'trienio': 'Ano_Trienio',
            'p1_pas1': 'P1_PAS1', 'p2_pas1': 'P2_PAS1', 'red_pas1': 'Red_PAS1',
            'p1_pas2': 'P1_PAS2', 'p2_pas2': 'P2_PAS2', 'red_pas2': 'Red_PAS2',
            # 'p1_pas3': 'P1_PAS3', 'p2_pas3': 'P2_PAS3', 'red_pas3': 'Red_PAS3', # Se existir
            'eb_pas1': 'EB_PAS1', 'eb_pas2': 'EB_PAS2', 'eb_pas3': 'EB_PAS3'
        }
        # Renomeia os que existem
        df_cloud = df_cloud.rename(columns=rename_map)
        
        return df_cloud
    except Exception as e:
        # st.error(f"Erro silencioso ao buscar dados da nuvem: {e}")
        return None

# Auto-Load na Inicialização
if 'df_global_escola' not in st.session_state or st.session_state['df_global_escola'] is None:
    df_nuvem = buscar_alunos_nuvem_global()
    if df_nuvem is not None:
        st.session_state['df_global_escola'] = df_nuvem
        # Sincroniza com st.session_state.df se este estiver vazio
        if st.session_state.df is None:
             st.session_state.df = df_nuvem.copy()



@st.cache_data
def load_cutoff_data_global():
    if not ARQUIVO_DADOS_GLOBAL.exists():
        return None
    try:
        df = pd.read_csv(ARQUIVO_DADOS_GLOBAL)
        # Padronização de Colunas
        if 'Min' in df.columns:
            df['Min'] = pd.to_numeric(df['Min'], errors='coerce')
        # Garante que Curso_Limpo existe (Sync Step 880)
        if 'Curso_Limpo' not in df.columns and 'Curso' in df.columns:
            df['Curso_Limpo'] = df['Curso']
        return df
    except Exception:
        return None

df_notas = load_cutoff_data_global()
if df_notas is None:
    st.warning("⚠️ Banco de notas de corte não encontrado. Algumas funcionalidades (Preditor, PDF) podem ser limitadas.")
    # Initialize as empty DF to avoid NameError/AttributeError
    df_notas = pd.DataFrame(columns=['Trienio', 'Semestre', 'Sistema_Nome', 'Curso_Limpo', 'Campus', 'Turno', 'Min', 'Chamada'])


# =============================================================================
# PÁGINA 1: ANÁLISE TEMPORAL
# =============================================================================



if page == "temporal":
    st.title(":material/analytics: Análise Temporal")
    
    # Configuração Padrão: Triênio Atual (Em Andamento)
    analysis_mode = "Triênio Atual (Em Andamento)"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Faça upload do arquivo da turma (CSV ou Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="O arquivo deve conter colunas: Nome, P1_PAS1, P2_PAS1, Red_PAS1, P1_PAS2, P2_PAS2, Red_PAS2"
        )
        
        # Feedback se já existe dados carregados
        if st.session_state.df is not None:
            st.info(f"✅ Base Atual: {len(st.session_state.df)} alunos carregados (Nuvem/Local).")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(f":material/check_circle: Arquivo carregado: {len(st.session_state.df)} alunos")
                
                # --- Normalização de Colunas de Cota (Novo Mapeamento) ---
                # Garante que colunas com nomes variados sejam mapeadas para 'Cota'
                df_temp = st.session_state.df
                possible_quota_cols = ['Sistema', 'Sistema_Nome', 'sistema', 'cota']
                for col in possible_quota_cols:
                    if col in df_temp.columns and 'Cota' not in df_temp.columns:
                        df_temp = df_temp.rename(columns={col: 'Cota'})
                st.session_state.df = df_temp

            except Exception as e:
                st.error(f":material/error: Erro ao ler arquivo: {e}")
    
    with col2:
        if st.button(":material/download: Usar Dados de Exemplo"):
            # Sempre carrega dados do triênio atual (sem PAS 3)
            include_pas3 = False
            st.session_state.df = load_sample_data(include_pas3=include_pas3)
            st.success(":material/check_circle: Dados de exemplo carregados!")
            
        # --- PERSISTÊNCIA GLOBAL (ERP INTELIGENTE) ---
        if st.session_state.df is not None:
            # Salva na "Memória Global" da Escola
            st.session_state['df_global_escola'] = st.session_state.df.copy()
            
            # Limpeza de Colunas (Trim spaces)
            st.session_state['df_global_escola'].columns = st.session_state['df_global_escola'].columns.str.strip()
            
            # Notificação Discreta
            st.toast(":material/database: Base Centralizada Atualizada! Disponível no Preditor.")

            # --- BOTÃO DE SALVAR NA NUVEM (SUPABASE) ---
            if supabase:
                if st.button("💾 Salvar Base na Nuvem", help="Salva a planilha atual no banco de dados para acesso global."):
                    try:
                        with st.spinner("Conectando ao banco de dados..."):
                            # 1. Prepara o DataFrame (Normalização para SQL)
                            df_to_upload = st.session_state['df_global_escola'].copy()
                            
                            # Mapeamento Explicito (App -> Banco)
                            # Garante que só enviamos o que tem no banco
                            col_map = {
                                'Nome': 'nome',
                                'Inscricao': 'inscricao',
                                'Unidade': 'unidade',
                                'Turma': 'turma',
                                'Curso_Alvo': 'curso_alvo',
                                'Curso Alvo': 'curso_alvo', # Caso tenha variação
                                'Ano_Trienio': 'trienio',
                                'Trienio': 'trienio',
                                'Sistema_Nome': 'cota',
                                'Cota': 'cota',
                                'P1_PAS1': 'p1_pas1', 'P2_PAS1': 'p2_pas1', 'Red_PAS1': 'red_pas1',
                                'P1_PAS2': 'p1_pas2', 'P2_PAS2': 'p2_pas2', 'Red_PAS2': 'red_pas2',
                                # 'P1_PAS3': 'p1_pas3', 'P2_PAS3': 'p2_pas3', 'Red_PAS3': 'red_pas3' # Se tiver no banco
                            }
                            
                            # Renomeia
                            df_to_upload = df_to_upload.rename(columns=col_map)
                            
                            # Filtra apenas colunas que existem no banco (evita erro de coluna extra)
                            cols_banco = [
                                'inscricao', 'nome', 'unidade', 'turma', 'curso_alvo', 'cota', 'trienio',
                                'p1_pas1', 'p2_pas1', 'red_pas1', 
                                'p1_pas2', 'p2_pas2', 'red_pas2'
                            ]
                            
                            # Mantém apenas as colunas que estão no df e no banco
                            cols_to_keep = [c for c in cols_banco if c in df_to_upload.columns]
                            df_final = df_to_upload[cols_to_keep].copy()

                            # Converte NaN para None (NULL no SQL)
                            df_final = df_final.replace({np.nan: None})
                            
                            # Converte para lista de dicionários
                            data_to_insert = df_final.to_dict(orient='records')
                            
                            # 2. Limpa a Tabela Mestra (Reset)
                            # Tenta limpar onde inscricao não é nula
                            if len(data_to_insert) > 0:
                                supabase.table("tabela_mestra").delete().neq("id", 0).execute() # Delete all rows logic
                                
                                # 3. Insere Novos Dados
                                supabase.table("tabela_mestra").insert(data_to_insert).execute()
                            
                                st.toast("Sucesso! Dados salvos na nuvem.", icon="☁️")
                                st.success(f"Base de {len(data_to_insert)} alunos sincronizada com sucesso!")
                            else:
                                st.warning("DataFrame vazio ou colunas não correspondentes.")
                            
                    except Exception as e:
                        st.error(f"Erro ao salvar na nuvem: {e}")
            else:
                st.warning("Conexão com Supabase não disponível.")
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Garante que a coluna 'Turma' seja a última se existir
        if 'Turma' in df.columns:
            cols = [c for c in df.columns if c != 'Turma'] + ['Turma']
            df = df[cols]
        
        # Estatísticas gerais (Prioridade para o Diretor)
        st.markdown("### :material/trending_up: Estatísticas Gerais")
        
        # Definição de colunas necessárias baseada no modo
        required_cols = ['P1_PAS1', 'P2_PAS1', 'P1_PAS2', 'P2_PAS2']
        if analysis_mode == "Triênios Concluídos (Histórico)":
            required_cols.extend(['P1_PAS3', 'P2_PAS3'])
            
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            st.warning(f":material/warning: Colunas faltando para o modo '{analysis_mode}': {', '.join(missing_cols)}")
            st.info("""
            :material/toc: **Colunas necessárias:**
            - P1_PAS1, P2_PAS1, Red_PAS1 (notas do PAS 1)
            - P1_PAS2, P2_PAS2, Red_PAS2 (notas do PAS 2)
            """ + ("- P1_PAS3, P2_PAS3, Red_PAS3 (notas do PAS 3)" if analysis_mode == "Triênios Concluídos (Histórico)" else "") + """
            
            :material/lightbulb: Use **Dados de Exemplo** para testar o sistema.
            """)
        else:
            # Cálculos de Escore Bruto
            df['EB_PAS1'] = df['P1_PAS1'] + df['P2_PAS1']
            df['EB_PAS2'] = df['P1_PAS2'] + df['P2_PAS2']
            
            cols_metrics = st.columns(4 if analysis_mode == "Triênio Atual (Em Andamento)" else 5)
            
            with cols_metrics[0]:
                with st.container(border=True):
                    st.metric("Total de Alunos", len(df))
            with cols_metrics[1]:
                with st.container(border=True):
                    st.metric("Média EB PAS 1", f"{df['EB_PAS1'].mean():.2f}")
            with cols_metrics[2]:
                with st.container(border=True):
                    st.metric("Média EB PAS 2", f"{df['EB_PAS2'].mean():.2f}")
            
            if analysis_mode == "Triênio Atual (Em Andamento)":
                with cols_metrics[3]:
                    with st.container(border=True):
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
                    color_discrete_map={'EB_PAS1': '#87CEEB', 'EB_PAS2': '#4682B4'}
                )
                st.plotly_chart(fig, use_container_width=True)

            else: # Triênios Concluídos
                df['EB_PAS3'] = df['P1_PAS3'] + df['P2_PAS3']
                
                with cols_metrics[3]:
                    with st.container(border=True):
                        st.metric("Média EB PAS 3", f"{df['EB_PAS3'].mean():.2f}")
                
                with cols_metrics[4]:
                    with st.container(border=True):
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
                    color_discrete_map={'EB_PAS1': '#87CEEB', 'EB_PAS2': '#4682B4', 'EB_PAS3': '#003366'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de Evolução Média (Enterprise Style)
                st.markdown("### :material/trending_up: Raio-X da Evolução (Média da Turma)")
                
                # Cálculo da Média Real (Ignorando Zeros/Faltantes)
                means_val = [
                    df[df['EB_PAS1'] > 0]['EB_PAS1'].mean(), 
                    df[df['EB_PAS2'] > 0]['EB_PAS2'].mean(), 
                    df[df['EB_PAS3'] > 0]['EB_PAS3'].mean()
                ]
                etapas_val = ['PAS 1', 'PAS 2', 'PAS 3']
                
                fig_line = go.Figure()
                
                fig_line.add_trace(go.Scatter(
                    x=etapas_val, 
                    y=means_val,
                    mode='lines+markers+text',
                    line=dict(color='#003366', width=4),
                    marker=dict(size=12, color='white', line=dict(width=2, color='#003366')),
                    text=[f"{v:.2f}" if i == 2 else "" for i, v in enumerate(means_val)],
                    textposition="top center",
                    textfont=dict(color='#003366', size=14, family="Arial Black")
                ))
                
                fig_line.update_layout(
                    title={'text': "Trajetória de Desempenho (Escore Bruto)", 'font': {'color': '#003366', 'size': 20}},
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=False, linecolor='#cccccc', tickfont=dict(color='#003366')),
                    yaxis=dict(showgrid=False, showticklabels=False), # Remove grade Y e labels para limpar visual
                    margin=dict(l=20, r=20, t=50, b=20),
                    showlegend=False
                )
                
                # Annotation para o último ponto
                last_mean = means_val[-1]
                fig_line.add_annotation(
                    x='PAS 3',
                    y=last_mean,
                    text=f"Média Final: {last_mean:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#003366",
                    ax=0,
                    ay=-40,
                    font=dict(color="#ffffff", size=12),
                    bgcolor="#003366",
                    bordercolor="#003366",
                    borderwidth=2,
                    borderpad=5,
                    opacity=0.9
                )
                
                st.plotly_chart(fig_line, use_container_width=True)

        # Tabela de Dados Brutos (Final da Página - Opção de Drill-down)
        st.markdown("---")
        with st.expander("📂 Visualizar Tabela de Dados Bruta"):
            st.dataframe(df, use_container_width=True)


# =============================================================================
# PÁGINA 2: GESTÃO DE ATIVOS (MONEYBALL)
# =============================================================================

elif page == "ativos":
    st.title(":material/work: Gestão de Ativos")
    
    st.info("""
    :material/target: **Lógica de Classificação (Duplo Corte: 1º e 2º Semestre):**
    - 🟢 **Baixo Risco**: Argumento previsto ≥ nota de corte do **1º Semestre** → Aprovado direto
    - 🟡 **Médio Risco / Oportunidade**: Argumento < corte do 1º Sem, mas ≥ corte do **2º Semestre** → Salvo pelo 2º Semestre
    - 🔴 **Alto Risco**: Argumento < ambos os cortes → Considerar redirecionamento
    """)
    
    if st.session_state.df is None:
        st.warning(":material/warning: Primeiro faça upload dos dados na página 'Análise Temporal'")
        st.stop()
    
    df = st.session_state.df.copy()
    
    # Verifica colunas necessárias
    required_cols = ['P1_PAS1', 'P2_PAS1', 'P1_PAS2', 'P2_PAS2', 'Red_PAS1', 'Red_PAS2']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        st.error(f":material/cancel: Colunas faltando: {', '.join(missing_cols)}")
        st.info(":material/lightbulb: Faça upload de um arquivo com P1/P2/Red para PAS 1 e PAS 2, ou use **Dados de Exemplo**.")
        st.stop()
    
    # Calcula EB se não existir
    if 'EB_PAS1' not in df.columns:
        df['EB_PAS1'] = df['P1_PAS1'] + df['P2_PAS1']
    if 'EB_PAS2' not in df.columns:
        df['EB_PAS2'] = df['P1_PAS2'] + df['P2_PAS2']
    
    # --- MOCK DATA: Adiciona colunas extras se não existirem ---
    n_rows = len(df)
    if 'Unidade' not in df.columns:
        np.random.seed(42)
        df['Unidade'] = np.random.choice(['Asa Sul', 'Taguatinga', 'Lago Sul'], n_rows, p=[0.4, 0.35, 0.25])
    if 'Turma' not in df.columns:
        np.random.seed(43)
        df['Turma'] = np.random.choice(['3º A', '3º B', '3º C'], n_rows)
    if 'Curso_Alvo' not in df.columns:
        np.random.seed(44)
        df['Curso_Alvo'] = np.random.choice([
            'MEDICINA (BACHARELADO)', 'DIREITO (BACHARELADO)', 'ENGENHARIA CIVIL (BACHARELADO)',
            'ADMINISTRAÇÃO (BACHARELADO)', 'PSICOLOGIA (BACHARELADO)', 'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)',
        ], n_rows, p=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
    if 'Cota' not in df.columns:
        # Fallback Determinístico: Se não tem coluna Cota, assume Universal para todos
        # (Usuário optou por remover a aleatoriedade)
        df['Cota'] = 'Sistema Universal'
    else:
        # Se existe a coluna, preenche apenas os buracos (NaN) com Universal
        df['Cota'] = df['Cota'].fillna('Sistema Universal')

    # Removemos a lógica de np.random.choice que causava discrepância
    # (Código legado de mock removido/substituído acima)
    
    # =================================================================
    # FILTROS HIERÁRQUICOS
    # =================================================================
    st.markdown("---")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        unidades_disp = ["Todas"] + sorted(df['Unidade'].unique().tolist())
        unidade_sel = st.selectbox(":material/business: Unidade", unidades_disp, key="ga_unidade")
    
    df_filtrado = df if unidade_sel == "Todas" else df[df['Unidade'] == unidade_sel]
    
    with col_f2:
        turmas_disp = ["Todas"] + sorted(df_filtrado['Turma'].unique().tolist())
        turma_sel = st.selectbox(":material/school: Turma", turmas_disp, key="ga_turma")
    
    if turma_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado['Turma'] == turma_sel]
    
    with col_f3:
        # Triênios Disponíveis (Dinâmico + Defaults)
        default_trienniums = ["2024-2026", "2023-2025"]
        available_trienniums = set(default_trienniums)
        
        if 'Ano_Trienio' in df.columns:
            available_trienniums.update(df['Ano_Trienio'].dropna().unique())
            
        # Ordena: Mais recente primeiro
        trienios_disp = sorted(list(available_trienniums), reverse=True)
        trienio_sel = st.selectbox(":material/calendar_today: Triênio", trienios_disp, key="ga_trienio")
        
    # Filtra por Triênio se a coluna existir
    if 'Ano_Trienio' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Ano_Trienio'] == trienio_sel]

    with col_f4:
        status_filter = st.selectbox(":material/traffic: Status", ["Todos", "🔴 Alto Risco", "🟡 Oportunidade (2º Sem)", "🟢 Baixo Risco"], key="ga_status")
        
    # --- PROJEÇÃO GLOBAL PARA 2024-2026 ---
    stats_p3_global_asset = None
    if trienio_sel == "2024-2026":
        # st.markdown("##### 🔮 Base de Projeção PAS 3 (Massa)")
        base_projecao_asset = st.radio(
            ":material/psychology: Cenário de Dificuldade PAS 3 (Simulação):",
            ["Replicar Padrão 2023-2025", "Utilizar Projeção Tendência"],
            horizontal=True,
            help="Defina qual estatística o sistema deve usar para projetar o futuro (PAS 3).",
            key="ga_base_projecao"
        )
        if base_projecao_asset == "Replicar Padrão 2023-2025":
            stats_p3_global_asset = TRIENNIUM_STATS["2023-2025"]["PAS3"]
        else:
            stats_p3_global_asset = STATS_PAS3_TREND
    
    # =================================================================
    # CÁLCULO DE MÉTRICAS POR ALUNO
    # =================================================================
    
    # Carrega notas de corte para o triênio mais recente
    data_dir = Path(__file__).parent.parent / "data"
    csv_corte = data_dir / "notas_corte_pas.csv"
    
    try:
        df_corte = pd.read_csv(csv_corte)
        # Determina o triênio de referência (anterior ao selecionado)
        try:
            start_year, end_year = map(int, trienio_sel.split('-'))
            trienio_ref = f"{start_year - 1}-{end_year - 1}"
            # Se não houver dados para o triênio anterior imediato, tenta o fallback consolidado
            if trienio_ref not in df_corte['Trienio'].unique():
                trienio_ref = "2022-2024"
        except:
            trienio_ref = "2022-2024"
        
        # --- Carrega cortes do 1º SEMESTRE (TODOS OS SISTEMAS) ---
        df_corte_1sem = df_corte[
            (df_corte['Trienio'] == trienio_ref) & 
            (df_corte['Semestre'] == '1°')
        ]
        # 1º Semestre: MENOR nota (Última Chamada) -> ascending=True
        df_corte_1sem = df_corte_1sem.sort_values('Min', ascending=True).drop_duplicates(
            subset=['Sistema_Nome', 'Curso_Limpo', 'Campus', 'Turno'], keep='first'
        )
        
        # Cria mapa aninhado: Sistema -> Curso (com Turno/Campus) -> Nota
        # chave: "DIREITO (BACHARELADO) - NOTURNO (DARCY RIBEIRO)"
        corte_1sem_map = {}
        for sistema, group in df_corte_1sem.groupby('Sistema_Nome'):
             # Cria dicionário com chaves compostas
             corte_1sem_map[sistema] = {}
             for _, row in group.iterrows():
                 # Chave composta para diferenciar turnos e campi
                 full_key = f"{row['Curso_Limpo']} - {row['Turno']} ({row['Campus']})"
                 corte_1sem_map[sistema][full_key] = row['Min']

        # --- Carrega cortes do 2º SEMESTRE (TODOS OS SISTEMAS) ---
        trienio_ref_2sem = trienio_ref
        has_2sem = not df_corte[(df_corte['Trienio'] == trienio_ref) & (df_corte['Semestre'] == '2°')].empty
        if not has_2sem:
            trienio_ref_2sem = "2022-2024"

        df_corte_2sem = df_corte[
            (df_corte['Trienio'] == trienio_ref_2sem) & 
            (df_corte['Semestre'] == '2°')
        ]
        # 2º Semestre: MAIOR nota (1ª Chamada) -> ascending=False
        df_corte_2sem = df_corte_2sem.sort_values('Min', ascending=False).drop_duplicates(
            subset=['Sistema_Nome', 'Curso_Limpo', 'Campus', 'Turno'], keep='first'
        )
        
        corte_2sem_map = {}
        for sistema, group in df_corte_2sem.groupby('Sistema_Nome'):
             corte_2sem_map[sistema] = {}
             for _, row in group.iterrows():
                 full_key = f"{row['Curso_Limpo']} - {row['Turno']} ({row['Campus']})"
                 corte_2sem_map[sistema][full_key] = row['Min']
        
        # Lista global de sistemas disponíveis
        available_systems = sorted(list(set(corte_1sem_map.keys()) | set(corte_2sem_map.keys())))
        
        # Compatibilidade: usa Universal como fallback para lista de cursos
        corte_por_curso = corte_1sem_map.get('Sistema Universal', {})
        available_courses = tuple(corte_por_curso.keys())
        
    except Exception as e:
        st.error(f"Erro ao carregar banco de cortes: {e}")
        corte_1sem_map = {}
        corte_2sem_map = {}
        corte_por_curso = {}
        available_courses = []
        available_systems = []
        trienio_ref = "N/A"
    
    # Função auxiliar de matching inline (sem cache do Streamlit)
    def _match_curso(nome_aluno_curso, dict_corte):
        """Resolve o nome do curso do aluno para a chave no dicionário de cortes."""
        if not nome_aluno_curso or nome_aluno_curso == 'Não informado':
            return None
        # 1. Exact match
        if nome_aluno_curso in dict_corte:
            return nome_aluno_curso
        # 2. Normaliza e tenta substring bidirecional
        norm_input = unicodedata.normalize('NFKD', nome_aluno_curso).encode('ASCII', 'ignore').decode('utf-8').upper().strip()
        best_match = None
        best_len = 0
        for key in dict_corte:
            norm_key = unicodedata.normalize('NFKD', key).encode('ASCII', 'ignore').decode('utf-8').upper().strip()
            if norm_input == norm_key:
                return key
            if norm_input in norm_key or norm_key in norm_input:
                if len(key) > best_len:
                    best_match = key
                    best_len = len(key)
        if best_match:
            return best_match
        # 3. Fallback: compara só o nome raiz (antes do parêntese)
        root_input = norm_input.split('(')[0].strip()
        for key in dict_corte:
            norm_key = unicodedata.normalize('NFKD', key).encode('ASCII', 'ignore').decode('utf-8').upper().strip()
            root_key = norm_key.split('(')[0].strip()
            if root_input == root_key:
                return key
        return None

    # Calcula métricas por aluno
    resultados = []
    debug_info = []  # Para o expander de debug
    for idx, row in df_filtrado.iterrows():
        nome = row.get('Nome', row.get('Inscricao', f'Aluno {idx}'))
        turma = row.get('Turma', 'N/A')
        unidade = row.get('Unidade', 'N/A')
        curso_alvo = row.get('Curso_Alvo', 'Não informado')
        cota_aluno = row.get('cota', row.get('Cota', row.get('sistema', row.get('sistema_nome', 'Sistema Universal'))))
        
        # --- Normalização Inteligente (Fuzzy Match) ---
        # 1. Normaliza Cota
        # Consolida sistemas disponíveis
        available_systems = list(set(corte_1sem_map.keys()) | set(corte_2sem_map.keys()))
        if available_systems:
            cota_aluno = find_best_match(str(cota_aluno), available_systems, cutoff=0.6)
            
        # 2. Normaliza Curso
        # Usa a cota já normalizada para listar cursos candidatos
        mapa_1 = corte_1sem_map.get(cota_aluno, corte_1sem_map.get('Sistema Universal', {}))
        mapa_2 = corte_2sem_map.get(cota_aluno, corte_2sem_map.get('Sistema Universal', {}))
        
        # Compatibilidade com código antigo (que usava corte_1sem)
        corte_1sem = mapa_1
        corte_2sem = mapa_2
        
        available_courses = list(mapa_1.keys())
        
        # Se não achou cursos na cota, pode ser que a cota esteja errada ou vazia, tenta todos os cursos únicos
        if not available_courses:
             all_courses = set()
             for c_map in corte_1sem_map.values():
                 all_courses.update(c_map.keys())
             available_courses = list(all_courses)
        
        if available_courses:
            curso_alvo = find_best_match(str(curso_alvo), available_courses, cutoff=0.4) # Cutoff baixo para pegar "audiovisual" -> "COMUNICAÇÃO..."
            
        # 3. Identifica corte específico para o curso (agora com chave composta)
        # O find_best_match já nos deu a chave completa: "DIREITO (...) - NOTURNO (...)"
        # Então o get direto deve funcionar
        nota_corte_1sem = mapa_1.get(curso_alvo)
        nota_corte_2sem = mapa_2.get(curso_alvo)
        
        # Tenta fallback de string se não achou (caso o match tenha sido "parcial" ou manual errado)
        if nota_corte_1sem is None and available_courses:
             # Tenta achar de novo nos cursos disponíveis (redudante mas seguro)
             match_retry = find_best_match(str(curso_alvo), available_courses, cutoff=0.6)
             nota_corte_1sem = mapa_1.get(match_retry)
             
        if nota_corte_2sem is None and available_courses:
             match_retry_2 = find_best_match(str(curso_alvo), available_courses, cutoff=0.6)
             nota_corte_2sem = mapa_2.get(match_retry_2)

        # Se ainda None, não temos corte para esse curso/turno específico
        if nota_corte_1sem is None:
            nota_corte_1sem = 0.0 # Indica "N/A"
        
        # Define qual nota de corte usar (1º sem ou 2º sem)
        # Regra Padrão: Usa 1º Semestre
        nota_corte = nota_corte_1sem
        
        # Armazena debug
        curso_matched_1 = curso_alvo if nota_corte_1sem else None
        curso_matched_2 = curso_alvo if nota_corte_2sem else None  # referência principal para cálculos
        
        # Prediz argumento final se modelo disponível
        arg_pred = 0.0
        gap = 0.0
        chance = 0.0
        
        if ARG_FINAL_MODEL is not None:
            try:
                # Features devem ser: [EB_PAS1, Red_PAS1, EB_PAS2, Red_PAS2, C_EB, C_Red]
                # EB = Escore Bruto = P1 + P2; C = variação entre etapas
                eb_p1 = float(row['P1_PAS1']) + float(row['P2_PAS1'])
                red_p1 = float(row.get('Red_PAS1', 6.0))
                eb_p2 = float(row['P1_PAS2']) + float(row['P2_PAS2'])
                red_p2 = float(row.get('Red_PAS2', 6.0))
                c_eb = eb_p2 - eb_p1
                c_red = red_p2 - red_p1
                
                features_aluno = np.array([[eb_p1, red_p1, eb_p2, red_p2, c_eb, c_red]])
                arg_pred = float(ARG_FINAL_MODEL.predict(features_aluno)[0])
                
                if nota_corte is not None:
                    gap = arg_pred - nota_corte
                    # Calcula probabilidade via Z-Score (Igual ao Diagnóstico)
                    if calculate_approval_probability:
                        chance = calculate_approval_probability(arg_pred, nota_corte, rmse=ARG_FINAL_MAE) * 100
                    else:
                        chance = 50.0  # fallback
                else:
                    gap = 0.0
                    chance = 0.0
            except Exception:
                arg_pred = 0.0
                gap = -999.0
                chance = 0.0
        
        # Reality Check (coorte histórica) - AGORA PADRONIZADO COM A CALCULADORA
        historico_pct = 0.0
        historico_err = ""
        df_hist_cohort_debug = pd.DataFrame() 
        
        if calculate_cohort_evolution_probability and nota_corte is not None and TargetCalculator:
            try:
                df_hist_cohort = load_cohort_data()
                df_hist_cohort_debug = df_hist_cohort
                
                if not df_hist_cohort.empty:
                    # Prepara dados para a calculadora (TargetCalculator)
                    notas_aluno = {
                        'P1_PAS1': float(row['P1_PAS1']), 'P2_PAS1': float(row['P2_PAS1']), 
                        'Red_PAS1': float(row.get('Red_PAS1', 6.0)),
                        'P1_PAS2': float(row['P1_PAS2']), 'P2_PAS2': float(row['P2_PAS2']), 
                        'Red_PAS2': float(row.get('Red_PAS2', 6.0))
                    }
                    
                    # Instancia calculadora
                    calc = TargetCalculator()
                    
                    # Define estatísticas baseado no triênio selacionado (GLOBAL)
                    # O seletor estará lá em cima, na configuração.
                    # Mas como isso é um loop, precisamos acessar a variável global de configuração.
                    # Vou assumir que 'stats_p3_global' foi definido antes do loop se for 2024-2026.
                    
                    stats_ciclo_calc = TRIENNIUM_STATS.get(trienio_sel) # trienio_sel vem do Page 5 config
                    
                    if trienio_sel == "2024-2026" and 'stats_p3_global_asset' in locals():
                        stats_p3_calc = stats_p3_global_asset
                    elif trienio_sel == "2023-2025":
                         stats_p3_calc = TRIENNIUM_STATS["2023-2025"]["PAS3"]
                    elif stats_ciclo_calc:
                        stats_p3_calc = stats_ciclo_calc["PAS3"]
                    else:
                        stats_p3_calc = STATS_PAS3_TREND # Fallback
                    
                    if stats_ciclo_calc:
                        # Lógica para determinar qual corte usar no Histórico
                        # Se o aluno é "Yellow" (não passa no 1º mas passa no 2º), usamos a nota do 2º sem
                        cutoff_historico = nota_corte
                        if gap < 0 and nota_corte_2sem is not None:
                            # Verifica se passaria no 2º (Gap 2 >= 0)
                            if arg_pred >= nota_corte_2sem:
                                cutoff_historico = nota_corte_2sem
                        
                        # Calcula o caminho exato para a nota de corte selecionada
                        result_path = calc.calculate_required_score(
                            notas_aluno, cutoff_historico,
                            stats_ciclo_calc["PAS1"], stats_ciclo_calc["PAS2"], stats_p3_calc
                        )
                        
                        # EB Total Necessário = P1_estimado + P2_necessario
                        eb_pas3_nec_real = result_path.p1_estimado + result_path.p2_necessario
                        
                        # Agora sim calcula a probabilidade histórica baseada na meta REAL
                        aluno_dados_hist = {
                            'eb_pas1': eb_p1, 
                            'eb_pas2': eb_p2
                        }
                        
                        prob_h, amostra_h = calculate_cohort_evolution_probability(
                            aluno_dados_hist, eb_pas3_nec_real, df_hist_cohort
                        )
                        historico_pct = prob_h
                        
            except Exception as e:
                historico_err = str(e)

        # Registra debug
        debug_info.append({
            'Nome': nome[:30],
            'Curso Input': curso_alvo[:40],
            'Matched 1ºSem': curso_matched_1[:40] if curso_matched_1 else '❌ NÃO ENCONTRADO',
            'Matched 2ºSem': curso_matched_2[:40] if curso_matched_2 else '❌ NÃO ENCONTRADO',
            'Corte 1ºSem': f"{nota_corte_1sem:.1f}" if nota_corte_1sem else 'N/A',
            'Corte 2ºSem': f"{nota_corte_2sem:.1f}" if nota_corte_2sem else 'N/A',
            'Arg Previsto': f"{arg_pred:.1f}",
            'Modelo?': '✅' if ARG_FINAL_MODEL is not None else '❌',
            'Histórico (%)': f"{historico_pct:.1f}%",
            'Histórico Err': historico_err if historico_err else "OK",
            'Cohort Size': f"{len(df_hist_cohort_debug)}" if not df_hist_cohort_debug.empty else "0",
        })
        
        # =============================================================
        # CLASSIFICAÇÃO DUPLO CORTE (1º e 2º Semestre)
        # =============================================================
        sugestao = ""
        
        if nota_corte_1sem is not None and arg_pred >= nota_corte_1sem:
            # Cenário 1: Aprovado direto no 1º semestre
            status = "🟢"
            status_level = 'green'
        elif nota_corte_2sem is not None and arg_pred >= nota_corte_2sem:
            # Cenário 2: Não passa no 1º, mas passa no 2º semestre
            status = "🟡"
            status_level = 'yellow'
            sugestao = "Aprovado no 2º Semestre"
        else:
            # Cenário 3: Não passa em nenhum dos dois
            status = "🔴"
            status_level = 'red'
        
        # Sugestão DUPLA (1º e 2º Semestre)
        # Se status for RED ou YELLOW, buscamos opções melhores.
        sugestao_1sem = None
        sugestao_2sem = None
        
        if status_level in ['red', 'yellow'] and calculate_approval_probability:
            # Busca melhor opção para 1º Semestre (que seja GREEN)
            melhor_gap_1 = -float('inf')
            
            # Ordena cursos por corte (do maior pro menor) para pegar o "melhor" possível que ele passa
            # Ou pegar o mais difícil que ele passa? Normalmente queremos o curso de maior prestígio (maior nota) que ele passa.
            for curso_alt, corte_alt in sorted(corte_1sem.items(), key=lambda x: x[1], reverse=True):
                 if curso_alt == curso_matched_1: continue
                 
                 # Checa se passa com segurança (ex: > 80% chance ou gap positivo)
                 # Usando gap positivo como critério base de "Green" simplificado
                 if arg_pred >= corte_alt:
                     # Verifica probabilidade para garantir
                     try:
                         prob = calculate_approval_probability(arg_pred, corte_alt, rmse=ARG_FINAL_MAE)
                         if prob >= 0.8:
                             sugestao_1sem = curso_alt.split(' (')[0]
                             break # Achou o curso com maior nota de corte que ele passa
                     except: pass
            
            # Busca melhor opção para 2º Semestre (que seja GREEN)
            for curso_alt, corte_alt in sorted(corte_2sem.items(), key=lambda x: x[1], reverse=True):
                 if curso_alt == curso_matched_2: continue
                 if arg_pred >= corte_alt:
                     try:
                         prob = calculate_approval_probability(arg_pred, corte_alt, rmse=ARG_FINAL_MAE)
                         if prob >= 0.8:
                             sugestao_2sem = curso_alt.split(' (')[0]
                             break 
                     except: pass

        # Formata string de sugestão
        sugestao_final_parts = []
        if sugestao_1sem:
            sugestao_final_parts.append(f"1º Sem: {sugestao_1sem}")
        if sugestao_2sem:
            sugestao_final_parts.append(f"2º Sem: {sugestao_2sem}")
            
        if sugestao_final_parts:
            sugestao = " | ".join(sugestao_final_parts)
        else:
            sugestao = ""
            
        # Se status for YELLOW, mostra a chance do 2º semestre explicitamente
        chance_display = f"{chance:.1f}%"
        if status_level == 'yellow':
             # Recalcula chance para o 2º semestre usando nota_corte_2sem
             if nota_corte_2sem:
                 try:
                     chance_2sem = calculate_approval_probability(arg_pred, nota_corte_2sem, rmse=ARG_FINAL_MAE) * 100
                     chance_display = f"1º: {chance:.1f}% | 2º: {chance_2sem:.1f}%"
                 except: pass
        
        resultados.append({
            'Status': status,
            'Status_Level': status_level,
            'Nome': nome,
            'Turma': turma,
            'Sistema de Concorrência': cota_aluno,
            'Curso Alvo': curso_alvo, 
            'Gap': round(gap, 1),
            'Chance': chance_display,
            'Histórico (%)': round(historico_pct, 1),
            'Sugestão': sugestao if sugestao else '—',
        })
    
    if not resultados:
        # Garante que as colunas existam mesmo que o DF esteja vazio
        df_result = pd.DataFrame(columns=[
            'Status', 'Status_Level', 'Nome', 'Turma', 'Sistema de Concorrência', 'Curso Alvo', 
            'Gap', 'Chance', 'Histórico (%)', 'Sugestão'
        ])
    else:
        df_result = pd.DataFrame(resultados)
    
    # Aplica filtro de status
    if status_filter == "🔴 Alto Risco":
        df_result = df_result[df_result['Status_Level'] == 'red']
    elif status_filter == "🟡 Oportunidade (2º Sem)":
        df_result = df_result[df_result['Status_Level'] == 'yellow']
    elif status_filter == "🟢 Baixo Risco":
        df_result = df_result[df_result['Status_Level'] == 'green']
    
    # =================================================================
    # KPIs (CARDS)
    # =================================================================
    col1, col2, col3, col4 = st.columns(4)
    
    total_alunos = len(df_result)
    n_red = (df_result['Status_Level'] == 'red').sum()
    n_yellow = (df_result['Status_Level'] == 'yellow').sum()
    n_green = (df_result['Status_Level'] == 'green').sum()
    
    with col1:
        with st.container(border=True):
            st.metric(":material/inventory_2: Total de Ativos", total_alunos)
    with col2:
        st.markdown(f"""
        <div style="background-color: #FFCDD2; padding: 15px; border-radius: 10px; text-align: center;">
            <p style="margin:0; font-size: 0.9em; color: #B71C1C;">🔴 Alto Risco</p>
            <h2 style="margin:0; color: #D32F2F;">{n_red}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background-color: #FFF9C4; padding: 15px; border-radius: 10px; text-align: center;">
            <p style="margin:0; font-size: 0.9em; color: #F57F17;">🟡 Oportunidade (2º Sem)</p>
            <h2 style="margin:0; color: #FFA000;">{n_yellow}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background-color: #C8E6C9; padding: 15px; border-radius: 10px; text-align: center;">
            <p style="margin:0; font-size: 0.9em; color: #1B5E20;">🟢 Baixo Risco</p>
            <h2 style="margin:0; color: #388E3C;">{n_green}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =================================================================
    # TABELA PRINCIPAL (PROFISSIONAL)
    # =================================================================
    st.markdown("### :material/dashboard: Painel de Ativos")
    st.caption(f"Referência: Triênio {trienio_ref} | Universal | Última Chamada")
    
    # Ordena: red primeiro, depois yellow, depois green
    order_map = {'red': 0, 'yellow': 1, 'green': 2}
    df_result['_order'] = df_result['Status_Level'].map(order_map)
    df_display = df_result.sort_values('_order').drop(columns=['Status_Level', '_order'])
    
    # Configuração profissional das colunas
    column_config = {
        'Status': st.column_config.TextColumn('Status', width='small'),
        'Nome': st.column_config.TextColumn('Nome', width='medium'),
        'Turma': st.column_config.TextColumn('Turma', width='small'),
        'Sistema de Concorrência': st.column_config.TextColumn('Sistema de Concorrência', width='medium'),
        'Curso Alvo': st.column_config.TextColumn('Curso Alvo', width='medium'),
        'Gap': st.column_config.NumberColumn(
            'Gap',
            format="%+.1f",
            help="Distância para a nota de corte (+ = acima, - = abaixo)",
        ),
        'Chance': st.column_config.TextColumn(
            'Chance',
            help="Probabilidade de aprovação baseada no modelo ML (Incerteza da Previsão)",
        ),
        'Histórico (%)': st.column_config.ProgressColumn(
            'Histórico',
            format="%.1f%%",
            min_value=0,
            max_value=100,
            help="Reality Check: % de alunos similares que alcançaram a Nota Exata necessária para este curso",
        ),
        'Sugestão': st.column_config.TextColumn(
            'Sugestão',
            width='medium',
            help="Curso alternativo na zona verde (≥80% de chance)",
        ),
    }
    
    st.dataframe(
        df_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=min(35 * len(df_display) + 38, 600),  # Auto-height com cap
    )
    
    # Rodapé com insights
    if n_red > 0:
        pct_risco = (n_red / total_alunos * 100) if total_alunos > 0 else 0
        st.warning(f":material/warning: **{n_red} alunos ({pct_risco:.0f}%)** estão na zona vermelha e podem precisar de redirecionamento de curso.")


# =============================================================================
# PÁGINA 3: PREDITOR PAS 3 (USANDO MODELO ML)
# =============================================================================

elif page == "preditor":
    st.title(":material/model_training: Preditor PAS 3")

    # --- INTEGRAÇÃO SUPABASE: JÁ REALIZADA GLOBALMENTE NO INÍCIO ---
    # Apenas verifica se tem dados
    if st.session_state.get('df_global_escola') is None:
         # Tenta buscar novamente (Fallback)
         with st.spinner("Buscando dados na nuvem..."):
            df_nuvem = buscar_alunos_nuvem_global()
            if df_nuvem is not None:
                st.session_state['df_global_escola'] = df_nuvem
                st.rerun()

    models_loaded = sum(1 for m in MODELS.values() if m is not None)
    if models_loaded == 0:
        st.error(":material/error: Nenhum modelo carregado. Verifique se os arquivos .joblib existem em models/")
        st.stop()

    # --- CONFIGURAÇÃO (GLOBAL) ---
    
    # 0. Preparação de Listas (Cota) - Definido antes p/ uso no Auto-Fill
    lista_cotas = sorted(df_notas['Sistema_Nome'].unique().astype(str).tolist())
    if 'Sistema Universal' in lista_cotas:
        lista_cotas.insert(0, lista_cotas.pop(lista_cotas.index('Sistema Universal')))

    # --- MODO DE OPERAÇÃO: BUSCA vs MANUAL ---
    st.markdown("### :material/search: Seleção do Aluno")
    
    # Verifica se existe base carregada
    tem_base = 'df_global_escola' in st.session_state and st.session_state['df_global_escola'] is not None
    
    # Seletor de Modo
    modo_selecao = st.radio(
        "Modo de Entrada:",
        ["🔍 Buscar na Base da Escola", "✍️ Entrada Manual"],
        index=0 if tem_base else 1,
        horizontal=True,
        disabled=not tem_base,
        help="Use 'Buscar' para carregar dados automaticamente da planilha da escola. Use 'Manual' para simular livremente."
    )
    
    if not tem_base:
        st.warning("⚠️ Nenhuma base escolar carregada. Vá para a aba **Análise Temporal** e faça upload da planilha para habilitar a busca.")

    # --- LÓGICA DE BUSCA (DRILL-DOWN) ---
    aluno_selecionado = None
    
    if modo_selecao == "🔍 Buscar na Base da Escola" and tem_base:
        df_escola = st.session_state['df_global_escola']
        
        c_unidade, c_turma, c_aluno = st.columns([1, 1, 2])
        
        # 1. Filtro Unidade (se existir coluna)
        if 'Unidade' in df_escola.columns:
            unidades = sorted(df_escola['Unidade'].dropna().unique())
            unidade_sel = c_unidade.selectbox("Unidade", ["Todas"] + list(unidades))
            if unidade_sel != "Todas":
                df_escola = df_escola[df_escola['Unidade'] == unidade_sel]
        else:
            c_unidade.info("Col. 'Unidade' não encontrada")

        # 2. Filtro Turma (se existir coluna)
        if 'Turma' in df_escola.columns:
            turmas = sorted(df_escola['Turma'].dropna().unique())
            turma_sel = c_turma.selectbox("Turma", ["Todas"] + list(turmas))
            if turma_sel != "Todas":
                df_escola = df_escola[df_escola['Turma'] == turma_sel]
        else:
            c_turma.info("Col. 'Turma' não encontrada")
            
        # 3. Filtro Aluno
        if 'Nome' in df_escola.columns:
            alunos = sorted(df_escola['Nome'].dropna().unique())
            nome_aluno = c_aluno.selectbox("Aluno", ["Selecione..."] + list(alunos))
            
            if nome_aluno != "Selecione...":
                aluno_selecionado = df_escola[df_escola['Nome'] == nome_aluno].iloc[0]
                st.success(f"Dados de **{nome_aluno}** carregados!")
                
                # --- AUTO-PREENCHIMENTO (SYNC STATE) ---
                # Atualiza as chaves do session_state que alimentam os inputs
                
                def safe_get(row, col):
                    val = row.get(col, 0.0)
                    return float(val) if pd.notnull(val) else 0.0

                st.session_state['input_p1_pas1'] = safe_get(aluno_selecionado, 'P1_PAS1')
                st.session_state['input_p2_pas1'] = safe_get(aluno_selecionado, 'P2_PAS1')
                st.session_state['input_red_pas1'] = safe_get(aluno_selecionado, 'Red_PAS1')
                
                st.session_state['input_p1_pas2'] = safe_get(aluno_selecionado, 'P1_PAS2')
                st.session_state['input_p2_pas2'] = safe_get(aluno_selecionado, 'P2_PAS2')
                st.session_state['input_red_pas2'] = safe_get(aluno_selecionado, 'Red_PAS2')
                
                # Metadados Opcionais (Cota, Curso)
                # Verifica se existe coluna de Cota e tenta selecionar
                col_cota = None
                for c in ['Sistema_Nome', 'Cota', 'Sistema', 'Sistema_Concorrencia']:
                    if c in df_escola.columns:
                        col_cota = c
                        break
                
                if col_cota:
                    cota_aluno = str(aluno_selecionado[col_cota]).strip()
                    # Usa Fuzzy Match para encontrar a cota oficial
                    match_cota = find_best_match(cota_aluno, lista_cotas)
                    if match_cota in lista_cotas:
                        st.session_state['input_cota'] = match_cota
                    else:
                         # Fallback (tenta conter)
                         found = False
                         for opt in lista_cotas:
                             if cota_aluno.lower() in opt.lower() or opt.lower() in cota_aluno.lower():
                                 st.session_state['input_cota'] = opt
                                 found = True
                                 break
                         if not found:
                             st.toast(f"Cota '{cota_aluno}' não encontrada na lista oficial.")
                                
                # Auto-fill Triênio
                col_trienio = None
                for c in ['Ano_Trienio', 'Trienio', 'Ciclo']:
                    if c in df_escola.columns:
                        col_trienio = c
                        break
                
                if col_trienio:
                    trienio_aluno = str(aluno_selecionado[col_trienio]).strip()
                    # Verifica se o triênio existe nas opções disponíveis
                    if trienio_aluno in TRIENNIUM_STATS:
                        st.session_state['input_trienio'] = trienio_aluno
                
                # Nome do Aluno (Novo Sync)
                if 'Nome' in df_escola.columns:
                     st.session_state['input_nome_aluno'] = str(aluno_selecionado['Nome'])
                
        else:
            st.error("Coluna 'Nome' obrigatória não encontrada na base.")


    # --- CONTROLES DE SIMULAÇÃO (CARREGAR ÚLTIMA) ---
    col_load, _ = st.columns([1, 3])
    with col_load:
        if st.button("🔄 Carregar Última Simulação"):
            if 'historico_ultimo_calculo' in st.session_state:
                hist = st.session_state['historico_ultimo_calculo']
                for k, v in hist.items():
                    st.session_state[k] = v
                st.toast("Simulação anterior restaurada!")
            else:
                st.warning("Nenhuma simulação salva na memória.")

    st.markdown("### :material/settings: Configuração do Candidato")
    
    col_sem, col_tri, col_cota = st.columns([1, 1, 2])
    
    with col_sem:
        st.markdown("**:material/calendar_month: Semestre**")
        semester_option = st.radio(
            "Semestre", ["1º Semestre", "2º Semestre"], 
            label_visibility="collapsed", horizontal=True
        )
        semester_db = "1°" if semester_option == "1º Semestre" else "2°"
        semester_int = 1 if semester_option == "1º Semestre" else 2

    with col_tri:
        st.markdown("**:material/school: Triênio**")
        ciclo_aluno = st.selectbox(
            "Triênio", list(TRIENNIUM_STATS.keys()), 
            label_visibility="collapsed",
            key="input_trienio" # Key vinculada ao auto-fill
        )
        stats_ciclo = TRIENNIUM_STATS[ciclo_aluno]
        # Lógica de referência (Ano Anterior)
        try:
            start_year, end_year = map(int, ciclo_aluno.split('-'))
            ref_triennium = f"{start_year - 1}-{end_year - 1}"
        except:
            ref_triennium = "2022-2024"

    with col_cota:
        st.markdown("**:material/label: Sistema de Concorrência (Cota)**")
        
        cota_selecionada = st.selectbox(

            "Cota", lista_cotas, 
            label_visibility="collapsed",
            key="input_cota" # Key vinculada ao auto-fill
        )

    st.caption(f":material/info: Referência: **{ref_triennium}** | Cota: **{cota_selecionada}**")




    # --- ABAS ---
    tab_diagnostico, tab_estrategia = st.tabs([":material/psychology: Diagnóstico Realista", ":material/track_changes: Calculadora de Estratégia"])

    # =========================================================================
    # ABA 1: DIAGNÓSTICO
    # =========================================================================
    with tab_diagnostico:
        
        col1, col2 = st.columns(2)
        
        # Helper para inicializar key se não existir (evita erro no primeiro render manual)
        def init_key(key, default=0.0):
            if key not in st.session_state:
                st.session_state[key] = default
        
        # Init Name Key (String)
        if 'input_nome_aluno' not in st.session_state:
            st.session_state['input_nome_aluno'] = ""

        init_key('input_p1_pas1'); init_key('input_p2_pas1'); init_key('input_red_pas1')
        init_key('input_p1_pas2'); init_key('input_p2_pas2'); init_key('input_red_pas2')

        # --- NOVO: Nome do Aluno (Manual Input) ---
        # Só exibe input se estiver em modo Manual (Em modo busca, o nome vem do selectbox e é read-only virtualmente)
        if modo_selecao == "✍️ Entrada Manual":
             st.text_input("Nome do Aluno", key="input_nome_aluno", placeholder="Digite o nome do estudante...")
        else:
             # Mostra nome carregado apenas como info visual
             st.info(f"Aluno Selecionado: **{st.session_state.get('input_nome_aluno', 'Estudante')}**")

        with col1:
            st.markdown("### :material/edit_note: Notas do PAS 1")
            
            p1_pas1 = st.number_input(
                "P1 PAS 1 (Língua Estrangeira)", -20.0, 20.0, step=0.001, format="%.3f",
                key="input_p1_pas1" # Key vinculada ao auto-fill
            )
            p2_pas1 = st.number_input(
                "P2 PAS 1 (Conhecimentos)", -100.0, 100.0, step=0.001, format="%.3f",
                key="input_p2_pas1"
            )
            red_pas1 = st.number_input(
                "Redação PAS 1", 0.0, 10.0, step=0.001, format="%.3f",
                key="input_red_pas1"
            )
            
        with col2:
            st.markdown("### :material/edit_note: Notas do PAS 2")
            p1_pas2 = st.number_input(
                "P1 PAS 2", -20.0, 20.0, step=0.001, format="%.3f",
                key="input_p1_pas2"
            )
            p2_pas2 = st.number_input(
                "P2 PAS 2", -100.0, 100.0, step=0.001, format="%.3f",
                key="input_p2_pas2"
            )
            red_pas2 = st.number_input(
                "Redação PAS 2", 0.0, 10.0, step=0.001, format="%.3f",
                key="input_red_pas2"
            )
        
        missing_data = any(v is None for v in [p1_pas1, p2_pas1, red_pas1, p1_pas2, p2_pas2, red_pas2])
        
        if not missing_data and st.button("🔮 Gerar Diagnóstico Oficial", type="primary"):
            try:
                # Recupera valores do session_state (garantindo float)
                def get_val(key): return float(st.session_state.get(key, 0.0))
                
                v_p1_pas1 = get_val('input_p1_pas1')
                v_p2_pas1 = get_val('input_p2_pas1')
                v_red_pas1 = get_val('input_red_pas1')
                v_p1_pas2 = get_val('input_p1_pas2')
                v_p2_pas2 = get_val('input_p2_pas2')
                v_red_pas2 = get_val('input_red_pas2')

                # Salva snapshot para "Carregar Última"
                # Salva snapshot para "Carregar Última"
                st.session_state['historico_ultimo_calculo'] = {
                    'input_p1_pas1': v_p1_pas1, 'input_p2_pas1': v_p2_pas1, 'input_red_pas1': v_red_pas1,
                    'input_p1_pas2': v_p1_pas2, 'input_p2_pas2': v_p2_pas2, 'input_red_pas2': v_red_pas2,
                    'input_cota': st.session_state.get('input_cota'),
                    'input_trienio': st.session_state.get('input_trienio'),
                    'input_nome_aluno': st.session_state.get('input_nome_aluno', '')
                }

                # Cálculo Original
                eb_pas1, eb_pas2 = v_p1_pas1 + v_p2_pas1, v_p1_pas2 + v_p2_pas2
                cresc_eb, cresc_red = eb_pas2 - eb_pas1, v_red_pas2 - v_red_pas1
                
                features = np.array([[eb_pas1, v_red_pas1, eb_pas2, v_red_pas2, cresc_eb, cresc_red]])
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
                        eb_pas1, v_red_pas1, eb_pas2, v_red_pas2,
                        cresc_eb, cresc_red,
                        abs(cresc_eb)/(abs(eb_pas1)+0.01), abs(cresc_red)/(abs(v_red_pas1)+0.01),
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
                    'red_pas1': v_red_pas1, 'red_pas2': v_red_pas2,
                    'p1_pas1': v_p1_pas1, 'p2_pas1': v_p2_pas1,
                    'p1_pas2': v_p1_pas2, 'p2_pas2': v_p2_pas2,
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
            st.markdown("### :material/calculate: Previsões do Modelo")
            
            c_eb, c_arg = st.columns(2)
            
            with c_eb:
                with st.container(border=True):
                    mae_eb = MODEL_MAE.get(recommended_model, 0.0)
                    st.metric("EB PAS 3 Previsto", f"{recommended_eb:.3f}", help=f"Modelo: {recommended_model.upper()}")
                    st.caption(f"Margem de erro estimada: ± {mae_eb:.2f}")
            
            with c_arg:
                with st.container(border=True):
                    st.metric("Argumento Final Previsto", f"{arg_final_pred:.3f}")
                    st.caption(f"Margem de erro estimada: ± {ARG_FINAL_MAE:.2f}")
            
            st.markdown("---")
            st.markdown("#### :material/tune: Ajuste de Cenário")
            arg_ajustado = st.slider(
                "🎯 Simulador de Meta: Ajuste para ver suas chances com notas maiores",
                min_value=float(arg_final_pred - ARG_FINAL_MAE),
                max_value=float(arg_final_pred + ARG_FINAL_MAE),
                value=float(arg_final_pred),
                step=0.1, format="%.3f"
            )
            
            # --- ANÁLISE DE PROBABILIDADE (ORIGINAL + COTA) ---
            st.markdown(f"#### :material/school: Análise de Probabilidade ({semester_option})")
            
            # 1. Filtra Dados pela COTA SELECIONADA (SEM FILTRO DE CHAMADA FIXA)
            df_cota_raw = df_notas[
                (df_notas['Trienio'] == ref_triennium) & 
                (df_notas['Semestre'] == semester_db) &
                (df_notas['Sistema_Nome'] == cota_selecionada)
            ].copy()
            
            # 2. LIMPEZA RIGOROSA (Step 540) - Evita duplicatas por espaços em branco
            for c in ['Curso_Limpo', 'Campus', 'Turno', 'Chamada']:
                if c in df_cota_raw.columns:
                    df_cota_raw[c] = df_cota_raw[c].astype(str).str.strip()
            
            # Cria identificador único para deduplicação
            df_cota_raw['Combo_Nome'] = df_cota_raw['Curso_Limpo'] + " (" + df_cota_raw['Campus'] + " - " + df_cota_raw['Turno'] + ")"
            
            # 3. DEDUPLICAÇÃO INTELIGENTE
            if not df_cota_raw.empty:
                # Extrai numeral da chamada para garantir ordenação correta (1ª < 2ª)
                df_cota_raw['Chamada_Num'] = df_cota_raw['Chamada'].str.extract('(\d+)').fillna(0).astype(int)
                
                if semester_int == 1:
                    # 1º Semestre: Prioridade para ÚLTIMA CHAMADA (Menor Nota = Piso de entrada)
                    df_cota_clean = df_cota_raw.sort_values(['Combo_Nome', 'Min'], ascending=[True, True]).drop_duplicates('Combo_Nome', keep='first')
                else:
                    # 2º Semestre: Prioridade para PRIMEIRA CHAMADA DISPONÍVEL (Maior Nota = Corte inicial)
                    df_cota_clean = df_cota_raw.sort_values(['Combo_Nome', 'Chamada_Num'], ascending=[True, True]).drop_duplicates('Combo_Nome', keep='first')
            else:
                df_cota_clean = df_cota_raw.copy()

            opcoes_lista = sorted(df_cota_clean['Combo_Nome'].unique().tolist())
            
            # Cria dicionário de referência para o selectbox
            ultimas_chamadas = {}
            for _, row in df_cota_clean.iterrows():
                ultimas_chamadas[row['Combo_Nome']] = {
                    'nota': row['Min'],
                    'chamada': row['Chamada']
                }
            
            # Fallback for empty dictionary if needed
            if not ultimas_chamadas and not df_cota_clean.empty:
                 for _, row in df_cota_clean.iterrows():
                     ultimas_chamadas[row['Combo_Nome']] = {'nota': row['Min'], 'chamada': row['Chamada']}
            
            # Seletor de Curso
            curso_combo_sel = st.selectbox(
                "Selecione um curso de interesse:", 
                ["Selecione..."] + opcoes_lista,
                format_func=lambda x: x if x == "Selecione..." else f"{x} [Corte ({ultimas_chamadas[x]['chamada']}): {ultimas_chamadas[x]['nota']:.3f}]"
            )
            
            if curso_combo_sel != "Selecione...":
                # Extrai os dados do curso selecionado via Combo_Nome
                row_sel = df_cota_raw[df_cota_raw['Combo_Nome'] == curso_combo_sel].iloc[0]
                curso_selecionado = row_sel['Curso_Limpo']
                campus_sel = row_sel['Campus']
                turno_sel = row_sel['Turno']
                
                # Busca a chamada correta baseada na regra de negócio
                # Reutilizamos df_cota_raw que já está filtrado por Cota/Ref/Semestre
                # Apenas filtramos pelo curso específico
                df_base_curso = df_cota_raw[df_cota_raw['Combo_Nome'] == curso_combo_sel]
                
                if semester_int == 1:
                    # 1º Semestre: Menor Nota (Última Chamada)
                    df_chamadas_curso = df_base_curso.sort_values('Min', ascending=True)
                else:
                    # 2º Semestre: Primeira Chamada Disponível (Maior Nota/Mais Restritiva)
                    df_chamadas_curso = df_base_curso.sort_values('Chamada', ascending=True)
                
                if not df_chamadas_curso.empty:
                    ultima_chamada = df_chamadas_curso.iloc[0]
                    nota_corte = ultima_chamada['Min']
                    chamada_ref = ultima_chamada['Chamada']
                else:
                    nota_corte = 0.0
                    chamada_ref = 'N/A'
                
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
                
                # --- NOVO: GRÁFICO DE EVOLUÇÃO DE CORTE (Real Data) ---
                # --- NOVO: GRÁFICO DE EVOLUÇÃO DE CORTE (Lógica Refeita - Step 932) ---
                st.divider()
                st.markdown(f"##### 📉 Análise de Tendência de Corte: {curso_selecionado}")

                try:
                    # --- 1. FILTRAGEM INICIAL E EXTRAÇÃO DO ANO ---
                    # Garante que estamos olhando apenas para o contexto exato
                    df_evolucao = df_notas[
                        (df_notas['Curso_Limpo'] == curso_selecionado) &
                        (df_notas['Campus'] == campus_sel) &
                        (df_notas['Turno'] == turno_sel) &
                        (df_notas['Semestre'] == semester_db)
                    ].copy()

                    coluna_trienio = 'Trienio' # Ajustado para o nome real
                    if coluna_trienio in df_evolucao.columns:
                        df_evolucao['Ano_X'] = df_evolucao[coluna_trienio].astype(str).apply(
                            lambda x: x.split('-')[-1].strip() if '-' in x else x.strip()
                        )
                    else:
                        # Fallback se a coluna for 'Ano'
                        df_evolucao['Ano_X'] = df_evolucao['Ano'].astype(str).str.strip()

                    # Filtro de Cota (Universal ou Selecionada)
                    cota_col = 'Sistema_Nome'
                    if cota_selecionada:
                         df_cota = df_evolucao[df_evolucao[cota_col] == cota_selecionada]
                         if not df_cota.empty:
                             df_evolucao = df_cota
                         else:
                             st.caption(f"Sem dados para a cota '{cota_selecionada}'. Mostrando Sistema Universal.")
                             df_evolucao = df_evolucao[df_evolucao[cota_col].astype(str).str.contains("Universal", case=False, na=False)]
                    else:
                        df_evolucao = df_evolucao[df_evolucao[cota_col].astype(str).str.contains("Universal", case=False, na=False)]

                    # --- 2. LÓGICA DE ÚLTIMA CHAMADA (CRÍTICO) ---
                    coluna_chamada = 'Chamada' # Ajustado para o nome real
                    
                    if not df_evolucao.empty and coluna_chamada in df_evolucao.columns:
                        # Extrai apenas o número da string (ex: "3ª Chamada" vira 3.0) para podermos ordenar matematicamente
                        # Regex extrai digitos. astype(float) lida com NaNs se houver.
                        df_evolucao['Chamada_Num'] = df_evolucao[coluna_chamada].astype(str).str.extract(r'(\d+)').astype(float)
                        
                        # Ordena o dataframe: Ano (Crescente) -> Chamada_Num (Decrescente)
                        # Assim, a linha da "3ª Chamada" fica acima da "1ª Chamada" no mesmo ano.
                        df_sorted = df_evolucao.sort_values(by=['Ano_X', 'Chamada_Num'], ascending=[True, False])
                        
                        # Remove as duplicatas de ano, mantendo apenas a primeira aparição (que será a maior/última chamada)
                        df_clean = df_sorted.drop_duplicates(subset=['Ano_X'], keep='first').copy()
                    else:
                        
                        coluna_nota = 'Min' # Ajuste para 'Min' se 'Nota' não existir
                        if not df_evolucao.empty:
                            df_clean = df_evolucao.sort_values(by=['Ano_X', coluna_nota], ascending=[True, True]).drop_duplicates(subset=['Ano_X'], keep='first')
                        else:
                            df_clean = pd.DataFrame()


                    
                    # --- BLOCO DE PLOTAGEM (CORREÇÃO DE COLUNAS - Step 944) ---
                    if not df_clean.empty:
                        # 1. Debug de Segurança: Limpeza Rigorosa de Ano (4 dígitos numéricos) e DEDUPLICAÇÃO FINAL
                        # Garante que '20212' vire '2021' e remove lixo
                        df_clean['Ano_X'] = df_clean['Ano_X'].astype(str).str.slice(0, 4)
                        
                        # Filtra apenas anos válidos (4 dígitos numéricos)
                        df_clean = df_clean[df_clean['Ano_X'].str.match(r'^\d{4}$')]
                        
                        # CRÍTICO: Agrupa novamente para garantir 1 ponto por ano
                        # 1º Semestre: Nota Mínima (Última Chamada) | 2º Semestre: Nota Máxima (1ª Chamada)
                        if semester_int == 1:
                            df_clean = df_clean.groupby('Ano_X', as_index=False)['Min'].min()
                        else:
                            df_clean = df_clean.groupby('Ano_X', as_index=False)['Min'].max()
                        
                        # Ordena e reseta index
                        df_clean = df_clean.sort_values('Ano_X').reset_index(drop=True)

                        # 2. Define o Y máximo para o gráfico não cortar a bolinha
                        if not df_clean.empty:
                            y_max = df_clean['Min'].max() 
                            y_min = df_clean['Min'].min()

                            fig = go.Figure()

                            # 3. CRIAÇÃO DA LINHA
                            # Usa listas Python puras para garantir que o Plotly não se confunda com Index ou Series
                            x_vals = df_clean['Ano_X'].tolist()
                            y_vals = df_clean['Min'].tolist()

                            fig.add_trace(go.Scatter(
                                x=x_vals, 
                                y=y_vals,
                                mode='lines+markers+text',
                                name='Nota de Corte',
                                line=dict(color='#003366', width=4, shape='linear'), 
                                marker=dict(size=10, color='white', line=dict(width=2, color='#003366')),
                                text=[f"{n:.2f}" for n in y_vals], 
                                textposition="top center"
                            ))

                            # 4. Annotation (Destaque do valor atual)
                            if len(x_vals) > 0:
                                ultimo_ano = x_vals[-1]
                                ultima_nota = y_vals[-1]

                                fig.add_annotation(
                                    x=ultimo_ano,
                                    y=ultima_nota,
                                    text=f"Atual: {ultima_nota:.2f}",
                                    showarrow=True,
                                    arrowhead=2,
                                    ax=0,
                                    ay=-40,
                                    bgcolor="#003366",
                                    bordercolor="#003366",
                                    font=dict(color="white")
                                )

                            # 5. Layout Limpo (SEM type='category' forçado)
                            fig.update_layout(
                                title=dict(
                                    text=f"Tendência: {curso_selecionado}",
                                    font=dict(size=18, color='#003366')
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(
                                    showgrid=False, 
                                    showline=True, 
                                    linecolor='#cccccc',
                                    # type='category' <--- REMOVIDO para evitar bugs com "20212"
                                    tickmode='linear' # Força mostrar todos os anos se possível
                                ),
                            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False, range=[y_min - 10, y_max + 10]),
                            margin=dict(l=20, r=20, t=60, b=20),
                            height=350
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("Dados históricos insuficientes para gerar gráfico de tendência.")
                    
                except Exception as e:
                    st.caption(f"Não foi possível gerar o gráfico de tendência: {e}")

                # --- NOVO: HISTÓRICO DE CHAMADAS (O que você pediu) ---
                st.markdown("##### :material/history: Histórico de Chamadas (Lista de Espera)")
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
            st.markdown(f"#### :material/domain: Cursos ao seu alcance (Top 10 no Sistema de Concorrência)")
            
            # Recalcula probabilidades para TODOS os cursos da cota (USANDO DATAFRAME LIMPO)
            if not df_cota_clean.empty and calculate_approval_probability:
                # Copiamos para evitar SettingWithCopyWarning
                df_recomenda = df_cota_clean.copy()
                
                df_recomenda['Chance %'] = df_recomenda['Min'].apply(
                    lambda x: calculate_approval_probability(arg_ajustado, x, rmse=ARG_FINAL_MAE) * 100
                )
                
                # Ordena pela proximidade da nota (Radar de cursos viáveis)
                df_recomenda['Dist'] = abs(df_recomenda['Min'] - arg_ajustado)
                closest = df_recomenda.sort_values('Dist').head(10)
                
                st.dataframe(
                    closest[['Curso_Limpo', 'Campus', 'Turno', 'Min', 'Chance %']].rename(columns={'Curso_Limpo': 'Curso', 'Min': 'Corte'}),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Chance %": st.column_config.ProgressColumn(
                            "Chance de Aprovação",
                            format="%d%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "Corte": st.column_config.NumberColumn(
                            "Nota de Corte",
                            format="%.3f"
                        )
                    }
                )

    # =========================================================================
    # ABA 2: CALCULADORA (MANTER ORIGINAL COM FILTRO DE COTA)
    # =========================================================================
    with tab_estrategia:
        
        if 'prediction_results' in st.session_state and TargetCalculator:
            res = st.session_state.prediction_results
            # Prepara notas (usa valores reais de P1 e P2 do session_state)
            def get_val(key): return float(st.session_state.get(key, 0.0))
            
            notas_validas = {
                'P1_PAS1': get_val('input_p1_pas1'), 
                'P2_PAS1': get_val('input_p2_pas1'), 
                'Red_PAS1': get_val('input_red_pas1'),
                'P1_PAS2': get_val('input_p1_pas2'), 
                'P2_PAS2': get_val('input_p2_pas2'), 
                'Red_PAS2': get_val('input_red_pas2')
            }
            calc = TargetCalculator()
            
            st.markdown(f"### :material/target: Meta ({semester_option} | {cota_selecionada})")
            
            # Filtro para Dropdown (Lógica Ajustada - Step Refactor)
            # 1. Filtra Triênio, Semestre e Cota
            df_estrat_base = df_notas[
                (df_notas['Trienio'] == ref_triennium) & 
                (df_notas['Semestre'] == semester_db) &
                (df_notas['Sistema_Nome'] == cota_selecionada)
            ]
            
            # 2. Aplica Lógica de Corte (Min vs Max)
            if semester_int == 1:
                # 1º Semestre: Menor Nota (Última Chamada) -> ascending=True
                 df_estrat = df_estrat_base.sort_values('Min', ascending=True).drop_duplicates(
                    subset=['Curso_Limpo', 'Campus', 'Turno'], keep='first'
                )
            else:
                 # 2º Semestre: Maior Nota (1ª Chamada) -> ascending=False
                 df_estrat = df_estrat_base.sort_values('Min', ascending=False).drop_duplicates(
                    subset=['Curso_Limpo', 'Campus', 'Turno'], keep='first'
                )
            
            # Ordena final para exibição
            df_estrat = df_estrat.sort_values(['Curso_Limpo', 'Campus', 'Turno'])
            
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
                with st.expander(":material/build: Customizar Estimativas (Parte 1 e Redação)", expanded=False):
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
                
                # --- NOVO: SELETOR DE ESTRATÉGIA PARA 2024-2026 (FUTURO) ---
                stats_p3_usado = None
                
                if ciclo_aluno == "2024-2026":
                    st.markdown("##### :material/model_training: Base de Projeção para o PAS 3 (Futuro)")
                    base_projecao = st.radio(
                        "Como você quer simular a dificuldade da prova?",
                        ["Replicar Padrão 2023-2025", "Utilizar Projeção Tendência"],
                        help="Replicar 2023-2025 assume que a prova será igual ao último ano. Tendência usa a média projetada estatisticamente."
                    )
                    
                    if base_projecao == "Replicar Padrão 2023-2025":
                        stats_p3_usado = TRIENNIUM_STATS["2023-2025"]["PAS3"]
                    else:
                        stats_p3_usado = STATS_PAS3_TREND
                elif ciclo_aluno == "2023-2025":
                     # Para 2023-2025, o PAS 3 já é histórico
                    stats_p3_usado = stats_ciclo["PAS3"]
                else:
                    # Para outros (2022-2024, etc), usa o histórico deles
                    stats_p3_usado = stats_ciclo["PAS3"]

                if st.button("🚀 Traçar Rota de Aprovação", type="primary"):
                    # Se, por algum motivo, stats_p3_usado for None (ex: erro de chave), fallback para trend
                    if stats_p3_usado is None:
                        stats_p3_usado = STATS_PAS3_TREND
                    
                    # Usa os overrides do slider
                    result = calc.calculate_required_score(
                        notas_validas, nota_alvo,
                        stats_ciclo["PAS1"], stats_ciclo["PAS2"], stats_p3_usado,
                        p1_override=p1_ov,
                        red_override=red_ov
                    )
                    
                    # --- CÁLCULO DE PROBABILIDADE HISTÓRICA (REALITY CHECK) ---
                    prob_hist = 0.0
                    amostra = 0
                    eb_pas3_necessario = result.p1_estimado + result.p2_necessario
                    
                    if calculate_cohort_evolution_probability:
                        try:
                            df_hist_cohort = load_cohort_data()
                            if not df_hist_cohort.empty:
                                aluno_dados = {'eb_pas1': res['eb_pas1'], 'eb_pas2': res['eb_pas2']}
                                prob_hist, amostra = calculate_cohort_evolution_probability(aluno_dados, eb_pas3_necessario, df_hist_cohort)
                        except: pass

                    # --- LÓGICA SEMÁFORO (COR DO RESULTADO) ---
                    # Verde: > 20% | Amarelo: 5-20% | Vermelho: < 5%
                    if prob_hist >= 20.0:
                        sem_cor = "success"
                        sem_icon = "🟢"
                        sem_msg = "Meta Alcançável!"
                    elif prob_hist >= 5.0:
                        sem_cor = "warning"
                        sem_icon = "🟡"
                        sem_msg = "Meta Desafiadora. Requer esforço acima da média."
                    else:
                        sem_cor = "error"
                        sem_icon = "🔴"
                        sem_msg = "Meta de Alto Risco. Estatisticamente improvável com essas notas."

                    # Banner de Resultado
                    getattr(st, sem_cor)(f"{sem_icon} {sem_msg}")

                    c1, c2, c3 = st.columns([1, 1, 1.2]) # Coluna 3 mais larga para P2
                    
                    with c1:
                        with st.container(border=True):
                            st.metric("P1 PAS 3 (Est.)", f"{result.p1_estimado:.2f}")
                    with c2:
                        with st.container(border=True):
                            st.metric("Redação (Est.)", f"{result.red_estimada:.2f}")
                    
                    # --- P2 COM ÊNFASE ---
                    p2_help_text = "Nota Acessível" if result.p2_necessario < 30 else ("Nota Desafiadora" if result.p2_necessario < 60 else "Nota Muito Alta")
                    p2_delta_color = "normal" if result.p2_necessario < 40 else "inverse"
                    
                    with c3:
                        with st.container(border=True):
                            st.metric(
                                "P2 NECESSÁRIA", 
                                f"{result.p2_necessario:.2f}",
                                delta=p2_help_text,
                                delta_color=p2_delta_color,
                                help="Esta é a nota que você precisa tirar na Parte 2 para atingir a meta."
                            )
                    
                    # --- INSIGHT CARD (REALITY CHECK) ---
                    if amostra > 0:
                        # Lógica de Mensagem Dinâmica (Step 762)
                        if prob_hist >= 40.0:
                            titulo_card = "Probabilidade: Alta"
                            msg_card = f"Perfil estatisticamente consolidado. Historicamente, **{prob_hist:.1f}%** dos candidatos com este escore obtiveram a vaga."
                            cor_card = "success"  # Verde
                            icon_card = "✅"
                        elif 10.0 <= prob_hist < 40.0:
                            titulo_card = "Probabilidade: Moderada (Competitivo)"
                            msg_card = f"Zona de concorrência acirrada. A taxa de conversão histórica para este perfil de nota é de **{prob_hist:.1f}%**."
                            cor_card = "warning"  # Amarelo
                            icon_card = "⚠️"
                        else:
                            titulo_card = "Probabilidade: Baixa (Atípico)"
                            msg_card = f"Cenário estatisticamente improvável (Outlier). Apenas **{prob_hist:.1f}%** da base histórica atingiu este resultado. Requer desempenho significativamente acima da média."
                            cor_card = "error"    # Vermelho
                            icon_card = "🚨"

                        st.info(f"""
                        **{titulo_card}**
                        
                        {msg_card}
                        
                        _(Base de análise: {amostra} alunos similares)_
                        """, icon=icon_card)
                    else:
                        st.caption(f":material/info: Sem dados históricos suficientes para análise de coorte (Amostra: {amostra}).")
            else:
                st.warning("Sem dados para esta cota.")
        else:
            st.warning("Preencha as notas na aba Diagnóstico primeiro.")


# =============================================================================
# PÁGINA 5: ANÁLISE DA ESCOLA (NOVA)
# =============================================================================

elif page == "escola":
    st.title(":material/domain: Análise da Escola vs População Geral")
    
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
    
    # --- LÓGICA DE DADOS (AUTO-LOAD GLOBAL) ---
    st.markdown("### 📂 Fonte de Dados")
    
    df_escola_input = None
    uploaded_file = None
    
    # 1. Tenta carregar do Estado Global (Supabase/Upload Anterior)
    if st.session_state.get('df_global_escola') is not None:
        st.info("✅ Usando dados carregados globalmente (Nuvem/Upload).")
        df_escola_input = st.session_state['df_global_escola'].copy()
        
        # Opção de sobrescrever
        if st.checkbox("Substituir por arquivo local (.xlsx)"):
            uploaded_file = st.file_uploader(
                ":material/upload: Upload da lista de alunos da escola (Excel)",
                type=['xlsx', 'xls'],
                help="O arquivo deve ter uma coluna 'Nome' com os nomes dos alunos."
            )
            if uploaded_file:
                try:
                     df_escola_input = pd.read_excel(uploaded_file)
                except:
                     df_escola_input = pd.read_csv(uploaded_file)
    else:
        # 2. Upload Manual (Fallback)
        uploaded_file = st.file_uploader(
            ":material/upload: Upload da lista de alunos da escola (Excel)",
            type=['xlsx', 'xls'],
            help="O arquivo deve ter uma coluna 'Nome' com os nomes dos alunos."
        )
        if uploaded_file:
             try:
                 df_escola_input = pd.read_excel(uploaded_file)
             except:
                 df_escola_input = pd.read_csv(uploaded_file)

    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(":material/download: Usar Exemplo de Escola"):
            example_path = Path(__file__).parent.parent / "data" / "exemplo_escola_1000_alunos.xlsx"
            if not example_path.exists():
                # Fallback para o caso de estar rodando na raiz
                example_path = Path("data/exemplo_escola_1000_alunos.xlsx")
                
            if example_path.exists():
                try:
                    escola_exemplo = pd.read_excel(example_path)
                    df_escola_input = escola_exemplo # Override
                    st.success(":material/check_circle: Carregado: 1000 alunos de exemplo")
                except Exception as e:
                    st.error(f"Erro ao ler arquivo de exemplo: {e}")
            else:
                st.error(":material/cancel: Arquivo de exemplo não encontrado.")
    
    # Processamento Principal (Se houver dados)
    if df_escola_input is not None:
        st.session_state.escola_df = df_escola_input
    
    if uploaded_file is not None:
        try:
            st.session_state.escola_df = pd.read_excel(uploaded_file)
            st.success(f":material/check_circle: Arquivo carregado: {len(st.session_state.escola_df)} nomes")
        except Exception as e:
            st.error(f":material/cancel: Erro ao ler arquivo: {e}")
    
    # Processa se houver dados da escola
    if 'escola_df' in st.session_state and st.session_state.escola_df is not None:
        escola_nomes = st.session_state.escola_df
        
        st.markdown("---")
        st.markdown("### :material/toc: Prévia dos nomes")
        cols_to_hide = ['id', 'created_at']
        df_display = escola_nomes.drop(columns=[c for c in cols_to_hide if c in escola_nomes.columns])
        st.dataframe(df_display.head(10), use_container_width=True)
        
        # Seleciona triênio - Ordem inversa (mais recente primeiro)
        trienios = sorted(df_geral['Ano_Trienio'].unique(), reverse=True)
        trienio_sel = st.selectbox(
            "Selecione o triênio para comparação:",
            trienios,
            index=0
        )
        
        df_trienio = df_geral[df_geral['Ano_Trienio'] == trienio_sel]
        
        if st.button(":material/search: Analisar Escola vs População", type="primary"):
            # Encontra os nomes na base geral
            if 'Nome' in escola_nomes.columns:
                nomes_escola = escola_nomes['Nome'].str.strip().str.upper()
                df_trienio_upper = df_trienio.copy()
                df_trienio_upper['Nome_Upper'] = df_trienio['Nome'].str.strip().str.upper()
                
                # Match por nome (inclui homônimos) mas remove duplicatas para contagem única de ALUNOS
                # Se um aluno da escola chama "JOAO SILVA" e na lista do PAS tem 2 "JOAO SILVA", conta como 1 encontrado.
                df_encontrados_unique = df_trienio_upper[df_trienio_upper['Nome_Upper'].isin(nomes_escola)].drop_duplicates(subset=['Nome_Upper'])
                
                # Mas para a análise de NOTAS, queremos todos os matches? 
                # Se a escola enviou "JOAO SILVA" e tem 2 no PAS, qual é o dele?
                # Por segurança, vamos manter todos os matches para cálculo de média (pode ser qualquer um dos dois),
                # mas para a TAXA DE MATCH, usamos o unique.
                
                df_escola = df_trienio_upper[df_trienio_upper['Nome_Upper'].isin(nomes_escola)]
                
                n_encontrados = len(df_encontrados_unique) # Conta CPFs únicos achados (pelo nome)
                n_total = len(escola_nomes)
                
                # Garante que não passa de 100%
                if n_encontrados > n_total: n_encontrados = n_total
                
                st.markdown("---")
                st.markdown("### :material/analytics: Resultados da Análise")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nomes enviados", n_total)
                with col2:
                    st.metric("Encontrados no PAS", n_encontrados)
                with col3:
                    taxa = (n_encontrados / n_total * 100) if n_total > 0 else 0
                    st.metric("Taxa de match", f"{taxa:.1f}%")
                
                if n_encontrados < 5:
                    st.warning(":material/warning: Poucos alunos encontrados. Verifique se os nomes estão corretos.")
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
                
                df_trienio['EB_Total'] = df_trienio['EB_PAS1'] + df_trienio['EB_PAS2'] + df_trienio['EB_PAS3']
                
                # =================================================================
                # 1. RESUMO EXECUTIVO (TOPO)
                # =================================================================
                
                # Cálculo de Médias Gerais
                media_escola = df_escola['Arg_Final'].mean()
                media_geral = df_trienio['Arg_Final'].mean()
                std_escola = df_escola['Arg_Final'].std()
                std_geral = df_trienio['Arg_Final'].std()
                diff = media_escola - media_geral

                # Banner de Resultado (Linguagem Simples)
                if diff > 0:
                    icon_name = ":material/celebration:"
                    texto_pos = "ACIMA"
                    style_box = "background-color: #d1e7dd; color: #0f5132; border-color: #badbcc;"
                    msg_header = f"PARABÉNS! Sua escola está {abs(diff):.1f} pontos ACIMA da média."
                else:
                    icon_name = ":material/trending_down:"
                    texto_pos = "ABAIXO"
                    style_box = "background-color: #f8d7da; color: #842029; border-color: #f5c2c7;"
                    msg_header = f"ATENÇÃO: Sua escola está {abs(diff):.1f} pontos ABAIXO da média."

                # Banner de Resultado (Linguagem Nativa para Ícones)
                if diff > 0:
                    st.success(f"### :material/celebration: {msg_header}")
                else:
                    st.error(f"### :material/trending_down: {msg_header}")

                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; margin-bottom: 25px; border-left: 5px solid {("#0f5132" if diff > 0 else "#842029")}; {style_box}">
                    <p style="margin: 0; font-size: 1.2em;">
                        A média de Argumento Final da sua escola (<strong>{media_escola:.2f}</strong>) supera a média geral do PAS (<strong>{media_geral:.2f}</strong>) em <strong>{abs(diff):.2f}</strong> pontos.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Cards de Detalhe (Argumento Final)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with st.container(border=True):
                        st.metric(":material/domain: Sua Escola", f"{media_escola:.2f}", help=f"Desvio Padrão: {std_escola:.2f}")
                with col2:
                    with st.container(border=True):
                        st.metric(":material/public: Média Geral PAS", f"{media_geral:.2f}", help=f"Desvio Padrão: {std_geral:.2f}")
                with col3:
                    with st.container(border=True):
                        st.metric(":material/compare_arrows: Diferença", f"{diff:+.2f}", delta=f"{diff:+.2f}")

                # =================================================================
                # 2. ANÁLISE DETALHADA POR ETAPA
                # =================================================================
                st.markdown("### :material/bar_chart: Desempenho por Etapa (Escore Bruto)")
                
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
                        ":material/looks_one: PAS 1",
                        f"{eb_escola_1:.1f}",
                        delta=f"{diff1:+.1f} vs média geral ({eb_geral_1:.1f})"
                    )
                
                with col2:
                    diff2 = eb_escola_2 - eb_geral_2
                    st.metric(
                        ":material/looks_two: PAS 2",
                        f"{eb_escola_2:.1f}",
                        delta=f"{diff2:+.1f} vs média geral ({eb_geral_2:.1f})"
                    )
                
                with col3:
                    diff3 = eb_escola_3 - eb_geral_3
                    st.metric(
                        ":material/looks_3: PAS 3",
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
                
                # Comparação de Argumento Final (REMOVIDO DAQUI POIS JÁ ESTÁ NO TOPO)
                # media_escola = df_escola['Arg_Final'].mean() ... (Movido para cima)
                
                # Teste estatístico mantido aqui como "Detalhe Técnico"
                st.markdown("---")
                st.markdown("### :material/science: Validação Estatística (Teste t)")
                
                try:
                    result = compare_groups(
                        group_a=df_escola['Arg_Final'].values,
                        group_b=df_trienio['Arg_Final'].values,
                        group_a_name="Sua Escola",
                        group_b_name="População Geral",
                        metric_name="Argumento Final"
                    )
                    
                    p_val_display = f"{result['p_value']:.4f}" if result['p_value'] >= 0.0001 else "< 0.0001"
                    
                    if result['statistically_significant']:
                        if diff > 0:
                            # Cálculo do Ganho Percentual
                            # Se a média geral for muito baixa (perto de zero), usamos o ganho absoluto para não distorcer
                            if media_geral > 0.1:
                                ganho_perc = (diff / abs(media_geral)) * 100
                                texto_ganho = f"**{ganho_perc:.0f}% acima** da média populacional"
                            else:
                                texto_ganho = f"**{diff:.1f} pontos acima** da média populacional"
                                
                            st.success(f":material/celebration: **Resultado Confirmado:** Sua escola pontuou {texto_ganho}, com alta significância estatística (p {p_val_display}).")
                        else:
                            st.error(f":material/warning: **Atenção:** O desempenho atual está estatisticamente abaixo da média (p {p_val_display}).")
                    else:
                        st.info(f":material/info: **Análise de Estabilidade:** A diferença observada não possui relevância estatística no momento (p = {p_val_display}).")
                    
                    with st.expander("🔬 Ver Detalhes Técnicos da Validação"):
                        st.write(f"**Valor-p:** {result['p_value']:.6f}")
                        st.write(f"**Tamanho da Amostra (n):** {n_encontrados}")
                        st.write(f"**Desvio Padrão (Escola):** {std_escola:.2f}")
                        st.caption("A validação estatística (Teste-t) garante que o resultado não foi por acaso.")
                        
                except Exception as e:
                    st.warning(f":material/warning: Não foi possível realizar teste estatístico: {e}")
                
                # ======================
                # VISUALIZAÇÕES DIDÁTICAS
                # ======================
                # 3. Ranking percentual e Resumo Visual (MOVIDOS DO FINAL PARA CÁ)
                percentil_escola = (df_trienio['Arg_Final'] < media_escola).mean() * 100
                
                col_v1, col_v2 = st.columns([1, 1])
                with col_v1:
                    st.info(f"### :material/emoji_events: Ranking: Top {100-percentil_escola:.0f}%")
                    st.markdown(f"A média da sua escola supera **{percentil_escola:.1f}%** de todos os candidatos.")
                    
                    # Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = percentil_escola,
                        number = {'suffix': "%"},
                        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1E88E5"}, 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_v2:
                    st.info("### :material/query_stats: Distribuição (Boxplot)")
                    st.markdown("Comparação da dispersão das notas.")
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(y=df_trienio['Arg_Final'], name='Geral', marker_color='#90A4AE', boxpoints=False))
                    fig_box.add_trace(go.Box(y=df_escola['Arg_Final'], name='Sua Escola', marker_color='#1E88E5'))
                    fig_box.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # 4. Histograma com destaque (avançado mas visual)
                st.markdown("---")
                st.markdown("### :material/query_stats: Onde sua escola se posiciona")
                
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=df_trienio['Arg_Final'],
                    name='Todos os candidatos',
                    marker_color='#90A4AE',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                # Linha vertical para média da escola (Destaque Visual)
                fig_hist.add_vline(
                    x=media_escola,
                    line_dash="solid", # Sólida para destaque
                    line_color="#0D47A1", # Azul Escuro Forte
                    line_width=4, # Mais grossa
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
                
                st.caption("📌 **Como ler:** A linha **AZUL ESCURA SÓLIDA** mostra a média da sua escola. A linha cinza pontilhada mostra a média geral.")
                
                # ============================================
                # HISTOGRAMAS POR ETAPA (PAS 1, 2, 3)
                # ============================================
                st.markdown("---")
                st.markdown("### :material/bar_chart: Distribuição por Etapa do PAS")
                
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
                    title="Distribuição Escore Bruto - PAS 1",
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
                    title="Distribuição Escore Bruto - PAS 2",
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
                    title="Distribuição Escore Bruto - PAS 3",
                    xaxis_title="Escore Bruto (P1 + P2)",
                    yaxis_title="Quantidade de candidatos",
                    showlegend=False, height=350
                )
                st.plotly_chart(fig_pas3, use_container_width=True)
                
                st.caption("📌 **Como ler:** As linhas **COLORIDAS TRACEJADAS** mostram a média da sua escola em cada etapa. A linha cinza pontilhada mostra a média geral.")
                
            else:
                st.warning("⚠️ O arquivo enviado não possui uma coluna 'Nome' válida.")
                
        # Rodapé: Tabela de Nomes
        st.markdown("---")
        with st.expander("📂 Ver Lista de Alunos Processados e Encontrados"):
            if 'escola_nomes' in locals():
                cols_to_hide = ['id', 'created_at']
                df_expander = escola_nomes.drop(columns=[c for c in cols_to_hide if c in escola_nomes.columns])
                st.dataframe(df_expander, use_container_width=True)
            else:
                st.info("Nenhum arquivo processado ainda.")


# =============================================================================
# PÁGINA 6: COMPARAÇÃO ENTRE GRUPOS (Teste A/B)
# =============================================================================

elif page == "comparacao":
    st.title(":material/trending_up: Comparação Entre Grupos")
    
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
        st.markdown("### :material/target: Métrica")
        # Prioriza EB_PAS2 se existir
        def_idx = num_cols.index('EB_PAS2') if 'EB_PAS2' in num_cols else 0
        metric = st.selectbox("Selecione a nota para comparar:", num_cols, index=def_idx)
        
    # --- SELEÇÃO DE GRUPOS HIERÁRQUICA ---
    st.markdown("### :material/groups: Definição dos Grupos")
    
    col_a, col_b = st.columns(2)
    
    # 1. Identifica colunas de Unidade e Turma
    col_uni = 'Unidade' if 'Unidade' in df.columns else None
    col_tur = 'Turma' if 'Turma' in df.columns else None
    
    if not col_uni and not col_tur:
        st.warning("⚠️ Colunas 'Unidade' ou 'Turma' não encontradas para agrupamento automático.")
        st.stop()
        
    def get_group_data(prefix, display_name):
        with st.container(border=True):
            st.markdown(f"**{display_name}**")
            
            # Filtro Unidade
            uni_options = ["Todas"] + sorted(df[col_uni].dropna().unique().tolist()) if col_uni else ["N/A"]
            sel_uni = st.selectbox(f"Unidade ({display_name})", uni_options, key=f"{prefix}_uni")
            
            # Filtro Turma (Dependente da Unidade)
            df_curr = df.copy()
            if sel_uni != "Todas" and col_uni:
                df_curr = df_curr[df_curr[col_uni] == sel_uni]
            
            tur_options = ["Todas"] + sorted(df_curr[col_tur].dropna().unique().tolist()) if col_tur else ["Todas"]
            sel_tur = st.selectbox(f"Turma ({display_name})", tur_options, key=f"{prefix}_tur")
            
            # Filtra data final
            df_final = df_curr.copy()
            if sel_tur != "Todas" and col_tur:
                df_final = df_final[df_final[col_tur] == sel_tur]
            
            y_vals = df_final[metric].dropna().values
            
            # Label para o gráfico
            label = f"{sel_uni}" if sel_tur == "Todas" else f"{sel_uni} - {sel_tur}"
            if sel_uni == "Todas" and sel_tur == "Todas": label = "Todos os Alunos"
            
            return y_vals, label

    with col_a:
        group_a, name_a = get_group_data("ga", "Grupo A")
        
    with col_b:
        group_b, name_b = get_group_data("gb", "Grupo B")
        
    if st.button(":material/analytics: Realizar Teste Estatístico", type="primary"):
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
                st.markdown(f"### :material/science: Resultado: {name_a} vs {name_b}")
                
                # Cards de Resumo
                ca, cb, cd = st.columns(3)
                ca.metric(f"Média {name_a}", f"{result['group_a_mean']:.2f}")
                cb.metric(f"Média {name_b}", f"{result['group_b_mean']:.2f}")
                cd.metric("Diferença", f"{result['difference']:+.2f}", delta=f"{result['difference']:+.2f}")
                
                # Interpretação Profissional
                p_val_simple = f"{result['p_value']:.4f}" if result['p_value'] >= 0.0001 else "< 0.0001"
                d_abs = abs(result['effect_size'])
                
                if result['statistically_significant']:
                    st.success(f"### ✅ Diferença de Performance Confirmada")
                    
                    # Frases de impacto baseadas no Cohen's d
                    if d_abs > 0.8:
                        impacto = "...com **Discrepância Crítica** entre as turmas."
                    elif d_abs > 0.5:
                        impacto = "...com **Diferença Considerável** de desempenho."
                    elif d_abs > 0.2:
                        impacto = "...com **Diferença Leve**, mas perceptível."
                    else:
                        impacto = "...com diferença mínima entre os grupos."
                        
                    st.markdown(f"> {impacto} *(p={p_val_simple})*")
                else:
                    st.info(f"### :material/info: Performance Equivalente")
                    st.markdown(f"> Não foram identificadas variações estatisticamente relevantes entre os grupos selecionados. *(p={p_val_simple})*")
                
                # Detalhes técnicos (Escondidos)
                with st.expander("🔬 Ver Laudo Estatístico (Técnico)"):
                    st.write(f"**Valor-p:** {result['p_value']:.6f}")
                    st.write(f"**Estatística t:** {result['t_statistic']:.4f}")
                    st.write(f"**Magnitude do Efeito:** {result['effect_size']:.2f}")
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

elif page == "pdf":
    st.title(":material/description: Gerador de Relatórios PDF")
    st.markdown("""
    > **Gera um PDF estilizado com sua projeção e metas.**
    """)
    
    tab_manual, tab_batch = st.tabs([":material/edit: Manual", ":material/package: Em Lote (Escola)"])
    
    pdf_gen = PDFGenerator()
    
    # ==========================================
    # 1. VISUAL PREVIEW & SMART INPUTS (MANUAL)
    # ==========================================
    with tab_manual:
        col_input, col_preview = st.columns([2, 1])
        
        with col_input:
            st.markdown("### Preenchimento Manual (Automático)")
            
            # --- INICIALIZAÇÃO DE ESTADO DO GERADOR DE PDF ---
            def init_pdf_state():
                # Filtros principais
                if 'pdf_sem_manual' not in st.session_state: st.session_state['pdf_sem_manual'] = 1
                if 'pdf_tri_manual' not in st.session_state: st.session_state['pdf_tri_manual'] = "2023-2025"
                if 'pdf_cota_manual' not in st.session_state: st.session_state['pdf_cota_manual'] = "Sistema Universal"
                
                # Campos do formulário
                if 'pdf_student_name' not in st.session_state: st.session_state['pdf_student_name'] = "Estudante"
                if 'pdf_p1_pas1' not in st.session_state: st.session_state['pdf_p1_pas1'] = 0.0
                if 'pdf_p2_pas1' not in st.session_state: st.session_state['pdf_p2_pas1'] = 0.0
                if 'pdf_red_pas1' not in st.session_state: st.session_state['pdf_red_pas1'] = 0.0
                if 'pdf_p1_pas2' not in st.session_state: st.session_state['pdf_p1_pas2'] = 0.0
                if 'pdf_p2_pas2' not in st.session_state: st.session_state['pdf_p2_pas2'] = 0.0
                if 'pdf_red_pas2' not in st.session_state: st.session_state['pdf_red_pas2'] = 0.0

            init_pdf_state()

            # Botão para recarregar dados do Preditor (MOVIDO PARA O TOPO para evitar Erro de SessionState)
            if st.button("🔄 Carregar Última Simulação", key="btn_load_last_sim_pdf_top"):
                if 'historico_ultimo_calculo' in st.session_state:
                    hist = st.session_state['historico_ultimo_calculo']
                    
                    # Mapeamento do Histórico para as Chaves do PDF
                    st.session_state['pdf_student_name'] = hist.get('input_nome_aluno', '')
                    st.session_state['pdf_p1_pas1'] = hist.get('input_p1_pas1', 0.0)
                    st.session_state['pdf_p2_pas1'] = hist.get('input_p2_pas1', 0.0)
                    st.session_state['pdf_red_pas1'] = hist.get('input_red_pas1', 0.0)
                    st.session_state['pdf_p1_pas2'] = hist.get('input_p1_pas2', 0.0)
                    st.session_state['pdf_p2_pas2'] = hist.get('input_p2_pas2', 0.0)
                    st.session_state['pdf_red_pas2'] = hist.get('input_red_pas2', 0.0)
                    
                    # Sync de Cota e Triênio (Se houver match nas listas)
                    # Nota: As listas ainda não foram geradas aqui, mas podemos confiar nos valores do session/histórico
                    # Se o valor não existir na lista quando o widget for criado, o Streamlit pode reclamar ou usar default.
                    # Mas como usamos index/key, se a key já estiver setada, ele tenta usar.
                    
                    cota_hist = hist.get('input_cota')
                    if cota_hist:
                        st.session_state['pdf_cota_manual'] = cota_hist
                        
                    tri_hist = hist.get('input_trienio')
                    if tri_hist:
                        st.session_state['pdf_tri_manual'] = tri_hist

                    st.toast("Dados da simulação carregados! O formulário será atualizado.")
                    st.rerun()
                else:
                    st.warning("Nenhuma simulação recente encontrada.")

            # --- SELEÇÃO DE CURSO (Manual) ---
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                target_semester = st.selectbox("Semestre de Ingresso", [1, 2], format_func=lambda x: f"{x}º Semestre", key="pdf_sem_manual")
            
            with col_f2:
                # Vou usar as mesmas variáveis de triênio disponíveis globalmente
                trienios_pdf_list = sorted(list(TRIENNIUM_STATS.keys()), reverse=True)
                ref_triennium_pdf = st.selectbox("Triênio de Referência", trienios_pdf_list, key="pdf_tri_manual")
                
            with col_f3:
                # Lista de cotas extraída diretamente do CSV de cortes
                try:
                    df_corte_full = pd.read_csv(Path(__file__).parent.parent / "data" / "notas_corte_pas.csv")
                    lista_cotas_pdf = sorted(df_corte_full['Sistema_Nome'].unique().tolist())
                    if 'Sistema Universal' in lista_cotas_pdf:
                        lista_cotas_pdf.insert(0, lista_cotas_pdf.pop(lista_cotas_pdf.index('Sistema Universal')))
                except:
                    lista_cotas_pdf = ["Sistema Universal", "Cota para Negros"]
                
                cota_pdf = st.selectbox("Sistema de Concorrência", lista_cotas_pdf, key="pdf_cota_manual")
                
            # --- LÓGICA DE CURSO SINCRONIZADA ABAIXO (DEPOIS DO CARREGAR ÚLTIMA) ---

            # Botão movido para o topo
            col_opts_1, col_opts_2 = st.columns(2)

            # ... Resto da UI ...
            
            # --- SELEÇÃO DE CURSO (Sync Total com Page 3) ---
            st.markdown("### Selecione o Curso Alvo")
            
            # Mapeia o semestre selecionado para o formato do DB
            semester_int_pdf = target_semester
            semester_db_pdf = "1º Semestre" if semester_int_pdf == 1 else "2º Semestre"
            
            # Mapeia o triênio selecionado para o formato do DB (ex: "2023-2025" -> "2022-2024")
            pdf_tri_sel = ref_triennium_pdf
            pdf_cota_sel = cota_pdf

            trienio_ref_pdf = ""
            start_year_pdf, end_year_pdf = map(int, pdf_tri_sel.split('-'))
            trienio_ref_pdf = f"{start_year_pdf - 1}-{end_year_pdf - 1}"
            
            # 1. Filtra Dados Iniciais
            # 1. Filtra Dados Iniciais (Usando função robusta encapsulada)
            # Reverte para load_course_stats para garantir compatibilidade de strings (1º vs 1°)
            df_cota_pdf = load_course_stats(semester=target_semester, triennium=trienio_ref_pdf, system=pdf_cota_sel)
            
            if df_cota_pdf is None:
                df_cota_pdf = pd.DataFrame()
            
            # 2. LIMPEZA RIGOROSA (Mesma da Page 3)
            # Garantir que a coluna de Curso existe (load_course_stats pode retornar 'Curso' ao invés de 'Curso_Limpo')
            if 'Curso_Limpo' not in df_cota_pdf.columns and 'Curso' in df_cota_pdf.columns:
                 df_cota_pdf['Curso_Limpo'] = df_cota_pdf['Curso']

            for c in ['Curso_Limpo', 'Campus', 'Turno', 'Chamada']:
                if c in df_cota_pdf.columns:
                    df_cota_pdf[c] = df_cota_pdf[c].astype(str).str.strip()

            # Cria identificador único
            df_cota_pdf['Combo_Nome'] = df_cota_pdf['Curso_Limpo'] + " (" + df_cota_pdf['Campus'] + " - " + df_cota_pdf['Turno'] + ")"

            # 3. DEDUPLICAÇÃO INTELIGENTE (Mesma da Page 3)
            if not df_cota_pdf.empty:
                df_cota_pdf['Chamada_Num'] = df_cota_pdf['Chamada'].str.extract(r'(\d+)').fillna(0).astype(int)
                
                if semester_int_pdf == 1:
                    # 1º Semestre: Prioridade para ÚLTIMA CHAMADA (Menor Nota)
                    df_cota_clean = df_cota_pdf.sort_values(['Combo_Nome', 'Min'], ascending=[True, True]).drop_duplicates('Combo_Nome', keep='first')
                else:
                    # 2º Semestre: Prioridade para PRIMEIRA CHAMADA DISPONÍVEL (Maior Nota - Conservadora)
                    # Nota: Na Page 3 usamos Chamada_Num asc -> keep first. Vamos manter igual.
                    df_cota_clean = df_cota_pdf.sort_values(['Combo_Nome', 'Chamada_Num'], ascending=[True, True]).drop_duplicates('Combo_Nome', keep='first')
                    
            else:
                df_cota_clean = df_cota_pdf.copy()
            
            cursos_lista = sorted(df_cota_clean['Combo_Nome'].unique().tolist())

            if cursos_lista:
                # Dicionário de referências (Nota e Chamada) para exibição
                course_info_pdf = {}
                for _, row in df_cota_clean.iterrows():
                    course_info_pdf[row['Combo_Nome']] = {
                        'nota': row['Min'],
                        'chamada': row.get('Chamada', 'N/A')
                    }
                
                def fmt_course_pdf(nome):
                    info = course_info_pdf.get(nome, {'nota': 0, 'chamada': 'N/A'})
                    return f"{nome} [{info['chamada']}: {info['nota']:.3f}]"
                
                # Tenta pré-selecionar se houver input salvo
                pre_index = None
                saved_course = st.session_state.get('input_curso_alvo')
                # Tenta match exato primeiro
                if saved_course and saved_course in cursos_lista:
                    pre_index = cursos_lista.index(saved_course)
                
                selected_combo = st.selectbox(
                    "Curso Pretendido", 
                    cursos_lista,
                    index=pre_index if pre_index is not None else 0, # Default to first if no saved course or not found
                    format_func=fmt_course_pdf,
                    help="Lista sincronizada com a regra: 1º Semestre (Última Chamada) | 2º Semestre (1ª Chamada)."
                )
                
                # Extrai nome limpo para o PDF e nota de corte
                if selected_combo:
                    selected_course_name = selected_combo
                    nota_corte_val = course_info_pdf.get(selected_combo, {}).get('nota', 0.0)
            else:
                st.warning("Nenhum curso encontrado para os filtros selecionados.")
                selected_combo = None
                selected_course_name = ""
                nota_corte_val = 0.0
            
            st.markdown("---")

            with st.form("pdf_manual_form"):
                col1, col2 = st.columns(2)
                
                # Helper para inicializar chaves do PDF no session state se não existirem
                def init_pdf_key(key, default):
                    if key not in st.session_state:
                        st.session_state[key] = default

                with col1:
                    aluno = st.text_input("Nome do Aluno", key="pdf_student_name")
                    
                    st.markdown("#### :material/edit_note: Notas PAS 1")
                    p1_pas1 = st.number_input("PAS 1 - P1 (Língua)", 0.0, 20.0, step=0.001, format="%.3f", key="pdf_p1_pas1")
                    p2_pas1 = st.number_input("PAS 1 - P2 (Gerais)", 0.0, 100.0, step=0.001, format="%.3f", key="pdf_p2_pas1")
                    red_pas1 = st.number_input("PAS 1 - Redação", 0.0, 10.0, step=0.001, format="%.3f", key="pdf_red_pas1")
                    
                with col2:
                    st.empty() # Spacer
                    st.markdown("#### :material/edit_note: Notas PAS 2")
                    p1_pas2 = st.number_input("PAS 2 - P1 (Língua)", 0.0, 20.0, step=0.001, format="%.3f", key="pdf_p1_pas2")
                    p2_pas2 = st.number_input("PAS 2 - P2 (Gerais)", 0.0, 100.0, step=0.001, format="%.3f", key="pdf_p2_pas2")
                    red_pas2 = st.number_input("PAS 2 - Redação", 0.0, 10.0, step=0.001, format="%.3f", key="pdf_red_pas2")
                    
                st.info(f"Nota de Corte Selecionada: **{nota_corte_val:.3f}** (Calculada automaticamente)")

                submitted = st.form_submit_button(":material/picture_as_pdf: Gerar PDF", type="primary")

        # --- PREVIEW COLUMN ---
        with col_preview:
            st.markdown("### :material/visibility: Prévia")
            try:
                img_path = Path(__file__).parent.parent / "assets" / "templates" / "modelo_pdf_previsaoPas.png"
                st.image(str(img_path), use_container_width=True)
            except Exception:
                st.markdown("""
                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9; text-align: center; height: 400px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                    <h3 style="color: #ccc;">📄</h3>
                    <p style="color: #666;">Modelo Dossiê Estratégico</p>
                    <small style="color: #999;">O PDF gerado conterá gráficos de desempenho e análise de chances.</small>
                </div>
                """, unsafe_allow_html=True)
            
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
                'sistema': pdf_cota_sel,
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
            
            # Gerar PDF
            try:
                pdf_bytes = pdf_gen.generate_single_pdf(data)
                
                if not pdf_bytes:
                    raise ValueError("O gerador retornou um PDF vazio. Verifique os logs do terminal.")
                
                st.success("PDF Gerado com Sucesso!")
                st.download_button(
                    label=":material/download: Baixar PDF",
                    data=pdf_bytes,
                    file_name=f"Relatorio_PAS_{aluno.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")

    # ==========================================
    # 2. BATCH MODE (ESCOLA)
    # ==========================================
    with tab_batch:
        st.markdown("### Processamento em Lote")
        
        # --- DOWNLOAD TEMPLATE BUTTON ---
        # Cria um DataFrame vazio com as colunas necessárias
        df_model = pd.DataFrame(columns=['Nome', 'Curso', 'P1_PAS1', 'P2_PAS1', 'Red_PAS1', 'P1_PAS2', 'P2_PAS2', 'Red_PAS2'])
        # Converte para Excel em memória
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_model.to_excel(writer, index=False, sheet_name='Modelo')
        model_data = output.getvalue()
        
        col_b_dl, col_b_info = st.columns([1, 3])
        with col_b_dl:
             st.download_button(
                label="📥 Baixar Planilha Modelo (.xlsx)",
                data=model_data,
                file_name="modelo_dados_alunos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Baixe este arquivo para preencher os dados corretamente."
            )
        with col_b_info:
            st.info("Para evitar erros, baixe o modelo padrão ao lado, preencha com os dados dos alunos e faça o upload.")
        
        st.markdown("---")
        st.markdown("#### :material/settings: Configuração do Lote")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            batch_semester = st.radio("Semestre de Ingresso (para corte)", [1, 2], index=0, horizontal=True, key="batch_sem")
        with col_b2:
            batch_ref_triennium = st.selectbox("Triênio de Referência (Corte)", list(TRIENNIUM_STATS.keys()), index=1, key="batch_tri")

        # --- LÓGICA DE DADOS (AUTO-LOAD GLOBAL) ---
        df_batch = None
        
        # 1. Tenta carregar do Estado Global
        if st.session_state.get('df_global_escola') is not None:
            st.info("✅ Usando dados carregados globalmente (Nuvem/Upload).")
            df_batch = st.session_state['df_global_escola'].copy()
            
            if st.checkbox("Substituir por arquivo local (CSV/Excel)", key="override_pdf_batch"):
                uploaded_batch = st.file_uploader("Upload de Arquivo de Dados", type=['csv', 'xlsx'], key="upload_pdf_batch")
                if uploaded_batch:
                    try:
                        df_batch = pd.read_csv(uploaded_batch) if uploaded_batch.name.endswith('.csv') else pd.read_excel(uploaded_batch)
                    except Exception as e:
                        st.error(f"Erro ao ler arquivo: {e}")
                        df_batch = None
        else:
             # 2. Upload Manual (Fallback)
            uploaded_batch = st.file_uploader("Upload de Arquivo de Dados", type=['csv', 'xlsx'], key="upload_pdf_batch")
            if uploaded_batch:
                try:
                    df_batch = pd.read_csv(uploaded_batch) if uploaded_batch.name.endswith('.csv') else pd.read_excel(uploaded_batch)
                except Exception as e:
                    st.error(f"Erro ao ler arquivo: {e}")

        if df_batch is not None:
            if st.button("Gerar PDFs em Lote"):
                try:
                    # df_batch já carregado acima

                    
                    # Normaliza colunas para minúsculo e remove espaços extras
                    df_batch.columns = df_batch.columns.str.strip().str.lower().str.replace(' ', '_')
                    
                    # Carrega notas de corte (TODOS os sistemas) do triênio de referência (Selecionado - 1)
                    try:
                        start_year_b, end_year_b = map(int, batch_ref_triennium.split('-'))
                        ref_tri_sync = f"{start_year_b - 1}-{end_year_b - 1}"
                    except:
                        ref_tri_sync = batch_ref_triennium # Fallback se falhar parciamento
                        
                    df_cursos_ref = load_course_stats(semester=batch_semester, triennium=ref_tri_sync, system=None)
                    
                    if df_cursos_ref is not None:
                        # --- FIX: Cria Combo_Nome para diferenciar Turnos/Campus ---
                        # E garante limpeza de "Sistema_Nome"
                        for col in ['Curso', 'Campus', 'Turno', 'Sistema_Nome']:
                            if col in df_cursos_ref.columns:
                                df_cursos_ref[col] = df_cursos_ref[col].astype(str).str.strip()
                        
                        # Cria lista de sistemas disponíveis para fuzzy match
                        if 'Sistema_Nome' in df_cursos_ref.columns:
                            available_systems = df_cursos_ref['Sistema_Nome'].unique().tolist()
                        else:
                            available_systems = ["Sistema Universal"]

                        # Cria coluna combinada se possível
                        if all(col in df_cursos_ref.columns for col in ['Curso', 'Campus', 'Turno']):
                             df_cursos_ref['Combo_Nome'] = df_cursos_ref['Curso'] + " (" + df_cursos_ref['Campus'] + " - " + df_cursos_ref['Turno'] + ")"
                             
                             # Cria mapa: (Combo_Nome, Sistema_Nome) -> Min
                             # Garante que temos as colunas
                             if 'Sistema_Nome' in df_cursos_ref.columns:
                                 # Remove duplicatas exatas de (Combo, Sistema) se houver (priorizando menor nota?)
                                 df_ref_unique = df_cursos_ref.sort_values('Min', ascending=True).drop_duplicates(['Combo_Nome', 'Sistema_Nome'], keep='first')
                                 
                                 # Dicionário compostos
                                 course_map = dict(zip(zip(df_ref_unique['Combo_Nome'], df_ref_unique['Sistema_Nome']), df_ref_unique['Min']))
                             else:
                                 # Fallback se não tiver sistema (não deveria acontecer se load_course_stats retornar tudo)
                                 # Mas se vier antigo...
                                 course_map = dict(zip(zip(df_cursos_ref['Combo_Nome'], ["Sistema Universal"]*len(df_cursos_ref)), df_cursos_ref['Min']))
                                 
                        else:
                             # Fallback para apenas Curso
                             if 'Sistema_Nome' in df_cursos_ref.columns:
                                 course_map = dict(zip(zip(df_cursos_ref['Curso'], df_cursos_ref['Sistema_Nome']), df_cursos_ref['Min']))
                             else:
                                 course_map = dict(zip(zip(df_cursos_ref['Curso'], ["Sistema Universal"]*len(df_cursos_ref)), df_cursos_ref['Min']))

                        # Lista de cursos únicos para match
                        available_courses = sorted(list(set(k[0] for k in course_map.keys())))
                    else:
                        course_map = {}
                        available_courses = []
                        available_systems = []
                    
                    # Stats base para cálculo (Triênio Selecionado)
                    stats_base = TRIENNIUM_STATS.get(batch_ref_triennium, TRIENNIUM_STATS["2023-2025"])
                    stats_pas3_proj = STATS_PAS3_TREND # Fonte da verdade global para projeções
                    calc = TargetCalculator()
                    
                    processed_data = []
                    
                    progress_bar = st.progress(0)
                    total_rows = len(df_batch)
                    
                    # Log de auditoria para o usuário ver na UI
                    with st.status("🚀 Iniciando Processamento em Lote...", expanded=True) as status:
                        status.write(f"📊 Base carregada com {total_rows} alunos.")
                        status.write(f"🎯 Sistemas (Cotas) encontrados: {', '.join(available_systems)}")
                        status.write(f"📚 Cursos mapeados: {len(available_courses)}")
                        
                        for idx, row in df_batch.iterrows():
                            # Extrai dados (com fallbacks seguros)
                            aluno_name = str(row.get('nome', row.get('aluno', f'Estudante {idx+1}')))
                            raw_course_name = str(row.get('curso', row.get('curso_alvo', 'Não informado')))
                            
                            # --- Fuzzy Match Logic ---
                            official_course_name = find_best_match(raw_course_name, available_courses)
                            
                            # Tenta encontrar o nome oficial do sistema (Cota)
                            raw_quota = str(row.get('cota', row.get('sistema', row.get('sistema_nome', row.get('sistema_concorrencia', 'Sistema Universal')))))
                            official_system = find_best_match(raw_quota, available_systems)
                            
                            # Chave de busca: (Curso, Sistema)
                            cutoff_key = (official_course_name, official_system)
                            nota_corte = course_map.get(cutoff_key, 0.0)
                            
                            match_info = f"👤 **{aluno_name}** | {official_course_name} | {official_system}"
                            
                            # Se não achou exato, tenta fallback para Sistema Universal no mesmo curso
                            if nota_corte == 0.0 and official_system != "Sistema Universal":
                                 fallback_key = (official_course_name, "Sistema Universal")
                                 nota_corte = course_map.get(fallback_key, 0.0)
                                 status.write(f"{match_info} ⚠️ Quota não encontrada, usando Universal. (Corte: {nota_corte:.2f})")
                            elif nota_corte == 0.0:
                                 status.write(f"{match_info} ❌ Nota de corte não encontrada!")
                            else:
                                 status.write(f"{match_info} ✅ OK (Corte: {nota_corte:.2f})")

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
                                'curso': official_course_name,
                                'sistema': official_system,
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
                        label=":material/archive: Baixar Arquivos (ZIP)",
                        data=zip_buffer,
                        file_name="relatorios_pas_batch.zip",
                        mime="application/zip"
                    )
                    
                except Exception as e:
                    st.error(f"Erro no processamento em lote: {e}")
