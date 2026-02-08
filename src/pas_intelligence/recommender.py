"""
Feature 3: Sistema de Recomendação de Cursos (KNN Vetorial)

Este módulo implementa um sistema de recomendação baseado em K-Nearest Neighbors
que encontra cursos onde estudantes aprovados tinham perfil similar ao do aluno atual.

A recomendação é baseada em:
1. Similaridade vetorial (perfil de notas do aluno vs aprovados históricos)
2. Filtro por viabilidade (Argumento do aluno >= corte do curso - margem)

IMPORTANTE: Os dados devem ser normalizados antes do treino do KNN para que
           a distância euclidiana seja significativa entre features de escalas diferentes.
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.neighbors import NearestNeighbors   # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore


@dataclass
class CourseRecommendation:
    """Representa uma recomendação de curso."""
    curso: str
    distancia: float
    argumento_corte: float
    margem: float  # Diferença entre argumento do aluno e corte
    probabilidade_aprovacao: str  # Alta, Média, Baixa


class CourseRecommender:
    """
    Sistema de recomendação de cursos baseado em KNN vetorial.
    
    O recomendador encontra cursos onde estudantes aprovados historicamente
    tinham perfis de notas similares ao do aluno em questão.
    
    Attributes:
        n_neighbors: Número de vizinhos para considerar na busca
        scaler: StandardScaler para normalização das features
        knn: Modelo NearestNeighbors treinado
        courses_df: DataFrame com dados dos cursos e perfis de aprovados
        feature_columns: Colunas usadas como features
    
    Example:
        >>> recommender = CourseRecommender(n_neighbors=10)
        >>> recommender.fit(courses_df, feature_cols=['EB_PAS1', 'EB_PAS2', 'EB_PAS3'])
        >>> recommendations = recommender.recommend(
        ...     student_profile=[30.0, 35.0, 40.0],
        ...     student_argument=50.0,
        ...     n_recommendations=5
        ... )
    """
    
    def __init__(self, n_neighbors: int = 10):
        """
        Inicializa o recomendador.
        
        Args:
            n_neighbors: Número de vizinhos a considerar na busca KNN.
                        Valores maiores = recomendações mais diversas.
        """
        self.n_neighbors = n_neighbors
        self.scaler: Optional[StandardScaler] = None
        self.knn: Optional[NearestNeighbors] = None
        self.courses_df: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.course_column: str = 'Curso'
        self.cutoff_column: str = 'Arg_Corte'
        self._fitted = False
    
    def fit(
        self,
        courses_df: pd.DataFrame,
        feature_columns: List[str],
        course_column: str = 'Curso',
        cutoff_column: str = 'Arg_Corte',
    ) -> 'CourseRecommender':
        """
        Treina o modelo KNN com os dados históricos de aprovados.
        
        IMPORTANTE: O scaler é fitado aqui nos dados de treino.
                   Novos alunos devem ser transformados com o mesmo scaler.
        
        Args:
            courses_df: DataFrame com registros de alunos aprovados.
                       Cada linha representa um aprovado em um curso.
            feature_columns: Colunas a usar como features (ex: ['EB_PAS1', 'EB_PAS2'])
            course_column: Nome da coluna com o curso
            cutoff_column: Nome da coluna com o argumento de corte do curso
        
        Returns:
            self (para permitir encadeamento)
        
        Raises:
            ValueError: Se colunas necessárias não existirem no DataFrame
        """
        # Valida colunas
        required_cols = feature_columns + [course_column, cutoff_column]
        missing = set(required_cols) - set(courses_df.columns)
        if missing:
            raise ValueError(f"Colunas faltando no DataFrame: {missing}")
        
        self.courses_df = courses_df.copy()
        self.feature_columns = feature_columns
        self.course_column = course_column
        self.cutoff_column = cutoff_column
        
        # Extrai features
        X = courses_df[feature_columns].values
        
        # NORMALIZAÇÃO: Crucial para KNN funcionar corretamente
        # Features de escalas diferentes (ex: EB vs Redação) seriam dominadas
        # pelas de maior magnitude sem normalização.
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X) # type: ignore
        
        # Treina KNN
        self.knn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(X)),
            metric='euclidean',
            algorithm='ball_tree'  # Eficiente para dimensões baixas
        )
        self.knn.fit(X_scaled) # type: ignore
        
        self._fitted = True
        return self
    
    def recommend(
        self,
        student_profile: Union[np.ndarray, List[float]],
        student_argument: float,
        n_recommendations: int = 5,
        passing_threshold: float = -0.5,
    ) -> List[Dict]:
        """
        Recomenda cursos para um aluno baseado em seu perfil.
        
        O algoritmo:
        1. Normaliza o perfil do aluno com o mesmo scaler do treino
        2. Encontra os K vizinhos mais próximos no espaço normalizado
        3. Agrupa por curso e calcula distância média
        4. Filtra cursos onde o aluno pode passar (arg >= corte - margem)
        5. Retorna top N ordenados por viabilidade
        
        Args:
            student_profile: Array com features do aluno (mesma ordem do treino)
            student_argument: Argumento Final do aluno (ou estimado)
            n_recommendations: Número de recomendações a retornar
            passing_threshold: Margem de tolerância (ex: -0.5 significa aceitar
                             cursos onde o aluno está até 0.5 pontos abaixo do corte)
        
        Returns:
            Lista de dicts ordenada por melhor fit, cada um contendo:
            - curso: Nome do curso
            - distancia_media: Distância média aos aprovados similares
            - argumento_corte: Nota de corte do curso
            - margem: Diferença entre argumento do aluno e corte
            - status: 'aprovado', 'margem', ou 'insuficiente'
            - probabilidade: 'Alta', 'Média', 'Baixa'
        
        Raises:
            ValueError: Se o modelo não foi treinado
        """
        if not self._fitted:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")
        
        # Prepara perfil do aluno
        profile = np.atleast_2d(student_profile)
        
        # Normaliza com o mesmo scaler do treino
        profile_scaled = self.scaler.transform(profile) # type: ignore
        
        # Busca K vizinhos mais próximos
        distances, indices = self.knn.kneighbors(profile_scaled) # type: ignore
        
        # Agrupa resultados por curso
        neighbors_df = self.courses_df.iloc[indices[0]].copy() # type: ignore
        neighbors_df['distancia'] = distances[0]
        
        # Agrega por curso
        course_stats = neighbors_df.groupby(self.course_column).agg({
            'distancia': 'mean',
            self.cutoff_column: 'first',  # Assume corte fixo por curso
        }).reset_index()
        
        # Calcula margem (positiva = acima do corte)
        course_stats['margem'] = student_argument - course_stats[self.cutoff_column]
        
        # Filtra por viabilidade
        viable = course_stats[course_stats['margem'] >= passing_threshold].copy()
        
        # Determina status e probabilidade
        def _classify(margem: float) -> tuple:
            if margem >= 5:
                return 'aprovado', 'Alta'
            elif margem >= 0:
                return 'aprovado', 'Média'
            elif margem >= passing_threshold:
                return 'margem', 'Baixa'
            else:
                return 'insuficiente', 'Muito Baixa'
        
        viable['status'], viable['probabilidade'] = zip(
            *viable['margem'].apply(_classify)
        )
        
        # Ordena: primeiro por probabilidade (Alta > Média > Baixa),
        # depois por distância (menor = mais similar)
        prob_order = {'Alta': 0, 'Média': 1, 'Baixa': 2, 'Muito Baixa': 3}
        viable['prob_order'] = viable['probabilidade'].map(prob_order)
        viable = viable.sort_values(['prob_order', 'distancia'])
        
        # Formata resultado
        recommendations = []
        for _, row in viable.head(n_recommendations).iterrows():
            recommendations.append({
                'curso': row[self.course_column],
                'distancia_media': round(row['distancia'], 4),
                'argumento_corte': round(row[self.cutoff_column], 3),
                'margem': round(row['margem'], 3),
                'status': row['status'],
                'probabilidade': row['probabilidade'],
            })
        
        return recommendations


def recommend_courses(
    student_profile: np.ndarray,
    course_profiles_df: pd.DataFrame,
    student_argument: float,
    feature_columns: List[str],
    course_column: str = 'Curso',
    cutoff_column: str = 'Arg_Corte',
    passing_threshold: float = -0.5,
    n_recommendations: int = 5,
) -> List[Dict]:
    """
    Função utilitária para recomendação one-shot de cursos.
    
    Cria um CourseRecommender, treina com os dados fornecidos, e retorna
    recomendações. Útil para uso rápido sem manter estado.
    
    Args:
        student_profile: Array com features do aluno
        course_profiles_df: DataFrame com histórico de aprovados
        student_argument: Argumento Final do aluno
        feature_columns: Colunas a usar como features
        course_column: Nome da coluna com o curso
        cutoff_column: Nome da coluna com argumento de corte
        passing_threshold: Margem de tolerância para viabilidade
        n_recommendations: Número de recomendações
    
    Returns:
        Lista de recomendações (ver CourseRecommender.recommend())
    
    Example:
        >>> recs = recommend_courses(
        ...     student_profile=np.array([30.0, 35.0, 40.0, 7.5]),
        ...     course_profiles_df=historico_df,
        ...     student_argument=45.5,
        ...     feature_columns=['EB_PAS1', 'EB_PAS2', 'EB_PAS3', 'Red_PAS3'],
        ... )
        >>> for rec in recs:
        ...     print(f"{rec['curso']}: {rec['probabilidade']}")
    """
    recommender = CourseRecommender(n_neighbors=20)
    recommender.fit(
        course_profiles_df,
        feature_columns=feature_columns,
        course_column=course_column,
        cutoff_column=cutoff_column,
    )
    
    return recommender.recommend(
        student_profile=student_profile,
        student_argument=student_argument,
        n_recommendations=n_recommendations,
        passing_threshold=passing_threshold,
    )
