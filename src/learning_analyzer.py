#!/usr/bin/env python3
"""
Learning Analyzer
Analyzes learner behavior, performs clustering, and identifies skill gaps
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


class LearningAnalyzer:
    """Analyze learner behavior and patterns"""
    
    def __init__(self, data_path: str = 'data/learner_data.csv'):
        """Initialize with learner data"""
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.clusters = None
        
        print(f"Loaded {len(self.df)} learners")
    
    def analyze_behavior(self) -> Dict:
        """Analyze overall learner behavior patterns"""
        
        analysis = {
            'total_learners': len(self.df),
            'avg_completion_rate': self.df['completion_rate'].mean(),
            'avg_quiz_score': self.df['quiz_average'].mean(),
            'avg_study_hours': self.df['total_study_hours'].mean(),
            'avg_time_to_proficiency': self.df['time_to_proficiency_weeks'].mean(),
            'at_risk_count': self.df['at_risk'].sum(),
            'at_risk_percentage': self.df['at_risk'].mean()
        }
        
        # Correlation analysis
        numeric_cols = ['study_hours_per_week', 'session_frequency', 'quiz_average', 
                       'completion_rate', 'total_study_hours']
        correlations = self.df[numeric_cols].corr()['completion_rate'].sort_values(ascending=False)
        
        analysis['top_correlations'] = correlations.to_dict()
        
        return analysis
    
    def cluster_learners(self, n_clusters: int = 4, save_plot: bool = True) -> np.ndarray:
        """
        Cluster learners based on behavior patterns
        
        Args:
            n_clusters: Number of clusters
            save_plot: Save visualization
        
        Returns:
            Cluster assignments
        """
        
        print(f"\nClustering learners into {n_clusters} groups...")
        
        # Select features for clustering
        feature_cols = [
            'study_hours_per_week', 'session_frequency', 'avg_session_duration',
            'quiz_average', 'completion_rate', 'courses_completed', 'skill_gap_score'
        ]
        
        X = self.df[feature_cols].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        self.df['cluster'] = self.clusters
        
        # Analyze clusters
        cluster_stats = self._analyze_clusters()
        
        # Visualize if requested
        if save_plot:
            self._plot_clusters(X_scaled, save_plot)
        
        return self.clusters
    
    def _analyze_clusters(self) -> Dict:
        """Analyze characteristics of each cluster"""
        
        cluster_analysis = {}
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            cluster_analysis[f'Cluster {cluster_id}'] = {
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(self.df) * 100,
                'avg_completion_rate': cluster_data['completion_rate'].mean(),
                'avg_quiz_score': cluster_data['quiz_average'].mean(),
                'avg_study_hours': cluster_data['total_study_hours'].mean(),
                'at_risk_percentage': cluster_data['at_risk'].mean() * 100
            }
        
        # Print cluster summary
        print("\n" + "="*60)
        print("Cluster Analysis Summary")
        print("="*60)
        for cluster_name, stats in cluster_analysis.items():
            print(f"\n{cluster_name}:")
            print(f"  Size: {stats['count']} learners ({stats['percentage']:.1f}%)")
            print(f"  Avg Completion Rate: {stats['avg_completion_rate']:.1%}")
            print(f"  Avg Quiz Score: {stats['avg_quiz_score']:.1f}")
            print(f"  Avg Study Hours: {stats['avg_study_hours']:.1f}")
            print(f"  At-Risk: {stats['at_risk_percentage']:.1f}%")
        
        return cluster_analysis
    
    def _plot_clusters(self, X_scaled: np.ndarray, save: bool = True):
        """Visualize clusters using PCA"""
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=self.clusters, cmap='viridis', 
                             alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Learner Clustering Visualization (PCA)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save:
            os.makedirs('results/visualizations', exist_ok=True)
            plt.savefig('results/visualizations/learner_clusters.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved cluster visualization to results/visualizations/learner_clusters.png")
        
        plt.close()
    
    def identify_skill_gaps(self, learner_id: str = None) -> Dict:
        """
        Identify skill gaps for a specific learner or all learners
        
        Args:
            learner_id: Specific learner ID, or None for all learners
        
        Returns:
            Skill gap analysis
        """
        
        skill_cols = ['skill_fundamentals', 'skill_intermediate', 'skill_advanced', 'skill_expert']
        
        if learner_id:
            learner = self.df[self.df['learner_id'] == learner_id].iloc[0]
            
            gaps = {
                'learner_id': learner_id,
                'skill_scores': {
                    'fundamentals': learner['skill_fundamentals'],
                    'intermediate': learner['skill_intermediate'],
                    'advanced': learner['skill_advanced'],
                    'expert': learner['skill_expert']
                },
                'weakest_domain': min(
                    [('fundamentals', learner['skill_fundamentals']),
                     ('intermediate', learner['skill_intermediate']),
                     ('advanced', learner['skill_advanced']),
                     ('expert', learner['skill_expert'])],
                    key=lambda x: x[1]
                )[0],
                'avg_skill_level': learner[skill_cols].mean(),
                'skill_gap_score': learner['skill_gap_score']
            }
            
            return gaps
        
        else:
            # Aggregate analysis
            avg_skills = self.df[skill_cols].mean()
            
            gaps = {
                'overall_avg_skills': avg_skills.to_dict(),
                'weakest_domain_overall': avg_skills.idxmin().replace('skill_', ''),
                'learners_with_high_gaps': len(self.df[self.df['skill_gap_score'] > 50]),
                'avg_skill_gap_score': self.df['skill_gap_score'].mean()
            }
            
            return gaps
    
    def plot_skill_gap_heatmap(self, save: bool = True):
        """Create heatmap of skill levels across learners"""
        
        skill_cols = ['skill_fundamentals', 'skill_intermediate', 'skill_advanced', 'skill_expert']
        
        # Sample 50 learners for visualization
        sample_df = self.df.sample(min(50, len(self.df)), random_state=42)
        skill_data = sample_df[skill_cols].values
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(skill_data, cmap='RdYlGn', vmin=0, vmax=100,
                   cbar_kws={'label': 'Skill Level'},
                   yticklabels=[f"L{i+1}" for i in range(len(sample_df))],
                   xticklabels=['Fundamentals', 'Intermediate', 'Advanced', 'Expert'])
        plt.title('Skill Level Heatmap (Sample of Learners)', fontsize=16, fontweight='bold')
        plt.xlabel('Skill Domain')
        plt.ylabel('Learner')
        plt.tight_layout()
        
        if save:
            os.makedirs('results/visualizations', exist_ok=True)
            plt.savefig('results/visualizations/skill_gap_heatmap.png', dpi=300, bbox_inches='tight')
            print("✓ Saved skill gap heatmap to results/visualizations/skill_gap_heatmap.png")
        
        plt.close()
    
    def plot_correlation_matrix(self, save: bool = True):
        """Plot correlation matrix of key features"""
        
        feature_cols = [
            'study_hours_per_week', 'session_frequency', 'quiz_average',
            'completion_rate', 'total_study_hours', 'skill_gap_score'
        ]
        
        corr_matrix = self.df[feature_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            os.makedirs('results/visualizations', exist_ok=True)
            plt.savefig('results/visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
            print("✓ Saved correlation matrix to results/visualizations/correlation_matrix.png")
        
        plt.close()
    
    def get_learner_profile(self, learner_id: str) -> Dict:
        """Get complete profile for a specific learner"""
        
        learner = self.df[self.df['learner_id'] == learner_id]
        
        if learner.empty:
            return {'error': f'Learner {learner_id} not found'}
        
        learner = learner.iloc[0]
        
        profile = {
            'learner_id': learner_id,
            'learner_type': learner['learner_type'],
            'study_pattern': {
                'hours_per_week': learner['study_hours_per_week'],
                'session_frequency': learner['session_frequency'],
                'avg_session_duration': learner['avg_session_duration']
            },
            'performance': {
                'quiz_average': learner['quiz_average'],
                'completion_rate': learner['completion_rate'],
                'courses_completed': learner['courses_completed']
            },
            'skill_levels': {
                'fundamentals': learner['skill_fundamentals'],
                'intermediate': learner['skill_intermediate'],
                'advanced': learner['skill_advanced'],
                'expert': learner['skill_expert']
            },
            'cluster': int(learner['cluster']) if 'cluster' in learner else None,
            'at_risk': bool(learner['at_risk'])
        }
        
        return profile


# Example usage
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Learning Analyzer - Test Run")
    print("="*60)
    
    # Initialize analyzer
    analyzer = LearningAnalyzer()
    
    # Analyze behavior
    print("\nAnalyzing learner behavior...")
    behavior = analyzer.analyze_behavior()
    print(f"\nTotal Learners: {behavior['total_learners']}")
    print(f"Avg Completion Rate: {behavior['avg_completion_rate']:.1%}")
    print(f"Avg Quiz Score: {behavior['avg_quiz_score']:.1f}")
    print(f"At-Risk Learners: {behavior['at_risk_count']} ({behavior['at_risk_percentage']:.1%})")
    
    # Cluster learners
    clusters = analyzer.cluster_learners(n_clusters=4)
    
    # Identify skill gaps
    print("\nIdentifying overall skill gaps...")
    gaps = analyzer.identify_skill_gaps()
    print(f"Weakest Domain: {gaps['weakest_domain_overall']}")
    print(f"Avg Skill Gap Score: {gaps['avg_skill_gap_score']:.1f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_skill_gap_heatmap()
    analyzer.plot_correlation_matrix()
    
    # Get specific learner profile
    print("\nGetting profile for L0001...")
    profile = analyzer.get_learner_profile('L0001')
    print(f"Learner Type: {profile['learner_type']}")
    print(f"Completion Rate: {profile['performance']['completion_rate']:.1%}")
    
    print("\n✅ Analysis complete!")
