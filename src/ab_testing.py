#!/usr/bin/env python3
"""
A/B Testing Simulator
Simulates comparison between traditional and adaptive learning paths
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import os

np.random.seed(42)


class ABTestSimulator:
    """Simulate A/B testing for adaptive learning paths"""
    
    def __init__(self):
        """Initialize A/B test simulator"""
        self.control_data = None
        self.treatment_data = None
    
    def simulate_traditional_path(self, n_learners: int = 500) -> pd.DataFrame:
        """
        Simulate traditional (fixed) learning path outcomes
        
        Args:
            n_learners: Number of learners in control group
        
        Returns:
            DataFrame with control group results
        """
        
        learners = []
        
        for i in range(n_learners):
            # Traditional path: lower completion rates, higher time
            completion_rate = np.random.beta(5, 3)  # Avg ~62%
            time_to_proficiency = np.random.gamma(10, 12)  # Avg ~120 hours
            quiz_score = np.random.normal(73, 12)  # Avg 73%
            
            learners.append({
                'learner_id': f'C{i+1:04d}',
                'group': 'control',
                'completed': 1 if completion_rate > 0.5 else 0,
                'completion_rate': completion_rate,
                'time_hours': time_to_proficiency,
                'quiz_score': max(0, min(100, quiz_score))
            })
        
        return pd.DataFrame(learners)
    
    def simulate_adaptive_path(self, n_learners: int = 500) -> pd.DataFrame:
        """
        Simulate adaptive learning path outcomes
        
        Args:
            n_learners: Number of learners in treatment group
        
        Returns:
            DataFrame with treatment group results
        """
        
        learners = []
        
        for i in range(n_learners):
            # Adaptive path: higher completion rates, lower time
            completion_rate = np.random.beta(8, 2)  # Avg ~84%
            time_to_proficiency = np.random.gamma(6, 12)  # Avg ~72 hours
            quiz_score = np.random.normal(93, 8)  # Avg 93%
            
            learners.append({
                'learner_id': f'T{i+1:04d}',
                'group': 'treatment',
                'completed': 1 if completion_rate > 0.5 else 0,
                'completion_rate': completion_rate,
                'time_hours': time_to_proficiency,
                'quiz_score': max(0, min(100, quiz_score))
            })
        
        return pd.DataFrame(learners)
    
    def run_simulation(self, control_size: int = 500, 
                      treatment_size: int = 500) -> Dict:
        """
        Run full A/B test simulation
        
        Args:
            control_size: Size of control group
            treatment_size: Size of treatment group
        
        Returns:
            Simulation results
        """
        
        print("\n" + "="*60)
        print("Running A/B Test Simulation")
        print("="*60)
        print(f"\nControl Group (Traditional Path): {control_size} learners")
        print(f"Treatment Group (Adaptive Path): {treatment_size} learners")
        
        # Generate data
        self.control_data = self.simulate_traditional_path(control_size)
        self.treatment_data = self.simulate_adaptive_path(treatment_size)
        
        # Calculate metrics
        results = self._calculate_metrics()
        
        # Statistical tests
        significance = self._perform_statistical_tests()
        results['statistical_tests'] = significance
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate key metrics for both groups"""
        
        control_completed = self.control_data['completed'].sum()
        treatment_completed = self.treatment_data['completed'].sum()
        
        metrics = {
            'control': {
                'completion_rate': self.control_data['completed'].mean(),
                'avg_time_hours': self.control_data['time_hours'].mean(),
                'avg_quiz_score': self.control_data['quiz_score'].mean(),
                'total_learners': len(self.control_data),
                'completed_learners': control_completed
            },
            'treatment': {
                'completion_rate': self.treatment_data['completed'].mean(),
                'avg_time_hours': self.treatment_data['time_hours'].mean(),
                'avg_quiz_score': self.treatment_data['quiz_score'].mean(),
                'total_learners': len(self.treatment_data),
                'completed_learners': treatment_completed
            }
        }
        
        # Calculate improvements
        metrics['improvements'] = {
            'completion_rate_lift': (
                (metrics['treatment']['completion_rate'] - metrics['control']['completion_rate']) /
                metrics['control']['completion_rate']
            ),
            'time_reduction': (
                (metrics['control']['avg_time_hours'] - metrics['treatment']['avg_time_hours']) /
                metrics['control']['avg_time_hours']
            ),
            'quiz_score_improvement': (
                (metrics['treatment']['avg_quiz_score'] - metrics['control']['avg_quiz_score']) /
                metrics['control']['avg_quiz_score']
            )
        }
        
        return metrics
    
    def _perform_statistical_tests(self) -> Dict:
        """Perform statistical significance tests"""
        
        # Completion rate (chi-square test)
        contingency_table = pd.crosstab(
            pd.concat([self.control_data, self.treatment_data])['group'],
            pd.concat([self.control_data, self.treatment_data])['completed']
        )
        chi2, p_completion, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Time to proficiency (t-test)
        t_stat_time, p_time = stats.ttest_ind(
            self.control_data['time_hours'],
            self.treatment_data['time_hours']
        )
        
        # Quiz scores (t-test)
        t_stat_quiz, p_quiz = stats.ttest_ind(
            self.control_data['quiz_score'],
            self.treatment_data['quiz_score']
        )
        
        # Effect size (Cohen's d) for time
        pooled_std = np.sqrt(
            (self.control_data['time_hours'].std()**2 + 
             self.treatment_data['time_hours'].std()**2) / 2
        )
        cohens_d = (
            self.control_data['time_hours'].mean() - 
            self.treatment_data['time_hours'].mean()
        ) / pooled_std
        
        return {
            'completion_rate': {
                'chi_square': chi2,
                'p_value': p_completion,
                'significant': p_completion < 0.05
            },
            'time_to_proficiency': {
                't_statistic': t_stat_time,
                'p_value': p_time,
                'significant': p_time < 0.05,
                'cohens_d': cohens_d
            },
            'quiz_scores': {
                't_statistic': t_stat_quiz,
                'p_value': p_quiz,
                'significant': p_quiz < 0.05
            }
        }
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        print("\nControl Group (Traditional Path):")
        print(f"  Completion Rate: {results['control']['completion_rate']:.1%}")
        print(f"  Avg Time: {results['control']['avg_time_hours']:.1f} hours")
        print(f"  Avg Quiz Score: {results['control']['avg_quiz_score']:.1f}%")
        
        print("\nTreatment Group (Adaptive Path):")
        print(f"  Completion Rate: {results['treatment']['completion_rate']:.1%}")
        print(f"  Avg Time: {results['treatment']['avg_time_hours']:.1f} hours")
        print(f"  Avg Quiz Score: {results['treatment']['avg_quiz_score']:.1f}%")
        
        print("\n" + "="*60)
        print("IMPROVEMENTS")
        print("="*60)
        print(f"  Completion Rate: +{results['improvements']['completion_rate_lift']:.1%}")
        print(f"  Time Reduction: -{results['improvements']['time_reduction']:.1%}")
        print(f"  Quiz Score Improvement: +{results['improvements']['quiz_score_improvement']:.1%}")
        
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE")
        print("="*60)
        
        tests = results['statistical_tests']
        
        print(f"\nCompletion Rate:")
        print(f"  p-value: {tests['completion_rate']['p_value']:.4f}")
        print(f"  Significant: {'YES ✓' if tests['completion_rate']['significant'] else 'NO'}")
        
        print(f"\nTime to Proficiency:")
        print(f"  p-value: {tests['time_to_proficiency']['p_value']:.4f}")
        print(f"  Cohen's d: {tests['time_to_proficiency']['cohens_d']:.3f}")
        print(f"  Significant: {'YES ✓' if tests['time_to_proficiency']['significant'] else 'NO'}")
        
        print(f"\nQuiz Scores:")
        print(f"  p-value: {tests['quiz_scores']['p_value']:.4f}")
        print(f"  Significant: {'YES ✓' if tests['quiz_scores']['significant'] else 'NO'}")
    
    def plot_comparison(self, save: bool = True):
        """Plot comparison visualizations"""
        
        if self.control_data is None or self.treatment_data is None:
            print("Run simulation first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Completion Rate Comparison
        ax1 = axes[0, 0]
        completion_data = pd.DataFrame({
            'Group': ['Control', 'Treatment'],
            'Completion Rate': [
                self.control_data['completed'].mean(),
                self.treatment_data['completed'].mean()
            ]
        })
        ax1.bar(completion_data['Group'], completion_data['Completion Rate'], 
               color=['#ff6b6b', '#51cf66'])
        ax1.set_ylabel('Completion Rate')
        ax1.set_title('Completion Rate Comparison', fontweight='bold')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(completion_data['Completion Rate']):
            ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        
        # 2. Time Distribution
        ax2 = axes[0, 1]
        ax2.hist(self.control_data['time_hours'], bins=30, alpha=0.6, 
                label='Control', color='#ff6b6b')
        ax2.hist(self.treatment_data['time_hours'], bins=30, alpha=0.6, 
                label='Treatment', color='#51cf66')
        ax2.set_xlabel('Time to Proficiency (hours)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Time Distribution', fontweight='bold')
        ax2.legend()
        ax2.axvline(self.control_data['time_hours'].mean(), color='#ff6b6b', 
                   linestyle='--', linewidth=2, label='Control Mean')
        ax2.axvline(self.treatment_data['time_hours'].mean(), color='#51cf66', 
                   linestyle='--', linewidth=2, label='Treatment Mean')
        
        # 3. Quiz Score Distribution
        ax3 = axes[1, 0]
        ax3.boxplot([self.control_data['quiz_score'], self.treatment_data['quiz_score']],
                   labels=['Control', 'Treatment'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        ax3.set_ylabel('Quiz Score (%)')
        ax3.set_title('Quiz Score Distribution', fontweight='bold')
        
        # 4. Summary Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        A/B TEST SUMMARY
        
        Control Group (Traditional):
        • Completion: {self.control_data['completed'].mean():.1%}
        • Avg Time: {self.control_data['time_hours'].mean():.0f} hrs
        • Avg Score: {self.control_data['quiz_score'].mean():.1f}%
        
        Treatment Group (Adaptive):
        • Completion: {self.treatment_data['completed'].mean():.1%}
        • Avg Time: {self.treatment_data['time_hours'].mean():.0f} hrs
        • Avg Score: {self.treatment_data['quiz_score'].mean():.1f}%
        
        Improvements:
        • Completion: +{((self.treatment_data['completed'].mean() - self.control_data['completed'].mean()) / self.control_data['completed'].mean()):.1%}
        • Time: -{((self.control_data['time_hours'].mean() - self.treatment_data['time_hours'].mean()) / self.control_data['time_hours'].mean()):.1%}
        • Score: +{((self.treatment_data['quiz_score'].mean() - self.control_data['quiz_score'].mean()) / self.control_data['quiz_score'].mean()):.1%}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        if save:
            os.makedirs('results/visualizations', exist_ok=True)
            plt.savefig('results/visualizations/ab_test_results.png', 
                       dpi=300, bbox_inches='tight')
            print("\n✓ Saved A/B test visualization")
        
        plt.close()


# Example usage
if __name__ == '__main__':
    simulator = ABTestSimulator()
    
    # Run simulation
    results = simulator.run_simulation(control_size=500, treatment_size=500)
    
    # Generate visualization
    simulator.plot_comparison()
    
    print("\n✅ A/B test simulation complete!")
