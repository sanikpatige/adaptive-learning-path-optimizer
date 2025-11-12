#!/usr/bin/env python3
"""
Main Analysis Script
Runs complete analysis pipeline
"""

import sys
import os

# Add src to path
sys.path.append('src')

from learning_analyzer import LearningAnalyzer
from recommender import LearningPathRecommender
from predictor import PerformancePredictor
from ab_testing import ABTestSimulator


def main():
    """Run complete analysis pipeline"""
    
    print("\n" + "="*70)
    print("ADAPTIVE LEARNING PATH OPTIMIZATION - COMPLETE ANALYSIS")
    print("="*70)
    
    # Check if data exists
    if not os.path.exists('data/learner_data.csv'):
        print("\n❌ Data not found. Please run: python generate_data.py")
        return
    
    print("\n📊 Step 1: Analyzing Learner Behavior")
    print("-" * 70)
    analyzer = LearningAnalyzer()
    
    # Behavior analysis
    behavior = analyzer.analyze_behavior()
    print(f"\n✓ Analyzed {behavior['total_learners']} learners")
    print(f"  - Avg Completion Rate: {behavior['avg_completion_rate']:.1%}")
    print(f"  - Avg Quiz Score: {behavior['avg_quiz_score']:.1f}")
    print(f"  - At-Risk Learners: {behavior['at_risk_count']} ({behavior['at_risk_percentage']:.1%})")
    
    # Clustering
    print(f"\n🔍 Step 2: Clustering Learners")
    print("-" * 70
