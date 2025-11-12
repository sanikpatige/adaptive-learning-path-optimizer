#!/usr/bin/env python3
"""
Performance Predictor
Predicts course completion and time-to-proficiency
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Tuple


class PerformancePredictor:
    """Predict learner performance and outcomes"""
    
    def __init__(self, data_path: str = 'data/learner_data.csv'):
        """Initialize with learner data"""
        
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        
        self.completion_model = None
        self.time_model = None
        
        print(f"Loaded {len(self.df)} learners for prediction")
    
    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for modeling
        
        Returns:
            X: Feature matrix
            y_completion: Binary completion target (1 if completion_rate > 0.7)
            y_time: Time to proficiency target
        """
        
        feature_cols = [
            'study_hours_per_week',
            'session_frequency',
            'avg_session_duration',
            'quiz_average',
            'courses_enrolled',
            'skill_fundamentals',
            'skill_intermediate',
            'skill_gap_score'
        ]
        
        X = self.df[feature_cols].values
        
        # Binary completion target (>70% = success)
        y_completion = (self.df['completion_rate'] > 0.7).astype(int).values
        
        # Time to proficiency target
        y_time = self.df['time_to_proficiency_weeks'].values
        
        return X, y_completion, y_time
    
    def train_completion_model(self, test_size: float = 0.2) -> Dict:
        """
        Train model to predict course completion probability
        
        Args:
            test_size: Proportion of test set
        
        Returns:
            Training metrics
        """
        
        print("\nTraining completion prediction model...")
        
        X, y_completion, _ = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_completion, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression
        self.completion_model = LogisticRegression(random_state=42, max_iter=1000)
        self.completion_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.completion_model.predict(X_test_scaled)
        y_pred_proba = self.completion_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(
                ['study_hours_per_week', 'session_frequency', 'avg_session_duration',
                 'quiz_average', 'courses_enrolled', 'skill_fundamentals',
                 'skill_intermediate', 'skill_gap_score'],
                self.completion_model.coef_[0]
            ))
        }
        
        print(f"✓ Completion model trained - Accuracy: {accuracy:.1%}")
        
        return metrics
    
    def train_time_model(self, test_size: float = 0.2) -> Dict:
        """
        Train model to predict time-to-proficiency
        
        Args:
            test_size: Proportion of test set
        
        Returns:
            Training metrics
        """
        
        print("\nTraining time-to-proficiency prediction model...")
        
        X, _, y_time = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_time, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train linear regression
        self.time_model = LinearRegression()
        self.time_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.time_model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            'r2_score': r2,
            'mae': mae,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✓ Time model trained - R²: {r2:.3f}, MAE: {mae:.1f} weeks")
        
        return metrics
    
    def predict_completion(self, learner_id: str) -> Dict:
        """
        Predict completion probability for a learner
        
        Args:
            learner_id: Target learner ID
        
        Returns:
            Prediction results
        """
        
        if self.completion_model is None:
            raise ValueError("Model not trained. Call train_completion_model() first.")
        
        learner = self.df[self.df['learner_id'] == learner_id]
        
        if learner.empty:
            return {'error': f'Learner {learner_id} not found'}
        
        learner = learner.iloc[0]
        
        # Prepare features
        features = np.array([[
            learner['study_hours_per_week'],
            learner['session_frequency'],
            learner['avg_session_duration'],
            learner['quiz_average'],
            learner['courses_enrolled'],
            learner['skill_fundamentals'],
            learner['skill_intermediate'],
            learner['skill_gap_score']
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Predict
        probability = self.completion_model.predict_proba(features_scaled)[0][1]
        prediction = self.completion_model.predict(features_scaled)[0]
        
        return {
            'learner_id': learner_id,
            'completion_probability': round(probability, 3),
            'predicted_outcome': 'complete' if prediction == 1 else 'at_risk',
            'actual_completion_rate': learner['completion_rate']
        }
    
    def predict_time_to_proficiency(self, learner_id: str) -> Dict:
        """
        Predict time to proficiency for a learner
        
        Args:
            learner_id: Target learner ID
        
        Returns:
            Prediction results
        """
        
        if self.time_model is None:
            raise ValueError("Model not trained. Call train_time_model() first.")
        
        learner = self.df[self.df['learner_id'] == learner_id]
        
        if learner.empty:
            return {'error': f'Learner {learner_id} not found'}
        
        learner = learner.iloc[0]
        
        # Prepare features
        features = np.array([[
            learner['study_hours_per_week'],
            learner['session_frequency'],
            learner['avg_session_duration'],
            learner['quiz_average'],
            learner['courses_enrolled'],
            learner['skill_fundamentals'],
            learner['skill_intermediate'],
            learner['skill_gap_score']
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predicted_weeks = self.time_model.predict(features_scaled)[0]
        
        return {
            'learner_id': learner_id,
            'predicted_weeks': round(predicted_weeks, 1),
            'predicted_hours': round(predicted_weeks * learner['study_hours_per_week'], 1),
            'actual_weeks': learner['time_to_proficiency_weeks']
        }
    
    def plot_feature_importance(self, save: bool = True):
        """Plot feature importance for completion model"""
        
        if self.completion_model is None:
            print("Model not trained yet")
            return
        
        feature_names = [
            'Study Hours/Week', 'Session Frequency', 'Avg Session Duration',
            'Quiz Average', 'Courses Enrolled', 'Skill: Fundamentals',
            'Skill: Intermediate', 'Skill Gap Score'
        ]
        
        coefficients = self.completion_model.coef_[0]
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(coefficients)), coefficients[sorted_idx])
        plt.yticks(range(len(coefficients)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance for Completion Prediction', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save:
            os.makedirs('results/visualizations', exist_ok=True)
            plt.savefig('results/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved feature importance plot")
        
        plt.close()
    
    def plot_prediction_accuracy(self, save: bool = True):
        """Plot actual vs predicted time-to-proficiency"""
        
        if self.time_model is None:
            print("Model not trained yet")
            return
        
        X, _, y_time = self.prepare_features()
        X_scaled = self.scaler.transform(X)
        y_pred = self.time_model.predict(X_scaled)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_time, y_pred, alpha=0.5, s=30)
        
        # Plot perfect prediction line
        min_val = min(y_time.min(), y_pred.min())
        max_val = max(y_time.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Time to Proficiency (weeks)')
        plt.ylabel('Predicted Time to Proficiency (weeks)')
        plt.title('Time-to-Proficiency Prediction Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            os.makedirs('results/visualizations', exist_ok=True)
            plt.savefig('results/visualizations/prediction_accuracy.png', dpi=300, bbox_inches='tight')
            print("✓ Saved prediction accuracy plot")
        
        plt.close()
    
    def save_models(self, directory: str = 'results/models'):
        """Save trained models"""
        
        os.makedirs(directory, exist_ok=True)
        
        if self.completion_model:
            joblib.dump(self.completion_model, f'{directory}/completion_model.pkl')
            joblib.dump(self.scaler, f'{directory}/scaler.pkl')
            print(f"✓ Saved completion model to {directory}")
        
        if self.time_model:
            joblib.dump(self.time_model, f'{directory}/time_model.pkl')
            print(f"✓ Saved time model to {directory}")
    
    def load_models(self, directory: str = 'results/models'):
        """Load trained models"""
        
        try:
            self.completion_model = joblib.load(f'{directory}/completion_model.pkl')
            self.time_model = joblib.load(f'{directory}/time_model.pkl')
            self.scaler = joblib.load(f'{directory}/scaler.pkl')
            print("✓ Models loaded successfully")
        except FileNotFoundError:
            print("Models not found. Train models first.")


# Example usage
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Performance Predictor - Test Run")
    print("="*60)
    
    # Initialize predictor
    predictor = PerformancePredictor()
    
    # Train models
    completion_metrics = predictor.train_completion_model()
    time_metrics = predictor.train_time_model()
    
    print(f"\nCompletion Model Accuracy: {completion_metrics['accuracy']:.1%}")
    print(f"Time Model R²: {time_metrics['r2_score']:.3f}")
    print(f"Time Model MAE: {time_metrics['mae']:.1f} weeks")
    
    # Test predictions
    test_learner = 'L0001'
    
    print(f"\nPredictions for {test_learner}:")
    completion_pred = predictor.predict_completion(test_learner)
    time_pred = predictor.predict_time_to_proficiency(test_learner)
    
    print(f"  Completion Probability: {completion_pred['completion_probability']:.1%}")
    print(f"  Predicted Time: {time_pred['predicted_weeks']:.1f} weeks")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    predictor.plot_feature_importance()
    predictor.plot_prediction_accuracy()
    
    # Save models
    predictor.save_models()
    
    print("\n✅ Prediction models trained and saved!")
