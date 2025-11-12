#!/usr/bin/env python3
"""
Generate Synthetic Learner Data
Creates realistic learning behavior data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)


def generate_learner_data(n_learners=1000):
    """Generate synthetic learner behavior data"""
    
    print(f"Generating data for {n_learners} learners...")
    
    learners = []
    
    # Learner types with different characteristics
    learner_types = {
        'fast': {'weight': 0.25, 'completion_rate': 0.9, 'study_hours': (40, 60), 'quiz_avg': (85, 95)},
        'steady': {'weight': 0.45, 'completion_rate': 0.75, 'study_hours': (60, 100), 'quiz_avg': (70, 85)},
        'struggling': {'weight': 0.20, 'completion_rate': 0.50, 'study_hours': (100, 160), 'quiz_avg': (55, 70)},
        'at_risk': {'weight': 0.10, 'completion_rate': 0.30, 'study_hours': (20, 50), 'quiz_avg': (40, 60)}
    }
    
    for i in range(n_learners):
        # Select learner type
        type_choice = random.choices(
            list(learner_types.keys()),
            weights=[t['weight'] for t in learner_types.values()]
        )[0]
        
        learner_profile = learner_types[type_choice]
        
        # Generate learner attributes
        learner_id = f"L{i+1:04d}"
        
        # Study behavior
        study_hours_per_week = random.uniform(5, 20)
        session_frequency = random.randint(2, 7)  # sessions per week
        avg_session_duration = random.uniform(0.5, 3.0)  # hours
        
        # Performance metrics
        quiz_average = random.uniform(*learner_profile['quiz_avg'])
        completion_rate = learner_profile['completion_rate'] + random.uniform(-0.1, 0.1)
        completion_rate = max(0, min(1, completion_rate))
        
        # Courses taken
        courses_enrolled = random.randint(3, 10)
        courses_completed = int(courses_enrolled * completion_rate)
        
        # Time metrics
        total_study_hours = random.uniform(*learner_profile['study_hours'])
        time_to_proficiency = total_study_hours / max(study_hours_per_week, 1)
        
        # Engagement metrics
        days_active = random.randint(30, 180)
        last_active_days_ago = random.randint(0, 30)
        
        # Skill domain scores (0-100)
        skill_domains = {
            'fundamentals': random.uniform(50, 100),
            'intermediate': random.uniform(40, 90),
            'advanced': random.uniform(30, 80),
            'expert': random.uniform(20, 70)
        }
        
        # Calculate skill gap score (lower is better)
        skill_gap_score = 100 - np.mean(list(skill_domains.values()))
        
        learner = {
            'learner_id': learner_id,
            'learner_type': type_choice,
            'study_hours_per_week': round(study_hours_per_week, 2),
            'session_frequency': session_frequency,
            'avg_session_duration': round(avg_session_duration, 2),
            'quiz_average': round(quiz_average, 2),
            'completion_rate': round(completion_rate, 3),
            'courses_enrolled': courses_enrolled,
            'courses_completed': courses_completed,
            'total_study_hours': round(total_study_hours, 2),
            'time_to_proficiency_weeks': round(time_to_proficiency, 2),
            'days_active': days_active,
            'last_active_days_ago': last_active_days_ago,
            'skill_fundamentals': round(skill_domains['fundamentals'], 2),
            'skill_intermediate': round(skill_domains['intermediate'], 2),
            'skill_advanced': round(skill_domains['advanced'], 2),
            'skill_expert': round(skill_domains['expert'], 2),
            'skill_gap_score': round(skill_gap_score, 2),
            'at_risk': 1 if type_choice == 'at_risk' else 0
        }
        
        learners.append(learner)
    
    df = pd.DataFrame(learners)
    return df


def generate_course_data(n_courses=50):
    """Generate synthetic course catalog data"""
    
    print(f"Generating data for {n_courses} courses...")
    
    courses = []
    
    course_categories = ['Cloud Fundamentals', 'Security', 'Machine Learning', 
                        'Data Analytics', 'DevOps', 'Architecture', 'Networking']
    difficulty_levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
    
    for i in range(n_courses):
        course_id = f"C{i+1:03d}"
        category = random.choice(course_categories)
        difficulty = random.choice(difficulty_levels)
        
        # Course metrics
        duration_hours = random.randint(5, 50)
        avg_completion_rate = random.uniform(0.5, 0.9)
        avg_quiz_score = random.uniform(65, 90)
        enrollment_count = random.randint(100, 5000)
        
        course = {
            'course_id': course_id,
            'course_name': f"{category} {difficulty} - Course {i+1}",
            'category': category,
            'difficulty': difficulty,
            'duration_hours': duration_hours,
            'avg_completion_rate': round(avg_completion_rate, 3),
            'avg_quiz_score': round(avg_quiz_score, 2),
            'enrollment_count': enrollment_count,
            'prerequisite_required': 1 if difficulty in ['Advanced', 'Expert'] else 0
        }
        
        courses.append(course)
    
    df = pd.DataFrame(courses)
    return df


def generate_assessment_data(learners_df, n_assessments_per_learner=10):
    """Generate synthetic assessment/quiz data"""
    
    print(f"Generating assessment data...")
    
    assessments = []
    
    for _, learner in learners_df.iterrows():
        n_assessments = random.randint(5, n_assessments_per_learner)
        
        for j in range(n_assessments):
            assessment_id = f"A{len(assessments)+1:05d}"
            
            # Score based on learner's average with some variance
            base_score = learner['quiz_average']
            score = base_score + random.uniform(-15, 15)
            score = max(0, min(100, score))
            
            # Time taken (minutes)
            time_taken = random.randint(10, 60)
            
            # Attempt number
            attempt = random.randint(1, 3)
            
            # Skill domain
            domain = random.choice(['fundamentals', 'intermediate', 'advanced', 'expert'])
            
            assessment = {
                'assessment_id': assessment_id,
                'learner_id': learner['learner_id'],
                'score': round(score, 2),
                'time_taken_minutes': time_taken,
                'attempt_number': attempt,
                'skill_domain': domain,
                'passed': 1 if score >= 70 else 0,
                'date': (datetime.now() - timedelta(days=random.randint(1, 180))).strftime('%Y-%m-%d')
            }
            
            assessments.append(assessment)
    
    df = pd.DataFrame(assessments)
    return df


def main():
    """Generate all datasets"""
    
    print("\n" + "="*60)
    print("Generating Synthetic Learning Data")
    print("="*60 + "\n")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate datasets
    learners_df = generate_learner_data(n_learners=1000)
    courses_df = generate_course_data(n_courses=50)
    assessments_df = generate_assessment_data(learners_df, n_assessments_per_learner=10)
    
    # Save to CSV
    learners_df.to_csv('data/learner_data.csv', index=False)
    courses_df.to_csv('data/course_data.csv', index=False)
    assessments_df.to_csv('data/assessment_data.csv', index=False)
    
    print("\n" + "="*60)
    print("Data Generation Complete!")
    print("="*60)
    print(f"\n📁 Generated files:")
    print(f"  - data/learner_data.csv ({len(learners_df)} learners)")
    print(f"  - data/course_data.csv ({len(courses_df)} courses)")
    print(f"  - data/assessment_data.csv ({len(assessments_df)} assessments)")
    
    print(f"\n📊 Summary Statistics:")
    print(f"  - Average completion rate: {learners_df['completion_rate'].mean():.2%}")
    print(f"  - Average quiz score: {learners_df['quiz_average'].mean():.1f}")
    print(f"  - Average study hours: {learners_df['total_study_hours'].mean():.1f}")
    print(f"  - Learners at risk: {learners_df['at_risk'].sum()} ({learners_df['at_risk'].mean():.1%})")
    
    print(f"\n✅ Ready for analysis! Run: python analyze.py\n")


if __name__ == '__main__':
    main()
