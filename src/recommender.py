#!/usr/bin/env python3
"""
Learning Path Recommender
Generates personalized course recommendations
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import os


class LearningPathRecommender:
    """Recommend personalized learning paths"""
    
    def __init__(self, learner_data_path: str = 'data/learner_data.csv',
                 course_data_path: str = 'data/course_data.csv'):
        """Initialize with learner and course data"""
        
        self.learners_df = pd.read_csv(learner_data_path)
        self.courses_df = pd.read_csv(course_data_path)
        
        # Create learner-course interaction matrix (simulated)
        self.interaction_matrix = self._create_interaction_matrix()
        self.similarity_matrix = None
        
        print(f"Loaded {len(self.learners_df)} learners and {len(self.courses_df)} courses")
    
    def _create_interaction_matrix(self) -> pd.DataFrame:
        """
        Create simulated learner-course interaction matrix
        1 = completed, 0 = not taken, 0.5 = in progress
        """
        
        # Simulate interactions based on learner completion rate
        interactions = []
        
        for _, learner in self.learners_df.iterrows():
            learner_id = learner['learner_id']
            n_courses_completed = int(learner['courses_completed'])
            n_courses_enrolled = int(learner['courses_enrolled'])
            
            # Sample completed courses
            completed_courses = np.random.choice(
                self.courses_df['course_id'].values,
                size=min(n_courses_completed, len(self.courses_df)),
                replace=False
            )
            
            # Sample in-progress courses
            in_progress = n_courses_enrolled - n_courses_completed
            if in_progress > 0:
                available_courses = [c for c in self.courses_df['course_id'].values 
                                    if c not in completed_courses]
                in_progress_courses = np.random.choice(
                    available_courses,
                    size=min(in_progress, len(available_courses)),
                    replace=False
                )
            else:
                in_progress_courses = []
            
            for course_id in completed_courses:
                interactions.append({
                    'learner_id': learner_id,
                    'course_id': course_id,
                    'interaction': 1.0
                })
            
            for course_id in in_progress_courses:
                interactions.append({
                    'learner_id': learner_id,
                    'course_id': course_id,
                    'interaction': 0.5
                })
        
        df = pd.DataFrame(interactions)
        
        # Pivot to matrix format
        matrix = df.pivot_table(
            index='learner_id',
            columns='course_id',
            values='interaction',
            fill_value=0
        )
        
        return matrix
    
    def calculate_similarity(self):
        """Calculate learner similarity matrix"""
        
        print("\nCalculating learner similarity...")
        
        # Use cosine similarity on interaction matrix
        self.similarity_matrix = cosine_similarity(self.interaction_matrix)
        
        # Convert to DataFrame for easier access
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=self.interaction_matrix.index,
            columns=self.interaction_matrix.index
        )
        
        print("✓ Similarity matrix calculated")
    
    def collaborative_filtering(self, learner_id: str, top_n: int = 5) -> List[Dict]:
        """
        Recommend courses based on similar learners (collaborative filtering)
        
        Args:
            learner_id: Target learner ID
            top_n: Number of recommendations
        
        Returns:
            List of recommended courses
        """
        
        if self.similarity_matrix is None:
            self.calculate_similarity()
        
        if learner_id not in self.similarity_df.index:
            return []
        
        # Get similar learners
        similar_learners = self.similarity_df[learner_id].sort_values(ascending=False)[1:11]
        
        # Get courses completed by similar learners
        similar_learner_ids = similar_learners.index.tolist()
        
        # Courses the target learner hasn't taken
        learner_courses = self.interaction_matrix.loc[learner_id]
        untaken_courses = learner_courses[learner_courses == 0].index.tolist()
        
        # Score courses by how many similar learners completed them
        course_scores = {}
        for course_id in untaken_courses:
            score = 0
            for sim_learner_id in similar_learner_ids:
                if self.interaction_matrix.loc[sim_learner_id, course_id] == 1.0:
                    # Weight by similarity
                    similarity = similar_learners[sim_learner_id]
                    score += similarity
            course_scores[course_id] = score
        
        # Sort and get top N
        top_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get course details
        recommendations = []
        for course_id, score in top_courses:
            course_info = self.courses_df[self.courses_df['course_id'] == course_id].iloc[0]
            recommendations.append({
                'course_id': course_id,
                'course_name': course_info['course_name'],
                'category': course_info['category'],
                'difficulty': course_info['difficulty'],
                'score': round(score, 3),
                'method': 'collaborative_filtering'
            })
        
        return recommendations
    
    def content_based_filtering(self, learner_id: str, top_n: int = 5) -> List[Dict]:
        """
        Recommend courses based on learner's skill gaps (content-based)
        
        Args:
            learner_id: Target learner ID
            top_n: Number of recommendations
        
        Returns:
            List of recommended courses
        """
        
        learner = self.learners_df[self.learners_df['learner_id'] == learner_id]
        
        if learner.empty:
            return []
        
        learner = learner.iloc[0]
        
        # Identify skill gaps
        skill_scores = {
            'Beginner': learner['skill_fundamentals'],
            'Intermediate': learner['skill_intermediate'],
            'Advanced': learner['skill_advanced'],
            'Expert': learner['skill_expert']
        }
        
        # Find weakest skill level
        weakest_level = min(skill_scores.items(), key=lambda x: x[1])[0]
        
        # Get courses the learner hasn't taken
        learner_courses = self.interaction_matrix.loc[learner_id]
        untaken_courses = learner_courses[learner_courses == 0].index.tolist()
        
        # Filter courses by difficulty matching skill gap
        candidate_courses = self.courses_df[
            (self.courses_df['course_id'].isin(untaken_courses)) &
            (self.courses_df['difficulty'] == weakest_level)
        ]
        
        # If not enough, include next level
        if len(candidate_courses) < top_n:
            next_levels = {
                'Beginner': 'Intermediate',
                'Intermediate': 'Advanced',
                'Advanced': 'Expert',
                'Expert': 'Expert'
            }
            next_level = next_levels[weakest_level]
            
            additional_courses = self.courses_df[
                (self.courses_df['course_id'].isin(untaken_courses)) &
                (self.courses_df['difficulty'] == next_level)
            ]
            candidate_courses = pd.concat([candidate_courses, additional_courses])
        
        # Sort by enrollment count (popularity proxy)
        candidate_courses = candidate_courses.sort_values('enrollment_count', ascending=False)
        
        # Get top N
        recommendations = []
        for _, course in candidate_courses.head(top_n).iterrows():
            recommendations.append({
                'course_id': course['course_id'],
                'course_name': course['course_name'],
                'category': course['category'],
                'difficulty': course['difficulty'],
                'targets_skill_gap': weakest_level,
                'method': 'content_based'
            })
        
        return recommendations
    
    def hybrid_recommendation(self, learner_id: str, top_n: int = 5) -> List[Dict]:
        """
        Hybrid approach combining collaborative and content-based filtering
        
        Args:
            learner_id: Target learner ID
            top_n: Number of recommendations
        
        Returns:
            List of recommended courses
        """
        
        # Get recommendations from both methods
        collab_recs = self.collaborative_filtering(learner_id, top_n=top_n*2)
        content_recs = self.content_based_filtering(learner_id, top_n=top_n*2)
        
        # Combine and deduplicate
        seen_courses = set()
        hybrid_recs = []
        
        # Interleave recommendations (prioritize collaborative slightly)
        all_recs = []
        for i in range(max(len(collab_recs), len(content_recs))):
            if i < len(collab_recs):
                all_recs.append(collab_recs[i])
            if i < len(content_recs):
                all_recs.append(content_recs[i])
        
        for rec in all_recs:
            if rec['course_id'] not in seen_courses:
                rec['method'] = 'hybrid'
                hybrid_recs.append(rec)
                seen_courses.add(rec['course_id'])
            
            if len(hybrid_recs) >= top_n:
                break
        
        return hybrid_recs
    
    def recommend_learning_path(self, learner_id: str, 
                                method: str = 'hybrid', 
                                top_n: int = 5) -> List[Dict]:
        """
        Main recommendation function
        
        Args:
            learner_id: Target learner ID
            method: 'collaborative', 'content_based', or 'hybrid'
            top_n: Number of recommendations
        
        Returns:
            List of recommended courses
        """
        
        if method == 'collaborative':
            return self.collaborative_filtering(learner_id, top_n)
        elif method == 'content_based':
            return self.content_based_filtering(learner_id, top_n)
        elif method == 'hybrid':
            return self.hybrid_recommendation(learner_id, top_n)
        else:
            raise ValueError(f"Unknown method: {method}")


# Example usage
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Learning Path Recommender - Test Run")
    print("="*60)
    
    # Initialize recommender
    recommender = LearningPathRecommender()
    
    # Test learner
    test_learner = 'L0001'
    
    print(f"\nGenerating recommendations for {test_learner}...")
    
    # Collaborative filtering
    print("\n1. Collaborative Filtering Recommendations:")
    collab_recs = recommender.collaborative_filtering(test_learner, top_n=3)
    for i, rec in enumerate(collab_recs, 1):
        print(f"   {i}. {rec['course_name']} ({rec['difficulty']})")
        print(f"      Score: {rec['score']:.3f}")
    
    # Content-based
    print("\n2. Content-Based Recommendations:")
    content_recs = recommender.content_based_filtering(test_learner, top_n=3)
    for i, rec in enumerate(content_recs, 1):
        print(f"   {i}. {rec['course_name']} ({rec['difficulty']})")
        print(f"      Targets: {rec['targets_skill_gap']}")
    
    # Hybrid
    print("\n3. Hybrid Recommendations:")
    hybrid_recs = recommender.hybrid_recommendation(test_learner, top_n=5)
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"   {i}. {rec['course_name']} ({rec['difficulty']})")
    
    print("\n✅ Recommendations generated!")
