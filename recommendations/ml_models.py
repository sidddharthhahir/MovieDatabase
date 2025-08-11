import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from django.conf import settings
from movies.models import Movie, Rating, Genre
from django.contrib.auth import get_user_model

User = get_user_model()

class MovieRecommendationSystem:
    def __init__(self):
        self.content_model = None
        self.collaborative_model = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_models')
        os.makedirs(self.model_path, exist_ok=True)
    
    def prepare_content_features(self):
        """Prepare content-based features for movies"""
        movies = Movie.objects.prefetch_related('genres').all()
        
        data = []
        for movie in movies:
            genres_str = ' '.join([genre.name for genre in movie.genres.all()])
            data.append({
                'movie_id': movie.id,
                'tmdb_id': movie.tmdb_id,
                'title': movie.title,
                'genres': genres_str,
                'overview': movie.overview or '',
                'vote_average': movie.vote_average,
                'popularity': movie.popularity,
                'runtime': movie.runtime or 0,
                'release_year': movie.release_date.year if movie.release_date else 2000
            })
        
        return pd.DataFrame(data)
    
    def prepare_collaborative_features(self):
        """Prepare collaborative filtering features"""
        ratings = Rating.objects.select_related('user', 'movie').all()
        
        data = []
        for rating in ratings:
            data.append({
                'user_id': rating.user.id,
                'movie_id': rating.movie.id,
                'rating': rating.rating,
                'tmdb_id': rating.movie.tmdb_id
            })
        
        return pd.DataFrame(data)
    
    def train_content_model(self):
        """Train content-based recommendation model"""
        df = self.prepare_content_features()
        
        # Create TF-IDF features from genres and overview
        text_features = df['genres'] + ' ' + df['overview']
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
        # Combine with numerical features
        numerical_features = df[['vote_average', 'popularity', 'runtime', 'release_year']].fillna(0)
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        # Save the model components
        joblib.dump(self.tfidf_vectorizer, os.path.join(self.model_path, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))
        joblib.dump(tfidf_matrix, os.path.join(self.model_path, 'tfidf_matrix.pkl'))
        joblib.dump(numerical_features_scaled, os.path.join(self.model_path, 'numerical_features.pkl'))
        joblib.dump(df, os.path.join(self.model_path, 'movies_df.pkl'))
        
        print("Content-based model trained and saved!")
    
    def train_collaborative_model(self):
        """Train collaborative filtering model using Random Forest"""
        ratings_df = self.prepare_collaborative_features()
        
        if len(ratings_df) < 100:  # Not enough data
            print("Not enough ratings data for collaborative filtering")
            return
        
        # Create user-movie matrix
        user_movie_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        
        # Prepare training data for binary classification (like/dislike)
        X = []
        y = []
        
        for _, row in ratings_df.iterrows():
            user_ratings = user_movie_matrix.loc[row['user_id']].values
            X.append(user_ratings)
            y.append(1 if row['rating'] >= 4 else 0)  # Binary: like (4-5) vs dislike (1-3)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.collaborative_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.collaborative_model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.collaborative_model, os.path.join(self.model_path, 'collaborative_model.pkl'))
        joblib.dump(user_movie_matrix, os.path.join(self.model_path, 'user_movie_matrix.pkl'))
        
        print(f"Collaborative model trained! Accuracy: {self.collaborative_model.score(X_test, y_test):.3f}")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.tfidf_vectorizer = joblib.load(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.pkl'))
            self.collaborative_model = joblib.load(os.path.join(self.model_path, 'collaborative_model.pkl'))
            return True
        except FileNotFoundError:
            return False
    
    def get_content_recommendations(self, movie_id, n_recommendations=10):
        """Get content-based recommendations for a movie"""
        try:
            tfidf_matrix = joblib.load(os.path.join(self.model_path, 'tfidf_matrix.pkl'))
            movies_df = joblib.load(os.path.join(self.model_path, 'movies_df.pkl'))
            
            # Find movie index
            movie_idx = movies_df[movies_df['movie_id'] == movie_id].index[0]
            
            # Calculate similarity
            cosine_sim = cosine_similarity(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
            
            # Get top similar movies
            similar_indices = cosine_sim.argsort()[-n_recommendations-1:-1][::-1]
            
            recommendations = []
            for idx in similar_indices:
                if idx != movie_idx:
                    recommendations.append({
                        'movie_id': movies_df.iloc[idx]['movie_id'],
                        'score': cosine_sim[idx],
                        'algorithm': 'content'
                    })
            
            return recommendations
        except Exception as e:
            print(f"Error in content recommendations: {e}")
            return []
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Get personalized recommendations for a user"""
        user_ratings = Rating.objects.filter(user_id=user_id).select_related('movie')
        
        if not user_ratings.exists():
            # Return popular movies for new users
            popular_movies = Movie.objects.order_by('-popularity')[:n_recommendations]
            return [{'movie_id': movie.id, 'score': movie.popularity/100, 'algorithm': 'popularity'} 
                   for movie in popular_movies]
        
        # Get content-based recommendations from user's highly rated movies
        recommendations = []
        high_rated_movies = user_ratings.filter(rating__gte=4)
        
        for rating in high_rated_movies[:3]:  # Use top 3 rated movies
            content_recs = self.get_content_recommendations(rating.movie.id, 5)
            recommendations.extend(content_recs)
        
        # Remove duplicates and movies user has already rated
        seen_movie_ids = set(user_ratings.values_list('movie_id', flat=True))
        unique_recs = {}
        
        for rec in recommendations:
            if rec['movie_id'] not in seen_movie_ids and rec['movie_id'] not in unique_recs:
                unique_recs[rec['movie_id']] = rec
        
        # Sort by score and return top N
        sorted_recs = sorted(unique_recs.values(), key=lambda x: x['score'], reverse=True)
        return sorted_recs[:n_recommendations]

# Initialize the recommendation system
recommendation_system = MovieRecommendationSystem()