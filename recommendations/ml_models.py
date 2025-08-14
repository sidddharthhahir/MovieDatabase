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
from django.db import models
from django.core.cache import cache
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
        """Get content-based recommendations for a movie (accepts internal id or tmdb_id)"""
        try:
            tfidf_matrix = joblib.load(os.path.join(self.model_path, 'tfidf_matrix.pkl'))
            movies_df = joblib.load(os.path.join(self.model_path, 'movies_df.pkl'))
            
            # Resolve by internal id OR tmdb_id (if caller passed tmdb_id)
            mask = (movies_df['movie_id'] == movie_id) | (movies_df['tmdb_id'] == movie_id)
            if not mask.any():
                print(f"Seed movie not found in movies_df for id/tmdb_id={movie_id}")
                return []

            movie_idx = movies_df.index[mask][0]
            
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
        """Get truly personalized recommendations for a user"""
        # Check cache first (but make it user-specific)
        cache_key = f"user_recs_{user_id}"
        cached_recs = cache.get(cache_key)
        if cached_recs:
            return cached_recs
        
        user_ratings = Rating.objects.filter(user_id=user_id).select_related('movie')
        
        if not user_ratings.exists():
            # Return popular movies for new users
            popular_movies = Movie.objects.order_by('-popularity')[:n_recommendations]
            recommendations = [{'movie_id': movie.id, 'score': movie.popularity/100, 'algorithm': 'popularity'} 
                             for movie in popular_movies]
        else:
            recommendations = self._get_personalized_recommendations(user_id, user_ratings, n_recommendations)
        
        # Cache for 30 minutes (will be cleared when user rates)
        cache.set(cache_key, recommendations, 1800)
        return recommendations
    
    def _get_personalized_recommendations(self, user_id, user_ratings, n_recommendations):
        """Generate personalized recommendations based on user's rating patterns"""
        recommendations = []
        seen_movie_ids = set(user_ratings.values_list('movie_id', flat=True))
        
        # Get user's preferences
        high_rated_movies = user_ratings.filter(rating__gte=4)
        user_avg_rating = user_ratings.aggregate(avg_rating=models.Avg('rating'))['avg_rating'] or 3.0
        
        # Strategy 1: Content-based from highly rated movies
        content_recs = []
        for rating in high_rated_movies.order_by('-rating')[:5]:  # Top 5 rated movies
            similar_movies = self.get_content_recommendations(rating.movie.id, 8)
            for rec in similar_movies:
                if rec['movie_id'] not in seen_movie_ids:
                    # Weight by user's rating of the seed movie
                    rec['score'] *= (rating.rating / 5.0)
                    rec['seed_movie'] = rating.movie.title
                    content_recs.append(rec)
        
        # Strategy 2: Genre-based recommendations
        genre_recs = self._get_genre_based_recommendations(user_id, user_ratings, seen_movie_ids, user_avg_rating)
        
        # Strategy 3: Collaborative filtering (if available)
        collab_recs = self._get_collaborative_recommendations(user_id, seen_movie_ids)
        
        # Combine and score all recommendations
        all_recs = {}
        
        # Add content-based (weight: 0.4)
        for rec in content_recs:
            movie_id = rec['movie_id']
            if movie_id not in all_recs:
                all_recs[movie_id] = {'movie_id': movie_id, 'total_score': 0, 'algorithms': []}
            all_recs[movie_id]['total_score'] += rec['score'] * 0.4
            all_recs[movie_id]['algorithms'].append(f"content({rec.get('seed_movie', 'unknown')})")
        
        # Add genre-based (weight: 0.3)
        for rec in genre_recs:
            movie_id = rec['movie_id']
            if movie_id not in all_recs:
                all_recs[movie_id] = {'movie_id': movie_id, 'total_score': 0, 'algorithms': []}
            all_recs[movie_id]['total_score'] += rec['score'] * 0.3
            all_recs[movie_id]['algorithms'].append('genre')
        
        # Add collaborative (weight: 0.3)
        for rec in collab_recs:
            movie_id = rec['movie_id']
            if movie_id not in all_recs:
                all_recs[movie_id] = {'movie_id': movie_id, 'total_score': 0, 'algorithms': []}
            all_recs[movie_id]['total_score'] += rec['score'] * 0.3
            all_recs[movie_id]['algorithms'].append('collaborative')
        
        # Sort by total score and return top N
        sorted_recs = sorted(all_recs.values(), key=lambda x: x['total_score'], reverse=True)
        
        final_recs = []
        for rec in sorted_recs[:n_recommendations]:
            final_recs.append({
                'movie_id': rec['movie_id'],
                'score': rec['total_score'],
                'algorithm': '+'.join(rec['algorithms'][:2])  # Show top 2 algorithms
            })
        
        return final_recs
    
    def _get_genre_based_recommendations(self, user_id, user_ratings, seen_movie_ids, user_avg_rating):
        """Get recommendations based on user's genre preferences"""
        # Calculate user's genre preferences
        genre_scores = {}
        for rating in user_ratings:
            weight = (rating.rating - user_avg_rating) / 2.0  # Normalize around user's average
            for genre in rating.movie.genres.all():
                if genre.name not in genre_scores:
                    genre_scores[genre.name] = []
                genre_scores[genre.name].append(weight)
        
        # Average scores per genre
        preferred_genres = {}
        for genre, scores in genre_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > 0:  # Only positive preferences
                preferred_genres[genre] = avg_score
        
        if not preferred_genres:
            return []
        
        # Find movies in preferred genres
        top_genres = sorted(preferred_genres.items(), key=lambda x: x[1], reverse=True)[:3]
        recommendations = []
        
        for genre_name, preference_score in top_genres:
            genre_movies = Movie.objects.filter(
                genres__name=genre_name
            ).exclude(
                id__in=seen_movie_ids
            ).order_by('-vote_average', '-popularity')[:10]
            
            for movie in genre_movies:
                recommendations.append({
                    'movie_id': movie.id,
                    'score': preference_score * (movie.vote_average / 10.0),
                    'algorithm': f'genre({genre_name})'
                })
        
        return recommendations
    
    def _get_collaborative_recommendations(self, user_id, seen_movie_ids):
        """Get collaborative filtering recommendations"""
        try:
            if not self.collaborative_model:
                self.load_models()
            
            user_movie_matrix = joblib.load(os.path.join(self.model_path, 'user_movie_matrix.pkl'))
            
            if user_id not in user_movie_matrix.index:
                return []
            
            user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)
            
            # Get prediction probabilities for all movies
            probabilities = self.collaborative_model.predict_proba(user_ratings)[0]
            
            recommendations = []
            for movie_idx, prob in enumerate(probabilities):
                movie_id = user_movie_matrix.columns[movie_idx]
                if movie_id not in seen_movie_ids and prob > 0.5:  # Only recommend if likely to like
                    recommendations.append({
                        'movie_id': movie_id,
                        'score': prob,
                        'algorithm': 'collaborative'
                    })
            
            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:20]
        
        except Exception as e:
            print(f"Error in collaborative recommendations: {e}")
            return []
    
    def clear_user_cache(self, user_id):
        """Clear user's recommendation cache when they rate a movie"""
        cache_key = f"user_recs_{user_id}"
        cache.delete(cache_key)

# Initialize the recommendation system
recommendation_system = MovieRecommendationSystem()