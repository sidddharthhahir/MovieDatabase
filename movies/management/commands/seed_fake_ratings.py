import random
from collections import Counter
from datetime import datetime
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.db import transaction
from movies.models import Movie, Rating, Genre

User = get_user_model()

def weighted_choice(items, weights):
    return random.choices(items, weights=weights, k=1)[0]

class Command(BaseCommand):
    help = "Seed synthetic users and ratings for faster model experimentation."

    def add_arguments(self, parser):
        parser.add_argument('--users', type=int, default=200, help='Number of synthetic users to create')
        parser.add_argument('--min_ratings', type=int, default=30, help='Min ratings per synthetic user')
        parser.add_argument('--max_ratings', type=int, default=80, help='Max ratings per synthetic user')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--clear', action='store_true', help='Delete existing synthetic data before seeding')
        parser.add_argument('--delete_synthetic_users', action='store_true', help='When used with --clear, also delete synthetic users')

    def handle(self, *args, **opts):
        random.seed(opts['seed'])

        # Prep movies and genres
        movies = list(Movie.objects.prefetch_related('genres').all())
        if not movies:
            self.stderr.write("No movies found. Run: python manage.py sync_tmdb --mode discover --languages hi --pages 50")
            return

        genres = list(Genre.objects.all())
        genre_names = [g.name for g in genres] if genres else []

        # Clear existing synthetic data if requested
        if opts['clear']:
            self.stdout.write("Clearing synthetic ratings...")
            Rating.objects.filter(user__username__startswith="demo_user_").delete()
            if opts['delete_synthetic_users']:
                self.stdout.write("Deleting synthetic users...")
                User.objects.filter(username__startswith="demo_user_").delete()

        # Create synthetic users with favorite_genres bias
        self.stdout.write(f"Creating {opts['users']} synthetic users…")
        created_users = []
        with transaction.atomic():
            for i in range(opts['users']):
                username = f"demo_user_{i+1:04d}"
                if User.objects.filter(username=username).exists():
                    user = User.objects.get(username=username)
                else:
                    favs = []
                    # Pick 2-3 favorite genres if available
                    if genre_names:
                        favs = random.sample(genre_names, k=min(len(genre_names), random.choice([2, 3])))
                    user = User.objects.create_user(
                        username=username,
                        email=f"{username}@example.com",
                        password="password123!",
                    )
                    # If CustomUser has favorite_genres field (as in your setup), set it.
                    if hasattr(user, 'favorite_genres'):
                        user.favorite_genres = favs
                        user.save()
                created_users.append(user)

        # Build a simple popularity distribution to bias sampling
        # More popular movies are more likely to be rated
        pops = [max(1.0, m.popularity or 1.0) for m in movies]
        pop_sum = sum(pops)
        weights = [p / pop_sum for p in pops]

        # Helper: genre match score for a movie given user's favorite genres
        def genre_match(movie, favs):
            mg = {g.name for g in movie.genres.all()}
            return len(mg.intersection(set(favs))) if favs else 0

        # Generate ratings
        self.stdout.write("Generating ratings with genre/popularity biases…")
        total_ratings = 0
        ratings_to_create = []
        for user in created_users:
            n = random.randint(opts['min_ratings'], opts['max_ratings'])
            # sample movies without replacement, weighted by popularity
            sampled = set()
            # To avoid expensive weighted sampling without replacement, do multiple draws with rejection
            while len(sampled) < n and len(sampled) < len(movies):
                m = weighted_choice(movies, weights)
                sampled.add(m)
            favs = getattr(user, 'favorite_genres', []) if hasattr(user, 'favorite_genres') else []
            for m in sampled:
                # Base rating around global trend: higher vote_average -> slightly higher user rating
                base = (m.vote_average or 5.5) / 2.0  # map 0-10 to ~0-5 scale
                # Add positive bias if movie matches user favorite genres
                bonus = 0.6 * min(2, genre_match(m, favs))
                # Add noise
                rating_val = base + bonus + random.gauss(0.0, 0.7)
                rating_val = max(1.0, min(5.0, rating_val))
                ratings_to_create.append(Rating(user=user, movie=m, rating=round(rating_val, 1)))
            total_ratings += n

        # Bulk create (in chunks)
        self.stdout.write(f"Creating ~{total_ratings} ratings…")
        CHUNK = 5000
        created = 0
        with transaction.atomic():
            for i in range(0, len(ratings_to_create), CHUNK):
                chunk = ratings_to_create[i:i+CHUNK]
                Rating.objects.bulk_create(chunk, ignore_conflicts=True)
                created += len(chunk)
        self.stdout.write(self.style.SUCCESS(f"Done. Created approximately {created} ratings for {len(created_users)} users."))