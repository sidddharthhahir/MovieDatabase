import time
import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from movies.models import Movie, Genre
from django.utils.dateparse import parse_date

TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_get(path, params=None, sleep=0.15):
    params = params or {}
    params['api_key'] = settings.TMDB_API_KEY
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    if sleep:
        time.sleep(sleep)  # be nice to the API
    return r.json()

def upsert_movie_from_tmdb_summary(summary):
    tmdb_id = summary['id']
    release_date = parse_date(summary.get('release_date')) if summary.get('release_date') else None
    obj, _ = Movie.objects.update_or_create(
        tmdb_id=tmdb_id,
        defaults={
            'title': summary.get('title') or summary.get('name') or '',
            'overview': summary.get('overview') or '',
            'release_date': release_date,
            'poster_path': summary.get('poster_path') or '',
            'backdrop_path': summary.get('backdrop_path') or '',
            'vote_average': summary.get('vote_average') or 0,
            'vote_count': summary.get('vote_count') or 0,
            'popularity': summary.get('popularity') or 0,
        }
    )
    # Fetch full details for runtime + genres
    detail = tmdb_get(f"/movie/{tmdb_id}", params={})
    obj.runtime = detail.get('runtime') or None
    obj.save()
    # genres
    obj.genres.clear()
    for g in detail.get('genres', []):
        genre = Genre.objects.filter(tmdb_id=g['id']).first()
        if genre:
            obj.genres.add(genre)
    return obj

class Command(BaseCommand):
    help = "Sync movies from TMDB: popular or discover (Bollywood/Indian languages)."

    def add_arguments(self, parser):
        # Mode
        parser.add_argument('--mode', choices=['popular', 'discover'], default='popular',
                            help='popular: TMDB popular feed; discover: filter by language/region/year/genre')
        # General
        parser.add_argument('--pages', type=int, default=5, help='Number of pages to fetch (20 per page)')
        parser.add_argument('--start_page', type=int, default=1, help='Start page (for continuing large syncs)')
        # Discover filters
        parser.add_argument('--languages', type=str, default='hi',
                            help='Comma-separated original language codes (e.g., hi,ta,te,ml,kn,bn,mr,ur)')
        parser.add_argument('--region', type=str, default='IN', help='Region/market code (e.g., IN)')
        parser.add_argument('--from_year', type=int, default=None, help='Filter release_date.gte (year)')
        parser.add_argument('--to_year', type=int, default=None, help='Filter release_date.lte (year)')
        parser.add_argument('--min_vote_count', type=int, default=0, help='Minimum TMDB vote_count')
        parser.add_argument('--min_vote_avg', type=float, default=0.0, help='Minimum TMDB vote_average')
        parser.add_argument('--with_genres', type=str, default='',
                            help='Comma-separated TMDB genre IDs to include (optional)')
        parser.add_argument('--sort_by', type=str, default='popularity.desc',
                            help='TMDB sort_by for discover (e.g., popularity.desc, vote_average.desc, release_date.desc)')

    def sync_genres(self):
        genres = tmdb_get("/genre/movie/list", params={}).get('genres', [])
        for g in genres:
            Genre.objects.update_or_create(tmdb_id=g['id'], defaults={'name': g['name']})
        self.stdout.write(f"Synced {len(genres)} genres")

    def handle(self, *args, **opts):
        api_key = settings.TMDB_API_KEY
        if not api_key:
            self.stderr.write("TMDB_API_KEY is not set in .env/settings")
            return

        mode = opts['mode']
        pages = opts['pages']
        start_page = opts['start_page']

        # Always ensure genres exist
        self.sync_genres()

        total_upserted = 0

        if mode == 'popular':
            self.stdout.write(f"Syncing POPULAR movies: pages {start_page}..{start_page+pages-1}")
            for page in range(start_page, start_page + pages):
                data = tmdb_get("/movie/popular", params={'page': page})
                results = data.get('results', [])
                for m in results:
                    obj = upsert_movie_from_tmdb_summary(m)
                    total_upserted += 1
                    self.stdout.write(f"[popular p{page}] {obj.title}")
            self.stdout.write(self.style.SUCCESS(f"Done. Upserted {total_upserted} movies."))

        elif mode == 'discover':
            languages = [l.strip() for l in opts['languages'].split(',') if l.strip()]
            region = opts['region']
            with_genres = opts['with_genres'].strip()
            sort_by = opts['sort_by']

            base_params = {
                'include_adult': 'false',
                'include_video': 'false',
                'watch_region': region,
                'region': region,
                'sort_by': sort_by,
                'vote_count.gte': opts['min_vote_count'],
                'vote_average.gte': opts['min_vote_avg'],
            }
            if opts['from_year']:
                base_params['primary_release_date.gte'] = f"{opts['from_year']}-01-01"
            if opts['to_year']:
                base_params['primary_release_date.lte'] = f"{opts['to_year']}-12-31"
            if with_genres:
                base_params['with_genres'] = with_genres

            self.stdout.write(f"Syncing DISCOVER for languages {languages}, region={region}, pages {start_page}..{start_page+pages-1}")
            for lang in languages:
                for page in range(start_page, start_page + pages):
                    params = dict(base_params)
                    params['with_original_language'] = lang
                    params['page'] = page
                    data = tmdb_get("/discover/movie", params=params)
                    results = data.get('results', [])
                    if not results:
                        self.stdout.write(f"[discover {lang}] no results on page {page}")
                        continue
                    for m in results:
                        obj = upsert_movie_from_tmdb_summary(m)
                        total_upserted += 1
                        self.stdout.write(f"[discover {lang} p{page}] {obj.title}")

            self.stdout.write(self.style.SUCCESS(f"Done. Upserted {total_upserted} movies."))

        else:
            self.stderr.write("Unsupported mode.")