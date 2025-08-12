MovieDatabase
A Django-based web app for exploring movies, managing user accounts, and generating personalized recommendations.
It uses Supabase (PostgreSQL) as the database and TMDB as the movie data source, with explainable ML recommendations.

Repository: https://github.com/sidddharthhahir/MovieDatabse

Features
User authentication and profiles (accounts), styled with crispy-forms and Bootstrap 5.
Movie browsing, details, images, and search (movies).
TMDB integration for titles, posters, and metadata.
Personalized recommendations (recommendations) powered by scikit-learn with explainability via SHAP and LIME.
Supabase (hosted PostgreSQL) for reliable storage.

Tech Stack
Django 4.2, Python 3.10+
Database: Supabase Postgres (via dj-database-url + psycopg2-binary)
ML/Analytics: scikit-learn, pandas, numpy, joblib, shap, lime, matplotlib, seaborn, plotly
UI: django-crispy-forms, crispy-bootstrap5
Images: Pillow
External API: TMDB

Project Structure
moviesite: Django project settings, URLs, WSGI/ASGI
accounts: auth, profiles, forms, views
movies: models, admin, views, TMDB integration helpers
recommendations: training/inference utils, loaders, explanations
ml_models: persisted model artifacts (.pkl), preprocessors
templates: reusable Django templates
manage.py and requirements.txt

Quickstart
Clone and create a virtual environment
git clone https://github.com/sidddharthhahir/MovieDatabse.git
cd MovieDatabse
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
Configure environment

Create a .env file in the project root:

# Django
SECRET_KEY=replace-with-a-strong-secret-key
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost

# Supabase (Postgres). Include sslmode=require for hosted DBs.
# You can find this in Supabase Project Settings > Database.
DATABASE_URL=postgres://USER:PASSWORD@HOST:PORT/DB_NAME?sslmode=require

# TMDB
TMDB_API_KEY=your_tmdb_api_key
TMDB_BASE_URL=https://api.themoviedb.org/3
TMDB_IMAGE_BASE_URL=https://image.tmdb.org/t/p
Notes for Supabase: ensure your IP is allowed if using connection restrictions, and keep sslmode=require. For local development without Supabase, omit DATABASE_URL to fall back to SQLite (if settings support it).

Apply migrations and create an admin user
python manage.py migrate
python manage.py createsuperuser
Run the server
python manage.py runserver
Visit http://127.0.0.1:8000/ and the admin at http://127.0.0.1:8000/admin/.

TMDB Integration
Set TMDB_API_KEY in .env. The app should use requests to call endpoints such as /search/movie, /movie/{id}, and image base paths from TMDB_IMAGE_BASE_URL (e.g., /w500).
If rate-limiting applies, consider simple caching of TMDB responses or nightly data pulls.
Recommendations and Models
Training: use pandas/numpy and scikit-learn to build similarity or hybrid models. Save artifacts to ml_models with joblib.dump.
Inference: the recommendations app should load artifacts at startup or lazily when requested.
Explainability: generate SHAP/LIME plots on demand. In headless environments, save plots to static files and embed them in templates.
Example artifact usage:

from joblib import load
model = load('ml_models/recommender.pkl')
Environment/Deployment
Production: set DEBUG=False, configure ALLOWED_HOSTS, keep DATABASE_URL pointing to Supabase with sslmode=require.
Static files: configure STATIC_ROOT and run collectstatic; consider WhiteNoise or a CDN.
Secrets: never commit .env. Use environment variables in your hosting platform.
Common Management Commands
# Create a new app
python manage.py startapp <app_name>

# Run tests (if present)
python manage.py test

Troubleshooting
Supabase connection errors: confirm DATABASE_URL format and sslmode=require; verify credentials and network access.
psycopg2-binary install issues: ensure build tools are available, or use prebuilt wheels; on Apple Silicon, recreate the venv with supported Python versions.
TMDB 401/403: check TMDB_API_KEY and ensure you’re sending the Authorization header or api_key parameter as your integration expects.
SHAP/LIME rendering: when running on servers without a display, set Matplotlib to Agg and save figures, then serve as static files.

Contributing
Open an issue to discuss changes. Please keep code style consistent and include concise descriptions in PRs.
