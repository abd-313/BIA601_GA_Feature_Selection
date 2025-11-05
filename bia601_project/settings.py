"""
Django settings for bia601_project project.
Organized for production deployment (Railway) and static files serving (WhiteNoise).
"""
import os
from pathlib import Path
import sys
import dj_database_url

# --- 1. CORE PATH CONFIGURATION ---

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Custom Path Configuration: Allows importing modules from the project's root ('src' directory).
# This is crucial for finding and importing joblib files from the 'data/' directory.
# This must be done early to affect imports.
sys.path.insert(0, os.path.join(BASE_DIR, '..'))


# --- 2. SECURITY AND DEBUGGING ---

# SECURITY WARNING: keep the secret key used in production secret!
# It's recommended to load this from an environment variable in production.
SECRET_KEY = 'django-insecure-w&^!s#$m0x1536&b%61v$9+0e58y2!q$78*!k)h03!*h$1&'

# SECURITY WARNING: don't run with debug turned on in production!
# Load DEBUG status from environment variables (e.g., set DEBUG=False in production)
DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'


# Host Configuration for Production (Railway)
# ALLOWED_HOSTS must include all possible hostnames, including Railway's URL.
ALLOWED_HOSTS = [
    # Get Railway's URL from the environment variable (if available), otherwise default to localhosts
    os.environ.get('RAILWAY_STATIC_URL', '127.0.0.1'),
    '.railway.app', # Wildcard for Railway deployment
    '0.0.0.0', 
    'localhost', 
    '127.0.0.1',
    '*', # Allows all hosts for easy deployment testing (for temporary use only)
]

# Required for reverse proxy setups (like Railway) to detect HTTPS properly
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
CSRF_TRUSTED_ORIGINS = ['https://*.railway.app', 'http://127.0.0.1'] 


# --- 3. APPLICATION DEFINITION ---

INSTALLED_APPS = [
    # Django Built-in Apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party Apps
    'widget_tweaks', # Added to enable easy styling of form fields
    
    # Custom Apps
    'analysis', # Custom application for our analysis logic
]

# --- 4. MIDDLEWARE (WhiteNoise and Security setup) ---

MIDDLEWARE = [
    # 1. Security (MUST be first)
    'django.middleware.security.SecurityMiddleware',
    
    # 2. WhiteNoise (MUST be second to serve static files efficiently)
    'whitenoise.middleware.WhiteNoiseMiddleware', 
    
    # 3. Standard Django Middleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# --- 5. TEMPLATES ---

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # CRITICAL: This tells Django to look inside the project's 'templates' folder.
        'DIRS': [os.path.join(BASE_DIR, 'templates')], 
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'bia601_project.wsgi.application'


# --- 6. DATABASE ---

# Uses dj-database-url to configure the database from the environment variable (DATABASE_URL).
# Defaults to SQLite for local development.
DATABASES = {
    'default': dj_database_url.config(
        default='sqlite:///db.sqlite3',
        conn_max_age=600,
        conn_health_checks=True,
    )
}


# --- 7. STATIC AND MEDIA FILES (CRITICAL FOR YOUR ISSUE) ---

# URL prefix for static files (e.g., /static/style.css)
STATIC_URL = 'static/'

# 1. STATIC_ROOT: The directory where 'collectstatic' will dump all files (on the server).
# WhiteNoise will serve files from this location.
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# 2. STATICFILES_DIRS: The directories where Django should look for project-level static files
# (e.g., your global 'style.css' in the root 'static/' folder).
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# 3. STATICFILES_STORAGE: Tells WhiteNoise to handle compression and caching.
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files (for user-uploaded content, if applicable)
MEDIA_URL = '/media/ga_jobs/'
MEDIA_ROOT = BASE_DIR / 'analysis' / 'ga_job_results'


# --- 8. AUTHENTICATION AND VALIDATION ---

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# --- 9. INTERNATIONALIZATION ---

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# --- 10. DEFAULT CONFIGURATION ---

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'