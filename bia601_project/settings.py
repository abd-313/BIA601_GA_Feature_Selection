import os
from pathlib import Path
import sys
import dj_database_url


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Custom Path Configuration: Allows importing modules from the project's root ('src' directory).

sys.path.insert(0, os.path.join(BASE_DIR, '..'))


#Forgot to make the env file lol):
SECRET_KEY = 'django-insecure-w&^!s#$m0x1536&b%61v$9+0e58y2!q$78*!k)h03!*h$1&'


DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'



ALLOWED_HOSTS = [
    
    os.environ.get('RAILWAY_STATIC_URL', '127.0.0.1'),
    '.railway.app', # Wildcard for Railway deployment
    '0.0.0.0', 
    'localhost', 
    '127.0.0.1',
    '*', # Allows all hosts for easy deployment testing (for temporary use only)
]


SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
CSRF_TRUSTED_ORIGINS = ['https://*.railway.app', 'http://127.0.0.1'] 




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



MIDDLEWARE = [
    
    'django.middleware.security.SecurityMiddleware',
    
    
    'whitenoise.middleware.WhiteNoiseMiddleware', 
    
    # 3. Standard Django Middleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        
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

DATABASES = {
    'default': dj_database_url.config(
        default='sqlite:///db.sqlite3',
        conn_max_age=600,
        conn_health_checks=True,
    )
}

STATIC_URL = 'static/'

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_URL = '/media/ga_jobs/'
MEDIA_ROOT = BASE_DIR / 'analysis' / 'ga_job_results'


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
LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'