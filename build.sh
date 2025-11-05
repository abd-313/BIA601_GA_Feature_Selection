#!/usr/bin/env bash

# File: build.sh
# Description: Script executed by Render during the build phase.

# Install all dependencies from requirements.txt
echo "--- Installing dependencies ---"
pip install -r requirements.txt

# Collect static files for production using whitenoise. The --noinput flag prevents interactive prompts.
echo "--- Collecting static files ---"
python manage.py collectstatic --noinput

# Run database migrations to set up the database structure
echo "--- Running migrations ---"
python manage.py migrate