#!/usr/bin/env bash

# File: build.sh
# Description: Script executed by Render during the build phase.


echo "--- Installing dependencies ---"
pip install -r requirements.txt


echo "--- Collecting static files ---"
python manage.py collectstatic --noinput


echo "--- Running migrations ---"
python manage.py migrate