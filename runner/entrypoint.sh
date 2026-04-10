#!/bin/sh
set -e

python /app/setup.py
exec "$@"
