"""Loads the credentials of the database from a local file (local deployment) or from the heroku credentials (real deployment)."""
import os

from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("DB_PASS") is None and os.path.exists(os.path.join(ROOT_DIR, '../.env')):
    load_dotenv(os.path.join(ROOT_DIR, '../.env'))
