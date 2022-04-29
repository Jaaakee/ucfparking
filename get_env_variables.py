import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from dotenv import load_dotenv

if (os.environ.get("DB_PASS") == None and os.path.exists(os.path.join(ROOT_DIR, 'config', 'conf', '.env'))):
 load_dotenv(os.path.join(ROOT_DIR, 'config', 'conf', '.env'))