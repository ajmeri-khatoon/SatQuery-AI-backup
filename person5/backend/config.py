import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the environment or .env file")

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is not set in the environment or .env file")
