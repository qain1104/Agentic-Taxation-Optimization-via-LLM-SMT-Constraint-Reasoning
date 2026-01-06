from __future__ import annotations
import os
from dotenv import load_dotenv

def load_env() -> None:
    try:
        load_dotenv()
    except Exception:
        pass

def get_openai_key() -> str:
    load_env()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Please export OPENAI_API_KEY or create a .env.")
    return key
