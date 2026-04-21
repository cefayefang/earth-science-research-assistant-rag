from pathlib import Path
from functools import lru_cache
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def get_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_path(key: str) -> Path:
    cfg = get_settings()
    parts = key.split(".")
    val = cfg
    for p in parts:
        val = val[p]
    return ROOT / val


def gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    return key
