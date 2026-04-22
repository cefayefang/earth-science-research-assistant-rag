from pathlib import Path
from functools import lru_cache
import yaml
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parent.parent.parent


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


def openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return key


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=openai_api_key())
