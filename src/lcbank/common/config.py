"""Config loading and the .env loader. Secrets are never printed, logged or saved (R4)."""
import os
from functools import lru_cache

import yaml

from .paths import CONFIG, ROOT


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    with open(CONFIG / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def last_available_sitting() -> int:
    return int(load("datasets")["last_available_sitting"])


def load_dotenv(path=None) -> list[str]:
    """Load KEY=VALUE lines from app-build/.env into os.environ without overriding existing values.

    Returns only the variable *names* that were set, never values.
    """
    path = path or ROOT / ".env"
    names = []
    if not os.path.exists(path):
        return names
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                names.append(key)
    return names
