"""Project configuration loaded from config.yaml."""
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML configuration file and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)