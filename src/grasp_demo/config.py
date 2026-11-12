"""Configuration loading with paths resolved relative to the YAML file."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import os


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"configuration does not exist: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("models.yaml must contain a mapping")
    config["_root"] = source.parent
    for section, key in (("retrieval", "corpus_config"), ("eval", "qa"), ("eval", "ids")):
        if config.get(section, {}).get(key):
            config[section][key] = str(resolve(config, config[section][key]))
    for name, adapter in config.get("proxy", {}).get("adapters", {}).items():
        config["proxy"]["adapters"][name] = str(resolve(config, adapter))
    if config.get("runs_dir"):
        config["runs_dir"] = str(resolve(config, config["runs_dir"]))
    return config


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config["_root"]) / path


def load_dotenv(root: str | Path) -> None:
    """Small dependency-free .env loader; environment variables win over disk."""
    path = Path(root) / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
