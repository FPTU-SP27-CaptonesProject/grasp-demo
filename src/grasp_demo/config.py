"""Configuration loading with paths resolved relative to the YAML file."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PATH_KEYS = (("retrieval", "corpus_config"), ("eval", "qa"), ("eval", "ids"), ("eval", "showcase"))


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"configuration does not exist: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("models.yaml must contain a mapping")
    config["_root"] = source.parent
    for section in ("reader", "retrieval", "proxy", "eval"):
        if config.get(section) is None:
            config[section] = {}
    for section, key in PATH_KEYS:
        if (config.get(section) or {}).get(key):
            config[section][key] = str(resolve(config, config[section][key]))
    adapters = (config.get("proxy") or {}).get("adapters") or {}
    for name, value in list(adapters.items()):
        entry = {"path": value} if isinstance(value, str) else dict(value or {})
        if not entry.get("path"):
            raise ValueError(f"proxy.adapters.{name} needs a path")
        entry["path"] = str(resolve(config, entry["path"]))
        entry["label"] = str(entry.get("label") or Path(entry["path"]).name)
        adapters[name] = entry
    if config.get("runs_dir"):
        config["runs_dir"] = str(resolve(config, config["runs_dir"]))
    return config


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config["_root"]) / path


def reader_api_key(values: dict[str, Any]) -> str:
    """Key from the environment variable named by `api_key_env`, else `api_key`, else EMPTY."""
    name = values.get("api_key_env")
    if name:
        return os.environ.get(name) or "EMPTY"
    return str(values.get("api_key") or "EMPTY")


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
