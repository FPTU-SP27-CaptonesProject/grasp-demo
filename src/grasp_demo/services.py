"""Resources shared by every column, loaded once. A component that fails its probe disables only the
columns that need it (with the reason); the server keeps running."""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import reader_api_key
from .proxy import HFProxy, VLLMProxy

NEEDS = {
    "rag_7b": ("reader", "retriever"),
    "rag_235b": ("strong", "retriever"),
    "ircot_7b": ("reader", "retriever"),
    "grasp_sft": ("reader", "retriever", "proxy"),
    "grasp_grpo": ("reader", "retriever", "proxy"),
}


def load_reader(services: DemoServices) -> None:
    from grasp_grpo.reader import GraspReader, chat_renderer, openai_completion

    values = services.config["reader"]
    complete = openai_completion(values["url"], reader_api_key(values), values["model"], temperature=0.0)
    complete("Say OK.", stop=None, max_tokens=4)  # probe: raises when the reader is down
    services.reader = GraspReader(complete, chat_renderer(values.get("tokenizer") or values["model"]))


def load_retriever(services: DemoServices) -> None:
    from grasp_grpo.retrieval import CorpusRouter

    from .allcorpora import RouterWithAll

    values = services.config["retrieval"]
    top_k = int(values.get("top_k", 10))
    router = CorpusRouter(values["corpus_config"], values.get("encoder", "intfloat/e5-large-v2"),
                          top_k=top_k, max_candidate_triples=int(values.get("max_candidate_triples", 20)),
                          encoder_device=values.get("encoder_device"))
    services.retriever = RouterWithAll(router, top_k)  # adds the merged corpus "all" for free questions


def load_proxy(services: DemoServices) -> None:
    values = services.config["proxy"]
    usable = {}
    for name, entry in values.get("adapters", {}).items():
        if Path(entry["path"], "adapter_config.json").is_file():
            usable[name] = entry
        else:
            services.adapter_errors[name] = f"missing adapter_config.json in {entry['path']}"
    if not usable:
        raise FileNotFoundError("; ".join(services.adapter_errors.values()) or "no proxy adapters configured")
    backend = values.get("backend", "hf")
    if backend == "hf":
        services.proxy = HFProxy(usable)
    elif backend == "vllm":
        key = os.environ.get(values.get("api_key_env", "READER_API_KEY")) or "EMPTY"
        services.proxy = VLLMProxy(values["url"], key, usable)
    else:
        raise ValueError(f"unknown proxy backend {backend!r} (hf or vllm)")


def load_strong(services: DemoServices) -> None:
    from openai import OpenAI

    from grasp_grpo.baselines import BEDROCK_URL, ChatReader

    key = os.environ.get("BEDROCK_API_KEY")
    if not key:
        raise RuntimeError("BEDROCK_API_KEY is not set")
    values = services.config["strong"]
    client = OpenAI(base_url=values.get("url", BEDROCK_URL), api_key=key, timeout=float(values.get("timeout", 60)), max_retries=int(values.get("retries", 1)))
    client.models.list()  # probe: GET /models checks the key and the endpoint
    services.strong_reader = ChatReader(client, values["model"])


LOADERS: dict[str, Callable[[Any], None]] = {"reader": load_reader, "retriever": load_retriever, "proxy": load_proxy, "strong": load_strong}


class DemoServices:
    def __init__(self, config: dict[str, Any], *, loaders: dict[str, Callable[[Any], None]] | None = None):
        self.config, self.loaders = config, LOADERS if loaders is None else loaders  # {} = load nothing (UI checks)
        self.reader = self.strong_reader = self.retriever = self.proxy = None
        self.retriever_lock = threading.Lock()
        self.health: dict[str, dict[str, Any]] = {}
        self.adapter_errors: dict[str, str] = {}
        adapters = (config.get("proxy") or {}).get("adapters") or {}
        self.adapter_metadata = {name: {"label": entry["label"], "path": entry["path"]} for name, entry in adapters.items()}

    @classmethod
    def from_config(cls, config: dict[str, Any], *, loaders: dict[str, Callable[[Any], None]] | None = None) -> DemoServices:
        services = cls(config, loaders=loaders)
        services.load()
        return services

    def load(self) -> None:
        for name, loader in self.loaders.items():
            try:
                loader(self)
                self.health[name] = {"ready": True}
            except Exception as error:  # a presentation server must survive one unavailable component
                self.health[name] = {"ready": False, "reason": f"{type(error).__name__}: {error}"}

    def enabled(self, column: str) -> tuple[bool, str | None]:
        missing = [item for item in NEEDS[column] if not self.health.get(item, {}).get("ready")]
        if missing:
            return False, "; ".join(self.health.get(item, {}).get("reason", f"{item} not loaded") for item in missing)
        if column.startswith("grasp_"):
            adapter = column.removeprefix("grasp_")
            if adapter not in self.proxy.names:
                return False, self.adapter_errors.get(adapter, f"adapter {adapter} is not loaded")
        return True, None

    def close(self) -> None:
        if self.proxy is not None:
            self.proxy.close()
        if self.retriever is not None:
            self.retriever.close()
