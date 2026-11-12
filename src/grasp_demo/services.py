"""One-process shared models used by all demo columns."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProxyPolicy:
    model: Any
    tokenizer: Any
    names: dict[str, str]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def generate(self, adapter: str):
        from grasp_grpo.batcher import role_generate_fn
        from grasp_grpo.model import active_adapter, generate

        if adapter not in self.names:
            raise RuntimeError(f"unknown proxy adapter: {adapter}")

        def raw(prompt: str, **kwargs):
            with self.lock, active_adapter(self.model, self.names[adapter]):
                return generate(self.model, self.tokenizer, prompt, **kwargs)

        return role_generate_fn(raw, temperature=0.0, top_p=1.0)


class DemoServices:
    """Load expensive resources once. Failed optional resources disable only their columns."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.reader = None
        self.strong_reader = None
        self.retriever = None
        self.proxy = None
        self.retriever_lock = threading.Lock()
        self.health: dict[str, dict[str, Any]] = {}
        self.adapter_metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DemoServices":
        services = cls(config)
        services._load()
        return services

    def _component(self, name: str, loader) -> None:
        try:
            loader()
            self.health[name] = {"ready": True}
        except Exception as error:  # a presentation server must survive one unavailable column
            self.health[name] = {"ready": False, "reason": f"{type(error).__name__}: {error}"}

    def _load(self) -> None:
        def reader() -> None:
            from grasp_grpo.reader import GraspReader, chat_renderer, openai_completion

            values = self.config["reader"]
            self.reader = GraspReader(
                openai_completion(values["url"], values.get("api_key", "EMPTY"), values["model"], temperature=0.0),
                chat_renderer(values.get("tokenizer") or values["model"]),
            )

        def retriever() -> None:
            from grasp_grpo.retrieval import CorpusRouter

            values = self.config["retrieval"]
            self.retriever = CorpusRouter(
                values["corpus_config"], values.get("encoder", "intfloat/e5-large-v2"),
                top_k=int(values.get("top_k", 10)),
                max_candidate_triples=int(values.get("max_candidate_triples", 20)),
                encoder_device=values.get("encoder_device"),
            )

        def proxy() -> None:
            from grasp_grpo.model import load_policy

            adapters = self.config["proxy"]["adapters"]
            sft = adapters["sft"]
            if not Path(sft, "adapter_config.json").is_file():
                raise FileNotFoundError(f"missing SFT adapter: {sft}")
            model, tokenizer, _ = load_policy(sft, trainable=False)
            names = {"sft": "default"}
            self.adapter_metadata["sft"] = _adapter_metadata(sft)
            grpo = adapters.get("grpo")
            if grpo and Path(grpo, "adapter_config.json").is_file():
                model.load_adapter(grpo, adapter_name="grpo", is_trainable=False)
                names["grpo"] = "grpo"
                self.adapter_metadata["grpo"] = _adapter_metadata(grpo)
            self.proxy = ProxyPolicy(model, tokenizer, names)

        def strong() -> None:
            from grasp_grpo.baselines import BEDROCK_URL, ChatReader
            from openai import OpenAI
            import os

            key = os.environ.get("BEDROCK_API_KEY")
            if not key:
                raise RuntimeError("BEDROCK_API_KEY is not set")
            values = self.config["strong"]
            self.strong_reader = ChatReader(OpenAI(base_url=values.get("url", BEDROCK_URL), api_key=key, timeout=60, max_retries=3), values["model"])

        self._component("reader", reader)
        self._component("retriever", retriever)
        self._component("proxy", proxy)
        self._component("strong", strong)

    def enabled(self, column: str) -> tuple[bool, str | None]:
        needed = {
            "rag_7b": ("reader", "retriever"),
            "rag_235b": ("strong", "retriever"),
            "ircot_7b": ("reader", "retriever"),
            "grasp_sft": ("reader", "retriever", "proxy"),
            "grasp_grpo": ("reader", "retriever", "proxy"),
        }[column]
        missing = [item for item in needed if not self.health.get(item, {}).get("ready")]
        if missing:
            return False, "; ".join(self.health[item].get("reason", item) for item in missing)
        if column == "grasp_grpo" and "grpo" not in self.proxy.names:
            return False, "GRPO adapter is not available"
        return True, None

    def close(self) -> None:
        if self.retriever is not None:
            self.retriever.close()


def _adapter_metadata(path: str) -> dict[str, Any]:
    """Small, optional labels for the UI; malformed metrics never block a model."""
    root = Path(path)
    result: dict[str, Any] = {"path": str(root), "label": root.name}
    config = root / "grpo_config.json"
    if not config.is_file():
        return result
    try:
        values = json.loads(config.read_text(encoding="utf-8"))
        result.update({key: values[key] for key in ("step", "f1", "checkpoint") if key in values})
    except (OSError, ValueError, TypeError):
        pass
    return result
