"""Where the 3B proxy runs: transformers in this process (hf) or a vLLM policy server (vllm).

Both expose `names`, `generate_fn(adapter)` (greedy, the trainer's role generate function) and `close()`.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

PREFIX = "demo-"


class HFProxy:
    """One base model with every adapter: the first adapter is PEFT's `default`, the others keep their names.
    One lock: one column generates at a time."""

    def __init__(self, adapters: dict[str, dict[str, Any]], *, load: Callable | None = None, generate: Callable | None = None):
        if load is None:
            from grasp_grpo.model import load_policy as load
        if generate is None:
            from grasp_grpo.model import generate
        self._generate = generate
        items = list(adapters.items())
        first, entry = items[0]
        self.model, self.tokenizer, _ = load(entry["path"], trainable=False)
        self.names = {first: "default"}
        for name, entry in items[1:]:
            self.model.load_adapter(entry["path"], adapter_name=name, is_trainable=False)
            self.names[name] = name
        self.lock = threading.Lock()

    def generate_fn(self, adapter: str):
        from grasp_grpo.batcher import role_generate_fn
        from grasp_grpo.model import active_adapter

        peft_name = self.names[adapter]

        def raw(prompt: str, **kwargs):
            with self.lock, active_adapter(self.model, peft_name):
                return self._generate(self.model, self.tokenizer, prompt, **kwargs)

        return role_generate_fn(raw, temperature=0.0, top_p=1.0)

    def close(self) -> None:
        return None


class VLLMProxy:
    """Adapters loaded on a running vLLM policy server as demo-<name>; unloaded on close.
    The adapter paths must exist on the server's machine (the demo runs there in vast mode)."""

    def __init__(self, url: str, api_key: str, adapters: dict[str, dict[str, Any]], *, admin=None, policy_factory: Callable | None = None):
        if admin is None:
            from grasp_grpo.policy_server import _Admin

            admin = _Admin(url, api_key)
        if policy_factory is None:
            def policy_factory(served: str, path: str):
                from grasp_grpo.model import load_tokenizer
                from grasp_grpo.policy_server import VLLMPolicy

                return VLLMPolicy(url, api_key, load_tokenizer(path), model=served, timeout=60.0)
        self.admin, self.names, self.policies = admin, {}, {}
        for name, entry in adapters.items():
            served = f"{PREFIX}{name}"
            self._load(served, entry["path"])
            self.names[name] = served
            self.policies[name] = policy_factory(served, entry["path"])

    def _load(self, served: str, path: str) -> None:
        payload = {"lora_name": served, "lora_path": path}
        try:
            self.admin.post("load_lora_adapter", payload)
        except RuntimeError:  # a previous demo server left the name loaded
            self.admin.post("unload_lora_adapter", {"lora_name": served})
            self.admin.post("load_lora_adapter", payload)

    def generate_fn(self, adapter: str):
        from grasp_grpo.batcher import role_generate_fn

        return role_generate_fn(self.policies[adapter].generate, temperature=0.0, top_p=1.0)

    def close(self) -> None:
        for served in self.names.values():
            try:
                self.admin.post("unload_lora_adapter", {"lora_name": served})
            except Exception:  # noqa: BLE001 - shutting down; the server may already be gone
                pass
