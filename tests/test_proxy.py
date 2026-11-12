import pytest
from grasp_grpo.model import Action

from grasp_demo.proxy import HFProxy, VLLMProxy

ADAPTERS = {"sft": {"path": "/ck/sft", "label": "SFT"}, "grpo": {"path": "/ck/s100", "label": "GRPO"}}


class FakeModel:
    def __init__(self):
        self.active_adapter, self.loaded, self.seen = "default", [], []

    def load_adapter(self, path, adapter_name, is_trainable):
        self.loaded.append((path, adapter_name, is_trainable))

    def set_adapter(self, name):
        self.active_adapter = name

    def parameters(self):
        return []


def test_hf_proxy_loads_the_first_adapter_as_base_and_switches():
    model = FakeModel()

    def generate(m, tokenizer, prompt, **kwargs):
        m.seen.append((m.active_adapter, kwargs["temperature"], kwargs["samples"]))
        return [Action("Yes", [], [])]

    proxy = HFProxy(ADAPTERS, load=lambda path, trainable: (model, "tok", "base"), generate=generate)
    assert proxy.names == {"sft": "default", "grpo": "grpo"}
    assert model.loaded == [("/ck/s100", "grpo", False)]
    proxy.generate_fn("grpo")("rd", "prompt", 1)
    proxy.generate_fn("sft")("rd", "prompt", 1)
    assert model.seen == [("grpo", 0.0, 1), ("default", 0.0, 1)]
    assert model.active_adapter == "default"


class FakeAdmin:
    def __init__(self, fail=None, fail_once=None):
        self.calls = []
        self.fail = fail or {}            # (endpoint, lora_name) -> message, raised on every such call
        self.fail_once = fail_once or {}  # same, raised only on the first such call

    def post(self, endpoint, payload):
        key = (endpoint, payload["lora_name"])
        self.calls.append(key)
        if key in self.fail_once:
            raise RuntimeError(self.fail_once.pop(key))
        if key in self.fail:
            raise RuntimeError(self.fail[key])


class FakePolicy:
    def __init__(self, served):
        self.served = served

    def generate(self, prompt, *, samples, max_new_tokens, temperature, top_p):
        return [Action(f"{self.served}:{temperature}", [], [])]


def test_vllm_proxy_loads_named_adapters_and_unloads_on_close():
    admin = FakeAdmin()
    proxy = VLLMProxy("http://x/v1", "key", ADAPTERS, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert proxy.names == {"sft": "demo-sft", "grpo": "demo-grpo"}
    assert proxy.generate_fn("grpo")("qf", "p", 1)[0].text == "demo-grpo:0.0"
    proxy.close()
    assert admin.calls == [("load_lora_adapter", "demo-sft"), ("load_lora_adapter", "demo-grpo"),
                           ("unload_lora_adapter", "demo-sft"), ("unload_lora_adapter", "demo-grpo")]


def test_vllm_proxy_reloads_a_name_left_by_an_old_server():
    admin = FakeAdmin(fail_once={("load_lora_adapter", "demo-sft"): "already loaded"})
    VLLMProxy("http://x/v1", "key", {"sft": ADAPTERS["sft"]}, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert admin.calls == [("load_lora_adapter", "demo-sft"), ("unload_lora_adapter", "demo-sft"), ("load_lora_adapter", "demo-sft")]


def test_vllm_proxy_reload_failure_surfaces_the_original_load_error():
    admin = FakeAdmin(fail_once={("load_lora_adapter", "demo-sft"): "bad path"},
                      fail={("unload_lora_adapter", "demo-sft"): "not found"})
    with pytest.raises(RuntimeError, match="bad path") as info:
        VLLMProxy("http://x/v1", "key", {"sft": ADAPTERS["sft"]}, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert "not found" not in str(info.value)
    assert str(info.value.__cause__) == "bad path"


def test_vllm_proxy_partial_failure_unloads_what_was_loaded():
    admin = FakeAdmin(fail={("load_lora_adapter", "demo-grpo"): "out of memory",
                            ("unload_lora_adapter", "demo-grpo"): "not found"})
    with pytest.raises(RuntimeError, match="out of memory"):
        VLLMProxy("http://x/v1", "key", ADAPTERS, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert ("unload_lora_adapter", "demo-sft") in admin.calls


def test_vllm_proxy_policy_failure_unloads_the_adapter_just_loaded():
    def factory(served, path):
        if served == "demo-grpo":
            raise RuntimeError("policy init failed")
        return FakePolicy(served)

    admin = FakeAdmin()
    with pytest.raises(RuntimeError, match="policy init failed"):
        VLLMProxy("http://x/v1", "key", ADAPTERS, admin=admin, policy_factory=factory)
    assert admin.calls == [("load_lora_adapter", "demo-sft"), ("load_lora_adapter", "demo-grpo"),
                           ("unload_lora_adapter", "demo-sft"), ("unload_lora_adapter", "demo-grpo")]
